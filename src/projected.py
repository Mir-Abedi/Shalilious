"""Local SGD with a trust-region ball around the last synchronized model.

Each round the server sets a radius

    R_t = rho * ||g_t|| * eta * K

and a client's local iterates are confined to B(w_t, R_t). A step that leaves the
ball is projected back onto its surface, and the client then stops until the next
aggregation -- so rho bounds how far a client may travel between syncs, which is
the quantity the Local SGD drift bounds actually care about.

Nothing here modifies Client or Server: rho unset takes the ordinary code path.
"""
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset

from client import Client
from server import Server


class ProjectedClient(Client):
    """A Client whose local iterates stay inside a ball around the incoming model."""

    def local_steps(self, state, n_steps, radius=None):
        """radius=None is plain Local SGD -- it delegates, so the control arm is not
        a second implementation that could drift from the real one.

        Sets `hit_step` (the local step at which the ball was hit, else None) and
        `last_delta` (the round's total displacement, flattened) for the server.
        """
        self.model.load_state_dict(state)
        center = parameters_to_vector(self.model.parameters()).detach().clone()
        self.hit_step = None

        if radius is None:
            out, grads = super().local_steps(state, n_steps)
        else:
            self.model.train()
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            grads = 0
            for k in range(n_steps):
                x, y = self._next_batch()
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                F.cross_entropy(self.model(x), y).backward()
                opt.step()
                grads += y.numel()
                v = parameters_to_vector(self.model.parameters()).detach() - center
                dist = v.norm()
                if dist > radius:
                    # land exactly where the step crosses the sphere, then freeze
                    vector_to_parameters(center + v * (radius / dist), self.model.parameters())
                    self.hit_step = k + 1
                    break
            out = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        self.last_delta = parameters_to_vector(self.model.parameters()).detach() - center
        return out, grads


class ProjectedServer(Server):
    """Server that sizes the ball each round and logs how the clients behaved in it."""

    def __init__(self, model, clients, dataset, pool, device, batch_size=512,
                 rho=None, grad_batch=2048):
        super().__init__(model, clients, dataset, pool, device, batch_size)
        self.rho = rho
        self.lr = clients[0].lr
        self.grad_loader = DataLoader(Subset(dataset, list(pool)), batch_size=grad_batch,
                                      shuffle=True, drop_last=True)
        self._git = iter(self.grad_loader)

    def _grad_estimate(self):
        """One large-minibatch gradient at the current global model.

        An unbiased estimate of g_t. The exact gradient would cost a full pass over the
        training set every round -- ~40x the whole training budget at K=1 -- and the
        radius only needs the scale of ||g_t||.
        """
        try:
            x, y = next(self._git)
        except StopIteration:
            self._git = iter(self.grad_loader)
            x, y = next(self._git)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        F.cross_entropy(self.model(x.to(self.device)), y.to(self.device)).backward()
        g = torch.cat([p.grad.reshape(-1) for p in self.model.parameters()]).detach().clone()
        self.model.zero_grad(set_to_none=True)
        return g

    def _exact_grad(self, state):
        """The true full-batch gradient at `state`, as the weighted mean of the clients'
        exact shard gradients (an identity -- the shards are a disjoint cover)."""
        gs = [c.full_gradient(state) for c in self.clients]
        w = torch.tensor([float(c.n) for c in self.clients], device=gs[0].device)
        w /= w.sum()
        g = torch.zeros_like(gs[0])
        for gi, wi in zip(gs, w):
            g += gi * wi
        return g

    @staticmethod
    def _cos(a, b):
        na, nb = a.norm(), b.norm()
        if na == 0 or nb == 0:
            return float("nan")
        return float(torch.dot(a, b) / (na * nb))

    def run(self, rounds, local_steps, eval_every, on_log, drift_every=0, exact_cos_every=0):
        grads = 0
        on_log(0, 0, self.evaluate(), None, None, {})
        for r in range(1, rounds + 1):
            state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            center = parameters_to_vector(self.model.parameters()).detach().clone()

            ghat = self._grad_estimate()
            radius = None if self.rho is None else self.rho * float(ghat.norm()) * self.lr * local_steps

            logging = (r % eval_every == 0) or r == rounds
            exact = self._exact_grad(state) if (exact_cos_every and logging
                                                and r % (eval_every * exact_cos_every) == 0) else None

            states, weights, hit_steps, cos_c, cos_c_exact = [], [], [], [], []
            for c in self.clients:
                s, g = c.local_steps(state, local_steps, radius)
                states.append(s)
                weights.append(c.n)
                grads += g
                if getattr(c, "hit_step", None) is not None:
                    hit_steps.append(c.hit_step)
                cos_c.append(self._cos(c.last_delta, -ghat))
                if exact is not None:
                    cos_c_exact.append(self._cos(c.last_delta, -exact))

            self.model.load_state_dict(self.aggregate(states, weights))
            delta_agg = parameters_to_vector(self.model.parameters()).detach() - center

            if logging:
                extra = {
                    "radius": radius,
                    "ball_hits": len(hit_steps),
                    "mean_hit_step": (sum(hit_steps) / len(hit_steps)) if hit_steps else None,
                    "cos_client": sum(cos_c) / len(cos_c),
                    "cos_agg": self._cos(delta_agg, -ghat),
                    "cos_client_exact": (sum(cos_c_exact) / len(cos_c_exact)) if cos_c_exact else None,
                }
                d = self.measure_drift(state) if drift_every and r % drift_every == 0 else (None, None)
                on_log(r, grads, self.evaluate(), d[0], d[1], extra)
