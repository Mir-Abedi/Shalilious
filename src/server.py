"""The coordinator. The server rule lives in aggregate -- override it to change it."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class Server:
    def __init__(self, model, clients, dataset, pool, device, batch_size=512):
        self.model = model.to(device)
        self.clients = clients
        self.device = device
        self.eval_loader = DataLoader(
            Subset(dataset, list(pool)), batch_size=batch_size, shuffle=False
        )

    def aggregate(self, states, weights):
        """Shard-size-weighted parameter mean (FedAvg)."""
        total = float(sum(weights))
        out = {}
        for k, v0 in states[0].items():
            if v0.is_floating_point():
                acc = torch.zeros_like(v0, dtype=torch.float64)
                for s, w in zip(states, weights):
                    acc += s[k].to(torch.float64) * (w / total)
                out[k] = acc.to(v0.dtype)
            else:
                out[k] = v0.clone()  # e.g. BN num_batches_tracked -- averaging is meaningless
        return out

    @torch.no_grad()
    def evaluate(self):
        """Mean training loss of the global model over the whole pool. The primary metric."""
        self.model.eval()
        total, n = 0.0, 0
        for x, y in self.eval_loader:
            x, y = x.to(self.device), y.to(self.device)
            total += F.cross_entropy(self.model(x), y, reduction="sum").item()
            n += y.numel()
        return total / n

    def measure_drift(self, state):
        """Client-gradient drift at `state`: (mean_i ||g_i - g||, ||g||).

        g_i is client i's exact full-shard gradient; g is the exact gradient of the global
        objective. g = sum_i (n_i/N) g_i is an IDENTITY, not an approximation -- the shards
        are a disjoint cover of the pool, so the global loss is the shard-size-weighted
        mean of the client losses and gradients pass straight through. Measuring the g_i
        therefore yields g for free.

        Every client is evaluated at the SAME point, which is what makes the spread
        attributable to the data partition rather than to where each client wandered.

        This is the bounded-gradient-dissimilarity constant from the Local SGD bounds, and
        it does not depend on the number of local steps: K decides which points the run
        visits, not what is measured at them.
        """
        # ponytail: holds one full gradient per client in memory (10 x 1.1M floats for
        # small_cnn). Chunk it or recompute in two passes if this ever runs resnet18 x many.
        gs = [c.full_gradient(state) for c in self.clients]
        w = torch.tensor([float(c.n) for c in self.clients], device=gs[0].device)
        w /= w.sum()
        g = torch.zeros_like(gs[0])
        for gi, wi in zip(gs, w):
            g += gi * wi
        drift = torch.stack([(gi - g).norm() for gi in gs]).mean()
        return float(drift), float(g.norm())

    def run(self, rounds, local_steps, eval_every, on_log, drift_every=0):
        """drift_every=0 disables drift measurement; each measurement costs one full pass
        over the training set, so it is deliberately not on by default."""
        grads = 0
        init = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        on_log(0, 0, self.evaluate(), *(self.measure_drift(init) if drift_every else (None, None)))
        for r in range(1, rounds + 1):
            state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            # Measured at the broadcast point, before any local step has been taken.
            measured = bool(drift_every) and r % drift_every == 0
            drift = self.measure_drift(state) if measured else (None, None)
            states, weights = [], []
            for c in self.clients:
                s, g = c.local_steps(state, local_steps)
                states.append(s)
                weights.append(c.n)
                grads += g
            self.model.load_state_dict(self.aggregate(states, weights))
            if r % eval_every == 0 or measured or r == rounds:
                on_log(r, grads, self.evaluate(), *drift)
