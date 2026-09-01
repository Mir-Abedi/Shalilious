"""One simulated client. The client rule lives in local_steps -- override it to change it."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class Client:
    def __init__(self, dataset, indices, model_fn, batch_size, lr, device):
        self.n = len(indices)
        self.lr = lr
        self.device = device
        self.model = model_fn().to(device)
        self.loader = DataLoader(
            Subset(dataset, list(indices)),
            batch_size=min(batch_size, self.n),
            shuffle=True,
            drop_last=False,
        )
        self._it = iter(self.loader)
        # Separate loader for drift measurement: no shuffling, so the accumulation order
        # (and thus the float rounding) is identical every time it is called.
        self.grad_loader = DataLoader(
            Subset(dataset, list(indices)), batch_size=min(512, self.n), shuffle=False
        )

    def _next_batch(self):
        """Endless stream over this client's shard: a small shard just gets revisited more."""
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)

    def local_steps(self, state, n_steps):
        """Run n_steps of SGD from `state`. Returns (new_state, per-sample gradient count).

        Fresh optimizer each round: plain SGD carries no state, and momentum would be
        a subclass, not a flag.
        """
        self.model.load_state_dict(state)
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        grads = 0
        for _ in range(n_steps):
            x, y = self._next_batch()
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            F.cross_entropy(self.model(x), y).backward()
            opt.step()
            grads += y.numel()
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}, grads

    def full_gradient(self, state):
        """Exact gradient of this client's mean loss over its WHOLE shard, at `state`.

        Full-shard, not mini-batch: a sampled estimate carries noise that would put a
        floor under the drift measurement even when clients genuinely agree. Accumulating
        `reduction="sum"` and dividing once at the end is the mean-loss gradient exactly.

        eval() mode so dropout and BatchNorm batch statistics stay out of a measurement
        that is supposed to depend only on the data and the weights.
        """
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        n = 0
        for x, y in self.grad_loader:
            x, y = x.to(self.device), y.to(self.device)
            F.cross_entropy(self.model(x), y, reduction="sum").backward()
            n += y.numel()
        flat = [
            (p.grad / n).reshape(-1) if p.grad is not None else torch.zeros(p.numel())
            for p in self.model.parameters()
        ]
        return torch.cat(flat).detach()
