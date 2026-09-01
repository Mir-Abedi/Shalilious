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

    def run(self, rounds, local_steps, eval_every, on_log):
        grads = 0
        on_log(0, 0, self.evaluate())
        for r in range(1, rounds + 1):
            state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            states, weights = [], []
            for c in self.clients:
                s, g = c.local_steps(state, local_steps)
                states.append(s)
                weights.append(c.n)
                grads += g
            self.model.load_state_dict(self.aggregate(states, weights))
            if r % eval_every == 0 or r == rounds:
                on_log(r, grads, self.evaluate())
