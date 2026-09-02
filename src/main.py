"""Entry point. A new variant is one class plus one dict entry below."""
import argparse
import csv
import os
import random

import numpy as np
import torch

from client import Client
from data import (by_superclass, by_target_h, dirichlet, iid, label_heterogeneity,
                  load_cifar100)
from models import resnet18, small_cnn
from projected import ProjectedClient, ProjectedServer
from server import Server

PARTITIONS = {"iid": iid, "dirichlet": dirichlet, "by_superclass": by_superclass,
              "target_h": by_target_h}
SERVERS = {"fedavg": Server}
MODELS = {"small_cnn": small_cnn, "resnet18": resnet18}


def parse_args():
    p = argparse.ArgumentParser(description="Mini-batch SGD vs Local SGD on CIFAR-100")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--partition", choices=PARTITIONS, default="iid")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet skew; lower is more skewed")
    p.add_argument("--target-h", type=float, default=0.5,
                   help="for --partition target_h: the H_label to generate, in [0, 1]")
    p.add_argument("--local-steps", type=int, default=1, help="1 is mini-batch SGD; >1 is Local SGD")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", choices=MODELS, default="small_cnn")
    p.add_argument("--server", choices=SERVERS, default="fedavg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-frac", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--rho", type=float, default=None,
                   help="trust-region ball: local iterates stay within "
                        "rho*||g_t||*lr*local_steps of the synced model. "
                        "Omit (or pass inf) for ordinary Local SGD.")
    p.add_argument("--grad-batch", type=int, default=2048,
                   help="minibatch used to estimate ||g_t|| for the ball radius")
    p.add_argument("--exact-cos-every", type=int, default=0,
                   help="every Nth logged round, also compute cosine against the exact "
                        "full-batch gradient (costs one pass over the training set); 0 disables")
    p.add_argument("--drift-every", type=int, default=0,
                   help="measure client-gradient drift every N rounds; 0 disables it. "
                        "Each measurement costs one full pass over the training set.")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default=None, help="CSV path; defaults to a name built from the flags")
    return p.parse_args()


def default_out(a):
    part = {"dirichlet": f"dirichlet{a.alpha}", "target_h": f"targeth{a.target_h}"}.get(
        a.partition, a.partition)
    if a.rho is not None:
        part += f"_rho{a.rho}"
    return f"runs/{a.model}_{part}_n{a.clients}_K{a.local_steps}_lr{a.lr}_s{a.seed}.csv"


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, pool, coarse = load_cifar100(a.data_root, a.data_frac, a.seed)
    kw = {"dirichlet": {"alpha": a.alpha}, "target_h": {"h": a.target_h}}.get(a.partition, {})
    shards = PARTITIONS[a.partition](coarse, pool, a.clients, rng, **kw)

    empty = [i for i, s in enumerate(shards) if len(s) == 0]
    if empty:
        print(f"warning: dropping {len(empty)} client(s) with empty shards (alpha={a.alpha})")
        shards = [s for s in shards if len(s) > 0]
    if not shards:
        raise SystemExit("every shard is empty -- raise --alpha or lower --clients")

    # rho=inf is ordinary Local SGD -- the ball is never binding, so take the plain path
    ball = a.rho is not None and a.rho != float("inf")
    cls = ProjectedClient if ball else Client
    clients = [cls(dataset, s, MODELS[a.model], a.batch_size, a.lr, device) for s in shards]
    if ball:
        server = ProjectedServer(MODELS[a.model](), clients, dataset, pool, device,
                                 rho=a.rho, grad_batch=a.grad_batch)
    else:
        server = SERVERS[a.server](MODELS[a.model](), clients, dataset, pool, device)
    h_label = label_heterogeneity(coarse, shards)

    out = a.out or default_out(a)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"{len(clients)} clients on {device}; shard sizes {[c.n for c in clients]}")
    print(f"H_label = {h_label:.4f} (measured over the 20 coarse superclasses) -> {out}")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        # h_label is constant per run; carried as a column so a plot needs only the CSVs.
        BALL_COLS = ["radius", "ball_hits", "mean_hit_step", "cos_client", "cos_agg",
                     "cos_client_exact"]
        w.writerow(["round", "grad_computations", "train_loss", "drift", "grad_norm",
                    "h_label"] + BALL_COLS)

        def fmt(v):
            return "" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))

        def log(r, grads, loss, drift=None, gnorm=None, extra=None):
            extra = extra or {}
            w.writerow([r, grads, f"{loss:.6f}",
                        "" if drift is None else f"{drift:.6e}",
                        "" if gnorm is None else f"{gnorm:.6e}",
                        f"{h_label:.6f}"] + [fmt(extra.get(c)) for c in BALL_COLS])
            f.flush()
            tail = "" if drift is None else f"  drift {drift:.4e}  |g| {gnorm:.4e}"
            if extra:
                tail += (f"  hits {extra['ball_hits']:>2}"
                         f"  cos {extra['cos_client']:+.3f}/{extra['cos_agg']:+.3f}")
            print(f"round {r:4d}  grads {grads:>10d}  loss {loss:.4f}{tail}", flush=True)

        if ball:
            server.run(a.rounds, a.local_steps, a.eval_every, log, a.drift_every,
                       a.exact_cos_every)
        else:
            server.run(a.rounds, a.local_steps, a.eval_every, log, a.drift_every)


if __name__ == "__main__":
    main()
