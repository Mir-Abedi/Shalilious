"""CIFAR-100 loading and client partitioning."""
import os
import pickle

import numpy as np
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
N_COARSE = 20


def coarse_labels(root):
    """The 20 superclass labels, read from the archive torchvision already downloaded.

    torchvision's CIFAR100 exposes only fine labels, but the raw pickle carries both.
    """
    path = os.path.join(root, "cifar-100-python", "train")
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    return np.array(entry["coarse_labels"], dtype=np.int64)


def load_cifar100(root="./data", data_frac=1.0, seed=0):
    """Return (dataset, pool, coarse). No augmentation: this is an optimization study."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.CIFAR100(root, train=True, download=True, transform=tf)
    coarse = coarse_labels(root)
    fine = np.array(ds.targets, dtype=np.int64)
    pool = np.arange(len(fine), dtype=np.int64)
    if data_frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = []
        for c in np.unique(fine):  # stratify so every fine label survives
            idx = pool[fine == c]
            rng.shuffle(idx)
            keep.append(idx[: max(1, int(round(len(idx) * data_frac)))])
        pool = np.sort(np.concatenate(keep))
    return ds, pool, coarse


def iid(coarse, pool, n_clients, rng):
    """Shuffle the pool and cut it into near-equal shards. `coarse` is unused by design."""
    idx = np.array(pool, dtype=np.int64).copy()
    rng.shuffle(idx)
    return [chunk.tolist() for chunk in np.array_split(idx, n_clients)]


def dirichlet(coarse, pool, n_clients, rng, alpha=0.5):
    """Label-skew partition: each superclass is split over clients by p ~ Dir(alpha).

    alpha -> 0 hands each superclass to a single client; large alpha approaches IID.
    Shard sizes come out unequal by construction -- aggregation weights by shard size.
    """
    pool = np.array(pool, dtype=np.int64)
    shards = [[] for _ in range(n_clients)]
    for c in range(N_COARSE):
        idx = pool[coarse[pool] == c]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        p = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for i, part in enumerate(np.split(idx, cuts)):
            shards[i].extend(part.tolist())
    return shards
