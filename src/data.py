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


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def label_heterogeneity(labels, shards):
    """Normalized label skew of a partition: H_label = I(C;Y)/H(Y) = 1 - H(Y|C)/H(Y).

    C is "which client did this example come from", Y is its label. 0 means every client
    has the same label distribution (knowing the client tells you nothing about the
    label); 1 means the client determines the label outright.

    Pass the COARSE superclasses as `labels` -- that is what the skew is drawn over, and
    it spans the full [0, 1]. Over the 100 fine labels the same partition gives the same
    curve scaled by log(20)/log(100).
    """
    labels = np.asarray(labels)
    idx = [np.asarray(s, dtype=np.int64) for s in shards if len(s) > 0]
    if not idx:
        raise ValueError("every shard is empty")
    all_idx = np.concatenate(idx)
    n_lab = int(labels.max()) + 1
    h_y = _entropy(np.bincount(labels[all_idx], minlength=n_lab) / len(all_idx))
    if h_y == 0:
        return 0.0  # one label everywhere: no information for the client to carry
    h_y_given_c = sum(
        (len(s) / len(all_idx)) * _entropy(np.bincount(labels[s], minlength=n_lab) / len(s))
        for s in idx
    )
    return float(1.0 - h_y_given_c / h_y)


def by_superclass(coarse, pool, n_clients, rng):
    """Deterministic pathological split: superclasses dealt out in contiguous blocks.

    With n_clients == 20 every client holds exactly one superclass, so H_label == 1 --
    the top of the range, which Dirichlet cannot reach at any alpha because it never
    produces a clean permutation by chance (alpha=0.003 over 20 clients still only
    averages 0.80, and starts emptying shards). `rng` is unused: this partition is fixed.
    """
    pool = np.array(pool, dtype=np.int64)
    groups = np.array_split(np.arange(N_COARSE), n_clients)
    return [pool[np.isin(coarse[pool], g)].tolist() for g in groups]
