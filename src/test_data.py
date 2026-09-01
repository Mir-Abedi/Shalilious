"""Self-checks for the data partitioners. Run: python src/test_data.py"""
import numpy as np

from data import dirichlet, iid


def _fake(n=2000, n_coarse=20, seed=0):
    """A stand-in for CIFAR-100's coarse labels: n images over n_coarse superclasses."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, n_coarse, size=n)
    pool = np.arange(n)
    return coarse, pool


def _entropy(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mean_client_entropy(shards, coarse, n_coarse=20):
    """Average superclass entropy within a client. Low = that client sees few superclasses."""
    out = []
    for s in shards:
        if len(s) == 0:
            continue
        counts = np.bincount(coarse[np.array(s)], minlength=n_coarse)
        out.append(_entropy(counts))
    return float(np.mean(out))


def test_partitions_are_disjoint_covers():
    coarse, pool = _fake()
    for name, shards in [
        ("iid", iid(coarse, pool, 5, np.random.default_rng(0))),
        ("dir", dirichlet(coarse, pool, 5, np.random.default_rng(0), alpha=0.5)),
    ]:
        assert len(shards) == 5, name
        flat = np.concatenate([np.array(s, dtype=np.int64) for s in shards])
        assert len(flat) == len(pool), f"{name}: {len(flat)} != {len(pool)}"
        assert len(np.unique(flat)) == len(pool), f"{name}: shards overlap"
        assert set(flat.tolist()) == set(pool.tolist()), f"{name}: not a cover"


def test_low_alpha_concentrates_superclasses():
    coarse, pool = _fake()
    skewed = dirichlet(coarse, pool, 5, np.random.default_rng(0), alpha=0.01)
    uniform = dirichlet(coarse, pool, 5, np.random.default_rng(0), alpha=100.0)
    lo = _mean_client_entropy(skewed, coarse)
    hi = _mean_client_entropy(uniform, coarse)
    assert lo < hi - 0.5, f"alpha=0.01 entropy {lo:.3f} not well below alpha=100 {hi:.3f}"


def test_iid_shards_are_near_equal():
    coarse, pool = _fake()
    sizes = np.array([len(s) for s in iid(coarse, pool, 7, np.random.default_rng(0))])
    assert sizes.max() - sizes.min() <= 1, f"iid sizes uneven: {sizes}"


if __name__ == "__main__":
    test_partitions_are_disjoint_covers()
    test_low_alpha_concentrates_superclasses()
    test_iid_shards_are_near_equal()
    print("ok")
