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


def test_models_output_100_logits():
    import torch

    from models import resnet18, small_cnn

    x = torch.randn(2, 3, 32, 32)
    for name, fn in [("small_cnn", small_cnn), ("resnet18", resnet18)]:
        out = fn()(x)
        assert out.shape == (2, 100), f"{name} gave {tuple(out.shape)}, want (2, 100)"


def test_aggregate_is_a_weighted_mean():
    import torch

    from server import Server

    srv = Server.__new__(Server)  # aggregate needs no constructed state
    a = {"w": torch.tensor([0.0, 10.0]), "n": torch.tensor(7)}
    b = {"w": torch.tensor([4.0, 2.0]), "n": torch.tensor(9)}
    out = srv.aggregate([a, b], [1.0, 3.0])
    assert torch.allclose(out["w"], torch.tensor([3.0, 4.0])), out["w"]
    assert out["n"].item() == 7, "integer buffers must be taken from the first client, not averaged"


def test_one_local_step_matches_plain_sgd():
    """local_steps=1 then averaging IS mini-batch SGD -- the comparison depends on it."""
    import copy

    import torch

    from client import Client
    from models import small_cnn

    torch.manual_seed(0)
    xs = torch.randn(8, 3, 32, 32)
    ys = torch.randint(0, 100, (8,))
    ds = list(zip(xs, ys))

    torch.manual_seed(1)
    ref = small_cnn()
    start = copy.deepcopy(ref.state_dict())
    opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    opt.zero_grad()
    torch.nn.functional.cross_entropy(ref(xs), ys).backward()
    opt.step()

    torch.manual_seed(1)
    c = Client(ds, list(range(8)), small_cnn, batch_size=8, lr=0.1, device="cpu")
    got, grads = c.local_steps(start, 1)
    assert grads == 8, grads
    for k, v in ref.state_dict().items():
        assert torch.allclose(got[k], v, atol=1e-5), f"{k} diverged from plain SGD"


if __name__ == "__main__":
    test_partitions_are_disjoint_covers()
    test_low_alpha_concentrates_superclasses()
    test_iid_shards_are_near_equal()
    test_models_output_100_logits()
    test_aggregate_is_a_weighted_mean()
    test_one_local_step_matches_plain_sgd()
    print("ok")
