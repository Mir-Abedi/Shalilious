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





def test_label_heterogeneity_bounds():
    """H_label = I(C;Y)/H(Y): 0 when clients look alike, 1 when the client names the label."""
    import numpy as np

    from data import label_heterogeneity

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    same = [[0, 1], [2, 3], [4, 5], [6, 7]]  # every client sees the same 50/50 mix
    assert abs(label_heterogeneity(labels, same)) < 1e-12, label_heterogeneity(labels, same)

    split = [[0, 2, 4, 6], [1, 3, 5, 7]]  # each client holds exactly one label
    assert abs(label_heterogeneity(labels, split) - 1.0) < 1e-12

    part = [[0, 1, 2, 4], [3, 5, 6, 7]]  # a partial skew lands strictly between
    h = label_heterogeneity(labels, part)
    assert 0.0 < h < 1.0, h


def _two_client_server(ds, n_each):
    from client import Client
    from models import small_cnn
    from server import Server

    a = Client(ds, list(range(n_each)), small_cnn, 3, 0.1, "cpu")
    b = Client(ds, list(range(n_each, 2 * n_each)), small_cnn, 3, 0.1, "cpu")
    return Server(small_cnn(), [a, b], ds, list(range(2 * n_each)), "cpu")


def test_drift_is_zero_for_identical_clients():
    """Same data on both clients => same exact gradient => drift is 0, not merely small."""
    import torch

    torch.manual_seed(0)
    xs = torch.randn(6, 3, 32, 32)
    ys = torch.randint(0, 100, (6,))
    ds = list(zip(xs, ys)) + list(zip(xs, ys))  # the same 6 examples, twice
    srv = _two_client_server(ds, 6)
    drift, gnorm = srv.measure_drift(srv.model.state_dict())
    assert drift < 1e-5, f"identical clients drifted by {drift}"
    assert gnorm > 0, "global gradient should not be zero at init"


def test_drift_is_positive_for_disagreeing_clients():
    """Positive control: label-skewed clients must register real drift."""
    import torch

    torch.manual_seed(0)
    xs = torch.randn(6, 3, 32, 32)
    ds = list(zip(xs, torch.zeros(6, dtype=torch.long))) + list(
        zip(xs, torch.full((6,), 7, dtype=torch.long))
    )
    srv = _two_client_server(ds, 6)
    drift, _ = srv.measure_drift(srv.model.state_dict())
    assert drift > 1e-3, f"disagreeing clients showed no drift: {drift}"


if __name__ == "__main__":
    test_partitions_are_disjoint_covers()
    test_low_alpha_concentrates_superclasses()
    test_iid_shards_are_near_equal()
    test_models_output_100_logits()
    test_label_heterogeneity_bounds()
    test_drift_is_zero_for_identical_clients()
    test_drift_is_positive_for_disagreeing_clients()
    test_aggregate_is_a_weighted_mean()
    test_one_local_step_matches_plain_sgd()
    print("ok")
