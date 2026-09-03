"""Fast checks for notebook-study controls; no dataset download required."""
from types import SimpleNamespace

import torch

from data import DeterministicGaussianNoise
from studies import StudyServer, loss_stability_metrics


def test_deterministic_noise_is_fixed_by_index():
    dataset = [(torch.zeros(1, 3, 3), 0), (torch.ones(1, 3, 3), 1)]
    a = DeterministicGaussianNoise(dataset, std=0.8, seed=12)
    b = DeterministicGaussianNoise(dataset, std=0.8, seed=12)
    c = DeterministicGaussianNoise(dataset, std=0.8, seed=13)
    assert torch.equal(a[0][0], a[0][0])
    assert torch.equal(a[1][0], b[1][0])
    assert not torch.equal(a[0][0], c[0][0])


def _aggregation_fixture():
    server = StudyServer.__new__(StudyServer)
    server.clients = [SimpleNamespace(n=1), SimpleNamespace(n=3)]
    server.client_js = [0.0, 0.5]
    server.beta = 1.0
    center = {"w": torch.tensor([0.0]), "counter": torch.tensor(4)}
    states = [
        {"w": torch.tensor([1.0]), "counter": torch.tensor(5)},
        {"w": torch.tensor([3.0]), "counter": torch.tensor(6)},
    ]
    return server, center, states


def test_uniform_and_data_weighting_are_distinct_on_unequal_shards():
    server, center, states = _aggregation_fixture()
    uniform = server.aggregate_round(center, states, [0, 1], [1, 1], "uniform")
    data = server.aggregate_round(center, states, [0, 1], [1, 1], "data")
    assert torch.allclose(uniform["w"], torch.tensor([2.0]))
    assert torch.allclose(data["w"], torch.tensor([2.5]))
    assert data["counter"].item() == 5


def test_fednova_reduces_to_data_weighting_at_equal_k():
    server, center, states = _aggregation_fixture()
    data = server.aggregate_round(center, states, [0, 1], [5, 5], "data")
    fednova = server.aggregate_round(center, states, [0, 1], [5, 5], "fednova")
    assert torch.allclose(fednova["w"], data["w"])


def test_selected_aggregation_weights_are_renormalized():
    server, _, _ = _aggregation_fixture()
    assert server.aggregation_weights([1], "data") == [1.0]
    weights = server.aggregation_weights([0, 1], "inverse_js")
    assert abs(sum(weights) - 1.0) < 1e-12
    assert weights[0] > 0 and weights[1] > 0


def test_loss_stability_metrics_measure_only_nonmonotone_regressions():
    history = [
        {"train_loss": 3.0},
        {"train_loss": 2.0},
        {"train_loss": 2.5},
        {"train_loss": 1.0},
    ]
    metrics = loss_stability_metrics(history)
    assert abs(metrics["loss_increase_fraction"] - 1 / 3) < 1e-12
    assert metrics["mean_positive_loss_jump"] == 0.5
    assert metrics["max_positive_loss_jump"] == 0.5
    assert metrics["loss_excess_path_variation"] == 1.0
