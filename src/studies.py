"""Reusable study runner for the focused Jupyter experiments.

The command-line path in :mod:`main` stays intentionally small.  This module adds the
experimental controls needed by the notebook while continuing to use the project's
Client, Server, models, partitions, training loss, and exact gradient-drift metric.
"""
from __future__ import annotations

import copy
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from client import Client
from data import by_target_h, dirichlet, iid, label_heterogeneity
from models import linear, small_cnn
from server import Server


MODELS = {"linear": linear, "small_cnn": small_cnn}


def seed_everything(seed):
    """Seed model initialization, client data streams, and participation sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_stability_metrics(history):
    """Quantify regressions and non-monotone motion in a checkpoint loss path.

    These complement, rather than replace, the loss curve.  A lower final loss is
    optimization progress; stability means fewer/smaller upward jumps and less path
    variation beyond the direct distance from initial to final loss.
    """
    losses = np.asarray([
        row["train_loss"] for row in history if row.get("train_loss") is not None
    ], dtype=float)
    if len(losses) < 2:
        return {
            "loss_increase_fraction": 0.0,
            "mean_positive_loss_jump": 0.0,
            "max_positive_loss_jump": 0.0,
            "loss_excess_path_variation": 0.0,
        }
    changes = np.diff(losses)
    increases = changes[changes > 0]
    total_variation = float(np.abs(changes).sum())
    direct_change = float(abs(losses[-1] - losses[0]))
    return {
        "loss_increase_fraction": float(np.mean(changes > 0)),
        "mean_positive_loss_jump": float(increases.mean()) if len(increases) else 0.0,
        "max_positive_loss_jump": float(increases.max()) if len(increases) else 0.0,
        "loss_excess_path_variation": max(0.0, total_variation - direct_change),
    }


def _js_divergence(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return float(np.sum(a[keep] * np.log(a[keep] / b[keep])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class StudyServer(Server):
    """FedAvg server with study-only aggregation and participation controls."""

    def __init__(self, model, clients, dataset, pool, device, labels, shards, beta=1.0):
        super().__init__(model, clients, dataset, pool, device)
        self.beta = float(beta)
        labels = np.asarray(labels)
        n_labels = int(labels.max()) + 1
        all_idx = np.concatenate([np.asarray(s, dtype=np.int64) for s in shards])
        global_dist = np.bincount(labels[all_idx], minlength=n_labels).astype(float)
        global_dist /= global_dist.sum()
        self.client_js = []
        for shard in shards:
            dist = np.bincount(labels[np.asarray(shard)], minlength=n_labels).astype(float)
            dist /= dist.sum()
            self.client_js.append(_js_divergence(dist, global_dist))

    def aggregation_weights(self, selected, policy):
        """Return normalized endpoint weights for policies that average endpoints."""
        if policy == "uniform":
            raw = np.ones(len(selected), dtype=float)
        elif policy == "data":
            raw = np.asarray([self.clients[i].n for i in selected], dtype=float)
        elif policy == "inverse_js":
            raw = np.asarray([
                self.clients[i].n * math.exp(-self.beta * self.client_js[i])
                for i in selected
            ], dtype=float)
        elif policy == "fednova":
            # FedNova uses data weights for its normalized DELTAS below.
            raw = np.asarray([self.clients[i].n for i in selected], dtype=float)
        else:
            raise ValueError(f"unknown aggregation policy: {policy}")
        return (raw / raw.sum()).tolist()

    def aggregate_round(self, center, states, selected, steps, policy):
        weights = self.aggregation_weights(selected, policy)
        if policy != "fednova":
            return self.aggregate(states, weights)

        # FedNova-style first-order correction for unequal local step counts:
        # w+ = w + sum_i p_i (Kbar/K_i) (w_i - w).
        kbar = sum(w * steps[i] for w, i in zip(weights, selected))
        out = {}
        for key, initial in center.items():
            if initial.is_floating_point():
                value = initial.to(torch.float64).clone()
                for state, weight, client_id in zip(states, weights, selected):
                    scale = weight * kbar / steps[client_id]
                    value += scale * (state[key].to(torch.float64) - initial.to(torch.float64))
                out[key] = value.to(initial.dtype)
            else:
                out[key] = initial.clone()
        return out

    @torch.no_grad()
    def evaluate_clients(self):
        """Evaluate the actual mixture of client objectives, including covariate shift."""
        state = self.model.state_dict()
        losses, accuracies = [], []
        for client in self.clients:
            client.model.load_state_dict(state)
            client.model.eval()
            loss_sum = correct = count = 0
            for x, y in client.grad_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = client.model(x)
                loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
                correct += int((logits.argmax(1) == y).sum())
                count += y.numel()
            losses.append(loss_sum / count)
            accuracies.append(correct / count)
        size_weights = np.asarray([c.n for c in self.clients], dtype=float)
        size_weights /= size_weights.sum()
        return {
            "train_loss": float(np.dot(size_weights, losses)),
            "train_accuracy": float(np.dot(size_weights, accuracies)),
            "client_losses": tuple(float(x) for x in losses),
            "client_accuracies": tuple(float(x) for x in accuracies),
        }

    def run_study(self, rounds, client_steps, aggregation="data", client_fraction=1.0,
                  eval_every=5, drift_rounds=(0,), seed=0):
        n_clients = len(self.clients)
        if len(client_steps) != n_clients or min(client_steps) < 1:
            raise ValueError("client_steps must contain one positive K_i per client")
        online = max(1, min(n_clients, int(math.ceil(client_fraction * n_clients))))
        rng = np.random.default_rng(seed)
        drift_rounds = set(drift_rounds)
        history = []
        examples_processed = local_updates = 0
        participation_counts = np.zeros(n_clients, dtype=int)

        def record(round_index, selected, weights, drift_pair=(None, None), evaluate=True):
            metrics = self.evaluate_clients() if evaluate else {
                "train_loss": None,
                "train_accuracy": None,
                "client_losses": None,
                "client_accuracies": None,
            }
            history.append({
                "round": round_index,
                "local_updates": local_updates,
                "examples_processed": examples_processed,
                "train_loss": metrics["train_loss"],
                "train_accuracy": metrics["train_accuracy"],
                "drift": drift_pair[0],
                "grad_norm": drift_pair[1],
                "selected_clients": tuple(selected),
                "aggregation_weights": tuple(weights),
                "client_losses": metrics["client_losses"],
                "client_accuracies": metrics["client_accuracies"],
            })

        initial = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        initial_drift = self.measure_drift(initial) if 0 in drift_rounds else (None, None)
        record(0, (), (), initial_drift)

        for round_index in range(1, rounds + 1):
            center = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if online == n_clients:
                selected = list(range(n_clients))
            else:
                selected = sorted(rng.choice(n_clients, size=online, replace=False).tolist())
            participation_counts[selected] += 1
            states = []
            for client_id in selected:
                state, seen = self.clients[client_id].local_steps(
                    center, client_steps[client_id]
                )
                states.append(state)
                examples_processed += seen
                local_updates += client_steps[client_id]
            weights = self.aggregation_weights(selected, aggregation)
            self.model.load_state_dict(
                self.aggregate_round(center, states, selected, client_steps, aggregation)
            )
            # The post-update model is the next broadcast point. Measuring here makes the
            # final drift and final objective refer to exactly the same checkpoint.
            drift = (self.measure_drift(self.model.state_dict())
                     if round_index in drift_rounds else (None, None))
            evaluate = (round_index % eval_every == 0 or round_index in drift_rounds
                        or round_index == rounds)
            # Keep a lightweight row every round so participation IDs and effective
            # online-client weights are fully auditable, not only visible at checkpoints.
            record(round_index, selected, weights, drift, evaluate=evaluate)

        return history, participation_counts


def make_partition(labels, pool, n_clients, seed, partition="target_h", target_h=0.0,
                   alpha=0.3):
    rng = np.random.default_rng(seed)
    if partition == "iid":
        return iid(labels, pool, n_clients, rng)
    if partition == "target_h":
        return by_target_h(labels, pool, n_clients, rng, h=target_h)
    if partition == "dirichlet":
        return dirichlet(labels, pool, n_clients, rng, alpha=alpha)
    raise ValueError(f"unknown partition: {partition}")


def run_federated(dataset, pool, labels, *, n_clients, seed, rounds, local_steps=5,
                  partition="target_h", target_h=0.0, alpha=0.3,
                  aggregation="data", aggregation_beta=1.0, client_fraction=1.0,
                  model="small_cnn", batch_size=64, lr=0.05, device=None,
                  eval_every=5, measure_final_drift=True, shards=None,
                  client_datasets=None):
    """Run one paired-seed condition and return checkpoint rows plus one summary row."""
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    shards = shards or make_partition(
        labels, pool, n_clients, seed, partition=partition, target_h=target_h, alpha=alpha
    )
    if any(len(s) == 0 for s in shards):
        raise ValueError("the requested partition produced an empty client")
    if isinstance(local_steps, int):
        client_steps = [local_steps] * n_clients
    else:
        client_steps = [int(k) for k in local_steps]
    if len(client_steps) != n_clients:
        raise ValueError("local_steps must be an int or have one value per client")

    model_fn = partial(MODELS[model], 1, 10, 28)
    initial_model = model_fn()
    datasets = client_datasets or [dataset] * n_clients
    clients = [
        Client(datasets[i], shards[i], model_fn, batch_size, lr, device,
               seed=seed * 10_000 + i)
        for i in range(n_clients)
    ]
    server = StudyServer(
        copy.deepcopy(initial_model), clients, dataset, pool, device, labels, shards,
        beta=aggregation_beta,
    )
    drift_rounds = (0, rounds) if measure_final_drift else (0,)
    history, counts = server.run_study(
        rounds, client_steps, aggregation=aggregation,
        client_fraction=client_fraction, eval_every=eval_every,
        drift_rounds=drift_rounds, seed=seed + 1_000_003,
    )
    sizes = np.asarray([len(s) for s in shards], dtype=float)
    final = history[-1]
    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "rounds": rounds,
        "client_fraction_requested": client_fraction,
        "clients_per_round": max(1, min(n_clients, int(math.ceil(client_fraction * n_clients)))),
        "client_fraction_actual": max(1, min(n_clients, int(math.ceil(client_fraction * n_clients)))) / n_clients,
        "partition": partition,
        "target_h": target_h if partition == "target_h" else None,
        "h_label": label_heterogeneity(labels, shards),
        "quantity_cv": float(sizes.std() / sizes.mean()),
        "aggregation": aggregation,
        "local_steps": tuple(client_steps),
        "mean_local_steps": float(np.mean(client_steps)),
        "final_train_loss": final["train_loss"],
        "final_train_accuracy": final["train_accuracy"],
        "final_drift": final["drift"],
        "final_grad_norm": final["grad_norm"],
        "local_updates": final["local_updates"],
        "examples_processed": final["examples_processed"],
        "participation_coverage": float(np.mean(counts > 0)),
        "participation_cv": float(counts.std() / counts.mean()) if counts.mean() else 0.0,
        "final_client_losses": final["client_losses"],
        "final_client_accuracies": final["client_accuracies"],
    }
    return history, summary
