"""Terminal runner for the four focused MNIST notebook studies.

Run from anywhere with, for example::

    python -u src/run_notebook_experiments.py --device cuda

The implementation intentionally calls :mod:`studies`, the same project-backed runner
used by the notebook, rather than maintaining a second federated training path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from data import DATASETS, DeterministicGaussianNoise
from plots.notebook_experiments import budget_curve_plot, hard_client_plot
from studies import loss_stability_metrics, make_partition, run_federated


STUDIES = ("hard_client", "client_count", "aggregation", "participation")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the four focused MNIST federated-learning studies"
    )
    parser.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument("--pilot", action="store_true",
                        help="one seed, 10%% of MNIST, and four reference rounds")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--data-frac", type=float, default=None)
    parser.add_argument("--data-pool-seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=None,
                        help="reference rounds; partial participation scales this value")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--model", choices=("small_cnn", "linear"), default="small_cnn")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "notebook_results_terminal")
    parser.add_argument("--no-final-drift", action="store_true",
                        help="skip the expensive exact final gradient-drift pass")

    parser.add_argument("--easy-k", type=int, default=5)
    parser.add_argument("--hard-k-values", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--noise-std", type=float, default=0.8)

    parser.add_argument("--client-counts", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--client-count-h", type=float, default=0.25)

    parser.add_argument("--aggregation-policies", nargs="+",
                        choices=("uniform", "data", "inverse_js", "fednova"),
                        default=["uniform", "data", "inverse_js"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--inverse-js-beta", type=float, default=2.0)

    parser.add_argument("--participation-fractions", nargs="+", type=float,
                        default=[0.2, 0.5, 1.0])
    parser.add_argument("--participation-h", type=float, default=0.8)
    return parser


def resolve_args(args):
    args.seeds = args.seeds if args.seeds is not None else ([0] if args.pilot else [0, 1, 2])
    args.data_frac = args.data_frac if args.data_frac is not None else (0.1 if args.pilot else 0.5)
    args.rounds = args.rounds if args.rounds is not None else (4 if args.pilot else 30)
    args.eval_every = args.eval_every if args.eval_every is not None else (1 if args.pilot else 5)
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a GPU")
    if not 0 < args.data_frac <= 1:
        raise SystemExit("--data-frac must be in (0, 1]")
    if args.rounds < 1 or args.eval_every < 1:
        raise SystemExit("--rounds and --eval-every must be positive")
    if any(not 0 < q <= 1 for q in args.participation_fractions):
        raise SystemExit("every --participation-fractions value must be in (0, 1]")
    return args


def tagged_history(history, **metadata):
    return [{**row, **metadata} for row in history]


def csv_value(value):
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value)
    return value


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def save_study(output_root, name, histories, summaries, figure):
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "history.csv", histories)
    write_csv(output / "summary.csv", summaries)
    figure.savefig(output / f"{name}.png", dpi=160, facecolor="#fcfcfb")
    import matplotlib.pyplot as plt
    plt.close(figure)
    print(f"saved {output}", flush=True)


def common_args(args):
    return {
        "model": args.model,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "measure_final_drift": not args.no_final_drift,
    }


def run_hard_client(args, dataset, pool, labels):
    histories, summaries = [], []
    for seed in args.seeds:
        shards = make_partition(labels, pool, 2, seed, partition="iid")
        hard_dataset = DeterministicGaussianNoise(
            dataset, std=args.noise_std, seed=100_000 + seed
        )
        for hard_k in args.hard_k_values:
            print(
                f"[hard_client] seed={seed} K_clean={args.easy_k} "
                f"tau_noisy={hard_k} rounds={args.rounds}",
                flush=True,
            )
            history, summary = run_federated(
                dataset, pool, labels, n_clients=2, seed=seed, rounds=args.rounds,
                local_steps=[args.easy_k, hard_k], partition="iid", shards=shards,
                client_datasets=[dataset, hard_dataset], aggregation="data",
                # Stability needs the actual round-to-round path, not five-round samples.
                eval_every=1, **common_args(args),
            )
            summary.update({
                "easy_k": args.easy_k,
                "hard_k": hard_k,
                "noise_std": args.noise_std,
                "hard_loss_gap": (
                    summary["final_client_losses"][1]
                    - summary["final_client_losses"][0]
                ),
                **loss_stability_metrics(history),
            })
            summaries.append(summary)
            histories.extend(tagged_history(history, seed=seed, hard_k=hard_k))
    save_study(args.output_root, "01_hard_client", histories, summaries,
               hard_client_plot(histories))


def run_client_count(args, dataset, pool, labels):
    conditions = {"H=0.00": 0.0, f"H={args.client_count_h:.2f}": args.client_count_h}
    histories, summaries = [], []
    for condition, target_h in conditions.items():
        for n_clients in args.client_counts:
            for seed in args.seeds:
                print(f"[client_count] {condition} N={n_clients} seed={seed}", flush=True)
                history, summary = run_federated(
                    dataset, pool, labels, n_clients=n_clients, seed=seed,
                    rounds=args.rounds, local_steps=5, partition="target_h",
                    target_h=target_h, aggregation="data", eval_every=args.eval_every,
                    **common_args(args),
                )
                summary["condition"] = condition
                summaries.append(summary)
                histories.extend(tagged_history(
                    history, seed=seed, client_count=n_clients, condition=condition
                ))
    figure = budget_curve_plot(
        histories, group_key="client_count", group_order=args.client_counts,
        group_labels={n: f"{n} clients" for n in args.client_counts},
        facet_key="condition", facet_order=list(conditions),
        facet_labels={key: key for key in conditions}, title="Effect of federation size",
    )
    save_study(args.output_root, "02_client_count", histories, summaries, figure)


def run_aggregation(args, dataset, pool, labels):
    histories, summaries = [], []
    for seed in args.seeds:
        shards = make_partition(
            labels, pool, 10, seed, partition="dirichlet", alpha=args.dirichlet_alpha
        )
        for policy in args.aggregation_policies:
            print(f"[aggregation] seed={seed} policy={policy}", flush=True)
            history, summary = run_federated(
                dataset, pool, labels, n_clients=10, seed=seed, rounds=args.rounds,
                local_steps=5, partition="dirichlet", alpha=args.dirichlet_alpha,
                shards=shards, aggregation=policy,
                aggregation_beta=args.inverse_js_beta, eval_every=args.eval_every,
                **common_args(args),
            )
            summaries.append(summary)
            histories.extend(tagged_history(
                history, seed=seed, aggregation_policy=policy
            ))
    figure = budget_curve_plot(
        histories, group_key="aggregation_policy",
        group_order=args.aggregation_policies,
        group_labels={policy: policy for policy in args.aggregation_policies},
        title="Communication-round weighting on unequal client shards",
    )
    save_study(args.output_root, "03_aggregation", histories, summaries, figure)


def run_participation(args, dataset, pool, labels):
    total_clients = 10
    conditions = {"H=0.00": 0.0, f"H={args.participation_h:.2f}": args.participation_h}
    histories, summaries = [], []
    for condition, target_h in conditions.items():
        for fraction in args.participation_fractions:
            online = math.ceil(fraction * total_clients)
            rounds = math.ceil(args.rounds * total_clients / online)
            eval_every = max(1, rounds // 6)
            for seed in args.seeds:
                print(
                    f"[participation] {condition} q={fraction} seed={seed} rounds={rounds}",
                    flush=True,
                )
                history, summary = run_federated(
                    dataset, pool, labels, n_clients=total_clients, seed=seed,
                    rounds=rounds, local_steps=5, partition="target_h",
                    target_h=target_h, aggregation="data", client_fraction=fraction,
                    eval_every=eval_every, **common_args(args),
                )
                summary["condition"] = condition
                summaries.append(summary)
                histories.extend(tagged_history(
                    history, seed=seed, participation_fraction=fraction,
                    condition=condition,
                ))
    figure = budget_curve_plot(
        histories, group_key="participation_fraction",
        group_order=args.participation_fractions,
        group_labels={q: f"q={q:g}" for q in args.participation_fractions},
        facet_key="condition", facet_order=list(conditions),
        facet_labels={key: key for key in conditions},
        title="Partial participation at a matched computation budget",
    )
    save_study(args.output_root, "04_participation", histories, summaries, figure)


def main():
    args = resolve_args(build_parser().parse_args())
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["data_root"] = os.fspath(args.data_root)
    config["output_root"] = os.fspath(args.output_root)
    (args.output_root / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"device={args.device} seeds={args.seeds} data_frac={args.data_frac}", flush=True)
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    dataset, pool, labels = DATASETS["mnist"]["load"](
        os.fspath(args.data_root), data_frac=args.data_frac, seed=args.data_pool_seed
    )
    print(f"loaded {len(pool):,} stratified MNIST examples", flush=True)

    runners = {
        "hard_client": run_hard_client,
        "client_count": run_client_count,
        "aggregation": run_aggregation,
        "participation": run_participation,
    }
    for study in args.studies:
        runners[study](args, dataset, pool, labels)
    print(f"all requested studies completed: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
