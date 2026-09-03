"""Regenerate the hard-client trajectory figure from an existing history CSV.

Usage:
    python plots/plot_hard_client_history.py history.csv [output.png]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plots.notebook_experiments import hard_client_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=Path("hard_client.png"))
    args = parser.parse_args()

    with args.history.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"round", "train_loss", "seed", "hard_k"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise SystemExit(f"history CSV is missing columns: {sorted(missing)}")

    history = []
    for row in rows:
        history.append({
            **row,
            "round": int(row["round"]),
            "seed": int(row["seed"]),
            "hard_k": int(row["hard_k"]),
            "train_loss": float(row["train_loss"]) if row["train_loss"] else None,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = hard_client_plot(history)
    figure.savefig(args.output, dpi=160, facecolor="#fcfcfb")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
