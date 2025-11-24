#!/usr/bin/env python3
"""Aggregate and visualize training curves from results/*.csv.

Looks for CSV files with columns: epoch,loss,accuracy. Produces overlay plots.
Optionally filters by prefix.
"""
import argparse
import csv
import glob
import os
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    plt = None


def load_curve(path: str) -> Tuple[List[int], List[float], List[float]]:
    epochs: List[int] = []
    losses: List[float] = []
    accs: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                epochs.append(int(row[0]))
                losses.append(float(row[1]))
                # Some files may have list-like accuracy (e.g. '[0.1, 0.2]'), take first element
                acc_field = row[2].strip()
                if acc_field.startswith("[") and acc_field.endswith("]"):
                    inner = acc_field[1:-1].split(",")
                    acc_val = (
                        float(inner[0]) if inner and inner[0].strip() else 0.0
                    )
                else:
                    acc_val = float(acc_field)
                accs.append(acc_val * 100.0)
            except Exception:
                continue
    return epochs, losses, accs


def main():
    ap = argparse.ArgumentParser(description="Visualize training CSV curves")
    ap.add_argument(
        "--dir", default="results", help="Directory containing CSVs"
    )
    ap.add_argument("--prefix", help="Only include files starting with prefix")
    ap.add_argument(
        "--out", default="results/overlay.png", help="Output plot path"
    )
    ap.add_argument(
        "--metric", choices=["loss", "accuracy"], default="accuracy"
    )
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib not installed")

    pattern = os.path.join(args.dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if args.prefix:
        files = [
            f for f in files if os.path.basename(f).startswith(args.prefix)
        ]
    if not files:
        raise SystemExit("No CSV files found matching criteria")

    plt.figure(figsize=(8, 5))
    for f in files:
        epochs, losses, accs = load_curve(f)
        label = os.path.basename(f).replace(".csv", "")
        if args.metric == "loss":
            plt.plot(epochs, losses, label=label)
        else:
            plt.plot(epochs, accs, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss" if args.metric == "loss" else "Accuracy (%)")
    plt.title(f"Overlay {args.metric.title()} Curves")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    print(f"Saved overlay plot to {args.out}")


if __name__ == "__main__":
    main()
