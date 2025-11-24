#!/usr/bin/env python3
"""Run multiple XOR training configurations and summarize convergence epochs."""
import argparse
import json
import os
import pathlib
import random
import sys
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bs.mlp import MLP, build_xor_dataset  # type: ignore


def run_config(
    hidden: int, lr: float, epochs: int
) -> Tuple[int, List[Tuple[int, float, float]]]:
    data = build_xor_dataset()
    mlp = MLP([2, hidden, 1], lr=lr)
    history = mlp.train(
        data, epochs=epochs, target_accuracy=1.0, verbose=False
    )
    final_epoch = history[-1][0]
    return final_epoch, history


def main():
    ap = argparse.ArgumentParser(description="Benchmark XOR MLP configs")
    ap.add_argument(
        "--hidden",
        nargs="*",
        type=int,
        default=[2, 3, 4, 5],
        help="Hidden sizes to test",
    )
    ap.add_argument(
        "--lr",
        nargs="*",
        type=float,
        default=[0.3, 0.5, 0.8],
        help="Learning rates to test",
    )
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--trials", type=int, default=3, help="Trials per config")
    ap.add_argument("--out", default="results/xor_benchmark.csv")
    args = ap.parse_args()

    rows = []
    for h in args.hidden:
        for lr in args.lr:
            for t in range(1, args.trials + 1):
                final_epoch, history = run_config(h, lr, args.epochs)
                rows.append(
                    {
                        "hidden": h,
                        "lr": lr,
                        "trial": t,
                        "epochs_to_converge": final_epoch,
                    }
                )
                print(
                    f"hidden={h} lr={lr} trial={t} converge_epoch={final_epoch}"
                )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("hidden,lr,trial,epochs_to_converge\n")
        for r in rows:
            f.write(
                f"{r['hidden']},{r['lr']},{r['trial']},{r['epochs_to_converge']}\n"
            )
    print(f"Saved benchmark CSV to {args.out}")

    # Optional quick plot if matplotlib installed
    try:
        import statistics

        import matplotlib.pyplot as plt  # type: ignore

        grouped = {}
        for r in rows:
            key = (r["hidden"], r["lr"])
            grouped.setdefault(key, []).append(r["epochs_to_converge"])
        plt.figure(figsize=(6, 4))
        xs = []
        ys = []
        labels = []
        for (h, lr), vals in grouped.items():
            xs.append(h + (lr / 10))
            ys.append(statistics.fmean(vals))
            labels.append(f"h{h}-lr{lr}")
        plt.scatter(xs, ys)
        for x, y, l in zip(xs, ys, labels):
            plt.text(x, y, l, fontsize=8)
        plt.xlabel("Hidden size (offset by lr)")
        plt.ylabel("Avg epochs to converge")
        plt.title("XOR MLP Convergence Benchmark")
        plot_path = args.out.replace(".csv", ".png")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Saved benchmark plot to {plot_path}")
    except ImportError:
        print("matplotlib not installed; skipping benchmark plot.")


if __name__ == "__main__":
    main()
