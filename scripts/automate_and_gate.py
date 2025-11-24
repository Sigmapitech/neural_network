#!/usr/bin/env python3
"""Automation script for AND gate perceptron training.

Runs multiple trials of training a 2-input perceptron on the AND gate dataset,
collects convergence phases and accuracy history, writes CSV summaries, and
optionally plots average learning curve (requires matplotlib).
"""
import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
from typing import List, Tuple

# Ensure project root in sys.path for direct script execution
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bs.my_perceptron import Perceptron, read_training_file, train_perceptron


def run_trials(
    dataset_path: str, trials: int, steps: int, lr: float, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    dataset = read_training_file(dataset_path, 2)
    trial_rows: List[Tuple[int, int, List[float], float]] = []
    # phase -> list[accuracy]
    accuracy_by_phase: dict[int, List[float]] = {}

    for t in range(1, trials + 1):
        perceptron = Perceptron.new(2, lr=lr)
        perceptron, history = train_perceptron(perceptron, dataset, steps)
        phases_to_converge = history[-1][0]
        trial_rows.append(
            (t, phases_to_converge, perceptron.weights.copy(), perceptron.bias)
        )
        for phase, acc in history:
            accuracy_by_phase.setdefault(phase, []).append(acc)

    # Write trials summary CSV
    trials_csv = os.path.join(output_dir, "and_gate_trials.csv")
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "phases_to_converge", "weights", "bias"])
        for row in trial_rows:
            w.writerow([row[0], row[1], json.dumps(row[2]), f"{row[3]:.6f}"])

    # Compute average accuracy per phase
    avg_acc: List[Tuple[int, float]] = []
    for phase in sorted(accuracy_by_phase.keys()):
        avg_acc.append((phase, statistics.fmean(accuracy_by_phase[phase])))

    acc_csv = os.path.join(output_dir, "and_gate_accuracy_curve.csv")
    with open(acc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "avg_accuracy_percent"])
        for phase, acc in avg_acc:
            w.writerow([phase, f"{acc:.2f}"])

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt  # type: ignore

        phases = [p for p, _ in avg_acc]
        accs = [a for _, a in avg_acc]
        plt.figure(figsize=(6, 4))
        plt.plot(phases, accs, marker="o")
        plt.title("AND Gate Perceptron Average Learning Curve")
        plt.xlabel("Training Phase")
        plt.ylabel("Average Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "and_gate_learning_curve.png")
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")

    print(f"Wrote trials summary to {trials_csv}")
    print(f"Wrote average accuracy curve to {acc_csv}")

    # Basic stats
    phases_list = [row[1] for row in trial_rows]
    print(
        f"Phases to converge: min={min(phases_list)}, max={max(phases_list)}, avg={statistics.fmean(phases_list):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Automate AND gate perceptron training trials"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/and_gate.txt",
        help="Path to AND gate dataset",
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of trials to run"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Training updates per phase"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for perceptron initialization",
    )
    parser.add_argument(
        "--out", default="results", help="Output directory for CSV/plots"
    )
    args = parser.parse_args()

    if args.trials <= 0 or args.steps <= 0:
        raise SystemExit("--trials and --steps must be > 0")

    run_trials(args.dataset, args.trials, args.steps, args.lr, args.out)


if __name__ == "__main__":
    main()
