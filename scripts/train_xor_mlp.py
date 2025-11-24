#!/usr/bin/env python3
"""Train a multilayer perceptron on XOR and optionally plot learning curve."""
import argparse
import os
import pathlib
import sys
from typing import List, Tuple

# Ensure root in path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bs.mlp import MLP, build_xor_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Train MLP on XOR gate")
    p.add_argument("--hidden", type=int, default=2, help="Hidden layer size")
    p.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    p.add_argument(
        "--epochs", type=int, default=5000, help="Max training epochs"
    )
    p.add_argument(
        "--no-plot", action="store_true", help="Disable matplotlib plot"
    )
    p.add_argument(
        "--out", default="results", help="Output directory for curve data"
    )
    p.add_argument("--save-model", help="Path to save trained model JSON")
    args = p.parse_args()

    if args.hidden <= 0 or args.lr <= 0 or args.epochs <= 0:
        raise SystemExit("--hidden, --lr, --epochs must be > 0")

    data = build_xor_dataset()
    mlp = MLP([2, args.hidden, 1], lr=args.lr)
    history = mlp.train(
        data, epochs=args.epochs, target_accuracy=1.0, verbose=True
    )

    os.makedirs(args.out, exist_ok=True)
    curve_path = os.path.join(args.out, "xor_mlp_curve.csv")
    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,accuracy\n")
        for epoch, loss, acc in history:
            f.write(f"{epoch},{loss:.6f},{acc:.4f}\n")
    print(f"Saved curve data to {curve_path}")

    print("-- Final evaluation --")
    for x, target in data:
        y = mlp.predict(x)[0]
        print(
            f"INPUT={x} PRED={y:.4f} BIN={(1 if y>=0.5 else 0)} EXPECTED={int(target[0])}"
        )

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            epochs = [e for e, _, _ in history]
            losses = [l for _, l, _ in history]
            accs = [a * 100 for _, _, a in history]
            plt.figure(figsize=(7, 4))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, losses)
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.subplot(1, 2, 2)
            plt.plot(epochs, accs)
            plt.title("Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("%")
            plt.tight_layout()
            plot_path = os.path.join(args.out, "xor_mlp_learning.png")
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping plot.")

    if args.save_model:
        model_path = args.save_model
        try:
            import json

            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(mlp.to_dict(), f)
            print(f"Saved model to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")


if __name__ == "__main__":
    main()
