#!/usr/bin/env python3
"""Train MLP on Tic-Tac-Toe position classification.

Modes:
  binary: win (any player) vs not yet win
  multi: ongoing vs draw vs win

Input encoding: 9 floats (X=1, O=-1, empty=0).
"""
import argparse
import os
import pathlib
import random
import sys
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bs.mlp import MLP  # type: ignore
from bs.tictactoe import (  # type: ignore
    build_binary_dataset,
    build_multiclass_dataset,
)


def split_dataset(
    data: List[Tuple[List[float], List[float]]], val_ratio: float = 0.2
):
    random.shuffle(data)
    n_val = int(len(data) * val_ratio)
    val = data[:n_val]
    train = data[n_val:]
    return train, val


def accuracy(model: MLP, data: List[Tuple[List[float], List[float]]]) -> float:
    correct = 0
    for x, y in data:
        out = model.predict(x)
        if len(y) == 1:
            pred = 1 if out[0] >= 0.5 else 0
            if pred == int(y[0]):
                correct += 1
        else:
            pred_idx = max(range(len(out)), key=lambda i: out[i])
            tgt_idx = max(range(len(y)), key=lambda i: y[i])
            if pred_idx == tgt_idx:
                correct += 1
    return correct / len(data) if data else 0.0


def main():
    p = argparse.ArgumentParser(description="Train MLP on TicTacToe")
    p.add_argument("--mode", choices=["binary", "multi"], default="binary")
    p.add_argument("--hidden", type=int, default=27, help="Hidden layer size")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument(
        "--batch", type=int, default=64, help="Mini-batch size (0=full batch)"
    )
    p.add_argument(
        "--lr-decay",
        type=float,
        default=0.0,
        help="Decay per epoch fraction (e.g. 0.001)",
    )
    p.add_argument(
        "--val", type=float, default=0.2, help="Validation split ratio"
    )
    p.add_argument(
        "--target",
        type=float,
        default=0.9,
        help="Target training accuracy to early-stop",
    )
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save-model", help="Path to save trained model JSON")
    args = p.parse_args()

    if args.hidden <= 0 or args.lr <= 0 or args.epochs <= 0:
        raise SystemExit("--hidden, --lr, --epochs must be > 0")

    if args.mode == "binary":
        data = build_binary_dataset()
        output_size = 1
    else:
        data = build_multiclass_dataset()
        output_size = 3

    train_set, val_set = split_dataset(data, val_ratio=args.val)
    print(
        f"Dataset size: total={len(data)} train={len(train_set)} val={len(val_set)} mode={args.mode}"
    )

    mlp = MLP([9, args.hidden, output_size], lr=args.lr)
    history = mlp.train(
        train_set,
        epochs=args.epochs,
        target_accuracy=args.target,
        verbose=True,
        batch_size=args.batch,
        lr_decay=args.lr_decay,
    )

    train_acc = accuracy(mlp, train_set)
    val_acc = accuracy(mlp, val_set)
    print(
        f"Final Train Accuracy: {train_acc*100:.2f}% | Validation Accuracy: {val_acc*100:.2f}%"
    )

    os.makedirs(args.out, exist_ok=True)
    curve_path = os.path.join(args.out, f"tictactoe_{args.mode}_curve.csv")
    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,accuracy\n")
        for epoch, loss, acc in history:
            f.write(f"{epoch},{loss:.6f},{acc:.4f}\n")
    print(f"Saved curve data to {curve_path}")

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
            plt.title("Train Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("%")
            plt.tight_layout()
            plot_path = os.path.join(
                args.out, f"tictactoe_{args.mode}_learning.png"
            )
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping plot.")

    if args.save_model:
        try:
            import json

            with open(args.save_model, "w", encoding="utf-8") as f:
                json.dump(mlp.to_dict(), f)
            print(f"Saved model to {args.save_model}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    # Show a few validation predictions
    print("-- Sample predictions (validation) --")
    for i, (x, y) in enumerate(val_set[:10]):
        out = mlp.predict(x)
        if output_size == 1:
            pred = 1 if out[0] >= 0.5 else 0
            print(f"{i}: PRED={pred} RAW={out[0]:.4f} TARGET={int(y[0])}")
        else:
            pred_idx = max(range(len(out)), key=lambda k: out[k])
            tgt_idx = max(range(len(y)), key=lambda k: y[k])
            print(
                f"{i}: PRED={pred_idx} OUT={','.join(f'{v:.3f}' for v in out)} TARGET={tgt_idx}"
            )


if __name__ == "__main__":
    main()
