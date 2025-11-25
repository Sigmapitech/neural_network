#!/usr/bin/env python3
"""Load a saved MLP model JSON and perform inference.

Usage examples:
  python3 scripts/infer_mlp.py --model saved_xor.json --input 0 1
  python3 scripts/infer_mlp.py --model saved_ttt.json --input 1 0 -1 0 1 -1 0 0 0
"""
import argparse
import json
import pathlib
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bs.mlp import MLP  # type: ignore


def main():
    p = argparse.ArgumentParser(description="MLP inference")
    p.add_argument("--model", required=True, help="Path to saved model JSON")
    p.add_argument("--input", nargs="+", type=float, help="Input values")
    p.add_argument(
        "--labels",
        action="store_true",
        help="Print human-readable class labels (for 3-class Tic-Tac-Toe)",
    )
    p.add_argument(
        "--ttt-gt",
        action="store_true",
        help="Compute Tic-Tac-Toe ground-truth class index for the provided 9-value input",
    )
    args = p.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        data = json.load(f)
    mlp = MLP.from_dict(data)
    x = args.input
    out = mlp.predict(x)
    print(
        f"Input={x}\nRaw Output={out}\nOutputActivation={getattr(mlp,'output_activation','sigmoid')}"
    )
    if len(out) == 1:
        print(f"Binary Thresholded={(1 if out[0] >= 0.5 else 0)}")
    else:
        pred_idx = max(range(len(out)), key=lambda i: out[i])
        print(f"Argmax Class={pred_idx}")
        if args.labels and len(out) == 3:
            labels = ["ongoing", "draw", "win"]
            print(f"Argmax Label={labels[pred_idx]}")

    if args.ttt_gt:
        if len(x) != 9:
            print("[ttt-gt] Expected 9 inputs for Tic-Tac-Toe ground truth.")
        else:
            try:
                from bs.tictactoe import check_winner, is_draw  # type: ignore

                b = [int(v) for v in x]
                gt = (
                    2
                    if check_winner(b) is not None
                    else (1 if is_draw(b) else 0)
                )
                labels = ["ongoing", "draw", "win"]
                print(
                    f"GroundTruth Class={gt} Label={labels[gt]} (0=ongoing,1=draw,2=win)"
                )
            except Exception as e:
                print(f"[ttt-gt] Unable to compute ground truth: {e}")


if __name__ == "__main__":
    main()
