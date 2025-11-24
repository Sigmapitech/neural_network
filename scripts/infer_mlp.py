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
    args = p.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        data = json.load(f)
    mlp = MLP.from_dict(data)
    x = args.input
    out = mlp.predict(x)
    print(f"Input={x}\nRaw Output={out}")
    if len(out) == 1:
        print(f"Binary Thresholded={(1 if out[0] >= 0.5 else 0)}")
    else:
        pred_idx = max(range(len(out)), key=lambda i: out[i])
        print(f"Argmax Class={pred_idx}")


if __name__ == "__main__":
    main()
