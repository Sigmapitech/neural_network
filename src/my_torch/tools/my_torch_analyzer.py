#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import TypeVar

from analyzer import predict_mode, predict_mode_profiled, train_mode
from my_torch import Network, TrainableNetwork

T = TypeVar("T", bound=Network | TrainableNetwork)

EPILOG = """
Examples:
  # Predict on chessboards
  ./my_torch_analyzer --predict my_torch_network.nn boards.txt

  # Train network
  ./my_torch_analyzer --train my_torch_network.nn training_data.txt --save trained.nn
"""


def run_predictor(args):
    network_configfile = Path(args.loadfile)

    network = Network.load(network_configfile)
    if network is None:
        return 84

    if args.profiled:
        print("Using profiled version, single threaded.")
        time.sleep(1)
        predict_mode_profiled(network, args.chessfile, encoding=args.encoding)
    else:
        predict_mode(network, args.chessfile, encoding=args.encoding)

    return 0


def run_trainer(args):
    network_configfile = Path(args.loadfile)

    network = TrainableNetwork.load(network_configfile)
    if network is None:
        return 84

    savefile = args.save if args.save else args.loadfile

    train_mode(
        network,
        args.chessfile,
        savefile,
        epochs=args.epochs,
        encoding=args.encoding,
    )


def main():
    parser = argparse.ArgumentParser(
        description="MY_TORCH Chess Position Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    # fmt: off
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--predict", action="store_true", help="Prediction mode")
    mode_group.add_argument("--train", action="store_true", help="Training mode")

    parser.add_argument("loadfile", metavar="LOADFILE", help="Neural network file to load")
    parser.add_argument("chessfile", metavar="CHESSFILE", help="File with FEN positions")

    parser.add_argument("--profiled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save", metavar="SAVEFILE", help="Save trained network (train mode only)")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs (default: 1000)")
    parser.add_argument(
        "--encoding",
        choices=["simple", "simple_extended", "piece_planes"],
        default="simple",
        help="Board encoding method (default: simple)",
    )

    # fmt: on
    args = parser.parse_args()

    if args.predict:
        run_predictor(args)
    else:
        run_trainer(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
