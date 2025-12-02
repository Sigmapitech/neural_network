#!/usr/bin/env python3
import argparse
import sys

from analyzer import predict_mode, train_mode
from my_torch import Network


def main():
    parser = argparse.ArgumentParser(
        description="MY_TORCH Chess Position Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on chessboards
  ./my_torch_analyzer --predict my_torch_network.nn boards.txt

  # Train network
  ./my_torch_analyzer --train my_torch_network.nn training_data.txt --save trained.nn
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--predict", action="store_true", help="Prediction mode"
    )
    mode_group.add_argument(
        "--train", action="store_true", help="Training mode"
    )

    parser.add_argument(
        "loadfile", metavar="LOADFILE", help="Neural network file to load"
    )
    parser.add_argument(
        "chessfile", metavar="CHESSFILE", help="File with FEN positions"
    )
    parser.add_argument(
        "--save",
        metavar="SAVEFILE",
        help="Save trained network (train mode only)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Training epochs (default: 1000)",
    )
    parser.add_argument(
        "--encoding",
        choices=["simple", "simple_extended", "piece_planes"],
        default="simple",
        help="Board encoding method (default: simple)",
    )

    args = parser.parse_args()

    try:
        print(f"Loading network from {args.loadfile}...")
        network = Network.load(args.loadfile)
        print(f"Loaded network: {network.layer_sizes}")
    except FileNotFoundError:
        print(f"Error: Network file '{args.loadfile}' not found")
        return 1
    except Exception as e:
        print(f"Error loading network: {e}")
        return 1

    if args.predict:
        predict_mode(network, args.chessfile, encoding=args.encoding)
    else:
        savefile = args.save if args.save else args.loadfile
        train_mode(
            network,
            args.chessfile,
            savefile,
            epochs=args.epochs,
            encoding=args.encoding,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
