#!/usr/bin/env python3
import argparse
import pathlib
import random
import sys
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chess_utils import GameState, fen_to_tensor, get_game_state, parse_fen

# Sample positions for quick testing - BALANCED across all 5 classes
SAMPLE_POSITIONS = [
    # Checkmate White (2 positions)
    (
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "Checkmate White",
    ),
    (
        "rnbqkb1r/ppp2ppp/3p1n2/8/3PP3/8/PPP2qPP/RNBQKB1R w KQkq - 0 5",
        "Checkmate White",
    ),
    # Checkmate Black (2 positions)
    (
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "Checkmate Black",
    ),
    (
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P2Q/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "Checkmate Black",
    ),
    # Check White (2 positions)
    (
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPq/RNBQKBNR w KQkq - 1 3",
        "Check White",
    ),
    (
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "Check White",
    ),
    # Check Black (2 positions)
    (
        "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        "Check Black",
    ),
    (
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        "Check Black",
    ),
    # Nothing (2 positions)
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Nothing",
    ),
    (
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "Nothing",
    ),
]


def generate_sample_dataset(output_file: str, count: int = 100) -> None:
    dataset = []

    while len(dataset) < count:
        fen, label = random.choice(SAMPLE_POSITIONS)
        dataset.append((fen, label))

    with open(output_file, "w", encoding="utf-8") as f:
        for fen, label in dataset:
            f.write(f"{fen} {label}\n")

    print(f"Generated {len(dataset)} positions to {output_file}")


def analyze_fen_file(
    input_file: str, output_file: str, detailed: bool = True
) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        fens = [line.strip() for line in f if line.strip()]

    results = []
    for fen in fens:
        try:
            state = get_game_state(fen, detailed=detailed)
            results.append((fen, state.value))
        except Exception as e:
            print(f"Warning: Could not analyze '{fen}': {e}")
            continue

    with open(output_file, "w", encoding="utf-8") as f:
        for fen, label in results:
            f.write(f"{fen} {label}\n")

    print(f"Analyzed {len(results)} positions, saved to {output_file}")


def stats_dataset(dataset_file: str) -> None:
    label_counts = {}
    total = 0

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.rsplit(maxsplit=2)
            if len(parts) >= 2:
                if len(parts) == 3 and parts[1] in ("Check", "Checkmate"):
                    label = f"{parts[1]} {parts[2]}"
                else:
                    label = parts[-1]

                label_counts[label] = label_counts.get(label, 0) + 1
                total += 1

    print(f"Dataset: {dataset_file}")
    print(f"Total: {total} positions")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate or analyze chess position datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    gen_sample = subparsers.add_parser(
        "sample", help="Generate sample dataset"
    )
    gen_sample.add_argument("output", help="Output dataset file")
    gen_sample.add_argument(
        "--count", type=int, default=100, help="Number of positions"
    )

    analyze = subparsers.add_parser(
        "analyze", help="Analyze FEN file and label positions"
    )
    analyze.add_argument("input", help="Input FEN file (one per line)")
    analyze.add_argument("output", help="Output labeled dataset")
    analyze.add_argument(
        "--basic",
        action="store_true",
        help="Use basic labels (Check/Checkmate) instead of detailed",
    )

    stats = subparsers.add_parser("stats", help="Show dataset statistics")
    stats.add_argument("dataset", help="Dataset file to analyze")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "sample":
        generate_sample_dataset(args.output, args.count)
    elif args.command == "analyze":
        analyze_fen_file(args.input, args.output, detailed=not args.basic)
    elif args.command == "stats":
        stats_dataset(args.dataset)

    return 0


if __name__ == "__main__":
    sys.exit(main())
