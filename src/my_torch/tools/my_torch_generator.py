#!/usr/bin/env python3
import argparse
import sys

from generator import generate_networks


def main():
    parser = argparse.ArgumentParser(
        description="Generate neural networks from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  ./my_torch_generator basic_network.conf 3

This will create: basic_network_1.nn, basic_network_2.nn, basic_network_3.nn
        """,
    )

    parser.add_argument(
        "configs",
        nargs="+",
        help="Pairs of config_file and count (e.g., config.json 3)",
    )

    args = parser.parse_args()

    if len(args.configs) % 2 != 0:
        print("Error: Arguments must be pairs of config_file and count")
        return 1

    for i in range(0, len(args.configs), 2):
        config_file = args.configs[i]
        try:
            count = int(args.configs[i + 1])
        except ValueError:
            print(f"Error: '{args.configs[i + 1]}' is not a valid number")
            return 1

        if count <= 0:
            print(f"Error: count must be positive, got {count}")
            return 1

        try:
            generate_networks(config_file, count)
        except FileNotFoundError:
            print(f"Error: Config file '{config_file}' not found")
            return 1
        except Exception as e:
            print(f"Error processing {config_file}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
