#!/usr/bin/env python3
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET = "dataset"

TESTS = {
    "Checkmate White": ("checkmate", "Checkmate White"),
    "Checkmate Black": ("checkmate", "Checkmate Black"),
    "Check White": ("check", "Check White"),
    "Check Black": ("check", "Check Black"),
    "Nothing": ("nothing", "Nothing"),
}


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return int(result.stdout.strip())


def count_lines_with_pattern(filepath, pattern):
    """Count lines in file that contain the pattern."""
    cmd = f'grep -c "{pattern}" {filepath}'
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return int(result.stdout.strip()) if result.returncode == 0 else 0


def evaluate_model(model):
    output = []
    output.append(model)

    total_correct = 0
    total_samples = 0
    signed_score = 0

    for label, (subdir, grep_pattern) in TESTS.items():
        datafile = os.path.join(DATASET, subdir, "many_pieces.txt")
        total = count_lines_with_pattern(datafile, grep_pattern)

        cmd = (
            f"./my_torch_analyzer --predict {model} {datafile} "
            f'| grep -c "{grep_pattern}"'
        )
        correct = run_cmd(cmd)

        acc = 100.0 * correct / total if total > 0 else 0.0

        output.append(
            f"- {label:<16}: {acc:6.2f}% | {correct:6d} / {total:6d}"
        )

        total_correct += correct
        total_samples += total
        signed_score += 2 * correct - total

    total_acc = (
        100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    )
    signed_pct = (
        100.0 * signed_score / total_samples if total_samples > 0 else 0.0
    )

    output.append(f"=> Total accuracy: {total_acc:.2f}%")
    output.append(f"   absolute: {signed_score:+d} -> {signed_pct:.2f}%\n")

    return "\n".join(output)


def main():
    models = sorted(glob.glob("*.nn") if len(sys.argv) == 1 else sys.argv[1:])
    print(", ".join(models), end="\n\n")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_model, m) for m in models]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
