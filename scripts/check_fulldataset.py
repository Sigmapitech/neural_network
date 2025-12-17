#!/usr/bin/env python3
import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET = "dataset"

TESTS = {
    "Checkmate": ("checkmate", "Checkmate"),
    # greping with an extra space to match only proper checks
    "Check": ("check", "Check "),
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


def count_lines(filepath):
    with open(filepath) as f:
        return sum(1 for _ in f)


def evaluate_model(model):
    output = []
    output.append(model)

    total_correct = 0
    total_samples = 0
    signed_score = 0

    for label, (subdir, grep_pattern) in TESTS.items():
        datafile = os.path.join(DATASET, subdir, "many_pieces.txt")
        total = count_lines(datafile)

        cmd = (
            f"./my_torch_analyzer --predict {model} {datafile} "
            f'| grep -c "{grep_pattern}"'
        )
        correct = run_cmd(cmd)

        acc = 100.0 * correct / total

        output.append(
            f"- {label:<10}: {acc:6.2f}% | {correct:6d} / {total:6d}"
        )

        total_correct += correct
        total_samples += total
        signed_score += 2 * correct - total

    total_acc = 100.0 * total_correct / total_samples
    signed_pct = 100.0 * signed_score / total_samples

    output.append(f"=> Total accuracy: {total_acc:.2f}%")
    output.append(f"   absolute: {signed_score:+d} -> {signed_pct:.2f}%\n")

    return "\n".join(output)


def main():
    models = sorted(glob.glob("*.nn"))
    print(", ".join(models))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_model, m) for m in models]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
