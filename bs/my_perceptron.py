#!/usr/bin/env python3
import json
import math
import random
import sys
from typing import List, Optional

USAGE = """USAGE
    ./my_perceptron [--new NB_INPUTS | --load LOADFILE] [--save SAVEFILE] [--train] [--steps N] [--max-phases M] FILE

DESCRIPTION
    --new    Creates a new perceptron with NB_INPUTS inputs.
    --load   Loads an existing perceptron from LOADFILE.
    --save   Saves the perceptron's state into SAVEFILE. If not provided, the state
                                         will be displayed on standard output.
        --train  Enables training mode (FILE must contain inputs plus expected output as last value).
        --steps  Number of random training updates per training phase (default: 50).
        --max-phases  Maximum training phases before stopping (useful for non-linearly separable data like XOR). Default: unlimited.
        FILE     File containing samples. In prediction mode: each line lists inputs. In training mode:
                         each line lists inputs followed by expected output (0 or 1) for AND gate.
"""


def error(msg: str, code: int = 84):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


class Perceptron:
    def __init__(
        self,
        weights: List[float],
        bias: float,
        lr: float = 0.1,
        activation: str = "heaviside",
    ):
        self.weights = weights
        self.bias = bias
        self.lr = lr
        self.activation = activation

    @staticmethod
    def new(n_inputs: int, lr: float = 0.1, activation: str = "heaviside"):
        if n_inputs <= 0:
            error("NB_INPUTS must be > 0")
        weights = [random.random() for _ in range(n_inputs)]
        bias = random.random()
        return Perceptron(weights, bias, lr, activation)

    @staticmethod
    def load(path: str):
        data: dict = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            error(f"Cannot load perceptron: {e}")
        required = {"weights", "bias", "learning_rate", "activation"}
        if not required.issubset(data):
            error("Invalid perceptron file format")
        return Perceptron(
            weights=data["weights"],
            bias=data["bias"],
            lr=data["learning_rate"],
            activation=data["activation"],
        )

    def dump(self) -> dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.lr,
            "activation": self.activation,
        }

    def _activate(self, x: float) -> int:
        if self.activation == "heaviside":
            return 1 if x >= 0 else 0
        elif self.activation == "tanh":
            t = math.tanh(x)
            return 1 if t >= 0 else 0
        else:
            return 1 if x >= 0 else 0

    def predict(self, inputs: List[float]) -> int:
        if len(inputs) != len(self.weights):
            error("Input vector size mismatch")
        s = sum(w * v for w, v in zip(self.weights, inputs)) + self.bias
        return self._activate(s)


def parse_args(argv: List[str]):
    if "--help" in argv or len(argv) == 1:
        print(USAGE)
        sys.exit(0)
    args = argv[1:]
    new_idx = load_idx = save_idx = steps_idx = maxph_idx = None
    train_flag = False
    file_arg: Optional[str] = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--new":
            if new_idx is not None:
                error("Duplicate --new")
            if i + 1 >= len(args):
                error("Missing NB_INPUTS after --new")
            new_idx = i
            i += 2
        elif a == "--load":
            if load_idx is not None:
                error("Duplicate --load")
            if i + 1 >= len(args):
                error("Missing LOADFILE after --load")
            load_idx = i
            i += 2
        elif a == "--save":
            if save_idx is not None:
                error("Duplicate --save")
            if i + 1 >= len(args):
                error("Missing SAVEFILE after --save")
            save_idx = i
            i += 2
        elif a == "--steps":
            if steps_idx is not None:
                error("Duplicate --steps")
            if i + 1 >= len(args):
                error("Missing number after --steps")
            steps_idx = i
            i += 2
        elif a == "--train":
            if train_flag:
                error("Duplicate --train")
            train_flag = True
            i += 1
        elif a == "--max-phases":
            if maxph_idx is not None:
                error("Duplicate --max-phases")
            if i + 1 >= len(args):
                error("Missing number after --max-phases")
            maxph_idx = i
            i += 2
        else:
            if file_arg is not None:
                error("Multiple FILE arguments")
            file_arg = a
            i += 1
    if (new_idx is None) == (load_idx is None):
        error("Specify exactly one of --new or --load")
    if file_arg is None:
        error("Missing FILE argument")
    return (
        args,
        new_idx,
        load_idx,
        save_idx,
        steps_idx,
        maxph_idx,
        train_flag,
        file_arg,
    )


def read_input_file(path: str) -> List[List[float]]:
    samples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace(",", " ")
                parts = [p for p in line.split() if p]
                try:
                    vec = [float(x) for x in parts]
                    samples.append(vec)
                except ValueError:
                    error(f"Invalid numeric value in input file: {line}")
    except Exception as e:
        error(f"Cannot read input FILE: {e}")
    return samples


def read_training_file(
    path: str, n_inputs: int
) -> List[tuple[List[float], int]]:
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace(",", " ")
                parts = [p for p in line.split() if p]
                nums: List[float] = []
                try:
                    nums = [float(x) for x in parts]
                except ValueError:
                    error(f"Invalid numeric value in training file: {line}")
                if len(nums) != n_inputs + 1:
                    error(
                        f"Training sample size mismatch (expected {n_inputs + 1} numbers): {line}"
                    )
                inputs = nums[:-1]
                expected_raw = nums[-1]
                if expected_raw not in (0.0, 1.0):
                    error(f"Expected output must be 0 or 1: {line}")
                data.append((inputs, int(expected_raw)))
    except Exception as e:
        error(f"Cannot read training FILE: {e}")
    if not data:
        error("Empty training file")
    return data


def train_perceptron(
    perceptron: Perceptron,
    dataset: List[tuple[List[float], int]],
    steps: int,
    max_phases: Optional[int] = None,
):
    """Train perceptron until perfect accuracy.

    Returns (perceptron, history) where history is a list of
    (phase_index, accuracy_percent). Phase 0 is before any updates.
    """
    history: List[tuple[int, float]] = []
    iteration = 0
    while True:
        # Compute accuracy at start of phase
        correct = sum(1 for x, y in dataset if perceptron.predict(x) == y)
        acc = correct / len(dataset) * 100.0
        history.append((iteration, acc))
        if acc == 100.0:
            print(f"Converged after {iteration} training phases.")
            break
        if max_phases is not None and iteration >= max_phases:
            print(
                f"Stopped after {iteration} phases (max reached). Best accuracy={acc:.1f}%."
            )
            break
        # Phase 2: Training (random single-sample updates)
        for _ in range(steps):
            x, y = random.choice(dataset)
            pred = perceptron.predict(x)
            err = y - pred
            if err != 0:
                for i in range(len(perceptron.weights)):
                    perceptron.weights[i] += perceptron.lr * err * x[i]
                perceptron.bias += perceptron.lr * err
        iteration += 1
        if iteration % 10 == 0:
            print(
                f"Phase {iteration}: accuracy={acc:.1f}% (steps per phase={steps})"
            )
    return perceptron, history


def main():
    (
        args,
        new_idx,
        load_idx,
        save_idx,
        steps_idx,
        maxph_idx,
        train_flag,
        file_arg,
    ) = parse_args(sys.argv)
    n_inputs = 0
    if new_idx is not None:
        try:
            n_inputs = int(args[new_idx + 1])
        except ValueError:
            error("NB_INPUTS must be an integer")
        perceptron = Perceptron.new(n_inputs)
    else:
        # load_idx cannot be None here (enforced in parse_args)
        assert load_idx is not None
        perceptron = Perceptron.load(args[load_idx + 1])
    n_inputs = len(perceptron.weights)
    assert file_arg is not None

    if train_flag:
        steps = 50
        if steps_idx is not None:
            try:
                steps = int(args[steps_idx + 1])
            except ValueError:
                error("--steps value must be an integer")
            if steps <= 0:
                error("--steps must be > 0")
        max_phases: Optional[int] = None
        if maxph_idx is not None:
            max_phases_val: Optional[int] = None
            try:
                max_phases_val = int(args[maxph_idx + 1])
            except ValueError:
                error("--max-phases value must be an integer")
            if max_phases_val is None or max_phases_val <= 0:
                error("--max-phases must be > 0")
            max_phases = max_phases_val
        dataset = read_training_file(file_arg, n_inputs)
        perceptron, history = train_perceptron(
            perceptron, dataset, steps, max_phases
        )
        # After training show final evaluation
        print("-- Final evaluation --")
        for x, y in dataset:
            print(f"INPUT={x} PREDICTED={perceptron.predict(x)} EXPECTED={y}")
        print("-- Accuracy history (phase, %) --")
        for phase, acc in history:
            print(f"{phase}\t{acc:.1f}")
    else:
        samples = read_input_file(file_arg)
        outputs = []
        cleaned_inputs = []
        for s in samples:
            # Allow prediction file lines to optionally include a trailing label (0/1) like training data.
            if len(s) == len(perceptron.weights):
                cleaned_inputs.append(s)
            elif len(s) == len(perceptron.weights) + 1 and s[-1] in (0.0, 1.0):
                cleaned_inputs.append(s[:-1])  # ignore provided label
            else:
                error(
                    "Sample size does not match perceptron input count (expected inputs or inputs+label)"
                )
        for inp in cleaned_inputs:
            outputs.append(perceptron.predict(inp))

        # Display predictions (show original line if label stripped)
        for raw, inp, out in zip(samples, cleaned_inputs, outputs):
            suffix = ""
            if len(raw) == len(perceptron.weights) + 1 and raw[-1] in (
                0.0,
                1.0,
            ):
                suffix = f" (label={int(raw[-1])} ignored)"
            print(f"INPUT={inp} OUTPUT={out}{suffix}")

    # Save or print state
    if save_idx is not None:
        save_path = args[save_idx + 1]
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(perceptron.dump(), f, indent=2)
        except Exception as e:
            error(f"Cannot save perceptron: {e}")
    else:
        print("-- Perceptron state --")
        print(json.dumps(perceptron.dump(), indent=2))


if __name__ == "__main__":
    main()
