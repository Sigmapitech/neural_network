# Neural Network From Scratch

This project incrementally builds neural network components and tooling from a single perceptron up to a multi‑layer perceptron (MLP) capable of classifying Tic‑Tac‑Toe board states. It includes scripts for training, benchmarking, visualization, serialization, and inference.

## Contents

- Perceptron (AND gate training, XOR limitation demo)
- Multi-Layer Perceptron (XOR solution, Tic-Tac-Toe classification)
- Dataset generation (logic gates + synthesized Tic-Tac-Toe boards)
- Training scripts and CSV logging
- Model serialization (JSON) and inference
- Accuracy & loss visualization overlays
- Hyperparameter benchmarking (XOR)

## Project Structure

```sh
flake.nix                # (Optional) Nix environment
pyproject.toml           # Python dependencies
Makefile                 # Convenience targets (if any defined)
bs/
  my_perceptron.py       # Perceptron CLI
  mlp.py                 # MLP implementation + serialization
  tictactoe.py           # Board generation & labeling utilities
scripts/
  automate_and_gate.py   # Multi-trial AND perceptron automation
  train_xor_mlp.py       # Train MLP on XOR
  train_tictactoe_mlp.py # Train MLP on Tic-Tac-Toe (binary or multi-class)
  infer_mlp.py           # Load serialized model and run inference
  visualize_results.py   # Overlay plots from multiple CSV logs
  benchmark_xor.py       # Systematic hyperparameter benchmarking for XOR
results/                 # (Generated) CSVs, plots, saved models
```

## Environment Setup

Use your preferred method:

### Option 1: Plain venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Option 2: Nix (if flake is configured)

```bash
nix develop
```

Dependencies (from `pyproject.toml`): `matplotlib` plus standard library only for core logic.

## Perceptron Usage (`bs/my_perceptron.py`)

Implements a single-layer perceptron with threshold activation.

### AND Gate Training

Input file format: each line `x1 x2 target` (space-separated).

```bash
python3 bs/my_perceptron.py --train data/and_gate.txt --steps 25 --learning-rate 0.1 --save perceptron_and.json
```

- Performs phased training; each phase loops over samples to update weights until accuracy == 1 or max steps per phase reached.
- Outputs accuracy history per phase.

### Load & Predict

```bash
python3 bs/my_perceptron.py --load perceptron_and.json --predict "1 0"
```

### XOR Limitation Demo

Show non-convergence (≈50% plateau):

```bash
python3 bs/my_perceptron.py --train data/xor_gate.txt --steps 25 --max-phases 40 --learning-rate 0.1
```

This demonstrates linear inseparability (single-layer perceptron cannot solve XOR).

## MLP Overview (`bs/mlp.py`)

- Architecture: Fully-connected feedforward network.
- Activation: Sigmoid for all layers.
- Loss: Mean Squared Error (MSE).
- Training: Mini-batch or full-batch epochs with optional learning rate decay.
- Serialization: JSON via `MLP.to_dict()` / `MLP.from_dict()`.

### Serialization Format (JSON)

```json
{
  "layer_sizes": [input_dim, hidden1, ..., output_dim],
  "learning_rate": 0.1,
  "weights": [ [ [..], ... ], ... ],  // per layer matrix rows (output neuron) of columns (inputs)
  "biases": [ [..], ... ]             // per layer bias vector
}
```

## Train MLP on XOR (`scripts/train_xor_mlp.py`)

```bash
python3 scripts/train_xor_mlp.py --hidden 4 --epochs 5000 --learning-rate 0.3 --save-model results/xor_model.json --log-csv results/xor_run.csv --plot results/xor_curve.png
```

- Adjust `--hidden` and `--learning-rate` if convergence is slow.
- Saves CSV with per-epoch loss & accuracy.

## Tic-Tac-Toe Classification (`scripts/train_tictactoe_mlp.py`)

Generates all legal board states internally (via `bs/tictactoe.py`). Two modes:

- Binary: Output 1 if X (first player) eventually wins, else 0.
- Multi-class: Classes (0: ongoing, 1: X win, 2: O win, 3: draw).

### Binary Training

```bash
python3 scripts/train_tictactoe_mlp.py --mode binary --hidden 64 --epochs 50 --batch-size 128 \
  --learning-rate 0.2 --decay 0.98 --save-model results/tictactoe_binary.json \
  --log-csv results/ttt_binary.csv --plot results/ttt_binary.png
```

### Multi-Class Training

```bash
python3 scripts/train_tictactoe_mlp.py --mode multi --hidden 96 --epochs 60 --batch-size 256 \
  --learning-rate 0.15 --decay 0.99 --save-model results/tictactoe_multi.json \
  --log-csv results/ttt_multi.csv --plot results/ttt_multi.png
```

The script reports training + validation accuracy.

## Inference (`scripts/infer_mlp.py`)

Pass a serialized model and feature vector(s).

```bash
python3 scripts/infer_mlp.py --model results/xor_model.json --inputs "0 1" "1 1"
```

- Outputs raw sigmoid values; script thresholds at 0.5 for binary.
- For multi-class Tic-Tac-Toe, highest activation index is predicted class.

## Visualization Overlay (`scripts/visualize_results.py`)

Aggregate curves from multiple CSV logs (each containing columns like `epoch,accuracy,loss`).

```bash
python3 scripts/visualize_results.py --dir results --metric accuracy --out results/overlay_accuracy.png
python3 scripts/visualize_results.py --dir results --metric loss --out results/overlay_loss.png
```

- Parses scalar or list-form metrics safely.
- Useful for comparing hyperparameter runs.

## XOR Benchmarking (`scripts/benchmark_xor.py`)

Run systematic trials across hidden sizes & learning rates.

```bash
python3 scripts/benchmark_xor.py --hidden-sizes 2 4 8 --learning-rates 0.1 0.3 0.5 --trials 5 \
  --epochs 3000 --out-csv results/xor_benchmark.csv --plot results/xor_benchmark.png
```

CSV includes convergence epoch (or max if not converged).

## CSV Log Format

Each training CSV typically includes:

```csv
epoch,loss,accuracy
0,0.735,0.25
1,0.684,0.50
...
```

Some automation scripts may store `accuracy_history` as a list per trial; visualization handles both simple scalars and list strings.

## Key Concepts Recap

- Perceptron: Linear decision boundary; fails on XOR.
- MLP: Hidden layers introduce non-linear separability enabling XOR & complex pattern recognition.
- Sigmoid Activation: Smooth non-linearity; outputs in (0,1).
- Backpropagation: Chain-rule gradient propagation layer by layer.
- Mini-Batch: Stabilizes updates and can improve convergence speed.
- Learning Rate Decay: Gradually lowers step size to refine towards minima.

## Suggested Extensions

- Replace MSE with cross-entropy + softmax for multi-class.
- Add confusion matrix & precision/recall metrics.
- Implement early stopping on validation loss plateau.
- Integrate grid/random search for hyperparameter tuning.

## Quick Commands Cheat Sheet

```bash
# AND perceptron
python3 bs/my_perceptron.py --train data/and_gate.txt --steps 25 --save perceptron_and.json

# XOR perceptron failure demo
python3 bs/my_perceptron.py --train data/xor_gate.txt --steps 25 --max-phases 40

# XOR MLP training
python3 scripts/train_xor_mlp.py --hidden 4 --epochs 5000 --learning-rate 0.3 --save-model results/xor_model.json

# Tic-Tac-Toe multi-class
python3 scripts/train_tictactoe_mlp.py --mode multi --hidden 96 --epochs 60 --save-model results/tictactoe_multi.json

# Inference
python3 scripts/infer_mlp.py --model results/xor_model.json --inputs "0 1"

# Overlay visualization
python3 scripts/visualize_results.py --dir results --metric accuracy --out results/overlay_accuracy.png

# XOR benchmarking
python3 scripts/benchmark_xor.py --hidden-sizes 2 4 8 --learning-rates 0.1 0.3 0.5 --trials 5 --epochs 3000
```

## Troubleshooting

- Slow XOR convergence: Increase hidden neurons or learning rate moderately (e.g., hidden=8, lr=0.4) without causing instability.
- Plateau on Tic-Tac-Toe validation: Slightly increase hidden size or epochs; ensure decay not too aggressive.
- NaN losses: Check for excessively large learning rate (>1.0); reduce it.
- Visualization parsing errors: Ensure CSV columns have headers `accuracy` and/or `loss`.
