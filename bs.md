# Neural Network From Scratch

This document explains the full journey of building neural models from first principles: a single perceptron, its training dynamics and limitations, then a handcrafted multi‑layer perceptron (MLP) with backpropagation used to solve XOR and classify Tic‑Tac‑Toe positions. It emphasizes UNDER‑THE‑HOOD mechanics (weight updates, gradient flow, dataset generation) rather than using external ML libraries.

## High‑Level Progression

1. Perceptron: linear separator; succeeds on AND; fails on XOR (proof of need for hidden layers).
2. Automation: statistical view of perceptron convergence behavior (multiple trials).
3. MLP Core: forward propagation + explicit manual backprop for arbitrary layer sizes.
4. XOR Solution: minimal hidden layer enables non‑linear separability.
5. Tic‑Tac‑Toe: synthetic state space generation; supervised classification.
6. Tooling: serialization, inference, visualization, benchmarking.

## Implemented Components

- Perceptron CLI with custom argument parser (no `argparse`): creation, training, prediction, save/load.
- AND gate multi‑trial automation: convergence phases, average accuracy curve.
- Minimal MLP: sigmoid activations, MSE loss, batch & mini‑batch gradient descent, optional learning rate decay, JSON persistence.
- XOR trainer script + benchmarking grid for convergence epoch statistics.
- Tic‑Tac‑Toe dataset generator enforcing legal game states (alternating turns, early stop at win, draw detection).
- Tic‑Tac‑Toe training script with validation split and early stopping by accuracy.
- Inference & curve visualization utilities (aggregated overlays).

---

## Mathematical Foundations

### Perceptron Decision Rule

Given inputs $x \in \mathbb{R}^n$, weights $w \in \mathbb{R}^n$, bias $b$:

$$
z = \sum_i w_i x_i + b,\quad
\hat y = H(z) = \begin{cases}
1 & z \ge 0 \\
0 & \text{otherwise}
\end{cases}
$$

Clarifying notation: $x \in \mathbb{R}^n$ means $x$ is an $n$‑dimensional real‑valued feature vector, i.e., $x = (x_1, x_2, \dots, x_n)$ with each $x_i \in \mathbb{R}$; similarly $w \in \mathbb{R}^n$ and $b \in \mathbb{R}$.

Alternate activation (tanh threshold) provided for experimentation.

### Perceptron Update (Online Misclassification Correction)

For a sample $(x, y)$ with prediction $\hat y$:

$$
  \text{error} = y - \hat y,\quad
w_i \leftarrow w_i + \mathrm{lr}\,\cdot\,\text{error}\,\cdot\, x_i,\quad
b \leftarrow b + \mathrm{lr}\,\cdot\,\text{error}.
$$

Only misclassified samples update parameters (classic perceptron rule, not gradient descent).

### MLP Forward Pass

Layer sizes: $L_0$ (input) … $L_k$ (output). For neuron $j$ in layer $L$:

$$
z_j^{(L)} = \sum_i w_{j,i}^{(L)}\, a_i^{(L-1)} + b_j^{(L)},\quad
a_j^{(L)} = \sigma\big(z_j^{(L)}\big) = \frac{1}{1 + e^{-z_j^{(L)}}}.
$$

Internal storage:

- `weights[L][j][i]` = weight from neuron i in previous layer to neuron j in current layer.
- `biases[L][j]` = bias for neuron j.
- Activations list includes input layer as index 0.

### Loss (Per Sample, MSE)

$$
\mathcal{L} = \sum_j \tfrac{1}{2}\,\big(a_j^{(\text{out})} - y_j\big)^2.
$$

Using MSE for simplicity (cross‑entropy would be more suitable for classification but requires softmax modifications).

### Backpropagation Derivatives

Sigmoid derivative given activation $a$: $\sigma'(a) = a(1-a)$.

Output deltas:

$$
\delta_j^{(\text{out})} = \big(a_j^{(\text{out})} - y_j\big)\,\sigma'\!\big(a_j^{(\text{out})}\big).
$$

Hidden layer deltas (chain rule):

$$
\delta_i^{(L)} = \left( \sum_j w_{j,i}^{(L+1)}\, \delta_j^{(L+1)} \right)\, \sigma'\!\big(a_i^{(L)}\big).
$$

Weight & bias gradients:

$$
\frac{\partial \mathcal{L}}{\partial w_{j,i}^{(L)}} = \delta_j^{(L)}\, a_i^{(L-1)},\quad
\frac{\partial \mathcal{L}}{\partial b_j^{(L)}} = \delta_j^{(L)}.
$$

Mini‑batch accumulation sums gradients across the batch; final update divides by batch size (average gradient).

### Training Loop (Epoch)

For each batch:

1. Initialize accumulators for grad_w, grad_b.
2. Forward each sample → activations.
3. Compute deltas via backward pass.
4. Accumulate gradients.
5. After batch: apply averaged update.
6. Track accuracy (#correct / #samples). Binary uses threshold; multi‑class uses argmax.

### Early Stopping

MLP training breaks early if `accuracy >= target_accuracy`.

---

## Project Layout

```sh
bs/
  my_perceptron.py       # Perceptron CLI (custom argument parsing)
  mlp.py                 # MLP core: forward, backprop, train
  tictactoe.py           # Dataset generation & labeling helpers
scripts/
  automate_and_gate.py   # Batch AND gate perceptron trials
  train_xor_mlp.py       # XOR training script (MLP)
  benchmark_xor.py       # Hyperparameter convergence benchmark for XOR
  visualize_results.py   # Overlay multiple CSV curves
  infer_mlp.py           # Load JSON model & predict
  train_tictactoe_mlp.py # Tic-Tac-Toe position classifier trainer
results/                 # Generated CSVs / plots / saved models
data/                    # AND & XOR gate truth tables
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional plotting via `matplotlib` (included in `pyproject.toml`).

## Perceptron Internals (`bs/my_perceptron.py`)

Equation: $\hat y = H\!\left(\sum_i w_i x_i + b\right)$.
Update rule (per misclassified sample): `w_i ← w_i + lr * (y - ŷ) * x_i`, `b ← b + lr * (y - ŷ)`.

Training uses PHASES:

1. Evaluate accuracy on full dataset.
2. Perform `--steps` random single-sample updates.
3. Repeat until 100% accuracy or `--max-phases` reached.

CLI (custom parser, no `argparse`):

```bash
# Train AND gate
python3 bs/my_perceptron.py --new 2 --train --steps 40 data/and_gate.txt --save perceptron_and.json

# Predict with saved model
python3 bs/my_perceptron.py --load perceptron_and.json data/and_gate.txt

# Demonstrate XOR failure (plateau)
python3 bs/my_perceptron.py --new 2 --train --steps 40 --max-phases 30 data/xor_gate.txt
```

Output includes per-phase accuracy history.

Prediction file format: lines may contain either just the inputs or `inputs + label` (0/1). In prediction mode the label, if present, is ignored, allowing reuse of the original training truth table without manual editing.

## AND Gate Automation (`scripts/automate_and_gate.py`)

Purpose: quantify variability in phases needed for convergence by repeated random initialization.

Flags:

- `--trials` number of independent runs
- `--steps` updates per phase (same semantics as perceptron CLI)
- `--lr` learning rate for each perceptron created
- `--out` output directory

Outputs:

- `and_gate_trials.csv`: columns `trial,phases_to_converge,weights,bias`
- `and_gate_accuracy_curve.csv`: phase vs average accuracy (%) across trials
- Optional plot `and_gate_learning_curve.png` if matplotlib installed.

Interpretation: Distribution of phases quantifies stability; fewer phases ⇒ weights closer to separating hyperplane early.

## Tic‑Tac‑Toe Dataset Generation (`bs/tictactoe.py`)

### Board Representation

Flat list of 9 integers: X=1, O=-1, empty=0. Row‑major ordering.

### Legality Rules Enforced

- X always moves first.
- Players strictly alternate (counts differ by at most 1; O never exceeds X).
- Search halts further expansion after first win (no post‑win moves included).
- Draw: board full with no winner.

### Generation Algorithm (Depth‑First Search)

```sh
rec(board):
  add board to visited
  if winner(board) or draw(board): mark finished; return
  player = next_player(board)
  for each empty cell i:
     board[i] = player
     if legal(board): rec(board)
     board[i] = 0
```

This prunes illegal turn orders and duplicated states, producing a compact supervised dataset of reachable positions.

### Labeling

- Binary: `[1]` if a win has already occurred else `[0]`.
- Multi: `[ongoing, draw, win]` one‑hot.

### Training Script (`scripts/train_tictactoe_mlp.py`)

Performs split into train/validation, mini‑batch gradient descent, optional plotting & saving.

Flags:

```sh
--mode {binary,multi}
--hidden HIDDEN_SIZE
--epochs N_EPOCHS
--lr LEARNING_RATE
--batch BATCH_SIZE (0 = full dataset)
--lr-decay DECAY_FRACTION
--val VALIDATION_RATIO
--target TARGET_TRAIN_ACCURACY (early stop)
--out OUTPUT_DIR
--no-plot (suppress plot)
--save-model MODEL_PATH.json
```

Examples:

```bash
# Binary classification (win vs not yet win)
python3 scripts/train_tictactoe_mlp.py --mode binary --hidden 64 --epochs 60 --lr 0.25 \
  --batch 128 --lr-decay 0.002 --target 0.95 --out results --save-model results/tictactoe_binary.json

# Multi-class (ongoing / draw / win)
python3 scripts/train_tictactoe_mlp.py --mode multi --hidden 96 --epochs 80 --lr 0.3 \
  --batch 256 --lr-decay 0.001 --target 0.90 --out results --save-model results/tictactoe_multi.json
```

Outputs:

- CSV: `tictactoe_<mode>_curve.csv` (epoch,loss,accuracy)
- Plot: `tictactoe_<mode>_learning.png` (loss & train accuracy) unless `--no-plot`
- JSON model (if `--save-model`)
- Console summary: final train & validation accuracy + sample validation predictions.
Hidden layer deltas:

$$
\delta_i^{(L)} = \left( \sum_j w_{j,i}^{(L+1)}\, \delta_j^{(L+1)} \right)\, \sigma'\!\big(a_i^{(L)}\big).
$$

Gradients:

$$
\frac{\partial \mathcal{L}}{\partial w_{j,i}^{(L)}} = \delta_j^{(L)}\, a_i^{(L-1)},\quad
\frac{\partial \mathcal{L}}{\partial b_j^{(L)}} = \delta_j^{(L)}.
$$

Batch update (average gradient over batch) using learning rate `lr`.

Mini‑batch: specify `batch_size` in `train_epoch`; if ≤0 uses full dataset.

Learning rate decay: each epoch $\mathrm{lr} \leftarrow \mathrm{lr}\,(1 - \mathrm{lr\_decay})$ when `lr_decay > 0`.

## XOR Training (`scripts/train_xor_mlp.py`)

Flags:

- `--hidden`: hidden layer size
- `--lr`: learning rate
- `--epochs`: max epochs
- `--no-plot`: disable plot
- `--out`: directory for `xor_mlp_curve.csv` & optional plot
- `--save-model`: JSON output path

Example:

```bash
python3 scripts/train_xor_mlp.py --hidden 4 --lr 0.5 --epochs 5000 --out results --save-model results/xor_model.json
```

CSV columns: `epoch,loss,accuracy` (accuracy in [0,1]).

## Benchmark XOR (`scripts/benchmark_xor.py`)

Grid of hidden sizes × learning rates × trials.
Flags: `--hidden`, `--lr`, `--epochs`, `--trials`, `--out`.
Output CSV: `hidden,lr,trial,epochs_to_converge`.
Optional scatter plot if matplotlib present.

## Inference (`scripts/infer_mlp.py`)

```bash
python3 scripts/infer_mlp.py --model results/xor_model.json --input 0 1
```

What it does: loads a saved JSON model and runs a forward pass on a single input vector you provide on the command line.

Usage:

```bash
python3 scripts/infer_mlp.py --model <path/to/model.json> --input <space-separated numbers>
```

- Input length must match the model’s input size (`layer_sizes[0]`).
- Tic‑Tac‑Toe inputs are 9 numbers with X=1, O=−1, empty=0 (row‑major).

Helpful flags:

- `--labels`: when output has 3 classes, also prints the class name (order `[ongoing, draw, win]`).
- `--ttt-gt`: for 9‑value inputs, computes and prints the Tic‑Tac‑Toe ground‑truth class using the rule helpers.

Output interpretation:

- Binary models: prints one sigmoid value in [0,1] and a line `Binary Thresholded=0|1` (threshold 0.5).
- Multi‑class (Tic‑Tac‑Toe): prints the list of three sigmoid values and a line `Argmax Class=<index>`. Class order used by training is `[ongoing, draw, win]`. Values do not necessarily sum to 1 (not softmax).

Examples:

```text
Input=[0, 1]
Raw Output=[0.842]
Binary Thresholded=1
```

```text
Input=[0, 0, 0, 0, 1, 0, 0, 0, 0]
Raw Output=[0.903, 0.001, 0.099]
Argmax Class=0   # class 0 = "ongoing"
```

With labels and ground truth:

```bash
python3 scripts/infer_mlp.py --model results/tictactoe_multi.json \
  --input 1 0 1 -1 -1 -1 0 1 0 --labels --ttt-gt
```

```text
Input=[1.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0]
Raw Output=[...]
Argmax Class=0
Argmax Label=ongoing
GroundTruth Class=2 Label=win (0=ongoing,1=draw,2=win)
```

If you prefer probability‑like outputs that sum to 1, switch the final layer to softmax (or add a softmax option for inference).

## Visualization (`scripts/visualize_results.py`)

Overlay curves across all CSVs in a directory.
Flags: `--dir`, `--prefix` (filter by filename prefix), `--metric` (`loss|accuracy`), `--out`.
Accuracy displayed as percentage (internally multiplies by 100).

```bash
python3 scripts/visualize_results.py --dir results --metric accuracy --out results/overlay_accuracy.png
```

## Serialization (`MLP.to_dict` / `MLP.from_dict`)

Structure:

```json
{
  "layer_sizes": [n_in, h1, ..., n_out],
  "learning_rate": 0.3,
  "weights": [[[...]]],
  "biases": [[...]]
}
```

### Save Model

```python
with open('model.json', 'w') as f:
    json.dump(mlp.to_dict(), f)
```

Persist with Python `json.dump`; load using `MLP.from_dict` (shape consistency validated).
    json.dump(mlp.to_dict(), f)

## Serialization Format

```json
{
  "layer_sizes": [2,4,1],
  "learning_rate": 0.5,
  "weights": [[[...],[...]], ...],
  "biases": [[...], ...]
}
```

Load with `MLP.from_dict()` via `infer_mlp.py`.

## CSV Conventions

All training scripts store epoch-level rows: `epoch,loss,accuracy` (accuracy ∈ [0,1]). Visualization multiplies accuracy by 100 for plotting.
Automation script for AND gate uses different CSVs: `and_gate_trials.csv` & `and_gate_accuracy_curve.csv`.

## Design Choices & Limitations

- Simplicity prioritized: raw Python lists for matrices; clear loops over vectorized operations.
- MSE for classification: fine on tiny problems; replace with cross‑entropy + softmax for multi‑class robustness.
- Sigmoid everywhere: introduces potential saturation; depth kept minimal to mitigate vanishing gradients.
- No momentum / Adam / weight decay: educational focus on base gradient descent.
- Early stopping uses accuracy only; could add validation loss patience.
- Deterministic dataset generation ensures reproducible Tic‑Tac‑Toe splits (random only in shuffling & weight init).

## Troubleshooting

- XOR does not converge: increase hidden size (`--hidden 4` or `8`) or adjust `--lr` (0.3–0.8). Extremely high `--lr` may destabilize.
- Loss plateaus: try lowering learning rate or using smaller batch (set `batch_size` < dataset length).
- NaN values: usually from large weights + high `lr`; restart with lower `lr`.
- Visualization finds no files: confirm CSVs in `results/` and metric columns present.

## Quick Reference Commands

```bash
# Perceptron AND training
python3 bs/my_perceptron.py --new 2 --train --steps 40 data/and_gate.txt --save perceptron_and.json

# Perceptron XOR failure
python3 bs/my_perceptron.py --new 2 --train --steps 40 --max-phases 30 data/xor_gate.txt

# XOR MLP training
python3 scripts/train_xor_mlp.py --hidden 4 --lr 0.5 --epochs 5000 --out results --save-model results/xor_model.json

# XOR benchmarking
python3 scripts/benchmark_xor.py --hidden 2 3 4 5 --lr 0.3 0.5 0.8 --epochs 5000 --trials 3 --out results/xor_benchmark.csv

# Inference
python3 scripts/infer_mlp.py --model results/xor_model.json --input 0 1

# Visualization overlay
python3 scripts/visualize_results.py --dir results --metric accuracy --out results/overlay_accuracy.png
```

## Extending Next

- Cross‑entropy + softmax output layer.
- Confusion matrix + precision/recall (macro + per class).
- Momentum / Adam optimizer abstraction.
- Gradient clipping for stability on larger hidden layers.
- Adjustable initialization (Xavier/He) to reduce early saturation.
- Export to ONNX or a lightweight inference format.
- Add command to compute Tic‑Tac‑Toe board feature importance (per input perturbation).
