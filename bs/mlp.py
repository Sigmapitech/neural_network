#!/usr/bin/env python3
"""Minimal Multilayer Perceptron implementation (no external NN libs).

Supports:
 - Arbitrary layer sizes (e.g. [2, 2, 1] for XOR)
 - Sigmoid activation for hidden & output layers
 - Mean Squared Error loss
 - Batch gradient descent (full dataset each epoch)

Designed for educational purposes on tiny datasets like XOR.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(y: float) -> float:
    """Derivative of sigmoid given output y (sigmoid(z))."""
    return y * (1.0 - y)


def softmax(vec: List[float]) -> List[float]:
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


class MLP:
    def __init__(
        self,
        layer_sizes: List[int],
        lr: float = 0.5,
        seed: int | None = None,
        output_activation: str = "sigmoid",  # or 'softmax'
        loss: str = "mse",  # 'mse' or 'ce' (cross-entropy only with softmax)
        class_weights: List[float] | None = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer")
        if any(s <= 0 for s in layer_sizes):
            raise ValueError("All layer sizes must be > 0")
        self.layer_sizes = layer_sizes
        self.lr = lr
        if output_activation not in ("sigmoid", "softmax"):
            raise ValueError(
                "output_activation must be 'sigmoid' or 'softmax'"
            )
        if loss not in ("mse", "ce"):
            raise ValueError("loss must be 'mse' or 'ce'")
        if output_activation == "softmax" and layer_sizes[-1] == 1:
            raise ValueError("softmax output requires >1 output neurons")
        if loss == "ce" and output_activation != "softmax":
            raise ValueError(
                "cross-entropy currently only supported with softmax output"
            )
        self.output_activation = output_activation
        self.loss = loss
        self.class_weights = class_weights
        if seed is not None:
            random.seed(seed)
        # Weights: list of matrices (next_layer_size x current_layer_size)
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = (
            []
        )  # one bias per neuron (excluding input layer)
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            self.weights.append(
                [
                    [random.uniform(-1.0, 1.0) for _ in range(in_size)]
                    for _ in range(out_size)
                ]
            )
            self.biases.append(
                [random.uniform(-0.5, 0.5) for _ in range(out_size)]
            )

    def to_dict(self) -> dict:
        return {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.lr,
            "weights": self.weights,
            "biases": self.biases,
            "output_activation": self.output_activation,
            "loss": self.loss,
            "class_weights": self.class_weights,
        }

    @staticmethod
    def from_dict(data: dict) -> "MLP":
        required = {"layer_sizes", "learning_rate", "weights", "biases"}
        if not required.issubset(data):
            raise ValueError("Invalid MLP model file")
        output_activation = data.get("output_activation", "sigmoid")
        loss = data.get("loss", "mse")
        mlp = MLP(
            data["layer_sizes"],
            lr=data["learning_rate"],
            output_activation=output_activation,
            loss=loss,
            class_weights=data.get("class_weights"),
        )  # initializes sizes
        if len(mlp.weights) != len(data["weights"]):
            raise ValueError("Weights shape mismatch")
        mlp.weights = data["weights"]
        mlp.biases = data["biases"]
        return mlp

    def forward(
        self, inputs: List[float]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError("Input size mismatch")
        activations: List[List[float]] = [inputs[:]]  # layer 0
        zs: List[List[float]] = []  # pre-activation values
        a = inputs
        for layer_idx in range(len(self.weights)):
            w_mat = self.weights[layer_idx]
            b_vec = self.biases[layer_idx]
            z_layer: List[float] = []
            for neuron_idx in range(len(w_mat)):
                w = w_mat[neuron_idx]
                z = (
                    sum(w_j * a_j for w_j, a_j in zip(w, a))
                    + b_vec[neuron_idx]
                )
                z_layer.append(z)
            # Activation choice: last layer may use softmax
            if (
                layer_idx == len(self.weights) - 1
                and self.output_activation == "softmax"
            ):
                a_next = softmax(z_layer)
            else:
                a_next = [sigmoid(z) for z in z_layer]
            zs.append(z_layer)
            activations.append(a_next)
            a = a_next
        return activations, zs

    def predict(self, inputs: List[float]) -> List[float]:
        activations, _ = self.forward(inputs)
        return activations[-1]

    def _backprop_sample(
        self, activations: List[List[float]], target: List[float]
    ) -> Tuple[List[List[List[float]]], List[List[float]], float, bool]:
        output = activations[-1]
        if self.loss == "mse":
            loss = sum(0.5 * (o - t) ** 2 for o, t in zip(output, target))
        else:  # cross-entropy with softmax output
            # Add small epsilon for numerical stability
            eps = 1e-12
            base = -sum(t * math.log(o + eps) for o, t in zip(output, target))
            if self.class_weights:
                # weight by true class
                w_true = 0.0
                for i, t in enumerate(target):
                    if t > 0.0:
                        w_true = self.class_weights[i]
                        break
                loss = w_true * base
            else:
                loss = base
        is_correct = False
        if len(target) == 1:
            pred_bin = 1 if output[0] >= 0.5 else 0
            is_correct = pred_bin == int(target[0])
        else:
            pred_idx = max(range(len(output)), key=lambda i: output[i])
            tgt_idx = max(range(len(target)), key=lambda i: target[i])
            is_correct = pred_idx == tgt_idx
        deltas: List[List[float]] = [None] * (len(self.layer_sizes) - 1)  # type: ignore
        last_layer_idx = len(self.layer_sizes) - 2
        out_acts = activations[-1]
        delta_out: List[float] = []
        for i in range(len(out_acts)):
            if self.output_activation == "softmax" and self.loss == "ce":
                # Softmax + cross-entropy simplifies gradient; apply class weighting if provided
                scale = 1.0
                if self.class_weights:
                    for k, t in enumerate(target):
                        if t > 0.0:
                            scale = self.class_weights[k]
                            break
                delta_out.append(scale * (out_acts[i] - target[i]))
            else:
                error = out_acts[i] - target[i]
                delta_out.append(error * sigmoid_derivative(out_acts[i]))
        deltas[last_layer_idx] = delta_out
        for layer_idx in range(last_layer_idx - 1, -1, -1):
            layer_deltas: List[float] = []
            for i in range(self.layer_sizes[layer_idx + 1]):
                s = 0.0
                for j in range(self.layer_sizes[layer_idx + 2]):
                    s += (
                        self.weights[layer_idx + 1][j][i]
                        * deltas[layer_idx + 1][j]
                    )
                a_val = activations[layer_idx + 1][i]
                layer_deltas.append(s * sigmoid_derivative(a_val))
            deltas[layer_idx] = layer_deltas
        grad_w = [
            [
                [0.0 for _ in range(len(self.weights[layer][0]))]
                for _ in range(len(self.weights[layer]))
            ]
            for layer in range(len(self.weights))
        ]
        grad_b = [[0.0 for _ in layer_biases] for layer_biases in self.biases]
        for layer_idx in range(len(self.weights)):
            for neuron_idx in range(len(self.weights[layer_idx])):
                for w_idx in range(len(self.weights[layer_idx][neuron_idx])):
                    grad_w[layer_idx][neuron_idx][w_idx] = (
                        deltas[layer_idx][neuron_idx]
                        * activations[layer_idx][w_idx]
                    )
                grad_b[layer_idx][neuron_idx] = deltas[layer_idx][neuron_idx]
        return grad_w, grad_b, loss, is_correct

    def train_epoch(
        self,
        dataset: List[Tuple[List[float], List[float]]],
        batch_size: int = 0,
    ) -> Tuple[float, float]:
        if batch_size <= 0 or batch_size > len(dataset):
            batch_size = len(dataset)
        total_loss = 0.0
        total_correct = 0
        random.shuffle(dataset)
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            acc_grad_w = [
                [
                    [0.0 for _ in range(len(self.weights[layer][0]))]
                    for _ in range(len(self.weights[layer]))
                ]
                for layer in range(len(self.weights))
            ]
            acc_grad_b = [
                [0.0 for _ in layer_biases] for layer_biases in self.biases
            ]
            for x, target in batch:
                activations, _ = self.forward(x)
                grad_w, grad_b, loss, is_correct = self._backprop_sample(
                    activations, target
                )
                total_loss += loss
                if is_correct:
                    total_correct += 1
                for li in range(len(self.weights)):
                    for ni in range(len(self.weights[li])):
                        for wi in range(len(self.weights[li][ni])):
                            acc_grad_w[li][ni][wi] += grad_w[li][ni][wi]
                        acc_grad_b[li][ni] += grad_b[li][ni]
            bsz = len(batch)
            for li in range(len(self.weights)):
                for ni in range(len(self.weights[li])):
                    for wi in range(len(self.weights[li][ni])):
                        self.weights[li][ni][wi] -= (
                            self.lr * acc_grad_w[li][ni][wi] / bsz
                        )
                    self.biases[li][ni] -= self.lr * acc_grad_b[li][ni] / bsz
        m = len(dataset)
        avg_loss = total_loss / m
        accuracy = total_correct / m if m else 0.0
        return avg_loss, accuracy

    def train(
        self,
        dataset: List[Tuple[List[float], List[float]]],
        epochs: int = 5000,
        target_accuracy: float = 1.0,
        verbose: bool = True,
        batch_size: int = 0,
        lr_decay: float = 0.0,
    ) -> List[Tuple[int, float, float]]:
        history: List[Tuple[int, float, float]] = []
        for epoch in range(1, epochs + 1):
            loss, acc = self.train_epoch(dataset, batch_size=batch_size)
            history.append((epoch, loss, acc))
            if verbose and (epoch % 500 == 0 or acc >= target_accuracy):
                print(f"Epoch {epoch}: loss={loss:.4f} acc={acc*100:.1f}%")
            if acc >= target_accuracy:
                break
            if lr_decay > 0:
                self.lr *= 1.0 - lr_decay
        return history


def build_xor_dataset() -> List[Tuple[List[float], List[float]]]:
    return [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]


def demo_xor(hidden: int = 2, lr: float = 0.5, epochs: int = 5000):
    mlp = MLP([2, hidden, 1], lr=lr)
    data = build_xor_dataset()
    history = mlp.train(data, epochs=epochs, target_accuracy=1.0, verbose=True)
    print("-- Final evaluation --")
    for x, t in data:
        y = mlp.predict(x)[0]
        print(
            f"INPUT={x} PRED={y:.4f} BIN={(1 if y>=0.5 else 0)} EXPECTED={int(t[0])}"
        )
    return mlp, history


if __name__ == "__main__":  # Manual demo
    demo_xor()
