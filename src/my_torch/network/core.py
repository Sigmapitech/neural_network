from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, List, Self, Tuple

import numpy as np

from .. import activations, losses
from ..optimizer import Optimizer, SGDOptimizer


class Network:
    def __init__(
        self,
        weights,
        biases,
        layer_sizes: List[int],
        hidden_activation: str = "sigmoid",
        output_activation: str = "sigmoid",
        loss: str = "mse",
        optimizer: Optimizer | None = None,
        class_weights: List[float] | None = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer")
        if any(s <= 0 for s in layer_sizes):
            raise ValueError("All layer sizes must be > 0")

        self.biases = biases
        self.weights = weights

        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss
        self.class_weights = class_weights
        self.optimizer = optimizer or SGDOptimizer(learning_rate=0.3)

        if output_activation == "softmax" and layer_sizes[-1] == 1:
            raise ValueError("Softmax requires >1 output neurons")
        if loss == "ce" and output_activation not in ("softmax",):
            raise ValueError("Cross-entropy requires softmax output")

    @classmethod
    def random_net(
        cls, layer_sizes: List[int], seed: int | None = None, **kwargs
    ) -> Self:
        if seed is not None:
            random.seed(seed)

        weights: List[List[List[float]]] = []
        biases: List[List[float]] = []

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            scale = (2.0 / in_size) ** 0.5
            weights.append(
                [
                    [random.gauss(0, scale) for _ in range(in_size)]
                    for _ in range(out_size)
                ]
            )
            biases.append([random.uniform(-0.1, 0.1) for _ in range(out_size)])

        return cls(
            layer_sizes=layer_sizes,
            weights=weights,
            biases=biases,
            **kwargs,
        )

    def _compute_loss(self, output: List[float], target: List[float]) -> float:
        if self.loss_fn == "mse":
            return losses.mse_loss(output, target)
        elif self.loss_fn == "ce":
            return losses.cross_entropy_loss(output, target)
        elif self.loss_fn == "weighted_ce" and self.class_weights:
            return losses.weighted_cross_entropy_loss(
                output, target, self.class_weights
            )
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def evaluate(
        self, dataset: List[Tuple[List[float], List[float]]]
    ) -> float:
        if not dataset:
            return 0.0

        correct = 0
        for x, target in dataset:
            output = self.predict(x)

            if len(target) == 1:
                pred = 1 if output[0] >= 0.5 else 0
                if pred == int(target[0]):
                    correct += 1
            else:
                pred_idx = max(range(len(output)), key=lambda i: output[i])
                tgt_idx = max(range(len(target)), key=lambda i: target[i])
                if pred_idx == tgt_idx:
                    correct += 1

        return correct / len(dataset)

    # Method stubs for dynamically attached methods (implemented in submodules)
    @staticmethod
    def get_activation(name: str) -> Callable[[float], float]:
        """Get activation function by name."""
        return {
            "sigmoid": activations.sigmoid,
            "relu": activations.relu,
            "tanh": activations.tanh,
            "linear": activations.linear,
        }[name]

    @staticmethod
    def get_activation_derivative(name: str) -> Callable[[float], float]:
        """Get activation derivative function by name."""
        return {
            "sigmoid": activations.sigmoid_derivative,
            "relu": activations.relu_derivative,
            "tanh": activations.tanh_derivative,
            "linear": activations.linear_derivative,
        }[name]

    def forward(
        self, inputs: List[float]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Forward pass through the network."""
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError(
                f"Input size {len(inputs)} doesn't match network input {self.layer_sizes[0]}"
            )

        # Convert inputs to numpy array
        a = np.array(inputs)

        # Store activations and z values for each layer
        activations_list = [a]
        zs = []

        for layer_idx in range(len(self.weights)):
            # Extract weight matrix and bias vector for current layer
            w_mat = self.weights[layer_idx]
            b_vec = self.biases[layer_idx]

            # Compute z values: z = W * a + b
            z_layer = np.dot(w_mat, a) + b_vec
            zs.append(z_layer)

            # Determine if it's the output layer
            is_output = layer_idx == len(self.weights) - 1

            # Activation function for output layer (e.g., softmax)
            if is_output and self.output_activation == "softmax":
                a_next = self.softmax(z_layer)
            else:
                # Use appropriate activation function for hidden layers
                act_fn = self.get_activation(
                    self.output_activation
                    if is_output
                    else self.hidden_activation
                )
                a_next = act_fn(z_layer)

            activations_list.append(a_next)
            a = a_next

        # Convert activations back to list of lists if needed
        activations_list = [a.tolist() for a in activations_list]
        zs = [z.tolist() for z in zs]

        return activations_list, zs

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_z = np.exp(z - np.max(z))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def get_activation(self, activation_type: str):
        """Returns the activation function based on the type."""
        if activation_type == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        elif activation_type == "tanh":
            return np.tanh
        elif activation_type == "relu":
            return lambda x: np.maximum(0, x)
        else:
            raise ValueError(
                f"Activation function {activation_type} not recognized"
            )

    def predict(self, inputs: List[float]) -> List[float]:
        """Make a prediction for given inputs."""
        activations_list, _ = self.forward(inputs)
        return activations_list[-1]

    def compute_output_delta(
        self, activations_list: List[List[float]], target: List[float]
    ) -> List[float]:
        out_acts = activations_list[-1]

        if self.output_activation == "softmax" and self.loss_fn in (
            "ce",
            "weighted_ce",
        ):
            scale = 1.0
            if self.loss_fn == "weighted_ce" and self.class_weights:
                for k, t in enumerate(target):
                    if t > 0.5:
                        scale = self.class_weights[k]
                        break
            return [
                scale * (out_acts[i] - target[i]) for i in range(len(out_acts))
            ]
        else:
            act_deriv = self.get_activation_derivative(self.output_activation)
            return [
                (out_acts[i] - target[i]) * act_deriv(out_acts[i])
                for i in range(len(out_acts))
            ]

    def compute_hidden_deltas(
        self, activations_list: List[List[float]], output_delta: List[float]
    ) -> List[List[float]]:
        """Backpropagate delta through hidden layers."""
        deltas: List[List[float]] = [
            [] for _ in range(len(self.layer_sizes) - 1)
        ]
        last_layer_idx = len(self.layer_sizes) - 2
        deltas[last_layer_idx] = output_delta

        act_deriv = self.get_activation_derivative(self.hidden_activation)
        for layer_idx in range(last_layer_idx - 1, -1, -1):
            layer_deltas = []
            for i in range(self.layer_sizes[layer_idx + 1]):
                s = sum(
                    self.weights[layer_idx + 1][j][i]
                    * deltas[layer_idx + 1][j]
                    for j in range(self.layer_sizes[layer_idx + 2])
                )
                a_val = activations_list[layer_idx + 1][i]
                layer_deltas.append(s * act_deriv(a_val))
            deltas[layer_idx] = layer_deltas

        return deltas

    def compute_gradients(
        self, activations_list: List[List[float]], deltas: List[List[float]]
    ) -> Tuple[List[List[List[float]]], List[List[float]]]:
        """Compute weight and bias gradients from deltas."""
        grad_w = [
            [[0.0 for _ in row] for row in layer] for layer in self.weights
        ]
        grad_b = [[0.0 for _ in layer] for layer in self.biases]

        for layer_idx in range(len(self.weights)):
            for neuron_idx in range(len(self.weights[layer_idx])):
                for w_idx in range(len(self.weights[layer_idx][neuron_idx])):
                    grad_w[layer_idx][neuron_idx][w_idx] = (
                        deltas[layer_idx][neuron_idx]
                        * activations_list[layer_idx][w_idx]
                    )
                grad_b[layer_idx][neuron_idx] = deltas[layer_idx][neuron_idx]

        return grad_w, grad_b

    def backprop_sample(
        self, activations_list: List[List[float]], target: List[float]
    ) -> Tuple[List[List[List[float]]], List[List[float]], float, bool]:
        """Full backpropagation for a single sample."""
        output = activations_list[-1]
        loss = self._compute_loss(output, target)

        # Check if prediction is correct
        is_correct = False
        if len(target) == 1:
            pred_bin = 1 if output[0] >= 0.5 else 0
            is_correct = pred_bin == int(target[0])
        else:
            pred_idx = max(range(len(output)), key=lambda i: output[i])
            tgt_idx = max(range(len(target)), key=lambda i: target[i])
            is_correct = pred_idx == tgt_idx

        # Compute deltas
        output_delta = self.compute_output_delta(activations_list, target)
        deltas = self.compute_hidden_deltas(activations_list, output_delta)

        # Compute gradients
        grad_w, grad_b = self.compute_gradients(activations_list, deltas)

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
                [[0.0 for _ in row] for row in layer] for layer in self.weights
            ]
            acc_grad_b = [[0.0 for _ in layer] for layer in self.biases]

            for x, target in batch:
                activations_list, _ = self.forward(x)
                grad_w, grad_b, loss, is_correct = self.backprop_sample(
                    activations_list, target
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
                        acc_grad_w[li][ni][wi] /= bsz
                    acc_grad_b[li][ni] /= bsz

            self.optimizer.update(
                self.weights, self.biases, acc_grad_w, acc_grad_b
            )

        self.optimizer.decay_lr()

        m = len(dataset)
        avg_loss = total_loss / m
        accuracy = total_correct / m if m else 0.0
        return avg_loss, accuracy

    def to_dict(self) -> dict:
        return {
            "version": "1.0",
            "architecture": {
                "layer_sizes": self.layer_sizes,
                "hidden_activation": self.hidden_activation,
                "output_activation": self.output_activation,
            },
            "hyperparameters": {
                "loss": self.loss_fn,
                "optimizer": self.optimizer.to_dict(),
                "class_weights": self.class_weights,
            },
            "parameters": {
                "weights": self.weights,
                "biases": self.biases,
            },
        }

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict):
        arch = data["architecture"]
        hyper = data.get("hyperparameters", {})
        params = data["parameters"]

        opt_data = hyper.get("optimizer", {})
        if opt_data.get("type") == "sgd":
            optimizer = SGDOptimizer.from_dict(opt_data)
        else:
            optimizer = SGDOptimizer(
                learning_rate=opt_data.get("learning_rate", 0.3)
            )

        return cls(
            weights=params["weights"],
            biases=params["biases"],
            layer_sizes=arch["layer_sizes"],
            hidden_activation=arch.get("hidden_activation", "sigmoid"),
            output_activation=arch.get("output_activation", "sigmoid"),
            loss=hyper.get("loss", "mse"),
            optimizer=optimizer,
            class_weights=hyper.get("class_weights"),
        )

    @classmethod
    def load(cls, filepath: Path) -> Self | None:
        try:
            print(f"Loading network from {filepath}...", file=sys.stderr)
            raw_config = filepath.read_text()

        except FileNotFoundError:
            print(
                f"Error: Network file '{filepath}' not found", file=sys.stderr
            )
            return None
        except Exception as e:
            print(f"Error loading network: {e}", file=sys.stderr)
            return None

        data = json.loads(raw_config)
        self = cls.from_dict(data)

        if self is None:
            print("Failed to load the network", file=sys.stderr)
            return None

        print(f"Loaded network: {self.layer_sizes}", file=sys.stderr)
        return self
