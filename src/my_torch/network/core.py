from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import losses
from ..optimizer import Optimizer, SGDOptimizer


class Network:
    def __init__(
        self,
        layer_sizes: List[int],
        hidden_activation: str = "sigmoid",
        output_activation: str = "sigmoid",
        loss: str = "mse",
        optimizer: Optimizer | None = None,
        class_weights: List[float] | None = None,
        seed: int | None = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer")
        if any(s <= 0 for s in layer_sizes):
            raise ValueError("All layer sizes must be > 0")

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

        if seed is not None:
            random.seed(seed)

        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            scale = (2.0 / in_size) ** 0.5
            self.weights.append(
                [
                    [random.gauss(0, scale) for _ in range(in_size)]
                    for _ in range(out_size)
                ]
            )
            self.biases.append(
                [random.uniform(-0.1, 0.1) for _ in range(out_size)]
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
    def _get_activation(name: str) -> Callable[[float], float]: ...

    @staticmethod
    def _get_activation_derivative(name: str) -> Callable[[float], float]: ...

    def forward(
        self, inputs: List[float]
    ) -> Tuple[List[List[float]], List[List[float]]]: ...

    def predict(self, inputs: List[float]) -> List[float]: ...

    def _backprop_sample(
        self, activations_list: List[List[float]], target: List[float]
    ) -> Tuple[List[List[List[float]]], List[List[float]], float, bool]: ...

    def train_epoch(
        self, dataset: List[Tuple[List[float], List[float]]], batch_size: int
    ) -> Tuple[float, float]: ...

    def train(
        self,
        dataset: List[Tuple[List[float], List[float]]],
        epochs: int = 1000,
        target_accuracy: float = 1.0,
        batch_size: int = 32,
        validation_data: Optional[
            List[Tuple[List[float], List[float]]]
        ] = None,
        verbose: bool = True,
    ) -> List[Tuple[int, float, float, float]]: ...

    def to_dict(self) -> Dict[str, Any]: ...

    def save(self, filepath: str) -> None: ...

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Network: ...

    @staticmethod
    def load(filepath: str) -> Network: ...
