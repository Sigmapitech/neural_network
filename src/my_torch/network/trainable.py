from __future__ import annotations

from typing import List, Tuple

from ..optimizer import Optimizer, SGDOptimizer
from .core import Network


class TrainableNetwork(Network):

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
        super().__init__(
            weights,
            biases,
            layer_sizes,
            hidden_activation,
            output_activation,
            loss,
            optimizer,
            class_weights,
        )

        self.optimizer: Optimizer = optimizer or SGDOptimizer(
            learning_rate=0.3
        )

    def train(
        self,
        dataset: List[Tuple[List[float], List[float]]],
        epochs: int = 1000,
        target_accuracy: float = 1.0,
        batch_size: int = 32,
        validation_data: List[Tuple[List[float], List[float]]] | None = None,
        verbose: bool = True,
    ) -> List[Tuple[int, float, float, float]]:
        history: List[Tuple[int, float, float, float]] = []

        for epoch in range(1, epochs + 1):
            loss, acc = self.train_epoch(dataset, batch_size=batch_size)

            val_acc = 0.0
            if validation_data:
                val_acc = self.evaluate(validation_data)

            history.append((epoch, loss, acc, val_acc))

            if verbose:
                val_str = (
                    f" val_acc={val_acc*100:.1f}%" if validation_data else ""
                )
                print(
                    f"Epoch {epoch}: loss={loss:.4f} train_acc={acc*100:.1f}%{val_str}"
                )

            if acc >= target_accuracy:
                if verbose:
                    print(
                        f"Reached target accuracy {target_accuracy*100:.1f}% at epoch {epoch}"
                    )
                break

        return history

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

        net = cls(
            weights=params["weights"],
            biases=params["biases"],
            layer_sizes=arch["layer_sizes"],
            hidden_activation=arch.get("hidden_activation", "sigmoid"),
            output_activation=arch.get("output_activation", "sigmoid"),
            loss=hyper.get("loss", "mse"),
            optimizer=optimizer,
            class_weights=hyper.get("class_weights"),
        )

        return net
