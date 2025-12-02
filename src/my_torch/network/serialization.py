from __future__ import annotations

import json

from ..optimizer import SGDOptimizer


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
        json.dump(to_dict(self), f, indent=2)


def from_dict(data: dict, network_cls):
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

    net = network_cls(
        layer_sizes=arch["layer_sizes"],
        hidden_activation=arch.get("hidden_activation", "sigmoid"),
        output_activation=arch.get("output_activation", "sigmoid"),
        loss=hyper.get("loss", "mse"),
        optimizer=optimizer,
        class_weights=hyper.get("class_weights"),
    )

    net.weights = params["weights"]
    net.biases = params["biases"]

    return net


def load(filepath: str, network_cls):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data, network_cls)
