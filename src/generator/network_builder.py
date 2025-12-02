from typing import Any, Dict

from my_torch import Network, SGDOptimizer


def create_network_from_config(config: Dict[str, Any]) -> Network:
    layer_sizes = config["layer_sizes"]
    hidden_activation = config.get("hidden_activation", "sigmoid")
    output_activation = config.get("output_activation", "sigmoid")
    loss = config.get("loss", "mse")

    opt_config = config.get("optimizer", {})
    opt_type = opt_config.get("type", "sgd")

    if opt_type == "sgd":
        optimizer = SGDOptimizer(
            learning_rate=opt_config.get("learning_rate", 0.1),
            momentum=opt_config.get("momentum", 0.0),
            lr_decay=opt_config.get("lr_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    class_weights = config.get("class_weights")

    network = Network(
        layer_sizes=layer_sizes,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        loss=loss,
        optimizer=optimizer,
        class_weights=class_weights,
        seed=config.get("seed"),
    )

    return network
