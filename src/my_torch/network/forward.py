from typing import Callable, List, Tuple

from .. import activations


def get_activation(name: str) -> Callable[[float], float]:
    """Get activation function by name."""
    return {
        "sigmoid": activations.sigmoid,
        "relu": activations.relu,
        "tanh": activations.tanh,
        "linear": activations.linear,
    }[name]


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

    activations_list: List[List[float]] = [inputs[:]]
    zs: List[List[float]] = []

    a = inputs
    for layer_idx in range(len(self.weights)):
        w_mat = self.weights[layer_idx]
        b_vec = self.biases[layer_idx]

        z_layer = [
            sum(w_j * a_j for w_j, a_j in zip(w_mat[ni], a)) + b_vec[ni]
            for ni in range(len(w_mat))
        ]
        zs.append(z_layer)

        is_output = layer_idx == len(self.weights) - 1
        if is_output and self.output_activation == "softmax":
            a_next = activations.softmax(z_layer)
        else:
            act_fn = (
                get_activation(self.output_activation)
                if is_output
                else get_activation(self.hidden_activation)
            )
            a_next = [act_fn(z) for z in z_layer]

        activations_list.append(a_next)
        a = a_next

    return activations_list, zs


def predict(self, inputs: List[float]) -> List[float]:
    """Make a prediction for given inputs."""
    activations_list, _ = forward(self, inputs)
    return activations_list[-1]
