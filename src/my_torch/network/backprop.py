from typing import List, Tuple


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
        act_deriv = self._get_activation_derivative(self.output_activation)
        return [
            (out_acts[i] - target[i]) * act_deriv(out_acts[i])
            for i in range(len(out_acts))
        ]


def compute_hidden_deltas(
    self, activations_list: List[List[float]], output_delta: List[float]
) -> List[List[float]]:
    """Backpropagate delta through hidden layers."""
    deltas: List[List[float]] = [[] for _ in range(len(self.layer_sizes) - 1)]
    last_layer_idx = len(self.layer_sizes) - 2
    deltas[last_layer_idx] = output_delta

    act_deriv = self._get_activation_derivative(self.hidden_activation)
    for layer_idx in range(last_layer_idx - 1, -1, -1):
        layer_deltas = []
        for i in range(self.layer_sizes[layer_idx + 1]):
            s = sum(
                self.weights[layer_idx + 1][j][i] * deltas[layer_idx + 1][j]
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
    grad_w = [[[0.0 for _ in row] for row in layer] for layer in self.weights]
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
    output_delta = compute_output_delta(self, activations_list, target)
    deltas = compute_hidden_deltas(self, activations_list, output_delta)

    # Compute gradients
    grad_w, grad_b = compute_gradients(self, activations_list, deltas)

    return grad_w, grad_b, loss, is_correct
