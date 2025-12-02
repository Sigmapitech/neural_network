import math
from typing import List


def mse_loss(output: List[float], target: List[float]) -> float:
    """Mean Squared Error loss.

    L = 1/n * Σ(output_i - target_i)²
    """
    if len(output) != len(target):
        raise ValueError("Output and target dimensions must match")
    return sum((o - t) ** 2 for o, t in zip(output, target)) / len(output)


def cross_entropy_loss(
    output: List[float], target: List[float], epsilon: float = 1e-12
) -> float:
    """Cross-entropy loss (for use with softmax output).

    L = -Σ(target_i * log(output_i))

    Args:
        output: Predicted probabilities (should sum to 1.0)
        target: One-hot encoded target
        epsilon: Small value for numerical stability
    """
    if len(output) != len(target):
        raise ValueError("Output and target dimensions must match")
    return -sum(t * math.log(max(o, epsilon)) for o, t in zip(output, target))


def weighted_cross_entropy_loss(
    output: List[float],
    target: List[float],
    class_weights: List[float],
    epsilon: float = 1e-12,
) -> float:
    """Weighted cross-entropy for imbalanced datasets.

    L = -Σ(weight_c * target_i * log(output_i))
    where c is the true class index.
    """
    if len(output) != len(target) or len(class_weights) != len(target):
        raise ValueError("Dimension mismatch")

    true_class = 0
    for i, t in enumerate(target):
        if t > 0.5:
            true_class = i
            break

    weight = class_weights[true_class]
    base_loss = -sum(
        t * math.log(max(o, epsilon)) for o, t in zip(output, target)
    )
    return weight * base_loss
