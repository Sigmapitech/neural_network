import math
from typing import List


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))


def sigmoid_derivative(y: float) -> float:
    return y * (1.0 - y)


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(y: float) -> float:
    return 1.0 - y * y


def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def linear(x: float) -> float:
    return x


def linear_derivative(_: float) -> float:
    return 1.0


def softmax(vec: List[float]) -> List[float]:
    if not vec:
        return []
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else exps
