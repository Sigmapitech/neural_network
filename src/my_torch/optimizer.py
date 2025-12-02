from typing import List


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def update(
        self,
        weights: List[List[List[float]]],
        biases: List[List[float]],
        grad_w: List[List[List[float]]],
        grad_b: List[List[float]],
    ) -> None:
        raise NotImplementedError

    def decay_lr(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "base",
            "learning_rate": self.lr,
        }


class SGDOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        lr_decay: float = 0.0,
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.velocity_w: List[List[List[float]]] | None = None
        self.velocity_b: List[List[float]] | None = None

    def update(
        self,
        weights: List[List[List[float]]],
        biases: List[List[float]],
        grad_w: List[List[List[float]]],
        grad_b: List[List[float]],
    ) -> None:
        if self.velocity_w is None or self.velocity_b is None:
            self.velocity_w = [
                [[0.0 for _ in row] for row in layer] for layer in weights
            ]
            self.velocity_b = [[0.0 for _ in layer] for layer in biases]

        for li in range(len(weights)):
            for ni in range(len(weights[li])):
                for wi in range(len(weights[li][ni])):
                    self.velocity_w[li][ni][wi] = (
                        self.momentum * self.velocity_w[li][ni][wi]
                        - self.lr * grad_w[li][ni][wi]
                    )
                    weights[li][ni][wi] += self.velocity_w[li][ni][wi]

                self.velocity_b[li][ni] = (
                    self.momentum * self.velocity_b[li][ni]
                    - self.lr * grad_b[li][ni]
                )
                biases[li][ni] += self.velocity_b[li][ni]

    def decay_lr(self) -> None:
        if self.lr_decay > 0:
            self.lr *= 1.0 - self.lr_decay

    def to_dict(self) -> dict:
        return {
            "type": "sgd",
            "learning_rate": self.lr,
            "momentum": self.momentum,
            "lr_decay": self.lr_decay,
        }

    @staticmethod
    def from_dict(data: dict) -> "SGDOptimizer":
        return SGDOptimizer(
            learning_rate=data.get("learning_rate", 0.01),
            momentum=data.get("momentum", 0.0),
            lr_decay=data.get("lr_decay", 0.0),
        )
