import random
from typing import List, Tuple


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
            grad_w, grad_b, loss, is_correct = self._backprop_sample(
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

        if verbose and (epoch % 100 == 0 or acc >= target_accuracy):
            val_str = f" val_acc={val_acc*100:.1f}%" if validation_data else ""
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
