from typing import List, Literal

Label = Literal[
    "Nothing",
    "Check White",
    "Check Black",
    "Checkmate White",
    "Checkmate Black",
]

LABELS: list[Label] = [
    "Nothing",
    "Check White",
    "Check Black",
    "Checkmate White",
    "Checkmate Black",
]


def label_to_vector(label: Label) -> List[float]:
    vec = [0.0] * 5
    vec[LABELS.index(label)] = 1.0

    return vec


def vector_to_label(vec: List[float]) -> Label:
    pc = -1
    pv = float("-inf")

    for c, v in enumerate(vec):
        if v > pv:
            pv = v
            pc = c

    return LABELS[pc]
