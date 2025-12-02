from typing import List


def label_to_vector(label: str) -> List[float]:
    label = label.strip()

    if label == "Nothing":
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    elif label in ("Check White", "CheckWhite"):
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    elif label in ("Check Black", "CheckBlack"):
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    elif label in ("Checkmate White", "CheckmateWhite"):
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    elif label in ("Checkmate Black", "CheckmateBlack"):
        return [0.0, 0.0, 0.0, 0.0, 1.0]
    elif label == "Check":
        return [0.0, 1.0, 0.0]
    elif label == "Checkmate":
        return [0.0, 0.0, 1.0]
    else:
        raise ValueError(f"Unknown label: {label}")


def vector_to_label(vec: List[float], mode: str = "auto") -> str:
    if mode == "auto":
        mode = "detailed" if len(vec) == 5 else "basic"

    pred_idx = max(range(len(vec)), key=lambda i: vec[i])

    if mode == "detailed" and len(vec) == 5:
        labels = [
            "Nothing",
            "Check White",
            "Check Black",
            "Checkmate White",
            "Checkmate Black",
        ]
        return labels[pred_idx]
    else:
        labels = ["Nothing", "Check", "Checkmate"]
        return labels[min(pred_idx, 2)]
