from typing import List, Tuple

from chess_utils import fen_to_tensor

from .labels import label_to_vector


def load_chessfile_predict(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_chessfile_train(
    filepath: str, encoding: str = "simple"
) -> List[Tuple[List[float], List[float]]]:

    dataset: List[Tuple[List[float], List[float]]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7 or len(parts) > 8:
                print(f"Warning: line {line_num} has invalid format, skipping")
                continue
            fen = " ".join(parts[:6])
            label_str = " ".join(parts[6:])
            try:
                x = fen_to_tensor(fen, encoding=encoding)
                y = label_to_vector(label_str)
                dataset.append((x, y))
            except Exception as e:
                print(f"Warning: line {line_num} parse error: {e}")
                continue

    return dataset
