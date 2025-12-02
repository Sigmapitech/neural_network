from typing import List, Tuple

from chess_utils import fen_to_tensor


def load_chessfile_predict(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_chessfile_train(
    filepath: str, encoding: str = "simple"
) -> List[Tuple[List[float], List[float]]]:
    from .labels import label_to_vector

    dataset: List[Tuple[List[float], List[float]]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.rsplit(maxsplit=2)
            if len(parts) < 2:
                print(f"Warning: line {line_num} has no label, skipping")
                continue

            if len(parts) == 3 and parts[1] in ("Check", "Checkmate"):
                fen = parts[0]
                label_str = f"{parts[1]} {parts[2]}"
            else:
                fen = " ".join(parts[:-1])
                label_str = parts[-1]

            try:
                x = fen_to_tensor(fen, encoding=encoding)
                y = label_to_vector(label_str)
                dataset.append((x, y))
            except Exception as e:
                print(f"Warning: line {line_num} parse error: {e}")
                continue

    return dataset
