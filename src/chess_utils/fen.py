from typing import List, Tuple

PIECE_TO_INDEX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


class BoardState:

    def __init__(
        self,
        board: List[List[str]],
        active_color: str,
        castling: str,
        en_passant: str,
        halfmove: int,
        fullmove: int,
    ):
        self.board = board
        self.active_color = active_color
        self.castling = castling
        self.en_passant = en_passant
        self.halfmove = halfmove
        self.fullmove = fullmove

    def get_piece(self, rank: int, file: int) -> str:
        """Get piece at position (rank 0-7, file 0-7). Returns "" if empty."""
        if 0 <= rank < 8 and 0 <= file < 8:
            return self.board[rank][file]
        return ""

    def set_piece(self, rank: int, file: int, piece: str) -> None:
        if 0 <= rank < 8 and 0 <= file < 8:
            self.board[rank][file] = piece

    def find_king(self, color: str) -> Tuple[int, int] | None:
        king = "K" if color == "w" else "k"
        for rank in range(8):
            for file in range(8):
                if self.board[rank][file] == king:
                    return (rank, file)
        return None

    def __str__(self) -> str:
        lines = []
        for rank in range(8):
            row = []
            for file in range(8):
                piece = self.board[rank][file]
                row.append(piece if piece else ".")
            lines.append(" ".join(row))
        return "\n".join(lines)


def parse_fen(fen: str) -> BoardState:
    parts = fen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Invalid FEN: {fen}")

    placement, active_color, castling, en_passant = (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
    )
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    board: List[List[str]] = []
    ranks = placement.split("/")

    if len(ranks) != 8:
        raise ValueError(f"Expected 8 ranks in FEN, got {len(ranks)}")

    for rank_str in ranks:
        rank_pieces: List[str] = []
        for char in rank_str:
            if char.isdigit():
                rank_pieces.extend([""] * int(char))
            else:
                rank_pieces.append(char)

        if len(rank_pieces) != 8:
            raise ValueError(
                f"Rank must have 8 squares, got {len(rank_pieces)}"
            )

        board.append(rank_pieces)

    return BoardState(
        board, active_color, castling, en_passant, halfmove, fullmove
    )


def fen_to_tensor(fen: str, encoding: str = "piece_planes") -> List[float]:
    board_state = parse_fen(fen)

    if encoding == "piece_planes":
        tensor = [0.0] * (12 * 64)
        for rank in range(8):
            for file in range(8):
                piece = board_state.get_piece(rank, file)
                if piece and piece in PIECE_TO_INDEX:
                    plane_idx = PIECE_TO_INDEX[piece]
                    pos = rank * 8 + file
                    tensor[plane_idx * 64 + pos] = 1.0
        return tensor

    elif encoding == "simple":
        piece_values = {
            "P": 1,
            "N": 2,
            "B": 3,
            "R": 4,
            "Q": 5,
            "K": 6,
            "p": -1,
            "n": -2,
            "b": -3,
            "r": -4,
            "q": -5,
            "k": -6,
        }
        tensor = []
        for rank in range(8):
            for file in range(8):
                piece = board_state.get_piece(rank, file)
                tensor.append(float(piece_values.get(piece, 0)))
        return tensor

    elif encoding == "simple_extended":
        piece_values = {
            "P": 1,
            "N": 2,
            "B": 3,
            "R": 4,
            "Q": 5,
            "K": 6,
            "p": -1,
            "n": -2,
            "b": -3,
            "r": -4,
            "q": -5,
            "k": -6,
        }
        tensor = []
        for rank in range(8):
            for file in range(8):
                piece = board_state.get_piece(rank, file)
                tensor.append(float(piece_values.get(piece, 0)))

        tensor.append(1.0 if board_state.active_color == "w" else -1.0)
        tensor.append(1.0 if "K" in board_state.castling else 0.0)
        tensor.append(1.0 if "Q" in board_state.castling else 0.0)
        tensor.append(1.0 if "k" in board_state.castling else 0.0)
        tensor.append(1.0 if "q" in board_state.castling else 0.0)
        return tensor

    else:
        raise ValueError(f"Unknown encoding: {encoding}")
