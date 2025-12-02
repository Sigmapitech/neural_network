from enum import Enum
from typing import List, Tuple

from .fen import BoardState, parse_fen


class GameState(Enum):

    NOTHING = "Nothing"
    CHECK_WHITE = "Check White"
    CHECK_BLACK = "Check Black"
    CHECKMATE_WHITE = "Checkmate White"
    CHECKMATE_BLACK = "Checkmate Black"
    CHECK = "Check"
    CHECKMATE = "Checkmate"


def is_square_attacked(
    board: BoardState, rank: int, file: int, by_color: str
) -> bool:
    for r in range(8):
        for f in range(8):
            piece = board.get_piece(r, f)
            if not piece:
                continue

            piece_color = "w" if piece.isupper() else "b"
            if piece_color != by_color:
                continue

            piece_type = piece.upper()

            if piece_type == "P":
                direction = -1 if by_color == "w" else 1
                if r + direction == rank and abs(f - file) == 1:
                    return True

            elif piece_type == "N":
                dr, df = abs(r - rank), abs(f - file)
                if (dr == 2 and df == 1) or (dr == 1 and df == 2):
                    return True

            elif piece_type == "B":
                if abs(r - rank) == abs(f - file) and abs(r - rank) > 0:
                    if _is_path_clear(board, r, f, rank, file):
                        return True

            elif piece_type == "R":
                if (r == rank or f == file) and not (r == rank and f == file):
                    if _is_path_clear(board, r, f, rank, file):
                        return True

            elif piece_type == "Q":
                if abs(r - rank) == abs(f - file) or r == rank or f == file:
                    if not (r == rank and f == file):
                        if _is_path_clear(board, r, f, rank, file):
                            return True

            elif piece_type == "K":
                if abs(r - rank) <= 1 and abs(f - file) <= 1:
                    if not (r == rank and f == file):
                        return True

    return False


def _is_path_clear(
    board: BoardState, r1: int, f1: int, r2: int, f2: int
) -> bool:
    dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
    df = 0 if f1 == f2 else (1 if f2 > f1 else -1)

    r, f = r1 + dr, f1 + df
    while r != r2 or f != f2:
        if board.get_piece(r, f):
            return False
        r += dr
        f += df

    return True


def is_check(board: BoardState, color: str) -> bool:
    king_pos = board.find_king(color)
    if not king_pos:
        return False

    enemy_color = "b" if color == "w" else "w"
    return is_square_attacked(board, king_pos[0], king_pos[1], enemy_color)


def get_possible_moves(
    board: BoardState, color: str
) -> List[Tuple[int, int, int, int]]:
    moves: List[Tuple[int, int, int, int]] = []

    for r in range(8):
        for f in range(8):
            piece = board.get_piece(r, f)
            if not piece:
                continue

            piece_color = "w" if piece.isupper() else "b"
            if piece_color != color:
                continue

            piece_type = piece.upper()

            if piece_type == "P":
                direction = -1 if color == "w" else 1
                new_r = r + direction
                if 0 <= new_r < 8:
                    if not board.get_piece(new_r, f):
                        moves.append((r, f, new_r, f))
                    for df in [-1, 1]:
                        new_f = f + df
                        if 0 <= new_f < 8:
                            target = board.get_piece(new_r, new_f)
                            if target and (
                                target.isupper() != piece.isupper()
                            ):
                                moves.append((r, f, new_r, new_f))

            elif piece_type == "N":
                for dr, df in [
                    (2, 1),
                    (2, -1),
                    (-2, 1),
                    (-2, -1),
                    (1, 2),
                    (1, -2),
                    (-1, 2),
                    (-1, -2),
                ]:
                    new_r, new_f = r + dr, f + df
                    if 0 <= new_r < 8 and 0 <= new_f < 8:
                        target = board.get_piece(new_r, new_f)
                        if not target or (target.isupper() != piece.isupper()):
                            moves.append((r, f, new_r, new_f))

            elif piece_type in ("B", "R", "Q"):
                directions = []
                if piece_type in ("B", "Q"):
                    directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                if piece_type in ("R", "Q"):
                    directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])

                for dr, df in directions:
                    new_r, new_f = r + dr, f + df
                    while 0 <= new_r < 8 and 0 <= new_f < 8:
                        target = board.get_piece(new_r, new_f)
                        if not target:
                            moves.append((r, f, new_r, new_f))
                            new_r += dr
                            new_f += df
                        elif target.isupper() != piece.isupper():
                            moves.append((r, f, new_r, new_f))
                            break
                        else:
                            break

            elif piece_type == "K":
                for dr in [-1, 0, 1]:
                    for df in [-1, 0, 1]:
                        if dr == 0 and df == 0:
                            continue
                        new_r, new_f = r + dr, f + df
                        if 0 <= new_r < 8 and 0 <= new_f < 8:
                            target = board.get_piece(new_r, new_f)
                            if not target or (
                                target.isupper() != piece.isupper()
                            ):
                                moves.append((r, f, new_r, new_f))

    return moves


def is_checkmate(board: BoardState, color: str) -> bool:
    if not is_check(board, color):
        return False

    for from_r, from_f, to_r, to_f in get_possible_moves(board, color):
        test_board = BoardState(
            [row[:] for row in board.board],
            board.active_color,
            board.castling,
            board.en_passant,
            board.halfmove,
            board.fullmove,
        )

        piece = test_board.get_piece(from_r, from_f)
        test_board.set_piece(to_r, to_f, piece)
        test_board.set_piece(from_r, from_f, "")

        if not is_check(test_board, color):
            return False

    return True


def get_game_state(fen: str, detailed: bool = True) -> GameState:
    board = parse_fen(fen)

    white_in_check = is_check(board, "w")
    black_in_check = is_check(board, "b")

    if white_in_check and is_checkmate(board, "w"):
        return GameState.CHECKMATE_BLACK if detailed else GameState.CHECKMATE

    if black_in_check and is_checkmate(board, "b"):
        return GameState.CHECKMATE_WHITE if detailed else GameState.CHECKMATE

    if white_in_check:
        return GameState.CHECK_WHITE if detailed else GameState.CHECK

    if black_in_check:
        return GameState.CHECK_BLACK if detailed else GameState.CHECK

    return GameState.NOTHING
