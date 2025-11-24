#!/usr/bin/env python3
"""Tic-Tac-Toe board generator and labeling utilities.

Board representation: list of 9 ints in row-major order.
Values: 1 = X, -1 = O, 0 = empty.
Turn rules enforced: X starts, players alternate, game stops at first win.

Labels:
 - Binary (win vs not): target = [1] if any player has already won, else [0]
 - Multi (ongoing, draw, win): one-hot 3-vector
     ongoing: [1,0,0]
     draw:    [0,1,0]
     win:     [0,0,1]
"""
from __future__ import annotations

from typing import Iterable, List, Set, Tuple

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diagonals
]


def check_winner(board: List[int]) -> int | None:
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:  # X wins
            return 1
        if s == -3:  # O wins
            return -1
    return None


def is_draw(board: List[int]) -> bool:
    return all(cell != 0 for cell in board) and check_winner(board) is None


def legal(board: List[int]) -> bool:
    x_count = sum(1 for v in board if v == 1)
    o_count = sum(1 for v in board if v == -1)
    if o_count > x_count or x_count - o_count > 1:
        return False
    # Cannot have both players winning
    winner = check_winner(board)
    if winner is not None:
        # after win, no further moves should be played (i.e. win occurs at earliest). We'll allow generation method to enforce.
        pass
    return True


def next_player(board: List[int]) -> int:
    x_count = sum(1 for v in board if v == 1)
    o_count = sum(1 for v in board if v == -1)
    return 1 if x_count == o_count else -1


def generate_all_positions() -> List[List[int]]:
    positions: Set[Tuple[int, ...]] = set()
    finished: Set[Tuple[int, ...]] = set()

    def rec(board: List[int]):
        t = tuple(board)
        if t in positions:
            return
        positions.add(t)
        winner = check_winner(board)
        if winner is not None or is_draw(board):
            finished.add(t)
            return
        player = next_player(board)
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                # Only continue if still legal and no winner prematurely
                if legal(board):
                    rec(board)
                board[i] = 0

    rec([0] * 9)
    return [list(p) for p in positions]


def encode_board(board: List[int]) -> List[float]:
    return [float(v) for v in board]


def build_binary_dataset() -> List[Tuple[List[float], List[float]]]:
    data: List[Tuple[List[float], List[float]]] = []
    for b in generate_all_positions():
        winner = check_winner(b)
        label = [1.0] if winner is not None else [0.0]
        data.append((encode_board(b), label))
    return data


def build_multiclass_dataset() -> List[Tuple[List[float], List[float]]]:
    data: List[Tuple[List[float], List[float]]] = []
    for b in generate_all_positions():
        winner = check_winner(b)
        if winner is not None:
            label = [0.0, 0.0, 1.0]  # win
        elif is_draw(b):
            label = [0.0, 1.0, 0.0]  # draw
        else:
            label = [1.0, 0.0, 0.0]  # ongoing
        data.append((encode_board(b), label))
    return data


if __name__ == "__main__":
    binary = build_binary_dataset()
    multi = build_multiclass_dataset()
    print(f"Binary positions: {len(binary)}; examples: {binary[:3]}")
    print(f"Multi positions: {len(multi)}; examples: {multi[:3]}")
