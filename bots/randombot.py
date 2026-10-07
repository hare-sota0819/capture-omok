"""RandomPlayer: plays a random empty point close to the stones on the board."""

import random

from player import Player

EMPTY = -1


class RandomPlayer(Player):
    """The weakest opponent. It only keeps its stones near the action."""

    REACH = 2

    def take_turn(self, board, time):
        size = len(board)
        stones = [(r, c) for r in range(size) for c in range(size) if board[r][c] != EMPTY]
        if not stones:
            return (size // 2, size // 2)
        nearby = set()
        for r, c in stones:
            for dr in range(-self.REACH, self.REACH + 1):
                for dc in range(-self.REACH, self.REACH + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < size and 0 <= cc < size and board[rr][cc] == EMPTY:
                        nearby.add((rr, cc))
        if not nearby:
            nearby = {(r, c) for r in range(size) for c in range(size)
                      if board[r][c] == EMPTY}
        return random.choice(sorted(nearby))
