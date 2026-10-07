"""HumanPlayer: asks a person at the terminal for each move."""

import re

from player import Player

EMPTY = -1
_MOVE_PATTERN = re.compile(r"(\d+),(\d+)")


class HumanPlayer(Player):
    """Reads moves typed as `row,col` (e.g. 13,4) and re-asks until one is valid.

    Only the `board` and `time` arguments are used, so this class does not
    depend on how the surrounding game infrastructure is written.
    """

    def take_turn(self, board, time):
        print("Time remaining: %s" % self._format_time(time))
        while True:
            text = input("Your move (row,col): ").strip()
            move, problem = self._parse_move(text, board)
            if move is not None:
                return move
            print("  Invalid move: %s Try again." % problem)

    @staticmethod
    def _parse_move(text, board):
        """Return ((row, col), None) if `text` is a playable move, else (None, reason)."""
        match = _MOVE_PATTERN.fullmatch(text)
        if match is None:
            return None, "type it as row,col with no spaces, like 13,4."
        row, col = int(match.group(1)), int(match.group(2))
        size = len(board)
        if not (0 <= row < size and 0 <= col < size):
            return None, "rows and columns go from 0 to %d." % (size - 1)
        if board[row][col] != EMPTY:
            return None, "(%d,%d) already has a stone." % (row, col)
        return (row, col), None

    @staticmethod
    def _format_time(time_ms):
        """Milliseconds -> 'm:ss.s'. A negative value means there is no limit."""
        if time_ms is None or time_ms < 0:
            return "no limit"
        tenths = int(time_ms // 100)  # round down so 59.96s never prints as 60.0
        return "%d:%04.1f" % (tenths // 600, (tenths % 600) / 10.0)
