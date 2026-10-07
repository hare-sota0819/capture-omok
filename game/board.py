"""Board and rules for capture omok (exact-five gomoku with pair captures).

The Board owns the grid and knows the rules that depend only on stone
positions: which moves are legal, which stones a move captures, and whether a
colour currently has an exact five-in-a-row. Deciding who won a *game*
(time-outs, illegal moves, simultaneous fives) is the GameManager's job.
"""

EMPTY = -1
BLACK = 0
WHITE = 1

BOARD_SIZE = 19
WIN_LENGTH = 5

# One representative per line orientation: horizontal, vertical, two diagonals.
LINE_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
# Captures are looked for outward from the new stone, so all 8 directions.
ALL_DIRECTIONS = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)

STONE_CHARS = {EMPTY: ".", BLACK: "X", WHITE: "O"}


def opponent_of(player):
    return WHITE if player == BLACK else BLACK


class Board:
    def __init__(self, size=BOARD_SIZE):
        self._size = size
        self._grid = [[EMPTY] * size for _ in range(size)]
        self._stone_count = 0

    # ------------------------------------------------------------- queries

    @property
    def size(self):
        return self._size

    def in_bounds(self, row, col):
        return 0 <= row < self._size and 0 <= col < self._size

    def get(self, row, col):
        return self._grid[row][col]

    def is_valid_move(self, row, col):
        return self.in_bounds(row, col) and self._grid[row][col] == EMPTY

    def is_full(self):
        return self._stone_count == self._size * self._size

    def to_array(self):
        """A fresh copy of the grid, safe to hand to a player."""
        return [row[:] for row in self._grid]

    # ------------------------------------------------------------- updates

    def place(self, row, col, player):
        """Put a stone down and remove whatever it captures.

        Returns the list of captured (row, col) positions, possibly empty.
        Raises ValueError if the position is off the board or occupied.
        """
        if not self.is_valid_move(row, col):
            raise ValueError("invalid move: (%s, %s)" % (row, col))
        self._grid[row][col] = player
        self._stone_count += 1

        captured = self._find_captures(row, col, player)
        for r, c in captured:
            self._grid[r][c] = EMPTY
        self._stone_count -= len(captured)
        return captured

    def _find_captures(self, row, col, player):
        """Pairs flanked by the new stone: new, opp, opp, own in a straight line.

        The far end must be the mover's own stone, so a run of three or more
        opponent stones is never captured, and neither is a pair with a gap.
        Stepping *into* a flank (own stone placed between two opponent stones)
        is safe because only the mover's captures are evaluated.
        """
        opponent = opponent_of(player)
        grid = self._grid
        captured = []
        for dr, dc in ALL_DIRECTIONS:
            end_r, end_c = row + 3 * dr, col + 3 * dc
            if not self.in_bounds(end_r, end_c):
                continue
            r1, c1 = row + dr, col + dc
            r2, c2 = row + 2 * dr, col + 2 * dc
            if (
                grid[r1][c1] == opponent
                and grid[r2][c2] == opponent
                and grid[end_r][end_c] == player
            ):
                captured.append((r1, c1))
                captured.append((r2, c2))
        return captured

    # ------------------------------------------------------- win detection

    def find_fives(self, player):
        """All lines of exactly five contiguous stones of `player`.

        Each line is a list of five (row, col) positions. A run of six or more
        does not count. The whole board is scanned because a capture can
        shorten the *opponent's* overline into a five far from the last move.
        """
        grid = self._grid
        size = self._size
        fives = []
        for dr, dc in LINE_DIRECTIONS:
            for row in range(size):
                for col in range(size):
                    if grid[row][col] != player:
                        continue
                    # Only count from the first stone of a run.
                    prev_r, prev_c = row - dr, col - dc
                    if self.in_bounds(prev_r, prev_c) and grid[prev_r][prev_c] == player:
                        continue
                    line = []
                    r, c = row, col
                    while self.in_bounds(r, c) and grid[r][c] == player:
                        line.append((r, c))
                        r += dr
                        c += dc
                    if len(line) == WIN_LENGTH:
                        fives.append(line)
        return fives

    def has_five(self, player):
        return bool(self.find_fives(player))

    # ------------------------------------------------------------- display

    def render(self, last_move=None, captured=()):
        """Text picture of the board.

        The last move is shown in [brackets]; points that were just emptied by
        a capture are shown as '*' so the capture is visible for one turn.
        """
        captured = set(captured)
        lines = [("    " + "".join("%2d " % c for c in range(self._size))).rstrip()]
        for row in range(self._size):
            cells = []
            for col in range(self._size):
                char = STONE_CHARS[self._grid[row][col]]
                if (row, col) == last_move:
                    cells.append("[%s]" % char)
                elif (row, col) in captured:
                    cells.append(" * ")
                else:
                    cells.append(" %s " % char)
            lines.append(("%2d  %s" % (row, "".join(cells))).rstrip())
        return "\n".join(lines)

    def __str__(self):
        return self.render()
