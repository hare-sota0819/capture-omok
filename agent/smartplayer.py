"""SmartPlayer: a capture omok agent (exact five wins, flanked pairs are captured).

How it thinks, in the order it is applied each turn:

1. Complete an exact five if one is available.
2. Look for a forced win: first by continuous fours, then by fours and open
   threes together. Every threat must be answered -- by blocking it or by
   capturing -- and the win must survive all of those answers.
3. Otherwise run an iterative-deepening alpha-beta search. Answers to threats
   are forced and cost little or no depth, so forcing lines are followed far
   beyond the nominal depth, and hanging captures are played out at the leaves.
4. Before committing, make sure the chosen move does not hand the opponent a
   forced win of the same kind; if it does, prefer a move that does not.

There are two engines that think this way. fastcore.py is compiled by numba
and searches some fifteen times more positions per second; it is used whenever
numba is installed. The engine in this file is plain Python: it is what plays
when numba is not there, and it is the reference the compiled one is checked
against.

What makes this one fast enough in pure Python is that nothing is recomputed from
scratch. The board is kept as 112 lines (rows, columns, diagonals), each stored
as one integer. Everything the engine wants to know about a line -- its score,
where a five can be completed, where a pair can be captured, where a four can
be made -- is looked up in a dictionary keyed by that integer, and a move only
changes the (at most four) lines through its point. Taking a move back restores
the saved integers without looking anything up.
"""

import importlib.util
import os
import random
import sys
import time as _time

from player import Player

SIZE = 19
CELLS = SIZE * SIZE
CENTRE = (SIZE // 2) * SIZE + SIZE // 2

WIN = 1000000
INFINITY = 10 * WIN

# Value of a five-cell window holding k own stones and no opponent stone.
WINDOW_VALUE = (0, 1, 8, 60, 220)
CAPTURE_VALUE = 45        # being able to capture a pair, per pair
DEFENCE_SHARE = 0.85      # how much denying the opponent's best use of a point is worth
OPEN_FOUR_BONUS = 4000    # ordering bonus for a move that makes two fives at once
CAPTURE_BONUS = 70        # ordering bonus per pair a move captures
RESCUE_BONUS = 110        # ... and per own pair it saves by taking the capturing point
FIVE_GAIN = 100000        # ordering value of completing (or stopping) a five
TEMPO_VALUE = 12
MOVER_WEIGHT = 1.0        # the side to move's own lines count this much more than the other's
THREAT_VALUE = 2500       # leaf bonus when the side to move can make an open four
MOVER_CAPTURE_VALUE = 30  # extra for each capture the side to move could make right now
DOUBLE_THREE_PENALTY = 1200
STONE_VALUE = 120         # per stone of difference: a captured pair is two lost moves
QUIESCE_DEPTH = 3         # captures followed at the leaves before judging a position

ROOT_WIDTH = 20
# Moves considered at a quiet node, by the depth (in half plies) still to go:
# wide wherever a lot of search hangs below the node, narrower near the leaves.
WIDTH_DEEP, WIDTH_MID, WIDTH_SHALLOW = 12, 10, 8
CACHE_WIDTH = 14          # how many ordered moves are remembered per position
QUIET_COST = 2            # depth is counted in half plies; a quiet move costs two
FOUR_COST = 2             # a move that makes a four (its answer is forced and free)
THREAT_REPLY_COST = 2     # an answer to an open three
REDUCE_AFTER = 4          # late-move reduction: from this move index on ...
REDUCE_MIN_DEPTH = 6      # ... when at least this much depth (half plies) is left
FRONTIER_MARGIN = 150     # at the last ply: static value this far above beta is a cut-off
FUTILITY_MARGIN = 300     # ... and a quiet move this far short of alpha is not tried
NULL_MOVE = False         # try passing first: if that is still good enough, cut off
NULL_MIN_DEPTH = 6
NULL_REDUCTION = 4
MAX_PLY = 60
VCF_DEPTH = 14            # attacking moves in a win by continuous fours
VCT_DEPTH = 8             # attacking moves in a win by fours and threes
SAFETY_DEPTH = 6          # the same, when checking what the opponent could do to us
DEFUSING_ENOUGH = 6       # stop looking for defences once this many are known

RESERVE_S = 2.5           # never planned to be spent
MOVES_AHEAD = 13          # own moves the remaining time is spread over at the start ...
MIN_MOVES_AHEAD = 8       # ... shrinking to this as the game goes on
MOVE_CAP_S = 5.0
OWN_VCT_SHARE = 0.15      # of a move's budget: looking for our own forced win ...
PROBE_SHARE = 0.15        # ... and for the opponent's
SOFT_SHARE = 0.6          # stop deepening after this share of the move's budget ...
SEARCH_SHARE = 0.85       # ... unless the result is unsettled; then up to this share
UNSETTLED_DROP = 100      # a fall in value this large between iterations is unsettled
NO_LIMIT_MOVE_S = 1.0
INSTANT_BELOW_S = 0.08
# White's answers to the first stone, by kind: next to it in a straight line,
# or diagonally. Which kind is better depends on how the opponent attacks (in
# test games each kind beat one strong attacker far more often than the other
# kind did, and anything further away lost almost every time), so the games
# of one match as White take the kinds in turn: see SmartPlayer._first_reply.
FIRST_REPLY_KINDS = (((0, 1), (0, -1), (1, 0), (-1, 0)),
                     ((1, 1), (1, -1), (-1, 1), (-1, -1)))
FIRST_REPLIES = FIRST_REPLY_KINDS[0] + FIRST_REPLY_KINDS[1]

# For experiments and self-play: when FIXED_NODES is set, every move gets the
# same amount of search (counted in nodes) instead of a share of the clock, so
# games are repeatable and do not depend on how fast the machine is.
FIXED_NODES = 0
NODE_RATE = 30000         # nodes that count as one second on that node clock

MEMO_LIMIT = 700000

# A learned evaluation, fitted to games the engine played against itself. When PATTERN_WIDTH is
# not 0, the static value of a position is the sum of PATTERN_SCORES over every
# stretch of PATTERN_WIDTH points on every line, read from the side to move
# (0 empty, 1 mine, 2 theirs, 3 off the board), plus the whole-board PATTERN_TERMS.
PATTERN_WIDTH = 7
PATTERN_SCORES = (
    0, 0, -20, 0, -20, -20, 20, 20, 0, 20, -20, 0, -20, -20, 14, 0,
    14, 14, 0, 0, 0, 0, 20, 20, 0, 20, 0, 0, 0, 0, 17, 17,
    0, 17, -20, 0, -20, -20, 14, 0, 14, 14, 0, 0, 0, 0, 14, 0,
    14, 14, 243, 0, 243, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 20, 20, 0, 20, 0, 0, 0, 0, 17, 17,
    0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 17,
    0, 17, 0, 0, 0, 0, -202, -202, 0, -202, 0, -20, -20, 14, 0, 14,
    14, 0, 0, 0, 0, 14, 0, 14, 14, 243, 0, 243, 243, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 14,
    243, 0, 243, 243, 0, 0, 0, 0, 243, 0, 243, 243, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 20, 0, 20, 0, 0, 0, 0, 17, 17, 0, 17, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 17, 0, 17, 0, 0, 0,
    0, -202, -202, 0, -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 17, 0, 17, 0, 0, 0, 0, -202, -202,
    0, -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -202, -202, 0,
    -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, -20, -20, 14, 0, 14, 14,
    0, 0, 0, 0, 0, 14, 14, 243, 0, 243, 243, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 243, 0, 243,
    243, 0, 0, 0, 0, 0, 243, 243, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 0,
    243, 243, 0, 0, 0, 0, 0, 243, 243, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 243, 243, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 20, 0, 0, 0, 17,
    17, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 17,
    0, 0, 0, -202, -202, 0, -202, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 17, 0, 17, 0, 0, 0, -202, -202, 0, -202, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, -202, 0, -202, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 17, 0, 0, 0,
    -202, 0, -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, -202, 0, -202, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, -202, 0, -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    20, 0, 20, 0, 0, 0, 17, 0, 17, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 20, 0, 20, 0, 0, 0, 17, 0, 17, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 17, 0, 17, 0, 0, 0, -202, 0, -202, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0,
    0, 0, 17, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 17,
    0, 0, 0, -202, 0, -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    17, 0, 0, 0, -202, 0, -202, 0, 0, 0, 0, 0, 0, 0, 0, -202,
    0, -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 17, 0, 17, 0, 0,
    0, 0, 0, 0, 0, 0, 17, 0, 0, -202, 0, -202, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 17, 0, 0, -202, 0, -202, 0, 0, 0, 0, 0, 0, 0, 0,
    -202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 17, 0, 0, 0, -202, 0, 0, 0, 0, 0, 0, 0, -202,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, -202, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -20, -20, 0,
    0, -20, -20, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, -20, -20, 14,
    14, 0, 0, 14, 14, 243, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, -20, 14, 14, 0, 0, 14, 14, 243, 243, 0, 0, 0, 0, 0, 0,
    0, 0, 14, 243, 243, 0, 0, 243, 243, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, -20, 14, 14, 0, 0, 14, 243, 243, 0, 0, 0, 0,
    0, 0, 0, 14, 243, 243, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 14, 243, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 243,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -20, 20, -20, 14, 0, 20, 0, 17, -20, 14, 0,
    14, 243, 0, 0, 0, 0, 20, 0, 17, 0, 0, 0, 17, 0, -202, 14,
    0, 14, 243, 0, 0, 0, 0, 243, 0, 243, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 17, 0, -202, 0, 0,
    0, 0, 0, 0, 0, 0, -202, 0, 0, -202, 0, 0, 14, 0, 243, 0,
    0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, -202, 0, 0, 0, -202,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
PATTERN_TERMS = {'tempo': 185, 'stones': 476, 'my_captures': 472, 'their_captures': -342, 'my_open': 1105, 'their_double': -236}


class _OutOfTime(Exception):
    pass


# --------------------------------------------------------------------------
# Static tables


def _build_lines():
    """Every straight line of four or more points, as a tuple of cell indices."""
    lines = []
    for r in range(SIZE):
        lines.append(tuple(r * SIZE + c for c in range(SIZE)))
    for c in range(SIZE):
        lines.append(tuple(r * SIZE + c for r in range(SIZE)))
    starts = [(0, c) for c in range(SIZE)] + [(r, 0) for r in range(1, SIZE)]
    for r, c in starts:                                  # down-right
        cells = []
        while r < SIZE and c < SIZE:
            cells.append(r * SIZE + c)
            r += 1
            c += 1
        if len(cells) >= 4:
            lines.append(tuple(cells))
    starts = [(0, c) for c in range(SIZE)] + [(r, SIZE - 1) for r in range(1, SIZE)]
    for r, c in starts:                                  # down-left
        cells = []
        while r < SIZE and c >= 0:
            cells.append(r * SIZE + c)
            r += 1
            c -= 1
        if len(cells) >= 4:
            lines.append(tuple(cells))
    return lines


LINE_CELLS = _build_lines()
LINE_COUNT = len(LINE_CELLS)
BYTE = (b"\x00", b"\x01", b"\x02")

# A line is stored as one integer, two bits per point, so a move changes it by
# an addition and it can be used directly as a dictionary key.
LINE_MEMO = {length: {} for length in range(4, SIZE + 1)}
THREE_MEMO = {length: {} for length in range(4, SIZE + 1)}

# Move-ordering gains are looked up by the (at most) 13 points around a point,
# six each side, which is everything a stone there can influence on one line.
GAIN_MEMO = {}            # (length, offset) -> {segment: gains}, see _fill_gain()


def _build_cell_tables():
    cell_lines = [[] for _ in range(CELLS)]
    cell_gains = [[] for _ in range(CELLS)]
    for line_id, cells in enumerate(LINE_CELLS):
        length = len(cells)
        for position, cell in enumerate(cells):
            cell_lines[cell].append((line_id, 2 * position, LINE_MEMO[length], length))
            low = max(0, position - 6)
            high = min(length, position + 7)
            shape = (high - low, position - low)
            mask = (1 << (2 * (high - low))) - 1
            memo = GAIN_MEMO.setdefault(shape, {})
            cell_gains[cell].append((line_id, 2 * low, mask, memo, shape[0], shape[1]))
    return [tuple(x) for x in cell_lines], [tuple(x) for x in cell_gains]


CELL_LINES, CELL_GAINS = _build_cell_tables()


def _decode(key, length):
    return bytes((key >> (2 * i)) & 3 for i in range(length))


def _build_neighbours():
    """Points up to two steps away along a line: where a reply can matter."""
    table = []
    for cell in range(CELLS):
        r, c = divmod(cell, SIZE)
        near = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                for step in (1, 2):
                    rr, cc = r + dr * step, c + dc * step
                    if 0 <= rr < SIZE and 0 <= cc < SIZE:
                        near.append(rr * SIZE + cc)
        table.append(tuple(near))
    return table


NEIGHBOURS = _build_neighbours()


def _build_capture_triples():
    """For each point, the (first, second, far end) cells in each of 8 directions."""
    table = []
    for cell in range(CELLS):
        r, c = divmod(cell, SIZE)
        triples = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + 3 * dr, c + 3 * dc
                if 0 <= rr < SIZE and 0 <= cc < SIZE:
                    triples.append(((r + dr) * SIZE + c + dc,
                                    (r + 2 * dr) * SIZE + c + 2 * dc,
                                    rr * SIZE + cc))
        table.append(tuple(triples))
    return table


CAPTURE_TRIPLES = _build_capture_triples()

_zobrist_source = random.Random(20261019)
ZOBRIST = ([0] * CELLS,
           [_zobrist_source.getrandbits(62) for _ in range(CELLS)],
           [_zobrist_source.getrandbits(62) for _ in range(CELLS)])
SIDE_KEY = (0, _zobrist_source.getrandbits(62), _zobrist_source.getrandbits(62))

# A small pull towards the middle, only enough to break ties between quiet moves.
CENTRE_BIAS = [(SIZE - abs(p // SIZE - SIZE // 2) - abs(p % SIZE - SIZE // 2)) // 5
               for p in range(CELLS)]

CAPTURE_PATTERNS = (None,
                    (bytes((1, 2, 2, 0)), bytes((0, 2, 2, 1))),
                    (bytes((2, 1, 1, 0)), bytes((0, 1, 1, 2))))

# --------------------------------------------------------------------------
# What one line is worth, and what can be done on it
#
# A line's entry is a tuple. Element 0 packs the numbers that are summed over
# the whole board into one integer (so a move updates all of them with a single
# addition); the rest are the positions behind those counts, per colour.

SCORE_BITS = 20
SCORE_MASK = (1 << SCORE_BITS) - 1
COUNT_MASK = (1 << 12) - 1
SCORE_SHIFT = (None, 0, 20)
FIVE_SHIFT = (None, 40, 52)       # lines holding an exact five
WIN_SHIFT = (None, 64, 76)        # points completing an exact five
OPEN_SHIFT = (None, 88, 100)      # points making two such points at once (an open four)
CAP_SHIFT = (None, 112, 124)      # points capturing a pair
FOUR_SHIFT = (None, 136, 148)     # points making a four

WIN_AT = (None, 1, 2)             # where each list sits in the entry
CAP_AT = (None, 3, 4)
FOUR_AT = (None, 5, 6)
OPEN_AT = (None, 7, 8)
GROWING_AT = (None, 9, 10)

EVAL_BITS = 26                    # the learned line value for each colour to move,
EVAL_MASK = (1 << EVAL_BITS) - 1  # stored with LINE_OFFSET added so it is never negative
EVAL_SHIFT = (None, 160, 160 + EVAL_BITS)
LINE_OFFSET = 1 << 18

SCORE_MEMO = {}


def _pattern_columns(width):
    """For every base-4 code of a pattern, its place in PATTERN_SCORES (-1: none).

    A pattern and its mirror image share a place; patterns without stones and
    patterns with an off-board point anywhere but at an end have none. The
    numbering is the one the scores were fitted with.
    """
    column = [-1] * (4 ** width)
    count = 0
    for code in range(4 ** width):
        digits = [(code >> (2 * k)) & 3 for k in range(width)]
        if (1 not in digits and 2 not in digits) or 3 in digits[1:-1]:
            continue
        mirror = 0
        for k in range(width):
            mirror |= digits[width - 1 - k] << (2 * k)
        if mirror < code:
            column[code] = column[mirror]
        else:
            column[code] = count
            count += 1
    return column


PATTERN_COLUMN = _pattern_columns(PATTERN_WIDTH) if PATTERN_WIDTH else ()


def _pattern_value(line, colour):
    """The learned value of one line for `colour` to move."""
    width = PATTERN_WIDTH
    swap = (0, 1, 2) if colour == 1 else (0, 2, 1)
    padded = [3] + [swap[v] for v in line] + [3]
    total = 0
    for start in range(len(padded) - width + 1):
        code = 0
        for k in range(width):
            code |= padded[start + k] << (2 * k)
        column = PATTERN_COLUMN[code]
        if column >= 0:
            total += PATTERN_SCORES[column]
    return total


def _analyse(line, whole=True):
    """Everything about one line of 0/1/2 bytes. See the entry layout above.

    `whole` is False for the short stretches the move ordering looks at, which
    never reach the evaluation and so need no learned value.
    """
    length = len(line)
    packed = 0
    lists = [None, (), (), (), (), (), (), (), (), False, False]
    for colour in (1, 2):
        other = 3 - colour
        score = 0
        fives = 0
        wins = set()
        makers = {}
        for i in range(length - 4):
            window = line[i:i + 5]
            if other in window:
                continue
            count = window.count(colour)
            if count == 0:
                continue
            # A neighbour of the same colour would turn this five into six or
            # more, which does not win, so the window is worth nothing.
            if i and line[i - 1] == colour:
                continue
            if i + 5 < length and line[i + 5] == colour:
                continue
            if count == 5:
                fives += 1
                continue
            score += WINDOW_VALUE[count]
            if count == 4:
                wins.add(i + window.index(0))
            elif count == 3:
                first = window.index(0)
                second = window.index(0, first + 1)
                makers.setdefault(i + first, set()).add(i + second)
                makers.setdefault(i + second, set()).add(i + first)
            elif count == 2:
                lists[GROWING_AT[colour]] = True
        captures = []
        near_pattern, far_pattern = CAPTURE_PATTERNS[colour]
        for i in range(length - 3):
            quad = line[i:i + 4]
            if quad == near_pattern:
                captures.append(i + 3)
            elif quad == far_pattern:
                captures.append(i)
        score += CAPTURE_VALUE * len(captures)
        opens = tuple(sorted(p for p, made in makers.items() if len(made) >= 2))

        lists[WIN_AT[colour]] = tuple(sorted(wins))
        lists[CAP_AT[colour]] = tuple(captures)
        lists[FOUR_AT[colour]] = tuple(sorted(makers))
        lists[OPEN_AT[colour]] = opens
        packed += (min(score, SCORE_MASK) << SCORE_SHIFT[colour]) \
            + (fives << FIVE_SHIFT[colour]) + (len(wins) << WIN_SHIFT[colour]) \
            + (len(opens) << OPEN_SHIFT[colour]) + (len(captures) << CAP_SHIFT[colour]) \
            + (len(makers) << FOUR_SHIFT[colour])
        if PATTERN_WIDTH and whole:
            packed += (LINE_OFFSET + _pattern_value(line, colour)) << EVAL_SHIFT[colour]
    lists[0] = packed
    return tuple(lists)


def _open_four_points(line, colour):
    """Positions where `colour` would make two win positions at once on this line."""
    length = len(line)
    other = 3 - colour
    makers = {}
    for i in range(length - 4):
        window = line[i:i + 5]
        if other in window or window.count(colour) != 3:
            continue
        if i and line[i - 1] == colour:
            continue
        if i + 5 < length and line[i + 5] == colour:
            continue
        first = window.index(0)
        second = window.index(0, first + 1)
        makers.setdefault(i + first, set()).add(i + second)
        makers.setdefault(i + second, set()).add(i + first)
    return [p for p, made in makers.items() if len(made) >= 2]


def _three_makers(line):
    """For each colour, the positions where a stone creates an open three.

    "Open three" here means exactly what matters tactically: afterwards the
    colour has a point on this line that would make an open four.
    """
    length = len(line)
    result = []
    for colour in (1, 2):
        other = 3 - colour
        found = []
        if not _open_four_points(line, colour):
            spots = set()
            for i in range(length - 4):
                window = line[i:i + 5]
                if other in window or window.count(colour) != 2:
                    continue
                for j in range(5):
                    if window[j] == 0:
                        spots.add(i + j)
            piece = BYTE[colour]
            for spot in sorted(spots):
                if _open_four_points(line[:spot] + piece + line[spot + 1:], colour):
                    found.append(spot)
        result.append(tuple(found))
    return tuple(result)


def _line_score(line):
    """Black's score minus White's on this line, counting a five as decisive."""
    score = SCORE_MEMO.get(line)
    if score is None:
        packed = _analyse(line, False)[0]
        score = (packed & SCORE_MASK) - ((packed >> SCORE_BITS) & SCORE_MASK) \
            + FIVE_GAIN * (((packed >> FIVE_SHIFT[1]) & COUNT_MASK)
                           - ((packed >> FIVE_SHIFT[2]) & COUNT_MASK))
        SCORE_MEMO[line] = score
    return score


def _fill_gain(segment, length, offset):
    """What a stone at `offset` of this segment does to the line's score.

    Returns (ordering value for Black, for White, score change for Black, for
    White). The ordering value is the mover's own change plus a share of what
    the opponent would have gained by playing there instead.
    """
    line = _decode(segment, length)
    before = _line_score(line)
    black = _line_score(line[:offset] + BYTE[1] + line[offset + 1:]) - before
    white = before - _line_score(line[:offset] + BYTE[2] + line[offset + 1:])
    gains = (int(black + DEFENCE_SHARE * white), int(white + DEFENCE_SHARE * black),
             black, white)
    GAIN_MEMO[(length, offset)][segment] = gains
    return gains


# --------------------------------------------------------------------------
# The engine


class _Engine:
    NODE_RATE = NODE_RATE                    # nodes that count as a second on the node clock

    def __init__(self):
        self.board = [0] * CELLS
        self.lines = [0] * LINE_COUNT
        self.entry = []
        for cells in LINE_CELLS:
            memo = LINE_MEMO[len(cells)]
            if 0 not in memo:
                memo[0] = _analyse(bytes(len(cells)))
            self.entry.append(memo[0])
        self.stats = sum(entry[0] for entry in self.entry)   # the packed numbers of all lines
        self.near = [0] * CELLS              # stones within two steps of each point
        self.count = [0, 0, 0]               # stones of each colour on the board
        self.hash = 0
        self.history = []                    # one record per move, enough to take it back
        self.table = {}
        self.order = {}                      # position -> its ordered quiet moves and their gains
        self.raw_gain = {}
        self.vcf_seen = {}
        self.vct_seen = {}
        self.nodes = 0
        self.now = _time.perf_counter        # the clock every time limit is measured on
        self.deadline = 0.0
        self.partial = None
        self.info = (0, 0, 0)                # depth, value and nodes of the last search
        self.ply_limit = MAX_PLY
        self.root_quiet = True
        self.killers = [[-1, -1] for _ in range(MAX_PLY + 2)]

    def _node_time(self):
        return self.nodes / float(NODE_RATE)

    def is_quiet(self):
        """No four, open four or immediate win for either side."""
        stats = self.stats
        return not any((stats >> shift[side]) & COUNT_MASK
                       for shift in (WIN_SHIFT, OPEN_SHIFT, FOUR_SHIFT) for side in (1, 2))

    def candidate_cells(self, colour, limit):
        return self.candidates(colour, limit)

    def use_node_clock(self, enabled=True):
        """Measure "time" in searched nodes (see FIXED_NODES) or on the real clock."""
        self.now = self._node_time if enabled else _time.perf_counter

    # ----------------------------------------------------------- the board

    def _put(self, cell, value, record):
        """Change one point, noting in `record` what is needed to change it back."""
        board = self.board
        old = board[cell]
        board[cell] = value
        self.hash ^= ZOBRIST[old][cell] ^ ZOBRIST[value][cell]
        lines = self.lines
        entries = self.entry
        change = value - old
        stats = self.stats
        for line_id, shift, memo, length in CELL_LINES[cell]:
            before = lines[line_id]
            current = entries[line_id]
            record += (line_id, before, current)
            line = before + (change << shift)
            lines[line_id] = line
            new = memo.get(line)
            if new is None:
                new = memo[line] = _analyse(_decode(line, length))
            entries[line_id] = new
            stats += new[0] - current[0]
        self.stats = stats
        near = self.near
        count = self.count
        count[old] -= 1
        count[value] += 1
        if old == 0:
            for other in NEIGHBOURS[cell]:
                near[other] += 1
        elif value == 0:
            for other in NEIGHBOURS[cell]:
                near[other] -= 1

    def play(self, cell, colour):
        """Place a stone and remove what it captures."""
        captured = ()
        if (self.stats >> CAP_SHIFT[colour]) & COUNT_MASK:
            board = self.board
            other = 3 - colour
            for first, second, far in CAPTURE_TRIPLES[cell]:
                if board[first] == other and board[second] == other and board[far] == colour:
                    captured += (first, second)
        record = [cell, colour, captured, self.stats, self.hash]
        self._put(cell, colour, record)
        for taken in captured:
            self._put(taken, 0, record)
        self.history.append(record)

    def undo(self):
        record = self.history.pop()
        cell, colour, captured = record[0], record[1], record[2]
        self.stats = record[3]
        self.hash = record[4]
        lines = self.lines
        entries = self.entry
        # Newest change first, so a line touched twice ends at its oldest state.
        for i in range(len(record) - 3, 4, -3):
            line_id = record[i]
            lines[line_id] = record[i + 1]
            entries[line_id] = record[i + 2]
        board = self.board
        near = self.near
        count = self.count
        board[cell] = 0
        count[colour] -= 1
        count[0] += 1
        for other in NEIGHBOURS[cell]:
            near[other] -= 1
        if captured:
            enemy = 3 - colour
            for taken in captured:
                board[taken] = enemy
                for other in NEIGHBOURS[taken]:
                    near[other] += 1
            count[enemy] += len(captured)
            count[0] -= len(captured)

    def unwind(self, length):
        while len(self.history) > length:
            self.undo()

    def load(self, grid):
        """Make the engine's position equal to the board handed to take_turn()."""
        self.history = []
        board = self.board
        scratch = []
        for r in range(SIZE):
            row = grid[r]
            base = r * SIZE
            for c in range(SIZE):
                value = int(row[c]) + 1          # -1/0/1  ->  0/1/2
                if board[base + c] != value:
                    self._put(base + c, value, scratch)
                    del scratch[:]
        self.count[0] = CELLS - self.count[1] - self.count[2]
        self._trim_memos()

    def _trim_memos(self):
        if sum(len(memo) for memo in LINE_MEMO.values()) > MEMO_LIMIT:
            for memo in LINE_MEMO.values():
                memo.clear()
            for line_id, cells in enumerate(LINE_CELLS):
                # Entries in use stay valid; they are simply looked up again.
                LINE_MEMO[len(cells)][self.lines[line_id]] = self.entry[line_id]
        if sum(len(memo) for memo in GAIN_MEMO.values()) > MEMO_LIMIT:
            for memo in GAIN_MEMO.values():
                memo.clear()
        if len(SCORE_MEMO) > MEMO_LIMIT:
            SCORE_MEMO.clear()
        if sum(len(memo) for memo in THREE_MEMO.values()) > MEMO_LIMIT:
            for memo in THREE_MEMO.values():
                memo.clear()
        if len(self.table) > MEMO_LIMIT:
            self.table.clear()
        if len(self.order) > MEMO_LIMIT // 4:
            self.order.clear()
        self.vcf_seen.clear()
        self.vct_seen.clear()

    # ----------------------------------------------- questions about lines

    def cells_of(self, index):
        """The points behind one of the counts, e.g. cells_of(WIN_AT[colour]).

        A point is listed once per line it appears on.
        """
        found = []
        line_id = 0
        for entry in self.entry:
            positions = entry[index]
            if positions:
                cells = LINE_CELLS[line_id]
                for position in positions:
                    found.append(cells[position])
            line_id += 1
        return found

    def three_moves(self, colour):
        """Points where `colour` would make an open three."""
        cells = set()
        lines = self.lines
        index = colour - 1
        growing = GROWING_AT[colour]
        line_id = 0
        for entry in self.entry:
            if entry[growing]:
                where = LINE_CELLS[line_id]
                memo = THREE_MEMO[len(where)]
                line = lines[line_id]
                makers = memo.get(line)
                if makers is None:
                    makers = memo[line] = _three_makers(_decode(line, len(where)))
                for position in makers[index]:
                    cells.add(where[position])
            line_id += 1
        return cells

    # ------------------------------------------------------- move ordering

    def ordered(self, cells, colour):
        """(value, cell, own score change) for the given empty cells, best first; and
        the points that must not be pruned: where we capture, and where the
        opponent would capture (playing there ourselves saves the pair)."""
        lines = self.lines
        index = colour - 1
        raw_index = index + 2
        gains = CELL_GAINS
        stats = self.stats
        captures = ()
        if (stats >> CAP_SHIFT[colour]) & COUNT_MASK:
            captures = self.cells_of(CAP_AT[colour])
        rescues = ()
        if (stats >> CAP_SHIFT[3 - colour]) & COUNT_MASK:
            rescues = self.cells_of(CAP_AT[3 - colour])
        open_fours = ()
        if (stats >> OPEN_SHIFT[colour]) & COUNT_MASK:
            open_fours = self.cells_of(OPEN_AT[colour])
        scored = []
        for cell in cells:
            value = CENTRE_BIAS[cell]
            raw = 0
            for line_id, shift, mask, memo, length, offset in gains[cell]:
                segment = (lines[line_id] >> shift) & mask
                gain = memo.get(segment)
                if gain is None:
                    gain = _fill_gain(segment, length, offset)
                value += gain[index]
                raw += gain[raw_index]
            if captures and cell in captures:
                value += CAPTURE_BONUS * captures.count(cell)
                raw += 2 * FUTILITY_MARGIN       # never dismissed as futile
            if rescues and cell in rescues:
                value += RESCUE_BONUS * rescues.count(cell)
                raw += 2 * FUTILITY_MARGIN
            if open_fours and cell in open_fours:
                value += OPEN_FOUR_BONUS
            scored.append((value, cell, raw))
        scored.sort(reverse=True)
        return scored, (captures + rescues if captures and rescues else captures or rescues)

    def candidates(self, colour, limit):
        """The most promising moves for `colour`, best first.

        Moves that capture, or that save a pair from capture, are kept even
        when they fall outside the limit: there are few of them and their real
        value only shows in the search.
        """
        board = self.board
        near = self.near
        scored, captures = self.ordered(
            [cell for cell in range(CELLS) if near[cell] and not board[cell]], colour)
        kept = scored[:limit]
        if captures:
            kept += [item for item in scored[limit:] if item[1] in captures]
        self.raw_gain = {cell: raw for _, cell, raw in kept}
        return [cell for _, cell, _ in kept]

    def defences(self, colour):
        """Moves worth considering when the opponent threatens an open four.

        Anything else loses to that four, so the only candidates are the points
        where the opponent would make a four, our own fours, and captures.
        """
        other = 3 - colour
        cells = set(self.cells_of(FOUR_AT[other]))
        stats = self.stats
        if (stats >> FOUR_SHIFT[colour]) & COUNT_MASK:
            cells.update(self.cells_of(FOUR_AT[colour]))
        if (stats >> CAP_SHIFT[colour]) & COUNT_MASK:
            cells.update(self.cells_of(CAP_AT[colour]))
        return [item[1] for item in self.ordered(cells, colour)[0]]

    def forced_replies(self, colour):
        """Answers to the opponent's four: block it, or capture something."""
        moves = list(dict.fromkeys(self.cells_of(WIN_AT[3 - colour])))
        if (self.stats >> CAP_SHIFT[colour]) & COUNT_MASK:
            for cell in self.cells_of(CAP_AT[colour]):
                if cell not in moves:
                    moves.append(cell)
        return moves

    # ---------------------------------------------------------- evaluation

    def evaluate(self, colour):
        """Static value for the side to move."""
        stats = self.stats
        count = self.count
        other = 3 - colour
        if PATTERN_WIDTH:
            terms = PATTERN_TERMS
            value = ((stats >> EVAL_SHIFT[colour]) & EVAL_MASK) - LINE_COUNT * LINE_OFFSET
            value += terms["tempo"] + terms["stones"] * (count[colour] - count[other])
            value += terms["my_captures"] * ((stats >> CAP_SHIFT[colour]) & COUNT_MASK)
            value += terms["their_captures"] * ((stats >> CAP_SHIFT[other]) & COUNT_MASK)
            if (stats >> OPEN_SHIFT[colour]) & COUNT_MASK:
                value += terms["my_open"]
            if ((stats >> OPEN_SHIFT[other]) & COUNT_MASK) >= 3 \
                    and not (stats >> FOUR_SHIFT[colour]) & COUNT_MASK:
                value += terms["their_double"]
            return value
        mine = (stats >> SCORE_SHIFT[colour]) & SCORE_MASK
        theirs = (stats >> SCORE_SHIFT[other]) & SCORE_MASK
        value = int(mine * MOVER_WEIGHT) - theirs + STONE_VALUE * (count[colour] - count[other])
        value += TEMPO_VALUE + MOVER_CAPTURE_VALUE * ((stats >> CAP_SHIFT[colour]) & COUNT_MASK)
        if (stats >> OPEN_SHIFT[colour]) & COUNT_MASK:
            return value + THREAT_VALUE
        # Two open threes against us cannot both be blocked; only our own
        # fours could still turn that around.
        if ((stats >> OPEN_SHIFT[other]) & COUNT_MASK) >= 3 \
                and not (stats >> FOUR_SHIFT[colour]) & COUNT_MASK:
            value -= DOUBLE_THREE_PENALTY
        return value

    def quiesce(self, alpha, beta, colour, ply, depth):
        """Leaf value once the pending captures have been played out.

        The side to move may decline to capture (taking the static value), so
        this only ever corrects positions where a capture is hanging.
        """
        self.nodes += 1
        stats = self.stats
        if (stats >> FIVE_SHIFT[colour]) & COUNT_MASK:
            return WIN - ply
        if (stats >> WIN_SHIFT[colour]) & COUNT_MASK:
            return WIN - ply - 1
        best = self.evaluate(colour)
        other = 3 - colour
        if depth <= 0 or best >= beta or not (stats >> CAP_SHIFT[colour]) & COUNT_MASK \
                or (stats >> WIN_SHIFT[other]) & COUNT_MASK:
            return best
        if best > alpha:
            alpha = best
        for move in dict.fromkeys(self.cells_of(CAP_AT[colour])):
            self.play(move, colour)
            value = -self.quiesce(-beta, -alpha, other, ply + 1, depth - 1)
            self.undo()
            if value > best:
                best = value
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        return best

    # -------------------------------------------------------------- search
    #
    # Depth is counted in half plies. A quiet move costs two. A move that makes
    # a four costs one and the reply to it nothing, so forcing sequences are
    # followed far beyond the nominal depth; the reply to an open three costs one.

    def quiet_moves(self, colour, key, width):
        cached = self.order.get(key)
        if cached is None:
            moves = self.candidates(colour, CACHE_WIDTH)
            cached = self.order[key] = (moves, self.raw_gain)
        moves = cached[0][:width]
        if len(cached[0]) > width:
            stats = self.stats
            if (stats >> CAP_SHIFT[1]) & COUNT_MASK or (stats >> CAP_SHIFT[2]) & COUNT_MASK:
                keep = self.cells_of(CAP_AT[1]) + self.cells_of(CAP_AT[2])
                moves += [move for move in cached[0][width:] if move in keep]
        return moves, cached[1]

    def search(self, depth, alpha, beta, colour, ply, allow_null=True):
        self.nodes += 1
        if not self.nodes & 255 and self.now() > self.deadline:
            raise _OutOfTime()
        stats = self.stats
        if (stats >> FIVE_SHIFT[colour]) & COUNT_MASK:
            return WIN - ply                 # the capture just made left me a five
        win_shift = WIN_SHIFT[colour]
        if (stats >> win_shift) & COUNT_MASK:
            return WIN - ply - 1
        other = 3 - colour
        in_check = (stats >> WIN_SHIFT[other]) & COUNT_MASK
        if ply >= self.ply_limit:
            return self.evaluate(colour)
        if depth <= 0 and not in_check:
            if (stats >> CAP_SHIFT[colour]) & COUNT_MASK:
                return self.quiesce(alpha, beta, colour, ply, QUIESCE_DEPTH)
            return self.evaluate(colour)

        key = self.hash ^ SIDE_KEY[colour]
        stored = self.table.get(key)
        first = -1
        futile = None
        if stored is not None:
            stored_depth, flag, value, first = stored
            if stored_depth >= depth:
                if flag == 0:
                    return value
                if flag == 1:
                    if value >= beta:
                        return value
                elif value <= alpha:
                    return value

        if in_check:
            moves = self.forced_replies(colour)
            reply_cost = 0
        elif (stats >> OPEN_SHIFT[other]) & COUNT_MASK:
            moves = self.defences(colour)
            reply_cost = THREAT_REPLY_COST
        else:
            if NULL_MOVE and allow_null and depth >= NULL_MIN_DEPTH and beta < WIN // 2 \
                    and self.evaluate(colour) >= beta:
                # Nothing is hanging over us; if even passing keeps the value
                # above beta, a real move will too.
                value = -self.search(depth - 2 - NULL_REDUCTION, -beta, 1 - beta, other,
                                     ply + 1, False)
                if value >= beta and value < WIN // 2:
                    return beta
            if depth >= 6:
                width = WIDTH_DEEP
            elif depth >= 4:
                width = WIDTH_MID
            else:
                width = WIDTH_SHALLOW
                if depth <= 2 and beta < WIN // 2 and alpha > -WIN // 2:
                    # Last ply before the leaves, with nothing hanging over us.
                    static = self.evaluate(colour)
                    if static - FRONTIER_MARGIN >= beta:
                        return static - FRONTIER_MARGIN
                    futile = alpha - static - FUTILITY_MARGIN
            moves, raw_gain = self.quiet_moves(colour, key, width)
            reply_cost = QUIET_COST
            if not moves:
                return 0
        killers = self.killers[ply]
        for preferred in (killers[1], killers[0], first):
            if preferred >= 0 and preferred in moves:
                moves.remove(preferred)
                moves.insert(0, preferred)

        original_alpha = alpha
        best = -INFINITY
        chosen = moves[0]
        play = self.play
        undo = self.undo
        search = self.search
        history = self.history
        open_shift = OPEN_SHIFT[colour]
        next_ply = ply + 1
        index = 0
        for move in moves:
            if futile is not None and index and raw_gain[move] <= futile:
                # Even if everything this move gains were kept, it would not
                # reach alpha.
                bound = static + raw_gain[move] + FUTILITY_MARGIN
                if bound > best:
                    best = bound
                index += 1
                continue
            play(move, colour)
            after = self.stats
            if (after >> win_shift) & COUNT_MASK:   # made a four: the answer is forced
                child = depth - (reply_cost if reply_cost < FOUR_COST else FOUR_COST)
                quiet = False
            else:
                child = depth - reply_cost
                quiet = reply_cost == QUIET_COST and not (after >> open_shift) & COUNT_MASK \
                    and not history[-1][2]
            if index == 0:
                value = -search(child, -beta, -alpha, other, next_ply)
            else:
                if quiet and index >= REDUCE_AFTER and depth >= REDUCE_MIN_DEPTH:
                    # Late quiet moves get a shallower look first.
                    value = -search(child - 2, -alpha - 1, -alpha, other, next_ply)
                    if value > alpha:
                        value = -search(child, -alpha - 1, -alpha, other, next_ply)
                else:
                    value = -search(child, -alpha - 1, -alpha, other, next_ply)
                if alpha < value < beta:
                    value = -search(child, -beta, -alpha, other, next_ply)
            undo()
            index += 1
            if value > best:
                best = value
                chosen = move
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        if killers[0] != move:
                            killers[1] = killers[0]
                            killers[0] = move
                        break
        flag = 1 if best >= beta else (2 if best <= original_alpha else 0)
        self.table[key] = (depth, flag, best, chosen)
        return best

    def search_root(self, moves, plies, colour):
        """Returns (best move, its value). Raises _OutOfTime, leaving self.partial set."""
        other = 3 - colour
        depth = 2 * plies
        alpha, beta = -INFINITY, INFINITY
        stats = self.stats
        if (stats >> WIN_SHIFT[other]) & COUNT_MASK:
            reply_cost = 0
        elif self.root_quiet:
            reply_cost = QUIET_COST
        else:
            reply_cost = THREAT_REPLY_COST
        self.ply_limit = min(MAX_PLY, 2 * plies + 8)
        win_shift = WIN_SHIFT[colour]
        best_move, best = moves[0], -INFINITY
        self.partial = None
        for index, move in enumerate(moves):
            self.play(move, colour)
            if (self.stats >> win_shift) & COUNT_MASK:
                child = depth - (reply_cost if reply_cost < FOUR_COST else FOUR_COST)
            else:
                child = depth - reply_cost
            if index == 0:
                value = -self.search(child, -beta, -alpha, other, 1)
            else:
                value = -self.search(child, -alpha - 1, -alpha, other, 1)
                if value > alpha:
                    value = -self.search(child, -beta, -alpha, other, 1)
            self.undo()
            if value > best:
                best, best_move = value, move
                self.partial = (move, value)
                if value > alpha:
                    alpha = value
        return best_move, best

    # --------------------------------------------- forced wins by fours (VCF)

    def vcf(self, colour, depth):
        """A move for `colour` that starts a win by continuous fours, or -1.

        Every attacking move makes a four. The defender may block it or make
        any capture; the attack must survive all of those replies.
        """
        self.nodes += 1
        if not self.nodes & 63 and self.now() > self.deadline:
            raise _OutOfTime()
        stats = self.stats
        other = 3 - colour
        if (stats >> WIN_SHIFT[colour]) & COUNT_MASK:
            return self.cells_of(WIN_AT[colour])[0]
        if depth <= 0 or not (stats >> FOUR_SHIFT[colour]) & COUNT_MASK:
            return -1
        key = self.hash ^ SIDE_KEY[colour]
        if self.vcf_seen.get(key, -1) >= depth:
            return -1
        fours = self.cells_of(FOUR_AT[colour])
        if (stats >> WIN_SHIFT[other]) & COUNT_MASK:
            threats = self.cells_of(WIN_AT[other])
            attacks = [cell for cell in dict.fromkeys(fours) if cell in threats]
        else:
            opens = self.cells_of(OPEN_AT[colour])
            attacks = sorted(dict.fromkeys(fours),
                             key=lambda cell: (cell in opens, fours.count(cell)), reverse=True)
        win_shift = WIN_SHIFT[colour]
        other_win_shift = WIN_SHIFT[other]
        for move in attacks:
            self.play(move, colour)
            after = self.stats
            if not (after >> win_shift) & COUNT_MASK or (after >> FIVE_SHIFT[other]) & COUNT_MASK \
                    or (after >> other_win_shift) & COUNT_MASK:
                self.undo()
                continue
            refuted = False
            for reply in self.forced_replies(other):
                self.play(reply, other)
                now = self.stats
                if (now >> FIVE_SHIFT[colour]) & COUNT_MASK or (now >> win_shift) & COUNT_MASK:
                    holds = True
                elif (now >> FIVE_SHIFT[other]) & COUNT_MASK:
                    holds = False
                else:
                    holds = self.vcf(colour, depth - 1) >= 0
                self.undo()
                if not holds:
                    refuted = True
                    break
            self.undo()
            if not refuted:
                return move
        self.vcf_seen[key] = depth
        return -1

    def try_vcf(self, colour, seconds):
        """vcf() with its own time slice. Returns -1 if nothing was proven in time."""
        saved_deadline = self.deadline
        self.deadline = min(saved_deadline, self.now() + seconds)
        base = len(self.history)
        try:
            return self.vcf(colour, VCF_DEPTH)
        except _OutOfTime:
            self.unwind(base)
            return -1
        finally:
            self.deadline = saved_deadline

    # ------------------------------- forced wins by fours and threes (VCT)

    def threat_moves(self, colour):
        """Every move of `colour` that makes a four or an open three, best first."""
        cells = self.three_moves(colour)
        if (self.stats >> FOUR_SHIFT[colour]) & COUNT_MASK:
            cells.update(self.cells_of(FOUR_AT[colour]))
        return [item[1] for item in self.ordered(cells, colour)[0]]

    def vct_attack(self, colour, depth):
        """True if `colour`, to move, wins by force using only fours and threes.

        `depth` is the number of attacking moves still allowed.
        """
        self.nodes += 1
        if not self.nodes & 63 and self.now() > self.deadline:
            raise _OutOfTime()
        stats = self.stats
        if (stats >> WIN_SHIFT[colour]) & COUNT_MASK:
            return True
        other = 3 - colour
        if (stats >> FIVE_SHIFT[other]) & COUNT_MASK:
            return False
        key = self.hash ^ SIDE_KEY[colour]
        seen = self.vct_seen.get(key)
        if seen is not None:
            if seen < 0:
                return True
            if seen >= depth:
                return False
        if depth <= 0:
            return False
        if (stats >> WIN_SHIFT[other]) & COUNT_MASK:
            # In check: the attack only continues through the forced answer.
            moves = self.forced_replies(colour)
        else:
            moves = self.threat_moves(colour)
        won = False
        for move in moves:
            self.play(move, colour)
            won = self.vct_defend(colour, depth - 1)
            self.undo()
            if won:
                break
        self.vct_seen[key] = -1 if won else depth
        return won

    def vct_defend(self, colour, depth):
        """True if attacker `colour` still wins whatever the defender, to move, does."""
        other = 3 - colour
        stats = self.stats
        if (stats >> FIVE_SHIFT[other]) & COUNT_MASK or (stats >> WIN_SHIFT[other]) & COUNT_MASK:
            return False                     # the defender wins first
        if (stats >> FIVE_SHIFT[colour]) & COUNT_MASK:
            return True
        if (stats >> WIN_SHIFT[colour]) & COUNT_MASK:
            replies = self.forced_replies(other)
        elif (stats >> OPEN_SHIFT[colour]) & COUNT_MASK:
            replies = self.defences(other)
        else:
            return False                     # the last move was not a threat
        for reply in replies:
            self.play(reply, other)
            won = self.vct_attack(colour, depth)
            self.undo()
            if not won:
                return False
        return True

    def find_vct(self, colour, seconds, max_depth=VCT_DEPTH):
        """A first move of a forced win for `colour` (to move), or -1.

        Searched with growing depth so the shortest win is found first and a
        time-out still leaves the shallower results valid.
        """
        saved_deadline = self.deadline
        self.deadline = min(saved_deadline, self.now() + seconds)
        base = len(self.history)
        other = 3 - colour
        try:
            if (self.stats >> WIN_SHIFT[colour]) & COUNT_MASK:
                return self.cells_of(WIN_AT[colour])[0]
            if (self.stats >> WIN_SHIFT[other]) & COUNT_MASK:
                moves = self.forced_replies(colour)
            else:
                moves = self.threat_moves(colour)
            for depth in range(1, max_depth + 1):
                for move in moves:
                    self.play(move, colour)
                    won = self.vct_defend(colour, depth - 1)
                    self.undo()
                    if won:
                        return move
            return -1
        except _OutOfTime:
            self.unwind(base)
            return -1
        finally:
            self.deadline = saved_deadline

    # ------------------------------------------------------------ choosing

    def quick_move(self, colour):
        """The best move without searching: used when there is no time to think."""
        other = 3 - colour
        if (self.stats >> WIN_SHIFT[other]) & COUNT_MASK:
            replies = self.forced_replies(colour)
            for move in replies:
                self.play(move, colour)
                after = self.stats
                safe = not (after >> WIN_SHIFT[other]) & COUNT_MASK \
                    and not (after >> FIVE_SHIFT[other]) & COUNT_MASK
                self.undo()
                if safe:
                    return move
            return replies[0]
        moves = self.candidates(colour, 1)
        return moves[0] if moves else -1

    def first_reply(self):
        """Answer the first stone on one of the eight points around it."""
        stone = next(cell for cell in range(CELLS) if self.board[cell])
        r, c = divmod(stone, SIZE)
        options = [(r + dr) * SIZE + c + dc for dr, dc in FIRST_REPLIES
                   if 2 <= r + dr < SIZE - 2 and 2 <= c + dc < SIZE - 2]
        if not options:                      # the stone is in a corner: step towards the middle
            r += 1 if r < SIZE // 2 else -1
            c += 1 if c < SIZE // 2 else -1
            return r * SIZE + c
        return random.choice(options)

    def defusing_moves(self, colour, threat, seconds, each):
        """Moves for `colour` after which the opponent's forced win is gone.

        `threat` is the first move of that win. Points on its path are tried
        first; the search stops when time is up or enough moves are known.
        """
        other = 3 - colour
        board = self.board
        cells = {threat}
        cells.update(self.cells_of(FOUR_AT[other]))
        cells.update(self.three_moves(other))
        cells.update(self.cells_of(FOUR_AT[colour]))
        cells.update(self.cells_of(CAP_AT[colour]))
        pool = [item[1] for item in self.ordered([c for c in cells if not board[c]], colour)[0]]
        if threat in pool:
            pool.remove(threat)
        pool.insert(0, threat)
        for move in self.candidates(colour, ROOT_WIDTH):
            if move not in pool:
                pool.append(move)
        stop = self.now() + seconds
        safe = []
        for move in pool:
            if self.now() >= stop or len(safe) >= DEFUSING_ENOUGH:
                break
            self.play(move, colour)
            self.vct_seen.clear()
            refutation = self.find_vct(other, each, SAFETY_DEPTH)
            self.undo()
            if refutation < 0:
                safe.append(move)
        return safe

    def choose(self, colour, budget):
        self.nodes = 0
        start = self.now()
        self.deadline = start + budget
        other = 3 - colour
        stones = self.count[1] + self.count[2]
        if stones == 0:
            return CENTRE
        if stones == 1:
            return self.first_reply()
        stats = self.stats
        if (stats >> WIN_SHIFT[colour]) & COUNT_MASK:
            return self.cells_of(WIN_AT[colour])[0]
        if budget < INSTANT_BELOW_S:
            return self.quick_move(colour)

        # 1. A forced win of our own.
        base = len(self.history)
        move = self.try_vcf(colour, budget * 0.05)
        if move >= 0:
            return move
        move = self.find_vct(colour, budget * OWN_VCT_SHARE)
        if move >= 0:
            return move

        # 2. What would the opponent do if we did nothing? If that wins by
        #    force, only moves that take the win away are worth searching.
        in_check = (stats >> WIN_SHIFT[other]) & COUNT_MASK
        defusing = None
        if not in_check:
            probe_start = self.now()
            self.vct_seen.clear()
            threat = self.find_vct(other, budget * PROBE_SHARE, SAFETY_DEPTH)
            if threat >= 0:
                took = self.now() - probe_start
                defusing = self.defusing_moves(colour, threat, budget * 0.3,
                                               max(2.0 * took, budget * 0.03))
            self.vct_seen.clear()

        self.root_quiet = False
        if in_check:
            moves = self.forced_replies(colour)
        elif defusing:
            moves = defusing
        elif (stats >> OPEN_SHIFT[other]) & COUNT_MASK:
            moves = self.defences(colour)
        else:
            moves = self.candidates(colour, ROOT_WIDTH)
            self.root_quiet = True
        if not moves:
            return self.quick_move(colour)
        if len(moves) == 1:
            return moves[0]
        for killers in self.killers:
            killers[0] = killers[1] = -1

        # 3. The main search over those moves.
        spent = self.now() - start
        left = budget - spent
        search_start = self.now()
        self.deadline = search_start + left * SEARCH_SHARE   # the rest is for the last check
        best, best_value = moves[0], 0
        ranking = list(moves)
        try:
            for plies in range(2, MAX_PLY):
                move, value = self.search_root(ranking, plies, colour)
                settled = move == best and value > best_value - UNSETTLED_DROP
                best, best_value = move, value
                self.info = (plies, value, self.nodes)
                ranking.remove(move)
                ranking.insert(0, move)
                if abs(value) >= WIN - 2 * MAX_PLY:
                    break
                # A new favourite or a falling value deserves another look.
                if settled and self.now() - search_start > left * SOFT_SHARE:
                    break
        except _OutOfTime:
            self.unwind(base)
            # The unfinished iteration searched the previous best move first,
            # so whatever it has found so far is at least as well founded.
            if self.partial is not None:
                best, best_value = self.partial
                ranking.remove(best)
                ranking.insert(0, best)

        # 4. Do not walk into a forced loss if another move avoids it. (Moves
        #    that came out of step 2 have already been checked.)
        self.deadline = start + budget
        if defusing or best_value <= -WIN // 2:
            return best
        tried = set()
        queue = ranking[:3]
        widened = False
        while queue:
            move = queue.pop(0)
            if move in tried:
                continue
            tried.add(move)
            remaining = self.deadline - self.now()
            if remaining <= 0.01:
                break
            self.play(move, colour)
            self.vct_seen.clear()
            refutation = self.find_vct(other, remaining * 0.4, SAFETY_DEPTH)
            self.undo()
            if refutation < 0:
                return move
            if not widened:
                widened = True
                cells = {refutation}
                cells.update(self.cells_of(FOUR_AT[other]))
                cells.update(self.three_moves(other))
                cells.update(self.cells_of(FOUR_AT[colour]))
                cells.update(self.cells_of(CAP_AT[colour]))
                board = self.board
                empty = [cell for cell in cells if not board[cell]]
                queue = [item[1] for item in self.ordered(empty, colour)[0]][:8] + queue
        return best


# --------------------------------------------------------------------------


# The compiled engine. FAST_MODULE names the file (without .py) that holds it;
# it is looked for next to this file.
FAST_MODULE = "fastcore"
FAST_THREADS = 0          # search threads for the compiled engine; 0 = from the number of cores
FAST_OPTIONS = {}         # settings of the compiled engine that differ from its own defaults
FAST_TABLE_BITS = 22      # the table of positions of a real game holds 2**22 of them (64 MB):
                          # the search threads tell each other what they found through it
EVAL_MODEL = {'lo': 6, 'hi': 18, 'shape_from': 5, 'windows': [[-30, -11, 53, 0, 33, 20, -134, 0], [-26, -27, -154, 0, 27, 42, 133, 0]], 'terms': [[482, 1140, 849, -614, 124, 120, -89, 186, -68, -20, 41, -124], [278, 403, 345, -265, 843, -547, -15, 25, -437, 213, -208, 650]], 'my': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -17, 64, -49, 85, 10, 105, 0, 0, 0, -1, -10, -23, 169, 18, -8, 16, 0, 0, 42, 256, -89, -68, -62, -102, -23, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -86, 4, -5, 42, 11, 51, 0, 0, 0, 287, 132, 141, 195, 182, 206, 19, 0, 0, 680, 265, 2, 124, -68, -165, -476, -406, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'their': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, -63, 4, -93, -79, -36, 0, 0, 0, 7, 43, 1, -163, -38, -196, -34, 0, 0, -10, -131, -142, -105, 154, 286, 50, 9, 0, -17, -376, -52, -341, -52, 6, -39, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 1, -3, -41, -12, -62, 0, 0, 0, -88, -120, -136, -152, -130, -145, 63, 0, 0, 160, -166, -90, -178, -21, 40, 285, 542, 0, -1611, -2076, -1997, -2022, -1867, -1775, -1534, -92, -7]]}


def _load_fast():
    """The compiled engine's module, or None if it cannot be used (no numba,
    file missing, or the learned table is not of the kind it understands)."""
    if PATTERN_WIDTH != 7:
        return None
    # The module is always known by the name of its file: numba's cache of
    # the compiled code remembers that name. (To have two different compiled
    # engines in one program, give their files different names.)
    if FAST_MODULE in sys.modules:
        return sys.modules[FAST_MODULE]
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here,):
        path = os.path.join(folder, FAST_MODULE + ".py")
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location(FAST_MODULE, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[FAST_MODULE] = module
                spec.loader.exec_module(module)
                return module
            except Exception:
                sys.modules.pop(FAST_MODULE, None)
                return None
    return None


_fast = _load_fast()


def _thread_count():
    if FAST_THREADS > 0:
        return FAST_THREADS
    wanted = os.environ.get("OMOK_THREADS", "")
    if wanted.isdigit() and int(wanted) > 0:
        return int(wanted)
    # (Measured on sixteen cores: twelve threads reach a given depth five
    # times as fast as one, eight threads four times. One core is left alone.)
    return max(1, min(12, (os.cpu_count() or 2) - 1))


def make_engine(threads=1):
    """The strongest engine available here: the compiled one, else the plain one.
    (For the tools. A mistake in FAST_OPTIONS is reported, not hidden.)"""
    if _fast is not None:
        return _fast.FastEngine(PATTERN_SCORES, PATTERN_TERMS, threads=threads,
                                options=FAST_OPTIONS, model=EVAL_MODEL)
    return _Engine()


# How many games this program has started as White (it begins at a random
# point, so that programs run one game at a time vary too).
_white_games = [random.randrange(len(FIRST_REPLY_KINDS))]


class SmartPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        self._engine = _Engine()
        self._fast = None
        if _fast is not None:
            try:
                self._fast = _fast.FastEngine(PATTERN_SCORES, PATTERN_TERMS,
                                              tt_bits=20 if FIXED_NODES else FAST_TABLE_BITS,
                                              threads=1 if FIXED_NODES else _thread_count(),
                                              options=FAST_OPTIONS, model=EVAL_MODEL)
            except Exception:
                self._fast = None

    def take_turn(self, board, time):
        start = _time.perf_counter()
        colour = self.color + 1
        move = -1
        if self.color == 1:
            reply = self._first_reply(board)
            if reply is not None:
                return reply
        if self._fast is not None:
            try:
                fast = self._fast
                fast.load(board)
                if FIXED_NODES:
                    fast.use_node_clock()
                    move = fast.choose(colour, FIXED_NODES / float(fast.NODE_RATE))
                else:
                    stones = fast.total(1, 5) + fast.total(2, 5)
                    budget = self._budget(time, stones) - (_time.perf_counter() - start)
                    move = fast.choose(colour, max(budget, 0.0))
            except Exception:
                self._fast = None            # the plain engine takes over for good
                move = -1
        if not (0 <= move < CELLS and board[move // SIZE][move % SIZE] == -1):
            engine = self._engine
            move = -1
            try:
                engine.load(board)
                if FIXED_NODES:
                    engine.use_node_clock()
                    move = engine.choose(colour, FIXED_NODES / float(NODE_RATE))
                else:
                    stones = engine.count[1] + engine.count[2]
                    budget = self._budget(time, stones) - (_time.perf_counter() - start)
                    move = engine.choose(colour, max(budget, 0.0))
            except Exception:
                move = -1
        if 0 <= move < CELLS and board[move // SIZE][move % SIZE] == -1:
            return (move // SIZE, move % SIZE)
        return self._any_legal_move(board)

    @staticmethod
    def _first_reply(board):
        """White's answer to the first stone, or None if this is not that moment.
        Successive games take the kinds of answer in turn: one win as White
        usually decides a match, and what fails against an opponent once is
        likely to fail against it again."""
        stone = None
        for r in range(SIZE):
            row = board[r]
            for c in range(SIZE):
                if row[c] != -1:
                    if stone is not None:
                        return None
                    stone = (r, c)
        if stone is None:
            return None
        r, c = stone
        turn = _white_games[0]
        _white_games[0] += 1
        for step in range(len(FIRST_REPLY_KINDS)):
            kind = FIRST_REPLY_KINDS[(turn + step) % len(FIRST_REPLY_KINDS)]
            options = [(r + dr, c + dc) for dr, dc in kind
                       if 2 <= r + dr < SIZE - 2 and 2 <= c + dc < SIZE - 2]
            if options:
                return random.choice(options)
        # The stone is in a corner: step towards the middle.
        return (r + (1 if r < SIZE // 2 else -1), c + (1 if c < SIZE // 2 else -1))

    @staticmethod
    def _budget(time_ms, stones):
        """Seconds to spend on this move, given the time left for the whole game."""
        if time_ms is None or time_ms < 0:
            return NO_LIMIT_MOVE_S
        usable = time_ms / 1000.0 - RESERVE_S
        moves_left = max(MIN_MOVES_AHEAD, MOVES_AHEAD - stones // 4)
        budget = min(MOVE_CAP_S, usable / moves_left)
        if stones < 4:
            budget *= 0.5                    # the first moves need little thought
        return max(budget, 0.0)

    @staticmethod
    def _any_legal_move(board):
        """Last resort: the empty point nearest the centre."""
        best = None
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] == -1:
                    distance = abs(r - SIZE // 2) + abs(c - SIZE // 2)
                    if best is None or distance < best[0]:
                        best = (distance, r, c)
        return (best[1], best[2]) if best else (0, 0)
