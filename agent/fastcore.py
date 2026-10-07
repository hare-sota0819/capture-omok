"""The compiled core of the capture omok agent. Needs numba (pip install numba).

Everything here works on plain arrays so that numba can turn it into machine
code. The board is kept on a 29 x 29 grid (the 19 x 19 board with a margin of
five "wall" points on every side), which lets every loop walk along a line
without checking for the edge.

For every point and each of the four line directions the engine keeps a
20-bit code of the ten points around it (five each way). Tables indexed by
that code say what a stone there would do on that line: complete an exact
five, make a four, make an open three, capture a pair. The tables are filled
in the first time a code is met.

The static evaluation is a learned table over every stretch of seven points
(fitted to self-play games), kept up to date as stones come and go.

All the numbers of one position live in two arrays (`m`, 32-bit, and `h`,
64-bit) at the offsets named below: calls between compiled functions are far
cheaper with a handful of arrays than with one array per quantity.
"""

from collections import namedtuple

import random
import threading
import time as _time

import numpy as np
from numba import njit, typeof, types

N = 19
PAD = 5
W = N + 2 * PAD
AREA = W * W
WALL = 3
DIRS = (1, W, W + 1, W - 1)
DIR8 = (1, -1, W, -W, W + 1, -W - 1, W - 1, -W + 1)

WIN = 1000000
INFINITY = 10 * WIN
MAX_PLY = 96
ROWS = MAX_PLY + 8
STACK = 512
MOVES = 368

# ---- m: the position and the work space of one search thread (int32)
M_BOARD = 0
M_NEAR = AREA                    # stones within two steps along a line
M_CODE = 2 * AREA                # + direction * AREA + point
M_INFO = 6 * AREA                # what an empty point would do, per direction
M_CWIN = 10 * AREA               # + colour * AREA + point: directions completing a five
M_CFOUR = 13 * AREA              # ... making a four
M_COPEN = 16 * AREA              # ... making an open four
M_CCAP = 19 * AREA               # pairs captured by a stone there
M_CTHR = 22 * AREA               # directions making an open three
M_SHP = 25 * AREA                # + colour * AREA + point: directions making an open three or better
M_SI = 28 * AREA                 # + colour * AREA + point: where the point is in the list below
M_G = 31 * AREA                  # + colour * 8 + one of the G_ columns
M_SN = M_G + 32                  # + colour: how many points the colour's list holds
M_SL = M_SN + 4                  # + (colour - 1) * 361 + i: the points where a stone of the colour
M_UST = M_SL + 722               # would make an open three or better, in no particular order
                                 # (then the undo stack, UNDO numbers per move: see make())
UNDO = 48
M_KIL = M_UST + STACK * UNDO
M_MV = M_KIL + ROWS * 2          # a list of moves for every ply
M_MS = M_MV + ROWS * MOVES       # and their ordering values
M_SIZE = M_MS + ROWS * MOVES

G_WIN = 0        # (point, direction) pairs where a stone completes an exact five
G_FOUR = 1       # ... makes a four (leaves at least one point completing a five)
G_OPEN = 2       # ... makes an open four (at least two such points)
G_CAP = 3        # pairs that could be captured right now
G_THR = 4        # (point, direction) pairs where a stone makes an open three
G_STONES = 5
G_FIVE = 6       # 1 if the colour has an exact five on the board
G_PTH = 7        # (point, side) pairs where a stone would threaten to capture a pair: . X X then here

# ---- h: 64-bit numbers of one search thread
H_HASH = 0
H_SP = 1                         # moves on the undo stack
H_NODES = 2
H_ABORT = 3
H_LIMIT = 4                      # stop when H_NODES reaches this
H_PLY_LIMIT = 5
H_GENERATION = 6                 # which forced-win search the entries of vt belong to
H_ROOT_MOVE = 7
H_ROOT_VALUE = 8
H_ROOT_DONE = 9
H_LOG = 10                       # where in m the next entry of the record of changes goes
H_EV = 12                        # + (side to move - 1): the sum over the stretches of seven points
                                 # (both stages in one number, see TC_PTAB)
H_UNDO = 16                      # + 4 * (place on the undo stack): H_HASH, the two H_EV and H_LOG
                                 # as they were before the move
H_LOGS = H_UNDO + 4 * STACK      # what the moves on the undo stack changed in m, one entry per
                                 # number: where it is (times 2**20) plus what it was before.
                                 # unmake() puts those numbers back, which is far cheaper than
                                 # working everything out again backwards.
# One stone on or off changes at most 805 numbers of m; a move with sixteen
# captured stones seventeen times that. Functions that make moves stop going
# deeper once less than LOG_ROOM is left, and none makes more than two moves
# before the next such check.
LOG_ROOM = 3 * 17 * 805
LOG_SIZE = 128 * 1024 + LOG_ROOM
LOG_LIMIT = H_LOGS + LOG_SIZE - LOG_ROOM
H_SIZE = H_LOGS + LOG_SIZE
LOG_MASK = (1 << 18) - 1         # (H_SIZE fits below this: see the note on masks at set_cell)
assert H_SIZE <= LOG_MASK + 1 and M_SIZE < (1 << 19)

# ---- tc: constant tables (int32)
TC_PTAB = 0                      # + (centre * 4096 + six neighbours) * 2 + view: what a stretch of
                                 # seven points is worth, the opening's number plus 2**20 times
                                 # the later one (one running sum then keeps both)
TC_WOK = 2 * 3 * 4096            # + direction * AREA + point: a seven-point stretch is centred here
TC_CELLS = TC_WOK + 4 * AREA     # the 361 points of the board
TC_BIAS = TC_CELLS + 361         # a small pull towards the middle
TC_SIZE = TC_BIAS + AREA

# ---- tz: 64-bit tables shared by all threads
TZ_ZOB = 0                       # + state * AREA + point
TZ_SIDE = 3 * AREA
TZ_PARAMS = 3 * AREA + 8
TZ_STOP = TZ_PARAMS + 40
TZ_TERMS = TZ_STOP + 8           # + stage * 16 + one of the S_ terms (stage 0 opening, 1 later)
TZ_SHAPE = TZ_TERMS + 32         # + stage * 162 + role * 81 + strongest * 9 + second: what an
TZ_WINDOWS = TZ_SHAPE + 2 * 162  # empty point of that kind is worth (role 0: to its side
TZ_EVAL_END = TZ_WINDOWS + 16    # when that side is to move; role 1: when it is not).
                                 # TZ_WINDOWS + stage * 8 + k: a window of five with k + 1 stones
                                 # of the side to move (k = 0..3) or of the other side (4..7);
                                 # only read when the evaluation is checked from scratch.
TZ_SHARING = TZ_EVAL_END         # not 0 while several threads search the same position
TZ_AGE = TZ_EVAL_END + 1         # goes up with every search; kept with each entry of the table of positions
TZ_BUSY = TZ_EVAL_END + 8        # + (mark & BUSY_MASK): the move some thread is searching right now
BUSY_MASK = 8191
TZ_SIZE = TZ_BUSY + BUSY_MASK + 1
SHARE_DEPTH = 6                  # threads only watch out for each other this far above the leaves

S_TEMPO = 0
S_STONES = 1
S_MY_CAPTURES = 2
S_THEIR_CAPTURES = 3
S_MY_OPEN = 4
S_THEIR_DOUBLE = 5
S_MY_PAIRS = 6
S_THEIR_PAIRS = 7
S_THEY_THREATEN = 8
S_I_HAVE_FOURS = 9
S_THEY_HAVE_FOURS = 10
S_IN_CHECK = 11
TERMS = ("tempo", "stones", "my_captures", "their_captures", "my_open", "their_double",
         "my_pairs", "their_pairs", "they_threaten", "i_have_fours", "they_have_fours",
         "in_check")
SHAPE_CLASSES = ("none", "one", "closed2", "open2", "closed3", "open3", "closed4", "open4", "five")

P_TEMPO = 0
P_STONES = 1
P_MY_CAPTURES = 2
P_THEIR_CAPTURES = 3
P_MY_OPEN = 4
P_THEIR_DOUBLE = 5
P_FRONTIER = 6
P_QUIESCE = 7
P_WIDTH_DEEP = 8
P_WIDTH_MID = 9
P_WIDTH_SHALLOW = 10
P_QUIET_COST = 11
P_FOUR_COST = 12
P_THREAT_COST = 13
P_REDUCE_AFTER = 14
P_REDUCE_MIN = 15
P_CAPTURE_BONUS = 16
P_RESCUE_BONUS = 17
P_OPEN_FOUR_BONUS = 18
P_REDUCE2_AFTER = 19
P_REDUCE2_MIN = 20
P_FRONTIER2 = 21
P_FUTILITY = 22
P_LEAF_VCF = 23
P_SHAPES = 24                    # 1: the evaluation looks at shapes
P_STAGE_LO = 25                  # up to this many stones the opening numbers count alone ...
P_STAGE_HI = 26                  # ... from this many on, the later ones
P_SHAPE_FROM = 27                # the weakest class that makes an empty point count as a shape (5..8)

# Everything that can be set per engine without compiling again. The first
# group is read by the compiled search (through tz), the second by the Python
# code that runs it. FastEngine(options={...}) changes any of them.
SEARCH_SLOTS = {
    "frontier": P_FRONTIER, "quiesce": P_QUIESCE,
    "width_deep": P_WIDTH_DEEP, "width_mid": P_WIDTH_MID, "width_shallow": P_WIDTH_SHALLOW,
    "quiet_cost": P_QUIET_COST, "four_cost": P_FOUR_COST, "threat_cost": P_THREAT_COST,
    "reduce_after": P_REDUCE_AFTER, "reduce_min": P_REDUCE_MIN,
    "reduce2_after": P_REDUCE2_AFTER, "reduce2_min": P_REDUCE2_MIN,
    "capture_bonus": P_CAPTURE_BONUS, "rescue_bonus": P_RESCUE_BONUS,
    "open_four_bonus": P_OPEN_FOUR_BONUS,
    "frontier2": P_FRONTIER2, "futility": P_FUTILITY, "leaf_vcf": P_LEAF_VCF,
}
OPTIONS = {
    "frontier": 150,             # at the last ply: static value this far above beta is a cut-off
    "frontier2": 0,              # the same one ply earlier (0: not used)
    "futility": 0,               # at the last ply: static value this far below alpha, and only
                                 # moves that threaten or capture are tried (0: not used)
    "quiesce": 3,                # captures followed at the leaves
    "leaf_vcf": 0,               # fours followed at the leaves, this many attacking moves deep
    "width_deep": 12, "width_mid": 10, "width_shallow": 8,
    "quiet_cost": 2, "four_cost": 2, "threat_cost": 2,
    "reduce_after": 4, "reduce_min": 6,          # late quiet moves: one ply less at first ...
    "reduce2_after": 99, "reduce2_min": 8,       # ... and two plies less from this index on
    "capture_bonus": 70, "rescue_bonus": 110, "open_four_bonus": 4000,
    # -- choosing a move
    "root_width": 20,
    "vcf_depth": 14,             # attacking moves in a win by continuous fours
    "vct_depth": 8,              # attacking moves in a win by fours and threes
    "safety_depth": 6,           # the same, when checking what the opponent could do to us
    "defusing_enough": 6,        # stop looking for defences once this many are known
    "own_vcf_share": 0.05,       # of a move's budget: looking for our own win by fours ...
    "own_vct_share": 0.15,       # ... by fours and threes ...
    "probe_share": 0.15,         # ... and for the opponent's
    "soft_share": 0.6,           # stop deepening after this share of what is left ...
    "search_share": 0.85,        # ... unless the result is unsettled; then up to this share
    "unsettled_drop": 100,       # a fall in value this large between iterations is unsettled
}

UNKNOWN = 0xFFFF
CLASSES = 2 << 20                # where the classes of a code are kept in otab (after the ordering values)
ORD_UNKNOWN = -2147483647
VCF_TAG = 0x5A5A5A5A5A5A
VCT_TAG = 0x3C3C3C3C3C3C

# Move ordering uses the hand-made window values of the first engine.
ORDER_WINDOW = (0, 1, 8, 60, 220)
ORDER_CAPTURE = 45
ORDER_FIVE = 100000
DEFENCE_SHARE = 0.85

# (The tuple's type carries the module's name: numba tells such types apart by
# name, and two versions of this file may be loaded side by side for a test.
# The class must also be found in this module under that name, or numba's
# cache of the compiled code cannot be read back and everything is compiled
# again on every start.)
Engine = namedtuple("Engine_" + __name__.replace(".", "_"), "m h itab otab tc tz tt vt")
globals()[Engine.__name__] = Engine

# Every compiled function is given its types here, and so is compiled (or read
# back from numba's cache) when this module is imported. Left to work the
# types out by itself, numba cannot reliably reload functions that call
# themselves from its cache.
ET = typeof(Engine(m=np.zeros(1, np.int32), h=np.zeros(1, np.int64), itab=np.zeros(1, np.uint16),
                   otab=np.zeros(1, np.int32), tc=np.zeros(1, np.int32), tz=np.zeros(1, np.int64),
                   tt=np.zeros(1, np.int64), vt=np.zeros(1, np.int64)))
I64 = types.int64


def _compiled(signature):
    """How this file's functions are compiled.

    numba counts who is using each array, and counts again whenever a
    function picks an array out of the engine's tuple. The counting is done
    with an instruction that costs as much as dozens of ordinary ones, and
    that search threads running side by side all want at once: measured, a
    third of the whole search's time went into it. No function compiled
    this way keeps an array beyond its own run or makes a new one, so there
    is nothing to count, and `_nrt=False` leaves the counting out. (numba
    refuses to compile a function that does need it, so this cannot go wrong
    quietly.) If some version of numba will not have the setting, or does
    refuse, the function is compiled the ordinary way.
    """
    def decorate(function):
        try:
            return njit(signature, cache=True, nogil=True, _nrt=False)(function)
        except Exception:
            return njit(signature, cache=True, nogil=True)(function)
    return decorate


# --------------------------------------------------------------------------
# What a stone would do on one line


@_compiled(types.void(I64, types.int64[::1]))
def _unpack(code, cells):
    """The eleven points a code stands for (index 5, the point itself, is empty)."""
    for j in range(5):
        cells[j] = (code >> (2 * j)) & 3
    cells[5] = 0
    for j in range(5, 10):
        cells[j + 1] = (code >> (2 * j)) & 3


@_compiled(I64(I64))
def _bits(mask):
    count = 0
    while mask:
        mask &= mask - 1
        count += 1
    return count


@_compiled(I64(types.int64[::1], I64))
def _completions(cells, colour):
    """Mask of the points that would complete an exact five through the centre
    (index 5), or -1 if there is such a five already."""
    mask = 0
    for a in range(1, 6):
        if cells[a - 1] == colour or cells[a + 5] == colour:
            continue                             # it would be six or more
        count = 0
        empty = -1
        bad = False
        for i in range(a, a + 5):
            v = cells[i]
            if v == colour:
                count += 1
            elif v == 0:
                empty = i
            else:
                bad = True
                break
        if bad:
            continue
        if count == 5:
            return -1
        if count == 4:
            mask |= 1 << empty
    return mask


@_compiled(types.boolean(types.int64[::1], I64))
def _open_four_next(cells, colour):
    """True if one more stone would leave two points completing a five."""
    for y in range(1, 10):
        if cells[y] == 0:
            cells[y] = colour
            mask = _completions(cells, colour)
            cells[y] = 0
            if mask > 0 and _bits(mask) >= 2:
                return True
    return False


@_compiled(I64(types.int64[::1], I64))
def _low_class(cells, colour):
    """For a line on which the centre stone makes no four and no open three:
    0 nothing, 1 a lone stone with room, 2 a closed two, 3 an open two (one
    more stone makes an open three), 4 a closed three."""
    best = 0
    for a in range(1, 6):
        if cells[a - 1] == colour or cells[a + 5] == colour:
            continue
        count = 0
        bad = False
        for i in range(a, a + 5):
            v = cells[i]
            if v == colour:
                count += 1
            elif v != 0:
                bad = True
                break
        if not bad and count > best:
            best = count
    if best <= 1:
        return best
    if best >= 3:
        return 4
    for y in range(1, 10):
        if cells[y] == 0:
            cells[y] = colour
            is_open = _completions(cells, colour) == 0 and _open_four_next(cells, colour)
            cells[y] = 0
            if is_open:
                return 3
    return 2


@njit(I64(I64), cache=True, nogil=True)
def analyse_info(code):
    """What a stone on the middle of these eleven points would do on this line.
    One byte per colour (Black low, White next): bit 0 exact five, bits 1-2
    completing points left (0, 1, 2+), bit 3 makes an open three, bits 4-5
    pairs captured along this line, bits 6-7 pairs it would threaten to
    capture. Above them, four bits per colour: the class of the line (see
    SHAPE_CLASSES)."""
    cells = np.empty(11, dtype=np.int64)
    result = 0
    for colour in (1, 2):
        other = 3 - colour
        _unpack(code, cells)
        cells[5] = colour
        five = False
        mask = 0
        for a in range(1, 6):
            if cells[a - 1] == colour or cells[a + 5] == colour:
                continue                         # it would be six or more
            count = 0
            empty = -1
            bad = False
            for i in range(a, a + 5):
                v = cells[i]
                if v == colour:
                    count += 1
                elif v == 0:
                    empty = i
                else:
                    bad = True
                    break
            if bad:
                continue
            if count == 5:
                five = True
            elif count == 4:
                mask |= 1 << empty
        left = _bits(mask)
        if left > 2:
            left = 2
        three = 0
        if not five and left == 0:
            # An open three: some further stone, in a window with this one,
            # would leave two completing points.
            for y in range(1, 10):
                if cells[y] != 0:
                    continue
                found = 0
                for a in range(max(1, y - 4), min(5, y) + 1):
                    if cells[a - 1] == colour or cells[a + 5] == colour:
                        continue
                    count = 0
                    empty = -1
                    bad = False
                    for i in range(a, a + 5):
                        v = cells[i]
                        if v == colour:
                            count += 1
                        elif v == 0:
                            if i != y:
                                empty = i
                        else:
                            bad = True
                            break
                    if not bad and count == 3 and empty >= 0:
                        found |= 1 << empty
                if _bits(found) >= 2:
                    three = 1
                    break
        captures = 0
        if cells[6] == other and cells[7] == other and cells[8] == colour:
            captures += 1
        if cells[4] == other and cells[3] == other and cells[2] == colour:
            captures += 1
        threats = 0
        if cells[6] == other and cells[7] == other and cells[8] == 0:
            threats += 1
        if cells[4] == other and cells[3] == other and cells[2] == 0:
            threats += 1
        byte = (1 if five else 0) | (left << 1) | (three << 3) | (captures << 4) | (threats << 6)
        result |= byte << (8 * (colour - 1))
        if five:
            kind = 8
        elif left >= 2:
            kind = 7
        elif left == 1:
            kind = 6
        elif three:
            kind = 5
        else:
            kind = _low_class(cells, colour)
        result |= kind << (16 + 4 * (colour - 1))
    return result


@_compiled(I64(types.int64[::1]))
def _segment_score(cells):
    """Black's hand-made line score minus White's on these eleven points."""
    total = 0
    for colour in (1, 2):
        other = 3 - colour
        score = 0
        for a in range(0, 7):
            if a > 0 and cells[a - 1] == colour:
                continue
            if a + 5 < 11 and cells[a + 5] == colour:
                continue
            count = 0
            bad = False
            for i in range(a, a + 5):
                v = cells[i]
                if v == colour:
                    count += 1
                elif v != 0:
                    bad = True
                    break
            if bad or count == 0:
                continue
            if count == 5:
                score += ORDER_FIVE
            else:
                score += ORDER_WINDOW[count]
        for i in range(0, 8):
            if cells[i + 1] == other and cells[i + 2] == other:
                if cells[i] == colour and cells[i + 3] == 0:
                    score += ORDER_CAPTURE
                elif cells[i] == 0 and cells[i + 3] == colour:
                    score += ORDER_CAPTURE
        if colour == 1:
            total += score
        else:
            total -= score
    return total


@njit(types.void(types.int32[::1], I64), cache=True, nogil=True)
def analyse_order(otab, code):
    """Fill in the ordering values of a stone of each colour on this line: the
    mover's own gain plus a share of what the opponent would have gained."""
    cells = np.empty(11, dtype=np.int64)
    _unpack(code, cells)
    before = _segment_score(cells)
    cells[5] = 1
    black = _segment_score(cells) - before
    cells[5] = 2
    white = before - _segment_score(cells)
    # White's entry first: readers test Black's to see whether both are there.
    otab[2 * code + 1] = int(white + DEFENCE_SHARE * black)
    otab[2 * code] = int(black + DEFENCE_SHARE * white)


@njit(cache=True, nogil=True, inline="always")
def order_value(m, otab, tc, tz, p, colour):
    """How promising a stone of `colour` on the empty point p looks.
    (About the masks: see set_cell.)"""
    p &= 1023
    colour &= 3
    value = tc[TC_BIAS + p]
    index = colour - 1
    for d in range(4):
        code = m[M_CODE + d * AREA + p] & 0xFFFFF
        if otab[2 * code] == ORD_UNKNOWN:
            analyse_order(otab, code)
        value += otab[2 * code + index]
    captures = m[M_CCAP + colour * AREA + p]
    if captures:
        value += tz[TZ_PARAMS + P_CAPTURE_BONUS] * captures
    rescues = m[M_CCAP + (3 - colour) * AREA + p]
    if rescues:
        value += tz[TZ_PARAMS + P_RESCUE_BONUS] * rescues
    if m[M_COPEN + colour * AREA + p]:
        value += tz[TZ_PARAMS + P_OPEN_FOUR_BONUS]
    return value


# --------------------------------------------------------------------------
# The position


@njit(cache=True, nogil=True, inline="always")
def apply_info(m, h, p, old, new, lp):
    """Point p's information on one line changed: keep the totals right.

    Every number of a single point that changes is first noted at h[lp]
    (where it is, what it was) for unmake(). The totals of the whole board
    and the lengths of the lists are not noted: make() keeps a copy of all of
    them. Returns where the next note goes."""
    for colour in (1, 2):
        shift = 8 * (colour - 1)
        a = (old >> shift) & 0xFF
        b = (new >> shift) & 0xFF
        if a != b:
            cell = colour * AREA + p
            g = M_G + colour * 8
            change = (b & 1) - (a & 1)
            if change:
                x = M_CWIN + cell
                v = m[x]
                h[lp & LOG_MASK] = (x << 20) | v
                lp += 1
                m[x] = v + change
                m[g + G_WIN] += change
            ka = (a >> 1) & 3
            kb = (b >> 1) & 3
            if ka != kb:
                change = (1 if kb >= 1 else 0) - (1 if ka >= 1 else 0)
                if change:
                    x = M_CFOUR + cell
                    v = m[x]
                    h[lp & LOG_MASK] = (x << 20) | v
                    lp += 1
                    m[x] = v + change
                    m[g + G_FOUR] += change
                change = (1 if kb >= 2 else 0) - (1 if ka >= 2 else 0)
                if change:
                    x = M_COPEN + cell
                    v = m[x]
                    h[lp & LOG_MASK] = (x << 20) | v
                    lp += 1
                    m[x] = v + change
                    m[g + G_OPEN] += change
            change = ((b >> 3) & 1) - ((a >> 3) & 1)
            if change:
                x = M_CTHR + cell
                v = m[x]
                h[lp & LOG_MASK] = (x << 20) | v
                lp += 1
                m[x] = v + change
                m[g + G_THR] += change
            change = ((b >> 4) & 3) - ((a >> 4) & 3)
            if change:
                x = M_CCAP + cell
                v = m[x]
                h[lp & LOG_MASK] = (x << 20) | v
                lp += 1
                m[x] = v + change
                m[g + G_CAP] += change
            change = ((b >> 6) & 3) - ((a >> 6) & 3)
            if change:
                m[g + G_PTH] += change
            # Does this line make an open three or better here? The points
            # where some line does are kept in a list: the evaluation and the
            # forced-win searches want exactly those points.
            change = (1 if (b & 15) != 0 else 0) - (1 if (a & 15) != 0 else 0)
            if change:
                x = M_SHP + cell
                lines = m[x]
                h[lp & LOG_MASK] = (x << 20) | lines
                lp += 1
                lines += change
                m[x] = lines
                if change > 0 and lines == 1:
                    n = m[M_SN + colour] & 511
                    x = M_SL + (colour - 1) * 361 + n
                    h[lp & LOG_MASK] = (x << 20) | m[x]
                    m[x] = p
                    x = M_SI + cell
                    h[(lp + 1) & LOG_MASK] = (x << 20) | m[x]
                    lp += 2
                    m[x] = n
                    m[M_SN + colour] = n + 1
                elif change < 0 and lines == 0:
                    n = (m[M_SN + colour] - 1) & 511
                    i = m[M_SI + cell] & 511
                    last = m[M_SL + (colour - 1) * 361 + n] & 1023
                    x = M_SL + (colour - 1) * 361 + i
                    h[lp & LOG_MASK] = (x << 20) | m[x]
                    m[x] = last
                    x = M_SI + colour * AREA + last
                    h[(lp + 1) & LOG_MASK] = (x << 20) | m[x]
                    lp += 2
                    m[x] = i
                    m[M_SN + colour] = n
    return lp


@njit(cache=True, nogil=True, inline="always")
def set_cell(m, h, itab, otab, tc, tz, p, new, lp):
    """Put a stone on p (new = 1 or 2) or take one off (new = 0), noting from
    h[lp] on what every changed number of m was (see apply_info). Returns
    where the next note goes.

    (The `& 1023`, `& 3` and the like below change nothing: the numbers are
    always that small. They are there for the compiler. numba lets a negative
    index count from the end of an array, as Python does, which costs a few
    instructions at every single access unless the compiler can see that the
    index cannot be negative; after such a mask it can.)"""
    p &= 1023
    new &= 3
    old = m[M_BOARD + p] & 3
    if old == 0:
        for d in range(4):
            x = M_INFO + d * AREA + p
            info = m[x] & 0xFFFF
            if info != 0:
                lp = apply_info(m, h, p, info, 0, lp)
                h[lp & LOG_MASK] = (x << 20) | info
                lp += 1
                m[x] = 0
    else:
        m[M_G + old * 8 + G_STONES] -= 1
    if new != 0:
        m[M_G + new * 8 + G_STONES] += 1
    e0 = 0
    e1 = 0
    for d in range(4):
        if tc[TC_WOK + d * AREA + p]:
            mid = (m[M_CODE + d * AREA + p] >> 4) & 0xFFF
            a = (old * 4096 + mid) * 2
            b = (new * 4096 + mid) * 2
            e0 += tc[b] - tc[a]
            e1 += tc[b + 1] - tc[a + 1]
    h[lp & LOG_MASK] = ((M_BOARD + p) << 20) | old
    lp += 1
    m[M_BOARD + p] = new
    h[H_HASH] ^= tz[TZ_ZOB + old * AREA + p] ^ tz[TZ_ZOB + new * AREA + p]
    step = 1 if old == 0 else -1
    for i in range(8):
        q = (p + DIR8[i]) & 1023
        if m[M_BOARD + q] != WALL:
            x = M_NEAR + q
            v = m[x]
            h[lp & LOG_MASK] = (x << 20) | v
            lp += 1
            m[x] = v + step
            q = (q + DIR8[i]) & 1023
            if m[M_BOARD + q] != WALL:
                x = M_NEAR + q
                v = m[x]
                h[lp & LOG_MASK] = (x << 20) | v
                lp += 1
                m[x] = v + step
    delta = new - old
    for d in range(4):
        D = DIRS[d]
        code_base = M_CODE + d * AREA
        info_base = M_INFO + d * AREA
        wok_base = TC_WOK + d * AREA
        for half in range(2):
            sign = 1 - 2 * half
            for k in range(1, 6):
                q = (p + sign * k * D) & 1023
                bq = m[M_BOARD + q] & 3
                if bq == WALL:
                    break
                offset = -sign * k               # where p lies, seen from q
                if offset < 0:
                    shift = 2 * (offset + 5)
                else:
                    shift = 2 * (offset + 4)
                c_old = m[code_base + q] & 0xFFFFF
                c_new = (c_old + delta * (1 << shift)) & 0xFFFFF
                h[lp & LOG_MASK] = ((code_base + q) << 20) | c_old
                lp += 1
                m[code_base + q] = c_new
                if k <= 3 and tc[wok_base + q]:
                    a = (bq * 4096 + ((c_old >> 4) & 0xFFF)) * 2
                    b = (bq * 4096 + ((c_new >> 4) & 0xFFF)) * 2
                    e0 += tc[b] - tc[a]
                    e1 += tc[b + 1] - tc[a + 1]
                if bq == 0:
                    info_new = int(itab[c_new])
                    if info_new == UNKNOWN:
                        # Met for the first time. The sixteen bits the totals
                        # depend on go into itab; the classes (which only the
                        # evaluation reads, at a few points) into otab.
                        info_new = analyse_info(c_new)
                        otab[CLASSES + c_new] = info_new >> 16
                        info_new &= 0xFFFF
                        itab[c_new] = info_new
                    info_old = m[info_base + q] & 0xFFFF
                    if info_new != info_old:
                        lp = apply_info(m, h, q, info_old, info_new, lp)
                        h[lp & LOG_MASK] = ((info_base + q) << 20) | info_old
                        lp += 1
                        m[info_base + q] = info_new
    h[H_EV] += e0
    h[H_EV + 1] += e1
    if new == 0:
        for d in range(4):
            c_own = m[M_CODE + d * AREA + p] & 0xFFFFF
            info = int(itab[c_own])
            if info == UNKNOWN:
                info = analyse_info(c_own)
                otab[CLASSES + c_own] = info >> 16
                info &= 0xFFFF
                itab[c_own] = info
            if info != 0:
                lp = apply_info(m, h, p, 0, info, lp)
            x = M_INFO + d * AREA + p
            h[lp & LOG_MASK] = (x << 20) | (m[x] & 0xFFFF)
            lp += 1
            m[x] = info
    return lp


@_compiled(types.void(ET, I64, I64))
def place(E, p, new):
    """set_cell for use from outside the compiled code (setting a position
    up): what it changes cannot be taken back."""
    h = E.h
    set_cell(E.m, h, E.itab, E.otab, E.tc, E.tz, p, new, h[H_LOG])


@_compiled(I64(ET, I64, I64))
def make(E, p, colour):
    """Play a stone, taking off the pairs it captures. Returns how many stones were taken.

    The undo stack gets, for unmake(): the point, the colour, the number of
    stones taken (at 0..2) and where they stood (5..20), and the totals of
    both colours and the lengths of their lists as they were (24..41); in h,
    where this move's record of changes starts, the hash and the two sums of
    the evaluation."""
    m = E.m
    h = E.h
    itab = E.itab
    otab = E.otab
    tc = E.tc
    tz = E.tz
    sp = h[H_SP] & (STACK - 1)
    u = M_UST + sp * UNDO
    other = 3 - colour
    lp = h[H_LOG]
    m[u] = p
    m[u + 1] = colour
    for i in range(16):
        m[u + 24 + i] = m[M_G + 8 + i]
    m[u + 40] = m[M_SN + 1]
    m[u + 41] = m[M_SN + 2]
    h[H_UNDO + 4 * sp] = h[H_HASH]
    h[H_UNDO + 4 * sp + 1] = h[H_EV]
    h[H_UNDO + 4 * sp + 2] = h[H_EV + 1]
    h[H_UNDO + 4 * sp + 3] = lp
    count = 0
    if m[M_CCAP + colour * AREA + p] > 0:
        for i in range(8):
            D = DIR8[i]
            if m[M_BOARD + p + D] == other and m[M_BOARD + p + 2 * D] == other \
                    and m[M_BOARD + p + 3 * D] == colour:
                m[u + 5 + count] = p + D
                m[u + 6 + count] = p + 2 * D
                count += 2
    m[u + 2] = count
    if m[M_CWIN + colour * AREA + p] > 0:
        m[M_G + colour * 8 + G_FIVE] = 1
    # The stone itself, then the stones it takes: one loop, so that the long
    # body of set_cell is compiled into this function only once.
    for i in range(-1, count):
        if i < 0:
            q = p
            state = colour
        else:
            q = m[u + 5 + i]
            state = 0
        lp = set_cell(m, h, itab, otab, tc, tz, q, state, lp)
    h[H_LOG] = lp
    if count:
        # Taking stones off can leave the opponent with exactly five.
        for i in range(count):
            r = m[u + 5 + i]
            for j in range(8):
                D = DIR8[j]
                q = r + D
                length = 0
                while m[M_BOARD + q] == other:
                    length += 1
                    q += D
                if length == 5:
                    m[M_G + other * 8 + G_FIVE] = 1
    h[H_SP] = sp + 1
    return count


@_compiled(types.void(ET))
def unmake(E):
    """Take the last move back: every number it changed gets its old value again."""
    m = E.m
    h = E.h
    sp = (h[H_SP] - 1) & (STACK - 1)
    h[H_SP] = sp
    u = M_UST + sp * UNDO
    start = h[H_UNDO + 4 * sp + 3]
    lp = h[H_LOG]
    while lp > start:
        lp -= 1
        entry = h[lp & LOG_MASK]
        m[(entry >> 20) & 0x7FFFF] = entry & 0xFFFFF
    h[H_LOG] = start
    for i in range(16):
        m[M_G + 8 + i] = m[u + 24 + i]
    m[M_SN + 1] = m[u + 40]
    m[M_SN + 2] = m[u + 41]
    h[H_HASH] = h[H_UNDO + 4 * sp]
    h[H_EV] = h[H_UNDO + 4 * sp + 1]
    h[H_EV + 1] = h[H_UNDO + 4 * sp + 2]


@njit(cache=True, nogil=True, inline="always")
def _stage_value(m, h, tz, colour, stage):
    """The static value with one of the two sets of numbers (0 opening, 1 later)."""
    other = 3 - colour
    gc = M_G + colour * 8
    go = M_G + other * 8
    t = TZ_TERMS + stage * 16
    packed = h[H_EV + colour - 1]                # the opening's sum + 2**20 * the later one
    value = packed & 0xFFFFF
    if value >= 0x80000:
        value -= 0x100000
    if stage != 0:
        value = (packed - value) >> 20
    value += tz[t + S_TEMPO] + tz[t + S_STONES] * (m[gc + G_STONES] - m[go + G_STONES])
    value += tz[t + S_MY_CAPTURES] * m[gc + G_CAP] + tz[t + S_THEIR_CAPTURES] * m[go + G_CAP]
    value += tz[t + S_MY_PAIRS] * m[go + G_PTH] + tz[t + S_THEIR_PAIRS] * m[gc + G_PTH]
    if m[gc + G_OPEN] > 0:
        value += tz[t + S_MY_OPEN]
    if m[go + G_OPEN] > 0:
        value += tz[t + S_THEY_THREATEN]
        if m[go + G_OPEN] >= 3 and m[gc + G_FOUR] == 0:
            value += tz[t + S_THEIR_DOUBLE]
    if m[gc + G_FOUR] > 0:
        value += tz[t + S_I_HAVE_FOURS]
    if m[go + G_FOUR] > 0:
        value += tz[t + S_THEY_HAVE_FOURS]
    if m[go + G_WIN] > 0:
        value += tz[t + S_IN_CHECK]
    return value


@njit(cache=True, nogil=True, inline="always")
def _shape_values(m, otab, tz, colour):
    """What the empty points where a stone would make an open three or better
    are worth, by their two strongest directions: for the side to move plus
    for the other side, with the opening's numbers and with the later ones
    (packed into one number, the later ones in the upper half)."""
    first = tz[TZ_PARAMS + P_SHAPE_FROM]
    v0 = 0
    v1 = 0
    for role in range(2):
        side = colour if role == 0 else 3 - colour
        shift = 4 * (side - 1)
        points = M_SL + (side - 1) * 361
        worth = TZ_SHAPE + role * 81
        for i in range(m[M_SN + side]):
            p = m[points + i] & 1023             # (about the masks: see set_cell)
            top = (otab[CLASSES + (m[M_CODE + p] & 0xFFFFF)] >> shift) & 15
            second = 0
            for d in range(1, 4):
                c = (otab[CLASSES + (m[M_CODE + d * AREA + p] & 0xFFFFF)] >> shift) & 15
                if c > top:
                    second = top
                    top = c
                elif c > second:
                    second = c
            if top >= first:
                v0 += tz[worth + top * 9 + second]
                v1 += tz[worth + 162 + top * 9 + second]
    return v0, v1


@njit(cache=True, nogil=True, inline="always")
def evaluate(m, h, otab, tz, colour):
    """Static value for the side to move: a learned number for every stretch
    of seven points (kept up to date as stones come and go), one for every
    empty point where a stone would make an open three or better (worked out
    here from the list of those points), and a few whole-board terms. There
    are two sets of numbers, for the opening and for later, blended by the
    number of stones on the board."""
    colour = 1 if colour == 1 else 2             # (for the compiler: see set_cell)
    stones = m[M_G + 8 + G_STONES] + m[M_G + 16 + G_STONES]
    lo = tz[TZ_PARAMS + P_STAGE_LO]
    hi = tz[TZ_PARAMS + P_STAGE_HI]
    s0 = 0
    s1 = 0
    if tz[TZ_PARAMS + P_SHAPES] != 0:
        s0, s1 = _shape_values(m, otab, tz, colour)
    if stones >= hi:
        return _stage_value(m, h, tz, colour, 1) + s1
    if stones <= lo:
        return _stage_value(m, h, tz, colour, 0) + s0
    return ((_stage_value(m, h, tz, colour, 0) + s0) * (hi - stones)
            + (_stage_value(m, h, tz, colour, 1) + s1) * (stones - lo)) // (hi - lo)


@_compiled(I64(ET, I64))
def evaluate_scan(E, colour):
    """The static value worked out from nothing but the board and the per-line
    information (slow; for checking the running sums). Only for evaluations
    given as window weights (install_model)."""
    m = E.m
    tc = E.tc
    tz = E.tz
    other = 3 - colour
    gc = M_G + colour * 8
    go = M_G + other * 8
    v0 = 0
    v1 = 0
    my_pairs = 0
    their_pairs = 0
    for i in range(361):
        p = tc[TC_CELLS + i]
        stone = m[M_BOARD + p]
        if stone == 0:
            for role in range(2):
                side = colour if role == 0 else other
                shift = 4 * (side - 1)
                top = 0
                second = 0
                for d in range(4):
                    c = (E.otab[CLASSES + m[M_CODE + d * AREA + p]] >> shift) & 15
                    if c > top:
                        second = top
                        top = c
                    elif c > second:
                        second = c
                if top >= tz[TZ_PARAMS + P_SHAPE_FROM]:
                    v0 += tz[TZ_SHAPE + role * 81 + top * 9 + second]
                    v1 += tz[TZ_SHAPE + 162 + role * 81 + top * 9 + second]
        else:
            for d in range(4):
                D = DIRS[d]
                if m[M_BOARD + p + D] == stone and m[M_BOARD + p - D] == 0 \
                        and m[M_BOARD + p + 2 * D] == 0:
                    if stone == colour:          # an open pair, . X X . : two points threaten it
                        my_pairs += 2
                    else:
                        their_pairs += 2
        for d in range(4):                       # the window of five starting at p
            D = DIRS[d]
            if m[M_BOARD + p + 4 * D] == WALL:
                continue
            mine = 0
            theirs = 0
            for k in range(5):
                v = m[M_BOARD + p + k * D]
                if v == colour:
                    mine += 1
                elif v == other:
                    theirs += 1
            before = m[M_BOARD + p - D]
            after = m[M_BOARD + p + 5 * D]
            if theirs == 0 and mine > 0 and mine < 5 and before != colour and after != colour:
                v0 += tz[TZ_WINDOWS + mine - 1]
                v1 += tz[TZ_WINDOWS + 8 + mine - 1]
            if mine == 0 and theirs > 0 and theirs < 5 and before != other and after != other:
                v0 += tz[TZ_WINDOWS + 4 + theirs - 1]
                v1 += tz[TZ_WINDOWS + 12 + theirs - 1]
    for stage in range(2):
        t = TZ_TERMS + stage * 16
        g = tz[t + S_TEMPO] + tz[t + S_STONES] * (m[gc + G_STONES] - m[go + G_STONES])
        g += tz[t + S_MY_CAPTURES] * m[gc + G_CAP] + tz[t + S_THEIR_CAPTURES] * m[go + G_CAP]
        g += tz[t + S_MY_PAIRS] * my_pairs + tz[t + S_THEIR_PAIRS] * their_pairs
        if m[gc + G_OPEN] > 0:
            g += tz[t + S_MY_OPEN]
        if m[go + G_OPEN] > 0:
            g += tz[t + S_THEY_THREATEN]
            if m[go + G_OPEN] >= 3 and m[gc + G_FOUR] == 0:
                g += tz[t + S_THEIR_DOUBLE]
        if m[gc + G_FOUR] > 0:
            g += tz[t + S_I_HAVE_FOURS]
        if m[go + G_FOUR] > 0:
            g += tz[t + S_THEY_HAVE_FOURS]
        if m[go + G_WIN] > 0:
            g += tz[t + S_IN_CHECK]
        if stage == 0:
            v0 += g
        else:
            v1 += g
    stones = m[gc + G_STONES] + m[go + G_STONES]
    lo = tz[TZ_PARAMS + P_STAGE_LO]
    hi = tz[TZ_PARAMS + P_STAGE_HI]
    if stones >= hi:
        return v1
    if stones <= lo:
        return v0
    return (v0 * (hi - stones) + v1 * (stones - lo)) // (hi - lo)


@_compiled(I64(ET, I64))
def static_value(E, colour):
    """evaluate for use from outside the compiled code."""
    return evaluate(E.m, E.h, E.otab, E.tz, colour)


@_compiled(types.void(ET))
def reset_position(E):
    """An empty board."""
    m = E.m
    h = E.h
    tc = E.tc
    itab = E.itab
    otab = E.otab
    for i in range(M_UST):
        m[i] = 0
    for p in range(AREA):
        m[M_BOARD + p] = WALL
    for i in range(361):
        m[M_BOARD + tc[TC_CELLS + i]] = 0
    h[H_HASH] = 0
    h[H_SP] = 0
    h[H_LOG] = H_LOGS
    h[H_EV] = 0
    h[H_EV + 1] = 0
    for i in range(361):
        p = tc[TC_CELLS + i]
        for d in range(4):
            D = DIRS[d]
            code = 0
            for k in range(1, 6):
                code |= m[M_BOARD + p - k * D] << (2 * (5 - k))
                code |= m[M_BOARD + p + k * D] << (2 * (k + 4))
            m[M_CODE + d * AREA + p] = code
            info = int(itab[code])
            if info == UNKNOWN:
                info = analyse_info(code)
                otab[CLASSES + code] = info >> 16
                info &= 0xFFFF
                itab[code] = info
            m[M_INFO + d * AREA + p] = info
            if info != 0:
                apply_info(m, h, p, 0, info, H_LOGS)


# --------------------------------------------------------------------------
# Lists of moves. Each writes into the row of `ply` and returns how many.


@njit(cache=True, nogil=True, inline="always")
def _sort_row(m, ply, n):
    """Order a row by its values, best first (the rows are short)."""
    ply &= 127
    base = M_MV + ply * MOVES
    values = M_MS + ply * MOVES
    for i in range(1, n):
        move = m[base + i]
        value = m[values + i]
        j = i - 1
        while j >= 0 and m[values + j] < value:
            m[base + j + 1] = m[base + j]
            m[values + j + 1] = m[values + j]
            j -= 1
        m[base + j + 1] = move
        m[values + j + 1] = value


@njit(cache=True, nogil=True, inline="always")
def _insert(m, base, n, p):
    """Put point p into the row of n points kept in board order, unless it is
    there already. Returns the new length."""
    n &= 511
    j = n
    while j > 0 and m[base + j - 1] > p:
        j -= 1
    if j > 0 and m[base + j - 1] == p:
        return n
    k = n
    while k > j:
        m[base + k] = m[base + k - 1]
        k -= 1
    m[base + j] = p
    return n + 1


@_compiled(I64(ET, I64, I64))
def gen_forced(E, colour, ply):
    """Answers to the opponent's four: block it, or capture something."""
    m = E.m
    tc = E.tc
    colour = 1 if colour == 1 else 2             # (1 or 2, said so that the compiler sees it;
    ply &= 127                                   # about this and the masks: see set_cell)
    other = 3 - colour
    base = M_MV + ply * MOVES
    win = M_CWIN + other * AREA
    cap = M_CCAP + colour * AREA
    n = 0
    points = M_SL + (other - 1) * 361            # the points completing their five are in their list
    for i in range(m[M_SN + other]):
        p = m[points + i] & 1023
        if m[win + p] > 0:
            n = _insert(m, base, n, p)
    left = m[M_G + colour * 8 + G_CAP]
    for i in range(361):
        if left <= 0:
            break
        p = tc[TC_CELLS + i] & 1023
        v = m[cap + p]
        if v > 0:
            left -= v
            if m[win + p] == 0:
                m[base + (n & 511)] = p
                n += 1
    return n


@_compiled(I64(ET, I64, I64))
def gen_defences(E, colour, ply):
    """Moves worth considering when the opponent threatens an open four: the
    points where the opponent would make a four, our own fours, and captures."""
    m = E.m
    tc = E.tc
    tz = E.tz
    otab = E.otab
    colour = 1 if colour == 1 else 2             # (about this and the masks: see set_cell)
    ply &= 127
    other = 3 - colour
    base = M_MV + ply * MOVES
    values = M_MS + ply * MOVES
    cap = M_CCAP + colour * AREA
    n = 0
    for side in (other, colour):
        four = M_CFOUR + side * AREA
        points = M_SL + (side - 1) * 361
        for i in range(m[M_SN + side]):
            p = m[points + i] & 1023
            if m[four + p] > 0:
                n = _insert(m, base, n, p)
    left = m[M_G + colour * 8 + G_CAP]
    for i in range(361):
        if left <= 0:
            break
        p = tc[TC_CELLS + i] & 1023
        v = m[cap + p]
        if v > 0:
            left -= v
            n = _insert(m, base, n, p)
    n &= 511
    for i in range(n):
        m[values + i] = order_value(m, otab, tc, tz, m[base + i], colour)
    _sort_row(m, ply, n)
    return n


@_compiled(I64(ET, I64, I64))
def gen_threats(E, colour, ply):
    """Every move of `colour` that makes a four or an open three, best first."""
    m = E.m
    tc = E.tc
    tz = E.tz
    otab = E.otab
    colour = 1 if colour == 1 else 2             # (about this and the masks: see set_cell)
    ply &= 127
    base = M_MV + ply * MOVES
    values = M_MS + ply * MOVES
    four = M_CFOUR + colour * AREA
    three = M_CTHR + colour * AREA
    n = 0
    points = M_SL + (colour - 1) * 361
    for i in range(m[M_SN + colour]):
        p = m[points + i] & 1023
        if m[four + p] > 0 or m[three + p] > 0:
            n = _insert(m, base, n, p)
    n &= 511
    for i in range(n):
        m[values + i] = order_value(m, otab, tc, tz, m[base + i], colour)
    _sort_row(m, ply, n)
    return n


@_compiled(I64(ET, I64, I64, I64))
def gen_quiet(E, colour, width, ply):
    """The `width` most promising moves near the stones, best first, and after
    them every move that captures or saves a pair (their worth only shows in
    the search)."""
    m = E.m
    tc = E.tc
    tz = E.tz
    otab = E.otab
    colour = 1 if colour == 1 else 2             # (about this and the masks: see set_cell)
    ply &= 127
    base = M_MV + ply * MOVES
    values = M_MS + ply * MOVES
    n = 0
    for i in range(361):
        p = tc[TC_CELLS + i] & 1023
        if m[M_NEAR + p] > 0 and m[M_BOARD + p] == 0:
            m[base + (n & 511)] = p
            m[values + (n & 511)] = order_value(m, otab, tc, tz, p, colour)
            n += 1
    n &= 511
    keep = (width if width < n else n) & 511
    for i in range(keep):
        best = i
        for j in range(i + 1, n):
            if m[values + j] > m[values + (best & 511)]:
                best = j
        best &= 511
        if best != i:
            move = m[base + i]
            m[base + i] = m[base + best]
            m[base + best] = move
            value = m[values + i]
            m[values + i] = m[values + best]
            m[values + best] = value
    if m[M_G + 8 + G_CAP] > 0 or m[M_G + 16 + G_CAP] > 0:
        for j in range(keep, n):
            p = m[base + j] & 1023
            if m[M_CCAP + AREA + p] > 0 or m[M_CCAP + 2 * AREA + p] > 0:
                m[base + j] = m[base + keep]
                m[base + keep] = p
                value = m[values + j]
                m[values + j] = m[values + keep]
                m[values + keep] = value
                keep += 1
    return keep


@njit(cache=True, nogil=True, inline="always")
def _to_front(m, base, n, move):
    """If `move` is in the row, bring it to the front without disturbing the rest."""
    for i in range(n):
        if m[base + i] == move:
            j = i
            while j > 0:
                m[base + j] = m[base + j - 1]
                j -= 1
            m[base] = move
            break


@_compiled(types.void(ET, I64, I64))
def sort_points(E, colour, n):
    """Order the n points in row 0 by how promising they look for `colour`."""
    m = E.m
    tc = E.tc
    tz = E.tz
    otab = E.otab
    for i in range(n):
        m[M_MS + i] = order_value(m, otab, tc, tz, m[M_MV + i], colour)
    _sort_row(m, 0, n)


@_compiled(I64(ET, I64))
def first_point(E, column_base):
    """The first point of the board with a non-zero entry at column_base, or -1."""
    m = E.m
    tc = E.tc
    column_base &= 0x1FFFF                       # (about the masks: see set_cell)
    for i in range(361):
        p = tc[TC_CELLS + i] & 1023
        if m[column_base + p] > 0:
            return p
    return -1


# ------------------------------------------------ forced wins by fours (VCF)


@_compiled(I64(ET, I64, I64, I64))
def vcf(E, colour, depth, ply):
    """A move for `colour` that starts a win by continuous fours, or -1.

    Every attacking move makes a four. The defender may block it or make any
    capture; the attack must survive all of those replies."""
    m = E.m
    h = E.h
    tz = E.tz
    colour = 1 if colour == 1 else 2             # (for the compiler, like the masks: see set_cell)
    ply &= 127
    h[H_NODES] += 1
    if (h[H_NODES] & 63) == 0:
        if tz[TZ_STOP] != 0 or h[H_NODES] >= h[H_LIMIT]:
            h[H_ABORT] = 1
    if h[H_ABORT] != 0:
        return -1
    other = 3 - colour
    gc = M_G + colour * 8
    go = M_G + other * 8
    if m[gc + G_WIN] > 0:
        return first_point(E, M_CWIN + colour * AREA)
    if depth <= 0 or m[gc + G_FOUR] == 0 or h[H_LOG] >= LOG_LIMIT:
        return -1
    vt = E.vt
    key = h[H_HASH] ^ tz[TZ_SIDE + colour] ^ VCF_TAG
    slot = (key & ((vt.shape[0] >> 1) - 1)) << 1
    if vt[slot] == key:
        seen = vt[slot + 1]
        if (seen >> 16) == h[H_GENERATION] and (seen & 0xFFFF) >= depth:
            return -1
    base = M_MV + ply * MOVES
    values = M_MS + ply * MOVES
    tc = E.tc
    four = M_CFOUR + colour * AREA
    blocks = M_CWIN + other * AREA
    in_check = m[go + G_WIN] > 0
    n = 0
    points = M_SL + (colour - 1) * 361
    for i in range(m[M_SN + colour]):
        p = m[points + i]
        if m[four + p] > 0:
            if in_check and m[blocks + p] == 0:
                continue                         # it must also stop their four
            n = _insert(m, base, n, p)
    for i in range(n):
        p = m[base + i]
        m[values + i] = m[four + p] + (100 if m[M_COPEN + colour * AREA + p] > 0 else 0)
    _sort_row(m, ply, n)
    reply_base = M_MV + (ply + 1) * MOVES
    for index in range(n):
        move = m[base + index]
        make(E, move, colour)
        if m[gc + G_WIN] == 0 or m[go + G_FIVE] != 0 or m[go + G_WIN] > 0:
            unmake(E)
            continue
        refuted = False
        replies = gen_forced(E, other, ply + 1)
        for j in range(replies):
            make(E, m[reply_base + j], other)
            if m[gc + G_FIVE] != 0 or m[gc + G_WIN] > 0:
                holds = True
            elif m[go + G_FIVE] != 0:
                holds = False
            else:
                holds = vcf(E, colour, depth - 1, ply + 2) >= 0
            unmake(E)
            if not holds:
                refuted = True
                break
        unmake(E)
        if h[H_ABORT] != 0:
            return -1
        if not refuted:
            return move
    vt[slot] = key
    vt[slot + 1] = (h[H_GENERATION] << 16) | depth
    return -1


# --------------------------------------------------------------------------
# The search
#
# Depth is counted in half plies. A quiet move costs two; a move that makes a
# four costs the same (its forced answer is free), and so does the answer to
# an open three.


@_compiled(I64(ET, I64, I64, I64, I64, I64))
def quiesce(E, alpha, beta, colour, ply, depth):
    """Leaf value once the pending captures have been played out. The side to
    move may decline to capture (taking the static value)."""
    m = E.m
    h = E.h
    colour = 1 if colour == 1 else 2             # (for the compiler, like the masks: see set_cell)
    ply &= 127
    h[H_NODES] += 1
    gc = M_G + colour * 8
    if m[gc + G_FIVE] != 0:
        return WIN - ply
    if m[gc + G_WIN] > 0:
        return WIN - ply - 1
    best = evaluate(m, h, E.otab, E.tz, colour)
    other = 3 - colour
    if depth <= 0 or best >= beta or m[gc + G_CAP] == 0 or m[M_G + other * 8 + G_WIN] > 0 \
            or h[H_LOG] >= LOG_LIMIT:
        return best
    if best > alpha:
        alpha = best
    base = M_MV + (ply & 127) * MOVES            # (about the masks: see set_cell)
    cap = M_CCAP + (colour & 3) * AREA
    tc = E.tc
    n = 0
    left = m[gc + G_CAP]
    for i in range(361):
        if left <= 0:
            break
        p = tc[TC_CELLS + i] & 1023
        v = m[cap + p]
        if v > 0:
            left -= v
            m[base + (n & 511)] = p
            n += 1
    n &= 511
    for i in range(n):
        make(E, m[base + i], colour)
        value = -quiesce(E, -beta, -alpha, other, ply + 1, depth - 1)
        unmake(E)
        if value > best:
            best = value
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    break
    return best


@_compiled(I64(ET, I64, I64, I64, I64, I64))
def search(E, depth, alpha, beta, colour, ply):
    m = E.m
    h = E.h
    tz = E.tz
    classes = E.otab
    colour = 1 if colour == 1 else 2             # (for the compiler, like the masks: see set_cell)
    ply &= 127
    h[H_NODES] += 1
    if (h[H_NODES] & 255) == 0:
        if tz[TZ_STOP] != 0 or h[H_NODES] >= h[H_LIMIT]:
            h[H_ABORT] = 1
    if h[H_ABORT] != 0:
        return 0
    other = 3 - colour
    gc = M_G + colour * 8
    go = M_G + other * 8
    if m[gc + G_FIVE] != 0:
        return WIN - ply                         # the capture just made left me a five
    if m[gc + G_WIN] > 0:
        return WIN - ply - 1
    in_check = m[go + G_WIN] > 0
    if ply >= h[H_PLY_LIMIT] or h[H_LOG] >= LOG_LIMIT:
        return evaluate(m, h, classes, tz, colour)
    if depth <= 0 and not in_check:
        if m[gc + G_CAP] > 0:
            value = quiesce(E, alpha, beta, colour, ply, tz[TZ_PARAMS + P_QUIESCE])
        else:
            value = evaluate(m, h, classes, tz, colour)
        leaf_vcf = tz[TZ_PARAMS + P_LEAF_VCF]
        if leaf_vcf > 0 and value < beta and m[gc + G_FOUR] > 0:
            # A win by continuous fours from here is a win, whatever the static value says.
            if vcf(E, colour, leaf_vcf, ply) >= 0:
                return WIN - ply - 2 * leaf_vcf
        return value

    # The table of positions. Every position has two places: the first keeps
    # the entry that was searched deepest, the second whatever came last.
    # (An entry is two numbers, `key ^ packed` and `packed`: if two threads
    # write the same place at once the halves do not fit together and the
    # entry is simply not believed.)
    tt = E.tt
    key = h[H_HASH] ^ tz[TZ_SIDE + colour]
    slot = (key & ((tt.shape[0] >> 2) - 1)) << 2
    first = -1
    packed = tt[slot + 1]
    if packed == 0 or (tt[slot] ^ packed) != key:
        packed = tt[slot + 3]
        if packed != 0 and (tt[slot + 2] ^ packed) != key:
            packed = 0
    if packed != 0:
        first = packed & 0x3FF
        if ((packed >> 12) & 0xFF) - 16 >= depth:
            flag = (packed >> 10) & 3
            value = ((packed >> 20) & 0xFFFFFFFF) - 2147483648
            if flag == 0:
                return value
            if flag == 1:
                if value >= beta:
                    return value
            elif value <= alpha:
                return value

    base = M_MV + ply * MOVES
    quiet_cost = tz[TZ_PARAMS + P_QUIET_COST]
    futile = False
    futile_value = 0
    if in_check:
        n = gen_forced(E, colour, ply)
        reply_cost = 0
    elif m[go + G_OPEN] > 0:
        n = gen_defences(E, colour, ply)
        reply_cost = tz[TZ_PARAMS + P_THREAT_COST]
    else:
        if depth >= 6:
            width = tz[TZ_PARAMS + P_WIDTH_DEEP]
        elif depth >= 4:
            width = tz[TZ_PARAMS + P_WIDTH_MID]
        else:
            width = tz[TZ_PARAMS + P_WIDTH_SHALLOW]
        if depth <= 4 and beta < WIN // 2 and alpha > -(WIN // 2):
            # Close to the leaves, with nothing hanging over us: trust the static value
            # when it is far outside the window.
            if depth <= 2:
                margin = tz[TZ_PARAMS + P_FRONTIER]
                futility = tz[TZ_PARAMS + P_FUTILITY]
            else:
                margin = tz[TZ_PARAMS + P_FRONTIER2]
                futility = 0
            if margin > 0 or futility > 0:
                static = evaluate(m, h, classes, tz, colour)
                if margin > 0 and static - margin >= beta:
                    return static - margin
                if futility > 0 and static + futility <= alpha:
                    futile = True
                    futile_value = static + futility
        n = gen_quiet(E, colour, width, ply)
        reply_cost = quiet_cost
        if futile:
            # Far below alpha: a quiet move will not make up the difference.
            kept = 0
            four = M_CFOUR + colour * AREA
            three = M_CTHR + colour * AREA
            cap = M_CCAP + colour * AREA
            rescue = M_CCAP + other * AREA
            for i in range(n):
                p = m[base + i] & 1023
                if m[four + p] > 0 or m[three + p] > 0 or m[cap + p] > 0 or m[rescue + p] > 0:
                    m[base + kept] = p
                    kept += 1
            n = kept
            if n == 0:
                return futile_value
    if n == 0:
        return 0
    node_quiet = reply_cost == quiet_cost
    killers = M_KIL + 2 * ply
    _to_front(m, base, n, m[killers + 1])
    _to_front(m, base, n, m[killers])
    if first > 0:
        _to_front(m, base, n, first)

    four_cost = tz[TZ_PARAMS + P_FOUR_COST]
    if reply_cost < four_cost:
        four_cost = reply_cost
    reduce_after = tz[TZ_PARAMS + P_REDUCE_AFTER]
    reduce_min = tz[TZ_PARAMS + P_REDUCE_MIN]
    reduce2_after = tz[TZ_PARAMS + P_REDUCE2_AFTER]
    reduce2_min = tz[TZ_PARAMS + P_REDUCE2_MIN]
    original_alpha = alpha
    best = -INFINITY
    if futile:
        best = futile_value                      # what the moves left out are taken to reach
    chosen = m[base]
    # Several threads searching the same tree would mostly do the same work in
    # the same order. So a thread that is about to search a move writes a
    # mark for it where the others can see it, and a thread that finds the
    # mark of another puts that move off until it has tried the rest: by then
    # the other thread's result is usually in the table of positions. (The
    # first move is never put off. `ranks` remembers where each move stood, so
    # that a move put off is searched as deep as it would have been.)
    sharing = depth >= SHARE_DEPTH and n > 1 and tz[TZ_SHARING] != 0
    ranks = M_MS + ply * MOVES
    if sharing:
        for i in range(n):
            m[ranks + i] = i
    position = h[H_HASH]
    later = n                                    # from this index on: moves that were put off
    index = 0
    while index < n:
        move = m[base + index] & 1023
        rank = index
        busy = 0
        mark = 0
        if sharing:
            rank = m[ranks + index]
            if index > 0:
                mark = (position ^ tz[TZ_ZOB + colour * AREA + move]) | 1
                busy = TZ_BUSY + (mark & BUSY_MASK)
                if index < later and tz[busy] == mark:
                    for j in range(index, n - 1):
                        m[base + j] = m[base + j + 1]
                        m[ranks + j] = m[ranks + j + 1]
                    m[base + n - 1] = move
                    m[ranks + n - 1] = rank
                    later -= 1
                    continue
                tz[busy] = mark
        taken = make(E, move, colour)
        if m[gc + G_WIN] > 0:                    # made a four: the answer is forced
            child = depth - four_cost
            quiet = False
        else:
            child = depth - reply_cost
            quiet = node_quiet and taken == 0 and m[gc + G_OPEN] == 0
        if index == 0:
            value = -search(E, child, -beta, -alpha, other, ply + 1)
        else:
            if quiet and rank >= reduce_after and depth >= reduce_min:
                # Late quiet moves get a shallower look first.
                less = 2
                if rank >= reduce2_after and depth >= reduce2_min:
                    less = 4
                value = -search(E, child - less, -alpha - 1, -alpha, other, ply + 1)
                if value > alpha:
                    value = -search(E, child, -alpha - 1, -alpha, other, ply + 1)
            else:
                value = -search(E, child, -alpha - 1, -alpha, other, ply + 1)
            if alpha < value and value < beta:
                value = -search(E, child, -beta, -alpha, other, ply + 1)
        unmake(E)
        if busy != 0 and tz[busy] == mark:
            tz[busy] = 0
        if h[H_ABORT] != 0:
            return 0
        if value > best:
            best = value
            chosen = move
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    if m[killers] != move:
                        m[killers + 1] = m[killers]
                        m[killers] = move
                    break
        index += 1
    if best >= beta:
        flag = 1
    elif best <= original_alpha:
        flag = 2
    else:
        flag = 0
    stored = depth
    if stored < -16:
        stored = -16
    elif stored > 200:
        stored = 200
    age = tz[TZ_AGE] & 63
    packed = chosen | (flag << 10) | ((stored + 16) << 12) | ((best + 2147483648) << 20) | (age << 52)
    old = tt[slot + 1]
    if old == 0 or (tt[slot] ^ old) == key or ((old >> 52) & 63) != age \
            or stored + 16 >= ((old >> 12) & 0xFF):
        if old != 0 and ((old >> 52) & 63) == age and (tt[slot] ^ old) != key:
            tt[slot + 3] = old                   # pushed out, but of this search: it moves down
            tt[slot + 2] = tt[slot]
        tt[slot + 1] = packed
        tt[slot] = key ^ packed
    else:
        tt[slot + 3] = packed
        tt[slot + 2] = key ^ packed
    return best


@_compiled(types.boolean(ET, I64, I64, I64, I64, I64))
def search_root(E, n, plies, colour, kind, max_ply):
    """One iteration over the n moves in row 0, to `plies` moves deep.

    kind: 0 the root is in check, 1 it is quiet, 2 it must answer a threat.
    Leaves the best move and its value in h (also when stopped half way: the
    first move searched is the best of the previous iteration, so what has
    been found by then is at least as well founded). Returns False if stopped."""
    m = E.m
    h = E.h
    tz = E.tz
    other = 3 - colour
    gc = M_G + colour * 8
    depth = 2 * plies
    if kind == 0:
        reply_cost = 0
    elif kind == 1:
        reply_cost = tz[TZ_PARAMS + P_QUIET_COST]
    else:
        reply_cost = tz[TZ_PARAMS + P_THREAT_COST]
    four_cost = tz[TZ_PARAMS + P_FOUR_COST]
    if reply_cost < four_cost:
        four_cost = reply_cost
    limit = 2 * plies + 8
    if limit > max_ply:
        limit = max_ply
    h[H_PLY_LIMIT] = limit
    h[H_ROOT_DONE] = 0
    alpha = -INFINITY
    beta = INFINITY
    best = -INFINITY
    sharing = n > 1 and tz[TZ_SHARING] != 0      # (see the note in search)
    position = h[H_HASH]
    later = n
    index = 0
    while index < n:
        move = m[M_MV + index] & 1023
        busy = 0
        mark = 0
        if sharing and index > 0:
            mark = (position ^ tz[TZ_ZOB + colour * AREA + move]) | 1
            busy = TZ_BUSY + (mark & BUSY_MASK)
            if index < later and tz[busy] == mark:
                for j in range(index, n - 1):
                    m[M_MV + j] = m[M_MV + j + 1]
                m[M_MV + n - 1] = move
                later -= 1
                continue
            tz[busy] = mark
        make(E, move, colour)
        if m[gc + G_WIN] > 0:
            child = depth - four_cost
        else:
            child = depth - reply_cost
        if index == 0:
            value = -search(E, child, -beta, -alpha, other, 1)
        else:
            value = -search(E, child, -alpha - 1, -alpha, other, 1)
            if value > alpha and h[H_ABORT] == 0:
                value = -search(E, child, -beta, -alpha, other, 1)
        unmake(E)
        if busy != 0 and tz[busy] == mark:
            tz[busy] = 0
        if h[H_ABORT] != 0:
            return False
        if value > best:
            best = value
            h[H_ROOT_MOVE] = move
            h[H_ROOT_VALUE] = value
            if value > alpha:
                alpha = value
        index += 1
        h[H_ROOT_DONE] = index
    return True


# ---------------------------------- forced wins by fours and threes (VCT)


@_compiled(types.boolean(ET, I64, I64, I64, I64))
def vct(E, colour, depth, ply, attacking):
    """Forced wins of attacker `colour` using only fours and open threes.

    attacking = 1: the attacker is to move; true if it wins within `depth`
    more attacking moves. attacking = 0: the defender is to move; true if the
    attacker still wins whatever the defender does. (One function for both
    sides, because compiled functions cannot call each other in a circle; and
    it calls itself with `1 - attacking`, never a written-out constant, which
    numba would treat as a different kind of call.)"""
    m = E.m
    h = E.h
    colour = 1 if colour == 1 else 2             # (for the compiler, like the masks: see set_cell)
    ply &= 127
    other = 3 - colour
    gc = M_G + colour * 8
    go = M_G + other * 8
    base = M_MV + ply * MOVES
    if attacking != 0:
        tz = E.tz
        h[H_NODES] += 1
        if (h[H_NODES] & 63) == 0:
            if tz[TZ_STOP] != 0 or h[H_NODES] >= h[H_LIMIT]:
                h[H_ABORT] = 1
        if h[H_ABORT] != 0:
            return False
        if m[gc + G_WIN] > 0:
            return True
        if m[go + G_FIVE] != 0:
            return False
        vt = E.vt
        key = h[H_HASH] ^ tz[TZ_SIDE + colour] ^ VCT_TAG
        slot = (key & ((vt.shape[0] >> 1) - 1)) << 1
        if vt[slot] == key:
            seen = vt[slot + 1]
            if (seen >> 16) == h[H_GENERATION]:
                known = seen & 0xFFFF
                if known == 0xFFFF:
                    return True
                if known >= depth:
                    return False
        if depth <= 0 or h[H_LOG] >= LOG_LIMIT:
            return False
        if m[go + G_WIN] > 0:
            n = gen_forced(E, colour, ply)       # in check: only through the forced answer
        else:
            n = gen_threats(E, colour, ply)
        won = False
        for i in range(n):
            make(E, m[base + i], colour)
            won = vct(E, colour, depth - 1, ply + 1, 1 - attacking)
            unmake(E)
            if h[H_ABORT] != 0:
                return False
            if won:
                break
        vt[slot] = key
        vt[slot + 1] = (h[H_GENERATION] << 16) | (0xFFFF if won else depth)
        return won
    if m[go + G_FIVE] != 0 or m[go + G_WIN] > 0:
        return False                             # the defender wins first
    if m[gc + G_FIVE] != 0:
        return True
    if m[gc + G_WIN] > 0:
        n = gen_forced(E, other, ply)
    elif m[gc + G_OPEN] > 0:
        n = gen_defences(E, other, ply)
    else:
        return False                             # the last move was not a threat
    if h[H_LOG] >= LOG_LIMIT:
        return False
    for i in range(n):
        make(E, m[base + i], other)
        won = vct(E, colour, depth, ply + 1, 1 - attacking)
        unmake(E)
        if h[H_ABORT] != 0:
            return False
        if not won:
            return False
    return True


@_compiled(I64(ET, I64, I64))
def find_vct(E, colour, max_depth):
    """A first move of a forced win for `colour` (to move), or -1. Searched
    with growing depth, so the shortest win is found first."""
    m = E.m
    h = E.h
    other = 3 - colour
    gc = M_G + colour * 8
    if m[gc + G_WIN] > 0:
        return first_point(E, M_CWIN + colour * AREA)
    if m[M_G + other * 8 + G_WIN] > 0:
        n = gen_forced(E, colour, 0)
    else:
        n = gen_threats(E, colour, 0)
    for depth in range(1, max_depth + 1):
        for i in range(n):
            move = m[M_MV + i]
            make(E, move, colour)
            won = vct(E, colour, depth - 1, 1, 0)
            unmake(E)
            if h[H_ABORT] != 0:
                return -1
            if won:
                return move
    return -1


@_compiled(I64(ET))
def warm_up(E):
    """Touch every compiled function once (used to compile ahead of a game)."""
    reset_position(E)
    E.h[H_ABORT] = 0
    E.h[H_NODES] = 0
    E.h[H_LIMIT] = 2000
    centre = (9 + PAD) * W + 9 + PAD
    make(E, centre, 1)
    make(E, centre + 1, 2)
    n = gen_quiet(E, 1, 8, 0)
    done = search_root(E, n, 2, 1, 1, 60)
    move = vcf(E, 1, 4, 0)
    other = find_vct(E, 1, 2)
    unmake(E)
    unmake(E)
    return n + move + other + (1 if done else 0)


# --------------------------------------------------------------------------
# Building the arrays (plain Python)


_built = {}


def new_shared(tt_bits=20):
    """The tables every thread of one engine shares."""
    rng = np.random.RandomState(20261019)
    tz = np.zeros(TZ_SIZE, dtype=np.int64)
    tz[TZ_ZOB + AREA:TZ_ZOB + 3 * AREA] = rng.randint(1, 2 ** 62, size=2 * AREA, dtype=np.int64)
    tz[TZ_SIDE + 1] = rng.randint(1, 2 ** 62, dtype=np.int64)
    tz[TZ_SIDE + 2] = rng.randint(1, 2 ** 62, dtype=np.int64)
    for name, slot in SEARCH_SLOTS.items():
        tz[TZ_PARAMS + slot] = OPTIONS[name]
    tc = np.zeros(TC_SIZE, dtype=np.int32)
    cells = [(r + PAD) * W + c + PAD for r in range(N) for c in range(N)]
    on_board = set(cells)
    tc[TC_CELLS:TC_CELLS + 361] = cells
    for p in cells:
        for d in range(4):
            D = DIRS[d]
            if all(q in on_board for q in (p - D, p - 2 * D, p + D, p + 2 * D)):
                tc[TC_WOK + d * AREA + p] = 1
    for r in range(N):
        for c in range(N):
            tc[TC_BIAS + (r + PAD) * W + c + PAD] = (N - abs(r - N // 2) - abs(c - N // 2)) // 5
    otab = np.full(3 << 20, ORD_UNKNOWN, dtype=np.int32)
    otab[CLASSES:] = 0
    return {"itab": np.full(1 << 20, UNKNOWN, dtype=np.uint16), "otab": otab,
            "tc": tc, "tz": tz, "tt": np.zeros(2 << tt_bits, dtype=np.int64)}


def new_engine(shared):
    """The arrays of one search thread, on top of the shared tables."""
    m = np.zeros(M_SIZE, dtype=np.int32)
    m[M_KIL:M_KIL + ROWS * 2] = -1
    h = np.zeros(H_SIZE, dtype=np.int64)
    h[H_LOG] = H_LOGS
    return Engine(m=m, h=h, itab=shared["itab"],
                  otab=shared["otab"], tc=shared["tc"], tz=shared["tz"], tt=shared["tt"],
                  vt=np.zeros(2 << 17, dtype=np.int64))


def pattern_columns(width):
    """The numbering of patterns, shared with the plain Python engine."""
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


def _use(shared, built):
    ptab, numbers, shape_from, lo, hi, shapes = built
    if np.abs(ptab).max() > 2047:
        raise ValueError("a stretch of seven points cannot be worth more than 2047")
    shared["tc"][TC_PTAB:TC_PTAB + ptab.shape[0]] = ptab[:, 0] + (ptab[:, 1] << 20)
    tz = shared["tz"]
    tz[TZ_TERMS:TZ_TERMS + len(numbers)] = numbers
    tz[TZ_PARAMS + P_STAGE_LO] = lo
    tz[TZ_PARAMS + P_STAGE_HI] = hi
    tz[TZ_PARAMS + P_SHAPE_FROM] = shape_from
    tz[TZ_PARAMS + P_SHAPES] = shapes


def install_evaluation(shared, scores, terms):
    """Use a learned table over stretches of seven points and the six
    whole-board terms that go with it (the evaluation of the plain engine in
    smartplayer.py): the same numbers at every stage, no shapes."""
    key = ("table", hash(tuple(scores)), tuple(sorted(terms.items())))
    if key not in _built:
        column = pattern_columns(7)
        swap = (0, 2, 1, 3)
        ptab = np.zeros((2 * 3 * 4096, 2), dtype=np.int64)       # for each stretch and view: the two stages
        for centre in range(3):
            for mid in range(4096):
                digits = [(mid >> (2 * t)) & 3 for t in range(6)]
                window = digits[:3] + [centre] + digits[3:]
                index = (centre * 4096 + mid) * 2
                for view in range(2):
                    seen = window if view == 0 else [swap[v] for v in window]
                    code = 0
                    for k in range(7):
                        code |= seen[k] << (2 * k)
                    where = column[code]
                    ptab[index + view] = int(scores[where]) if where >= 0 else 0
        numbers = np.zeros(TZ_EVAL_END - TZ_TERMS, dtype=np.int64)
        for stage in range(2):
            for slot, name in enumerate(TERMS[:6]):
                numbers[stage * 16 + slot] = terms[name]
        _built[key] = (ptab, numbers, 99, 0, 0, 0)
    _use(shared, _built[key])


def install_model(shared, model):
    """Use a fitted evaluation:

    {"lo": stones, "hi": stones, "shape_from": class, "windows": [opening, later],
     "terms": [opening, later], "my": [opening, later], "their": [opening, later]}

    A `windows` entry is eight numbers: what a window of five is worth that
    holds 1, 2, 3, 4 stones of the side to move and none of the other side
    (and could become an exact five), then the same for the other side's. A
    `terms` entry lists the whole-board terms in the order of TERMS. A `my` /
    `their` entry gives what an empty point is worth by the classes of its two
    strongest directions (81 numbers, strongest * 9 + second); points whose
    strongest direction is below `shape_from` (at least 5, an open three) do
    not count. Up to `lo` stones
    on the board the first entry of each list is used, from `hi` on the
    second, in between a blend; one entry means the same numbers throughout."""
    key = ("model", repr(sorted(model.items())))
    if key not in _built:
        def stage_of(name, stage):
            return model[name][min(stage, len(model[name]) - 1)]

        ptab = np.zeros((2 * 3 * 4096, 2), dtype=np.int64)
        for centre in range(3):
            for mid in range(4096):
                digits = [(mid >> (2 * t)) & 3 for t in range(6)]
                seven = digits[:3] + [centre] + digits[3:]
                five = seven[1:6]
                if 3 in five:
                    continue
                index = (centre * 4096 + mid) * 2
                for view in range(2):
                    me, other = 1 + view, 2 - view
                    slots = []
                    if other not in five and seven[0] != me and seven[6] != me and 1 <= five.count(me) <= 4:
                        slots.append(five.count(me) - 1)
                    if me not in five and seven[0] != other and seven[6] != other \
                            and 1 <= five.count(other) <= 4:
                        slots.append(4 + five.count(other) - 1)
                    for stage in range(2):
                        ptab[index + view, stage] = sum(
                            int(stage_of("windows", stage)[slot]) for slot in slots)
        numbers = np.zeros(TZ_EVAL_END - TZ_TERMS, dtype=np.int64)
        shape_from = int(model.get("shape_from", 5))
        shapes = 0
        for stage in range(2):
            for slot, value in enumerate(stage_of("terms", stage)):
                numbers[stage * 16 + slot] = int(value)
            if numbers[stage * 16 + S_MY_PAIRS] or numbers[stage * 16 + S_THEIR_PAIRS]:
                shapes = 1
            for role, name in enumerate(("my", "their")):
                table = stage_of(name, stage)
                for top in range(max(5, shape_from), 9):
                    for second in range(top + 1):
                        value = int(table[top * 9 + second])
                        numbers[TZ_SHAPE - TZ_TERMS + stage * 162 + role * 81 + top * 9 + second] = value
                        if value:
                            shapes = 1
            for slot in range(8):
                numbers[TZ_WINDOWS - TZ_TERMS + stage * 8 + slot] = int(stage_of("windows", stage)[slot])
        _built[key] = (ptab, numbers, max(5, shape_from), int(model.get("lo", 0)),
                       int(model.get("hi", 0)), shapes)
    _use(shared, _built[key])


def to_point(row, col):
    return (row + PAD) * W + col + PAD


def to_cell(p):
    """(row, col) of a point of the padded grid."""
    return p // W - PAD, p % W - PAD


# --------------------------------------------------------------------------
# Choosing a move (plain Python around the compiled searches)

CENTRE = to_point(N // 2, N // 2)
NODE_RATE = 500000        # nodes that count as one second when search is measured in nodes
SEARCH_PLY = 60           # no line is followed further than this
INSTANT_BELOW_S = 0.02    # with less time than this, play the best-looking move at once
INSTANT_BELOW_NODES = 2000
FIRST_REPLIES = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


class FastEngine:
    """One position and the machinery to pick a move in it."""

    NODE_RATE = NODE_RATE

    def __init__(self, scores=None, terms=None, tt_bits=20, threads=1, options=None, model=None):
        self.shared = new_shared(tt_bits)
        if model is not None:
            install_model(self.shared, model)
        elif scores is not None:
            install_evaluation(self.shared, scores, terms)
        self.options = dict(OPTIONS)
        for name, value in (options or {}).items():
            if name not in OPTIONS:
                raise ValueError("unknown engine option: %r" % (name,))
            self.options[name] = value
        for name, slot in SEARCH_SLOTS.items():
            self.shared["tz"][TZ_PARAMS + slot] = int(self.options[name])
        self.root_width = int(self.options["root_width"])
        self.vcf_depth = int(self.options["vcf_depth"])
        self.vct_depth = int(self.options["vct_depth"])
        self.safety_depth = int(self.options["safety_depth"])
        self.defusing_enough = int(self.options["defusing_enough"])
        self.own_vcf_share = float(self.options["own_vcf_share"])
        self.own_vct_share = float(self.options["own_vct_share"])
        self.probe_share = float(self.options["probe_share"])
        self.soft_share = float(self.options["soft_share"])
        self.search_share = float(self.options["search_share"])
        self.unsettled_drop = int(self.options["unsettled_drop"])
        self.E = new_engine(self.shared)
        reset_position(self.E)
        self.info = None                     # (depth, value, nodes) of the last search
        self.trace = []                      # (depth, seconds, nodes) as each depth was finished
        self.node_clock = False
        self._token = 0
        # Extra threads search the same position at the same time, each with
        # its own copy of it; what they find reaches the others through the
        # shared transposition table.
        self.threads = max(1, int(threads))
        self.helpers = [new_engine(self.shared) for _ in range(self.threads - 1)]
        self.helper_nodes = 0

    # ------------------------------------------------------------ the clock

    def use_node_clock(self, enabled=True):
        """Measure "time" in searched nodes, so that games can be repeated exactly."""
        self.node_clock = enabled

    def now(self):
        if self.node_clock:
            return self.E.h[H_NODES] / float(NODE_RATE)
        return _time.perf_counter()

    def _expire(self, token):
        if token == self._token:
            self.E.tz[TZ_STOP] = 1

    def _begin(self, seconds):
        """Allow the compiled code to run for this long from now."""
        h = self.E.h
        h[H_ABORT] = 0
        self.E.tz[TZ_STOP] = 0
        self._token += 1
        if self.node_clock:
            h[H_LIMIT] = h[H_NODES] + max(64, int(seconds * NODE_RATE))
            return None
        h[H_LIMIT] = 1 << 62
        if seconds <= 0.0005:
            self.E.tz[TZ_STOP] = 1
            return None
        timer = threading.Timer(seconds, self._expire, (self._token,))
        timer.daemon = True
        timer.start()
        return timer

    def _end(self, timer):
        """Returns True if the compiled code was stopped before it finished."""
        if timer is not None:
            timer.cancel()
        self._token += 1
        h = self.E.h
        stopped = bool(h[H_ABORT])
        h[H_ABORT] = 0
        self.E.tz[TZ_STOP] = 0
        return stopped

    # --------------------------------------------------------- the position

    def load(self, grid):
        """Make the position equal to the board handed to take_turn()."""
        E = self.E
        m = E.m
        for r in range(N):
            row = grid[r]
            base = (r + PAD) * W + PAD
            for c in range(N):
                value = int(row[c]) + 1          # -1/0/1  ->  0/1/2
                p = base + c
                current = m[M_BOARD + p]
                if current != value:
                    if current != 0 and value != 0:
                        place(E, p, 0)
                    place(E, p, value)
        E.h[H_SP] = 0
        E.h[H_LOG] = H_LOGS
        m[M_G + 8 + G_FIVE] = 0
        m[M_G + 16 + G_FIVE] = 0

    def total(self, colour, column):
        return int(self.E.m[M_G + colour * 8 + column])

    def points(self, base, colour):
        """The points with a non-zero entry in one of the per-point tables."""
        start = base + colour * AREA
        return [int(p) for p in np.nonzero(self.E.m[start:start + AREA])[0]]

    def _row(self, n):
        return [int(p) for p in self.E.m[M_MV:M_MV + n]]

    def candidates(self, colour, limit):
        return self._row(gen_quiet(self.E, colour, limit, 0))

    def candidate_cells(self, colour, limit):
        """The same as row * 19 + col, for use from outside."""
        cells = []
        for p in self.candidates(colour, limit):
            r, c = to_cell(p)
            cells.append(r * N + c)
        return cells

    def forced_replies(self, colour):
        return self._row(gen_forced(self.E, colour, 0))

    def defences(self, colour):
        return self._row(gen_defences(self.E, colour, 0))

    def ordered(self, points, colour):
        m = self.E.m
        points = [p for p in points if m[M_BOARD + p] == 0]
        m[M_MV:M_MV + len(points)] = points
        sort_points(self.E, colour, len(points))
        return self._row(len(points))

    def is_quiet(self):
        """No four, open four or immediate win for either side."""
        m = self.E.m
        return not any(m[M_G + side * 8 + column] for side in (1, 2)
                       for column in (G_WIN, G_FOUR, G_OPEN))

    # ------------------------------------------------------- extra threads

    def _help(self, E, colour, kind, ranking, first_plies):
        """Body of a helper thread: the same deepening search as the main one."""
        h = E.h
        n = len(ranking)
        ranking = list(ranking)
        for plies in range(first_plies, SEARCH_PLY):
            E.m[M_MV:M_MV + n] = ranking
            if not search_root(E, n, plies, colour, kind, SEARCH_PLY):
                break
            move = int(h[H_ROOT_MOVE])
            ranking.remove(move)
            ranking.insert(0, move)
            if abs(int(h[H_ROOT_VALUE])) >= WIN - 2 * SEARCH_PLY:
                break

    def _start_helpers(self, colour, kind, ranking):
        if not self.helpers or self.node_clock:
            return []
        main = self.E
        started = []
        main.tz[TZ_BUSY:TZ_BUSY + BUSY_MASK + 1] = 0
        main.tz[TZ_SHARING] = 1
        for index, E in enumerate(self.helpers):
            E.m[:M_UST] = main.m[:M_UST]
            E.m[M_KIL:M_KIL + ROWS * 2] = -1
            E.h[:H_SIZE] = 0
            E.h[H_LOG] = H_LOGS
            E.h[H_HASH] = main.h[H_HASH]
            E.h[H_EV] = main.h[H_EV]
            E.h[H_EV + 1] = main.h[H_EV + 1]
            E.h[H_LIMIT] = 1 << 62
            thread = threading.Thread(target=self._help, daemon=True,
                                      args=(E, colour, kind, ranking, 2 + (index + 1) % 2))
            thread.start()
            started.append(thread)
        return started

    def _stop_helpers(self, started):
        if not started:
            return
        self.E.tz[TZ_STOP] = 1
        for thread in started:
            thread.join()
        self.E.tz[TZ_SHARING] = 0
        self.helper_nodes = sum(int(E.h[H_NODES]) for E in self.helpers)

    # ------------------------------------------------- the compiled searches

    def try_vcf(self, colour, seconds):
        timer = self._begin(seconds)
        move = vcf(self.E, colour, self.vcf_depth, 0)
        return -1 if self._end(timer) else int(move)

    def try_vct(self, colour, seconds, max_depth=0):
        self.E.h[H_GENERATION] += 1
        timer = self._begin(seconds)
        move = find_vct(self.E, colour, max_depth or self.vct_depth)
        return -1 if self._end(timer) else int(move)

    # ------------------------------------------------------------- choosing

    def quick_move(self, colour):
        """The best move without searching: used when there is no time to think."""
        E = self.E
        other = 3 - colour
        if self.total(other, G_WIN) > 0:
            replies = self.forced_replies(colour)
            for move in replies:
                make(E, move, colour)
                safe = self.total(other, G_WIN) == 0 and self.total(other, G_FIVE) == 0
                unmake(E)
                if safe:
                    return move
            return replies[0]
        moves = self.candidates(colour, 1)
        return moves[0] if moves else -1

    def first_reply(self):
        """Answer the first stone on one of the eight points around it. (The
        agent itself decides this move before the engine is asked: see
        SmartPlayer._first_reply.)"""
        m = self.E.m
        stone = next(int(p) for p in self.E.tc[TC_CELLS:TC_CELLS + 361] if m[M_BOARD + p] != 0)
        r, c = to_cell(stone)
        options = [to_point(r + dr, c + dc) for dr, dc in FIRST_REPLIES
                   if 2 <= r + dr < N - 2 and 2 <= c + dc < N - 2]
        if not options:                      # the stone is in a corner: step towards the middle
            r += 1 if r < N // 2 else -1
            c += 1 if c < N // 2 else -1
            return to_point(r, c)
        return random.choice(options)

    def defusing_moves(self, colour, threat, seconds, each):
        """Moves for `colour` after which the opponent's forced win is gone.
        `threat` is the first move of that win; points on its path are tried first."""
        E = self.E
        other = 3 - colour
        cells = {threat}
        cells.update(self.points(M_CFOUR, other))
        cells.update(self.points(M_CTHR, other))
        cells.update(self.points(M_CFOUR, colour))
        cells.update(self.points(M_CCAP, colour))
        pool = self.ordered(cells, colour)
        if threat in pool:
            pool.remove(threat)
        pool.insert(0, threat)
        for move in self.candidates(colour, self.root_width):
            if move not in pool:
                pool.append(move)
        stop = self.now() + seconds
        safe = []
        for move in pool:
            if self.now() >= stop or len(safe) >= self.defusing_enough:
                break
            make(E, move, colour)
            refutation = self.try_vct(other, each, self.safety_depth)
            unmake(E)
            if refutation < 0:
                safe.append(move)
        return safe

    def choose_point(self, colour, budget):
        """The move for `colour` (1 Black, 2 White) as a point of the padded grid."""
        E = self.E
        m = E.m
        h = E.h
        h[H_NODES] = 0
        self.info = None
        self.trace = []
        start = self.now()
        other = 3 - colour
        stones = self.total(1, G_STONES) + self.total(2, G_STONES)
        if stones == 0:
            return CENTRE
        if stones == 1:
            return self.first_reply()
        if self.total(colour, G_WIN) > 0:
            return int(first_point(E, M_CWIN + colour * AREA))
        if self.node_clock:
            if budget * NODE_RATE < INSTANT_BELOW_NODES:
                return self.quick_move(colour)
        elif budget < INSTANT_BELOW_S:
            return self.quick_move(colour)

        E.tz[TZ_AGE] = (int(E.tz[TZ_AGE]) + 1) & 63      # entries of earlier searches may now be replaced
        # 1. A forced win of our own.
        h[H_GENERATION] += 1
        move = self.try_vcf(colour, budget * self.own_vcf_share)
        if move >= 0:
            return move
        move = self.try_vct(colour, budget * self.own_vct_share)
        if move >= 0:
            return move

        # 2. What would the opponent do if we did nothing? If that wins by
        #    force, only moves that take the win away are worth searching.
        in_check = self.total(other, G_WIN) > 0
        defusing = None
        if not in_check:
            probe_start = self.now()
            threat = self.try_vct(other, budget * self.probe_share, self.safety_depth)
            if threat >= 0:
                took = self.now() - probe_start
                defusing = self.defusing_moves(colour, threat, budget * 0.3,
                                               max(2.0 * took, budget * 0.03))
        if in_check:
            moves = self.forced_replies(colour)
            kind = 0
        elif defusing:
            moves = defusing
            kind = 2
        elif self.total(other, G_OPEN) > 0:
            moves = self.defences(colour)
            kind = 2
        else:
            moves = self.candidates(colour, self.root_width)
            kind = 1
        if not moves:
            return self.quick_move(colour)
        if len(moves) == 1:
            return moves[0]
        m[M_KIL:M_KIL + ROWS * 2] = -1

        # 3. The main search over those moves, one ply deeper each time.
        left = budget - (self.now() - start)
        search_start = self.now()
        best, best_value = moves[0], 0
        ranking = list(moves)
        timer = self._begin(left * self.search_share)
        helpers = self._start_helpers(colour, kind, ranking)
        self.helper_nodes = 0
        for plies in range(2, SEARCH_PLY):
            m[M_MV:M_MV + len(ranking)] = ranking
            finished = search_root(E, len(ranking), plies, colour, kind, SEARCH_PLY)
            if not finished:
                # The unfinished iteration looked at the previous best move
                # first, so what it found so far is at least as well founded.
                if h[H_ROOT_DONE] >= 1:
                    best, best_value = int(h[H_ROOT_MOVE]), int(h[H_ROOT_VALUE])
                    ranking.remove(best)
                    ranking.insert(0, best)
                break
            move, value = int(h[H_ROOT_MOVE]), int(h[H_ROOT_VALUE])
            settled = move == best and value > best_value - self.unsettled_drop
            best, best_value = move, value
            self.info = (plies, value, int(h[H_NODES]))
            self.trace.append((plies, self.now() - start, int(h[H_NODES])))
            ranking.remove(move)
            ranking.insert(0, move)
            if abs(value) >= WIN - 2 * SEARCH_PLY:
                break
            # A new favourite or a falling value deserves another look.
            if settled and self.now() - search_start > left * self.soft_share:
                break
        self._stop_helpers(helpers)
        self._end(timer)

        # 4. Do not walk into a forced loss if another move avoids it. (Moves
        #    that came out of step 2 have already been checked.)
        deadline = start + budget
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
            remaining = deadline - self.now()
            if remaining <= 0.01 * (budget if self.node_clock else 1.0):
                break
            make(E, move, colour)
            refutation = self.try_vct(other, remaining * 0.4, self.safety_depth)
            unmake(E)
            if refutation < 0:
                return move
            if not widened:
                widened = True
                cells = {refutation}
                cells.update(self.points(M_CFOUR, other))
                cells.update(self.points(M_CTHR, other))
                cells.update(self.points(M_CFOUR, colour))
                cells.update(self.points(M_CCAP, colour))
                queue = self.ordered(cells, colour)[:8] + queue
        return best

    def choose(self, colour, budget):
        """The move for `colour` as row * 19 + col, thinking for `budget` seconds."""
        p = self.choose_point(colour, budget)
        if p < 0:
            return -1
        r, c = to_cell(p)
        return r * N + c
