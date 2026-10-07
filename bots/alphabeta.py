"""Classic alpha-beta gomoku engine for capture omok (exact-five win, pair captures).

Design
------
* Board is a flat padded list (23x23, two wall cells on every side).
* Every one of the 112 board lines carries a base-3 key that is updated
  incrementally on make/undo.  A module-level cache maps a line key to
    (packed stats, ordering values for black-to-move, for white-to-move,
     raw move values for black, raw move values for white)
  so static evaluation is O(1) at the leaves and move ordering is four table
  lookups per candidate cell.
* Line stats come from 5-cell windows that hold no enemy stone and are not
  flanked by an own stone (so filling them gives *exactly* five): fives,
  fours (winning cells), open-four creation points, and capture chances.
* Search: iterative-deepening negamax alpha-beta with a transposition table,
  candidate moves near stones, width-limited ordering that always keeps
  tactical moves, forced-reply extension when the opponent has a four
  (replies = blocks + captures), captures simulated in make/undo.
* Time: a slice of the remaining clock per move (capped), checked inside the
  search; falls back to the best move of the last finished iteration.
"""

import random as _random
import time as _time

from player import Player

N = 19
W = N + 4
SZ = W * W
EMPTY = 2
WALL = 3

P3 = [3 ** i for i in range(20)]
P19 = P3[19]

CELLS = [(r + 2) * W + (c + 2) for r in range(N) for c in range(N)]
DIR8 = (1, -1, W, -W, W + 1, -W - 1, W - 1, -W + 1)
STAR = tuple(d * k for d in DIR8 for k in (1, 2))

# ---------------------------------------------------------------- lines ----
LINE_LEN = []
_cell_lines = {p: [] for p in CELLS}


def _build_lines():
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(N):
            for c in range(N):
                pr, pc = r - dr, c - dc
                if 0 <= pr < N and 0 <= pc < N:
                    continue  # not the start of a line
                lid = len(LINE_LEN)
                pos = 0
                rr, cc = r, c
                while 0 <= rr < N and 0 <= cc < N:
                    _cell_lines[(rr + 2) * W + (cc + 2)].append((lid, pos))
                    pos += 1
                    rr += dr
                    cc += dc
                LINE_LEN.append(pos)


_build_lines()
NLINES = len(LINE_LEN)

# CP[p] = (l0, pos0, l1, pos1, l2, pos2, l3, pos3)
CP = [None] * SZ
# CL[color][p] = ((line, key delta), ...)
CL = [[None] * SZ, [None] * SZ]
for _p in CELLS:
    _ls = _cell_lines[_p]
    CP[_p] = tuple(x for lp in _ls for x in lp)
    for _c in (0, 1):
        CL[_c][_p] = tuple((l, (_c + 1) * P3[pos]) for l, pos in _ls)

_rng = _random.Random(20240917)
Z = [[_rng.getrandbits(64) for _ in range(SZ)] for _ in range(2)]
ZSIDE = _rng.getrandbits(64)

# ------------------------------------------------------- packed formats ----
# line stats (summed over all lines into one int):
#   bits   0-19 score black      20-39 score white
#   bits  40-49 fives black      50-59 fives white
#   bits  60-69 fours black      70-79 fours white     (winning cells)
#   bits  80-89 open-four spots black, 90-99 white
#   bits 100-109 capture chances black, 110-119 white
M20 = (1 << 20) - 1
M10 = (1 << 10) - 1
N5MASK = ((1 << 20) - 1) << 40
STAT_SHIFT = ((0, 40, 60, 80, 100), (20, 50, 70, 90, 110))

# per-cell move value on one line (summed over the 4 lines of a cell):
#   bits 0-6   quiet potential (3 per one-stone window, 1 per empty window)
#   bits 7-11  windows with two own stones (makes a three)
#   bits 12-15 pairs captured
#   bits 16-23 windows with three own stones (makes a four)
#   bits 24+   windows with four own stones (makes five)
MV_ADD = (1, 3, 1 << 7, 1 << 16, 1 << 24)
MV_CAP = 1 << 12
CAP_MASK = 0xF000
V_WIN = (0, 1, 7, 40, 200)

WIN = 1000000
INF = 10 * WIN

LC = {}
LC_LIMIT = 220000
DEBUG = False


def _line_eval(key):
    """Slow path: analyse one line (cached by key)."""
    L, rest = divmod(key, P19)
    cells = []
    for _ in range(L):
        rest, r = divmod(rest, 3)
        cells.append(r)
    mvs = [None, [0] * L, [0] * L]
    stats = 0
    for c in (1, 2):
        o = 3 - c
        m = mvs[c]
        score = n5 = n4 = 0
        for i in range(L - 4):
            w = cells[i:i + 5]
            if o in w:
                continue
            if i > 0 and cells[i - 1] == c:
                continue
            if i + 5 < L and cells[i + 5] == c:
                continue
            k = w.count(c)
            if k == 5:
                n5 += 1
                continue
            score += V_WIN[k]
            if k == 4:
                n4 += 1
            add = MV_ADD[k]
            for j in range(5):
                if w[j] == 0:
                    m[i + j] += add
        t3 = 0
        for x in m:
            if ((x >> 16) & 255) >= 2:
                t3 += 1
        cap = 0
        for i in range(L - 3):
            if cells[i + 1] == o and cells[i + 2] == o:
                a = cells[i]
                d = cells[i + 3]
                if a == 0 and d == c:
                    m[i] += MV_CAP
                    cap += 1
                elif a == c and d == 0:
                    m[i + 3] += MV_CAP
                    cap += 1
        sh = STAT_SHIFT[c - 1]
        stats += (score << sh[0]) + (n5 << sh[1]) + (n4 << sh[2]) + (t3 << sh[3]) + (cap << sh[4])
    mb = mvs[1]
    mw = mvs[2]
    entry = (
        stats,
        tuple([3 * mb[i] + 2 * mw[i] for i in range(L)]),
        tuple([3 * mw[i] + 2 * mb[i] for i in range(L)]),
        tuple(mb),
        tuple(mw),
    )
    LC[key] = entry
    return entry


class _TimeUp(Exception):
    pass


class AlphaBetaPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        self.last_depth = 0
        self.last_nodes = 0
        self.last_score = 0
        self.last_time = 0.0
        self.errors = 0
        self.last_error = None

    # ------------------------------------------------------------ public --
    def take_turn(self, board, time):
        t0 = _time.perf_counter()
        move = None
        try:
            move = self._choose(board, time, t0)
        except Exception as exc:
            self.errors += 1
            self.last_error = repr(exc)
            move = None
        try:
            if move is not None:
                r, c = move
                if 0 <= r < N and 0 <= c < N and board[r][c] == -1:
                    self.last_time = _time.perf_counter() - t0
                    return (r, c)
        except Exception:
            pass
        return self._fallback(board)

    # ---------------------------------------------------------- fallback --
    @staticmethod
    def _fallback(board):
        best = None
        for r in range(N):
            for c in range(N):
                if board[r][c] != -1:
                    continue
                near = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < N and 0 <= cc < N and board[rr][cc] != -1:
                            near += 1
                key = (near > 0, -(abs(r - 9) + abs(c - 9)))
                if best is None or key > best[0]:
                    best = (key, (r, c))
        return best[1] if best else (9, 9)

    # ------------------------------------------------------------ search --
    def _choose(self, grid, time_ms, t0):
        me = self.color
        op = 1 - me

        # ---- time budget
        if time_ms is None or time_ms < 0:
            budget = 1.0
            rem = 1e9
        else:
            rem = time_ms / 1000.0
            if rem < 1.0:
                budget = 0.0
            else:
                budget = min(1.2, max(0.02, (rem - 2.0) / 22.0))

        # ---- build state
        if len(LC) > LC_LIMIT:
            LC.clear()
        board = [WALL] * SZ
        for p in CELLS:
            board[p] = EMPTY
        linekey = [L * P19 for L in LINE_LEN]
        near = [0] * SZ
        hsh = 0
        nstones = 0
        rmin = cmin = N
        rmax = cmax = -1
        for r in range(N):
            row = grid[r]
            base = (r + 2) * W + 2
            for c in range(N):
                v = row[c]
                if v == 0 or v == 1:
                    p = base + c
                    board[p] = v
                    nstones += 1
                    if r < rmin:
                        rmin = r
                    if r > rmax:
                        rmax = r
                    if c < cmin:
                        cmin = c
                    if c > cmax:
                        cmax = c
                    hsh ^= Z[v][p]
                    for l, dl in CL[v][p]:
                        linekey[l] += dl
                    for off in STAR:
                        near[p + off] += 1
        if nstones == 0:
            return (9, 9)
        if nstones >= N * N:
            return None
        total = 0
        lc_get = LC.get
        for l in range(NLINES):
            k = linekey[l]
            e = lc_get(k)
            if e is None:
                e = _line_eval(k)
            total += e[0]
        if nstones < 4:
            budget *= 0.5
        # the search only looks at cells inside the stones' bounding box + 4
        AREA = [(r + 2) * W + (c + 2)
                for r in range(max(0, rmin - 4), min(N, rmax + 5))
                for c in range(max(0, cmin - 4), min(N, cmax + 5))]

        deadline = t0 + budget
        nodes = 0
        maxply = 8
        TT = {}
        CL0 = CL
        clock = _time.perf_counter

        # ------------------------------------------------------ make/undo
        def make(p, c):
            nonlocal total, hsh
            board[p] = c
            hsh ^= Z[c][p]
            t = total
            for l, dl in CL0[c][p]:
                k = linekey[l]
                kn = k + dl
                linekey[l] = kn
                e = lc_get(kn)
                if e is None:
                    e = _line_eval(kn)
                t += e[0] - LC[k][0]
            for off in STAR:
                near[p + off] += 1
            o = 1 - c
            caps = None
            for d in DIR8:
                q = p + d
                if board[q] == o:
                    q2 = q + d
                    if board[q2] == o and board[q2 + d] == c:
                        if caps is None:
                            caps = [q, q2]
                        else:
                            caps.append(q)
                            caps.append(q2)
            if caps is not None:
                zo = Z[o]
                clo = CL0[o]
                for q in caps:
                    board[q] = EMPTY
                    hsh ^= zo[q]
                    for l, dl in clo[q]:
                        k = linekey[l]
                        kn = k - dl
                        linekey[l] = kn
                        e = lc_get(kn)
                        if e is None:
                            e = _line_eval(kn)
                        t += e[0] - LC[k][0]
                    for off in STAR:
                        near[q + off] -= 1
            total = t
            return caps

        def undo(p, c, caps, t_saved, h_saved):
            nonlocal total, hsh
            if caps is not None:
                o = 1 - c
                clo = CL0[o]
                for q in caps:
                    board[q] = o
                    for l, dl in clo[q]:
                        linekey[l] += dl
                    for off in STAR:
                        near[q + off] += 1
            board[p] = EMPTY
            for l, dl in CL0[c][p]:
                linekey[l] -= dl
            for off in STAR:
                near[p + off] -= 1
            total = t_saved
            hsh = h_saved

        # ------------------------------------------------------- movegen
        def gen(S, in_check, width):
            """Ordered candidate moves for side S."""
            out = []
            app = out.append
            a = 1 + S
            if in_check:
                raw = 3 + S
                blk = 1 << 25  # opponent five-cell contributes 2 << 24
                for p in AREA:
                    if near[p] and board[p] == 2:
                        l0, i0, l1, i1, l2, i2, l3, i3 = CP[p]
                        e0 = LC[linekey[l0]]
                        e1 = LC[linekey[l1]]
                        e2 = LC[linekey[l2]]
                        e3 = LC[linekey[l3]]
                        sc = e0[a][i0] + e1[a][i1] + e2[a][i2] + e3[a][i3]
                        if sc >= blk:
                            app((sc, p))
                        elif (e0[raw][i0] + e1[raw][i1] + e2[raw][i2] + e3[raw][i3]) & CAP_MASK:
                            app((sc, p))
                out.sort(reverse=True)
                return [x[1] for x in out]
            for p in AREA:
                if near[p] and board[p] == 2:
                    l0, i0, l1, i1, l2, i2, l3, i3 = CP[p]
                    app((LC[linekey[l0]][a][i0] + LC[linekey[l1]][a][i1]
                         + LC[linekey[l2]][a][i2] + LC[linekey[l3]][a][i3], p))
            out.sort(reverse=True)
            n = len(out)
            if n > width:
                lim = min(n, 2 * width)
                k = width
                while k < lim and out[k][0] >= 8192:
                    k += 1
                del out[k:]
            return [x[1] for x in out]

        # ------------------------------------------------------ evaluate
        def evaluate(S, t):
            if S == 0:
                v = ((t & M20) - ((t >> 20) & M20)
                     + 80 * ((t >> 100) & M10) - 50 * ((t >> 110) & M10))
                if (t >> 80) & M10:
                    v += 5000
                elif ((t >> 90) & M10) >= 3:
                    v -= 600
            else:
                v = (((t >> 20) & M20) - (t & M20)
                     + 80 * ((t >> 110) & M10) - 50 * ((t >> 100) & M10))
                if (t >> 90) & M10:
                    v += 5000
                elif ((t >> 80) & M10) >= 3:
                    v -= 600
            return v + 15

        # ------------------------------------------------------- negamax
        def negamax(depth, alpha, beta, S, ply):
            nonlocal nodes
            nodes += 1
            if not nodes & 255 and clock() > deadline:
                raise _TimeUp()
            t = total
            if S == 0:
                n4s = (t >> 60) & M10
                n4o = (t >> 70) & M10
            else:
                n4s = (t >> 70) & M10
                n4o = (t >> 60) & M10
            if n4s:
                return WIN - ply
            if depth <= 0:
                if not n4o:
                    return evaluate(S, t)
                if ply >= maxply:
                    return evaluate(S, t) - 4000
            elif depth == 1 and not n4o and beta < 500000:
                # static null-move pruning at frontier nodes: comfortably above
                # beta with the move in hand and no enemy open-four threat
                if not (t >> (90 - 10 * S)) & M10:
                    sv = evaluate(S, t) - 120
                    if sv >= beta:
                        return sv
            h0 = hsh
            key = h0 ^ ZSIDE if S else h0
            tte = TT.get(key)
            ttm = 0
            if tte is not None:
                td, tf, tv, ttm = tte
                if td >= depth:
                    if tf == 0:
                        return tv
                    if tf == 1:
                        if tv >= beta:
                            return tv
                    elif tv <= alpha:
                        return tv
            if depth >= 3:
                width = 12
            elif depth == 2:
                width = 10
            else:
                width = 8
            O = 1 - S
            nd = depth if n4o else depth - 1
            best = -INF
            a0 = alpha
            first = True
            if ttm and board[ttm] == 2:
                # try the hash move before paying for move generation
                cur = (ttm,)
                moves = None
            else:
                ttm = 0
                cur = moves = gen(S, n4o, width)
                if not moves:
                    if n4o:
                        return -(WIN - ply - 1)
                    return 0
            bestm = cur[0]
            while True:
                cut = False
                for m in cur:
                    caps = make(m, S)
                    if total & N5MASK:
                        mine = (total >> (40 + 10 * S)) & M10
                        theirs = (total >> (40 + 10 * O)) & M10
                        if mine and theirs:
                            v = 0
                        elif mine:
                            v = WIN - ply
                        else:
                            v = -(WIN - ply)
                    elif first:
                        v = -negamax(nd, -beta, -alpha, O, ply + 1)
                    else:
                        v = -negamax(nd, -alpha - 1, -alpha, O, ply + 1)
                        if alpha < v < beta:
                            v = -negamax(nd, -beta, -alpha, O, ply + 1)
                    first = False
                    undo(m, S, caps, t, h0)
                    if v > best:
                        best = v
                        bestm = m
                        if v > alpha:
                            alpha = v
                            if v >= beta:
                                cut = True
                                break
                if cut or moves is not None:
                    break
                moves = gen(S, n4o, width)
                if ttm in moves:
                    moves.remove(ttm)
                cur = moves
            if best <= a0:
                flag = 2
            elif best >= beta:
                flag = 1
            else:
                flag = 0
            TT[key] = (depth, flag, best, bestm)
            return best

        # ---------------------------------------------------------- root
        def rc(p):
            return (p // W - 2, p % W - 2)

        t = total
        n4me = (t >> (60 + 10 * me)) & M10
        n4op = (t >> (60 + 10 * op)) & M10

        # 1. immediate win
        if n4me:
            raw = 3 + me
            for p in CELLS:
                if board[p] != 2:
                    continue
                l0, i0, l1, i1, l2, i2, l3, i3 = CP[p]
                v = (LC[linekey[l0]][raw][i0] + LC[linekey[l1]][raw][i1]
                     + LC[linekey[l2]][raw][i2] + LC[linekey[l3]][raw][i3])
                if v >> 24:
                    h0 = hsh
                    caps = make(p, me)
                    mine = (total >> (40 + 10 * me)) & M10
                    theirs = (total >> (40 + 10 * op)) & M10
                    undo(p, me, caps, t, h0)
                    if mine and not theirs:
                        self.last_depth = 0
                        self.last_score = WIN
                        return rc(p)

        # 2. candidate list (forced replies only, if the opponent has a four)
        moves = gen(me, n4op, 22)
        if not moves:
            moves = [p for p in CELLS if board[p] == 2 and near[p]]
            if not moves:
                moves = [p for p in CELLS if board[p] == 2]
            if not moves:
                return None
        best_move = moves[0]
        self.last_depth = 0
        self.last_nodes = 0
        if len(moves) == 1 or budget <= 0.0:
            return rc(best_move)

        best_score = 0
        depth = 1
        h_root = hsh
        if DEBUG:
            snap = (board[:], linekey[:], near[:], total, hsh)
        try:
            while depth <= 30:
                maxply = depth + 8
                alpha = -INF
                it_best = None
                it_score = -INF
                for m in moves:
                    caps = make(m, me)
                    t5 = total & N5MASK
                    if t5:
                        mine = (total >> (40 + 10 * me)) & M10
                        theirs = (total >> (40 + 10 * op)) & M10
                        if mine and theirs:
                            v = 0
                        elif mine:
                            v = WIN
                        else:
                            v = -WIN
                    elif it_best is None:
                        v = -negamax(depth - 1, -INF, INF, op, 1)
                    else:
                        v = -negamax(depth - 1, -alpha - 1, -alpha, op, 1)
                        if v > alpha:
                            v = -negamax(depth - 1, -INF, -alpha, op, 1)
                    undo(m, me, caps, t, h_root)
                    if v > it_score:
                        it_score = v
                        it_best = m
                        # a move that finished searching and beats the rest so
                        # far is safe to play even if the iteration is cut off
                        best_move = m
                        best_score = v
                        if v > alpha:
                            alpha = v
                self.last_depth = depth
                if it_best is not None and it_best != moves[0]:
                    moves.remove(it_best)
                    moves.insert(0, it_best)
                if it_score >= WIN - 100 or it_score <= -(WIN - 100):
                    break
                now = clock()
                if now - t0 > budget * 0.45:
                    break
                if len(TT) > 400000:
                    TT.clear()
                depth += 1
            if DEBUG:
                assert snap == (board, linekey, near, total, hsh), "state not restored"
                fresh = 0
                for l in range(NLINES):
                    fresh += LC[linekey[l]][0]
                assert fresh == total
        except _TimeUp:
            pass
        self.last_nodes = nodes
        self.last_score = best_score
        return rc(best_move)
