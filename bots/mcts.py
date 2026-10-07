"""MctsPlayer: heuristic-guided Monte Carlo tree search (PUCT).

Design
------
* Board is kept as a flat padded list (0 empty, 1 black, 2 white, 3 border).
* Every candidate point gets a threat score for both colours from a lazily
  built line-pattern table (11-cell window, so the exact-five rule and
  overlines are handled exactly), plus capture / capturable-pair features.
* Tree search is PUCT with those scores as priors.  Leaves are evaluated
  statically from the same scores (no random playouts).  Immediate fives are
  proven inside the tree (small MCTS-solver), nodes where the opponent
  threatens a five only expand blocks and captures, and captures are
  simulated exactly on the way down.
* Before searching the root handles the tactical musts with exact
  simulation: win now, stop an immediate loss (block or capture), play a
  verified open four / double four.
"""
import math
import random
from time import perf_counter

from player import Player

_N = 19
_W = 24                      # row stride: 19 cells + 5 padding cells
_OFF = 6                     # padding rows above the board
_SIZE = (_N + 12) * _W

_AX = (1, _W, _W + 1, _W - 1)
_DIRS = (1, -1, _W, -_W, _W + 1, -_W - 1, _W - 1, -_W + 1)

_TEMPLATE = [3] * _SIZE
_CELLS = []
for _r in range(_N):
    for _c in range(_N):
        _TEMPLATE[(_r + _OFF) * _W + _c] = 0
        _CELLS.append((_r + _OFF) * _W + _c)

_NEIGH = {}                  # points at distance 1-2 along the 8 directions
_RAY = {}                    # points at distance 1-5 along the 8 directions
for _p in _CELLS:
    _near = []
    _ray = []
    for _d in _DIRS:
        _q = _p
        for _k in range(1, 6):
            _q += _d
            if _TEMPLATE[_q] == 3:
                break
            _ray.append(_q)
            if _k <= 2:
                _near.append(_q)
    _NEIGH[_p] = tuple(_near)
    _RAY[_p] = tuple(_ray)

_FIVE = 1000000
_OPEN4 = 100000
_FOUR3 = 50000
_DBL3 = 10000

_B_O3 = 1 << 12
_B_F = 1 << 15
_B_O4 = 1 << 18
_B_5 = 1 << 21

_CAP_BONUS = 400
_CAP_THREAT = 40
_VUL_PEN = 300
_LAM = 0.8                   # weight of the defensive part of a prior
_CP = 1.2
_K_NODE = 8
_K_ROOT = 16


# ----------------------------------------------------------------------------
# line pattern table
# ----------------------------------------------------------------------------
def _run(arr, i):
    lo = i
    while lo > 0 and arr[lo - 1] == 1:
        lo -= 1
    hi = i
    while hi < 10 and arr[hi + 1] == 1:
        hi += 1
    return lo, hi


def _winpoints(arr):
    """Empty cells that complete an exact five containing the centre."""
    cnt = 0
    for q in range(1, 10):
        if arr[q] == 0:
            arr[q] = 1
            lo, hi = _run(arr, q)
            arr[q] = 0
            if hi - lo == 4 and lo <= 5 <= hi:
                cnt += 1
    return cnt


def _line_info(arr):
    """Classify the line through a freshly placed own stone at arr[5].

    arr: 11 codes, 0 empty / 1 own / 2 blocked.  Returns a packed int.
    """
    lo, hi = _run(arr, 5)
    n = hi - lo + 1
    if n == 5:
        return _B_5
    if n >= 6:
        return 0
    w = _winpoints(arr)
    if w >= 2:
        return _B_O4
    if w == 1:
        return _B_F
    n_open = n_closed = 0
    for q in range(1, 10):
        if arr[q] == 0:
            arr[q] = 1
            lo, hi = _run(arr, 5)
            if hi - lo + 1 < 6:
                w = _winpoints(arr)
                if w >= 2:
                    n_open += 1
                elif w == 1:
                    n_closed += 1
            arr[q] = 0
    if n_open:
        return _B_O3 + 10 * min(n_open, 3)
    twos = ones = 0
    for s in range(1, 6):
        if arr[s - 1] == 1 or arr[s + 5] == 1:
            continue
        win = arr[s:s + 5]
        if 2 in win:
            continue
        k = win.count(1)
        if k == 2:
            twos += 1
        elif k == 1:
            ones += 1
    minor = 14 * twos + 2 * ones
    if n_closed:
        minor += 100 + 25 * min(n_closed, 3)
    return minor


class _PatternTable(dict):
    def __missing__(self, key):
        raw = [(key >> (18 - 2 * i)) & 3 for i in range(10)]
        out = 0
        for colour in (1, 2):
            arr = [0 if v == 0 else (1 if v == colour else 2) for v in raw]
            arr.insert(5, 1)
            info = _line_info(arr)
            out |= info << (24 * (colour - 1))
        self[key] = out
        return out


_PAT = _PatternTable()


def _comb(p):
    if p >> 21:
        return _FIVE
    o4 = (p >> 18) & 7
    f = (p >> 15) & 7
    o3 = (p >> 12) & 7
    m = p & 4095
    if o4 or f >= 2:
        return _OPEN4
    if f and o3:
        return _FOUR3
    if o3 >= 2:
        return _DBL3
    if f:
        return 1000 + m
    return 1500 + m


def _cell(b, p, D=_PAT):
    """(attack1, attack2, vul1, vul2, caps1, caps2) for the empty point p."""
    t = (D[b[p - 5] << 18 | b[p - 4] << 16 | b[p - 3] << 14 | b[p - 2] << 12
           | b[p - 1] << 10 | b[p + 1] << 8 | b[p + 2] << 6 | b[p + 3] << 4
           | b[p + 4] << 2 | b[p + 5]]
         + D[b[p - 120] << 18 | b[p - 96] << 16 | b[p - 72] << 14
             | b[p - 48] << 12 | b[p - 24] << 10 | b[p + 24] << 8
             | b[p + 48] << 6 | b[p + 72] << 4 | b[p + 96] << 2 | b[p + 120]]
         + D[b[p - 125] << 18 | b[p - 100] << 16 | b[p - 75] << 14
             | b[p - 50] << 12 | b[p - 25] << 10 | b[p + 25] << 8
             | b[p + 50] << 6 | b[p + 75] << 4 | b[p + 100] << 2 | b[p + 125]]
         + D[b[p - 115] << 18 | b[p - 92] << 16 | b[p - 69] << 14
             | b[p - 46] << 12 | b[p - 23] << 10 | b[p + 23] << 8
             | b[p + 46] << 6 | b[p + 69] << 4 | b[p + 92] << 2 | b[p + 115]])
    a1 = t & 0xFFFFFF
    a2 = t >> 24
    if a1 >= 4096:
        a1 = _comb(a1)
    if a2 >= 4096:
        a2 = _comb(a2)
    c1 = c2 = v1 = v2 = 0
    for d in _DIRS:
        x = b[p + d]
        if x == 1:
            y = b[p + d + d]
            if y == 1:
                z = b[p + 3 * d]
                if z == 2:
                    c2 += 1
                elif z == 0:
                    a2 += _CAP_THREAT
            elif y == 0:
                if b[p - d] == 2:
                    v1 += _VUL_PEN
            elif y == 2:
                if b[p - d] == 0:
                    v1 += _VUL_PEN
        elif x == 2:
            y = b[p + d + d]
            if y == 2:
                z = b[p + 3 * d]
                if z == 1:
                    c1 += 1
                elif z == 0:
                    a1 += _CAP_THREAT
            elif y == 0:
                if b[p - d] == 1:
                    v2 += _VUL_PEN
            elif y == 1:
                if b[p - d] == 0:
                    v2 += _VUL_PEN
    if c1:
        a1 += _CAP_BONUS * c1
    if c2:
        a2 += _CAP_BONUS * c2
    return (a1, a2, v1, v2, c1, c2)


# ----------------------------------------------------------------------------
# exact rules on the flat board
# ----------------------------------------------------------------------------
def _place(b, p, c):
    """Put colour c on p, remove captured pairs, return captured cells."""
    b[p] = c
    o = 3 - c
    caps = None
    for d in _DIRS:
        if b[p + d] == o:
            q2 = p + d + d
            if b[q2] == o and b[q2 + d] == c:
                b[p + d] = 0
                b[q2] = 0
                if caps is None:
                    caps = [p + d, q2]
                else:
                    caps.append(p + d)
                    caps.append(q2)
    return caps


def _wins_at(b, p, c):
    for d in _AX:
        n = 1
        q = p + d
        while b[q] == c:
            n += 1
            q += d
        q = p - d
        while b[q] == c:
            n += 1
            q -= d
        if n == 5:
            return True
    return False


def _five_near(b, cells, c):
    """Exact five of colour c in a run that touches a just-emptied cell."""
    for x in cells:
        for d in _DIRS:
            y = x + d
            if b[y] == c:
                n = 0
                while b[y] == c:
                    n += 1
                    y += d
                if n == 5:
                    return True
    return False


def _outcome(b, p, c):
    """Play c at p on a copy.  Returns (code, board, caps).

    code: 1 mover wins, -1 mover loses, 2 draw, 0 game goes on.
    """
    nb = b[:]
    caps = _place(nb, p, c)
    mine = _wins_at(nb, p, c)
    theirs = bool(caps) and _five_near(nb, caps, 3 - c)
    if mine and theirs:
        return 2, nb, caps
    if mine:
        return 1, nb, caps
    if theirs:
        return -1, nb, caps
    return 0, nb, caps


def _can_capture(b, p, c):
    o = 3 - c
    for d in _DIRS:
        if b[p + d] == o and b[p + d + d] == o and b[p + 3 * d] == c:
            return True
    return False


# ----------------------------------------------------------------------------
# tree
# ----------------------------------------------------------------------------
class _Node(object):
    __slots__ = ('mover', 'sc', 'moves', 'P', 'N', 'Wt', 'ch', 'total',
                 'wsum', 'value', 'proven')


def _squash(s):
    if s <= 1.0:
        return 1.0
    if s <= 2000.0:
        return s
    return 2000.0 * (1.0 + math.log2(s / 2000.0))


def _make_node(mover, sc, K, allowed=None, noise=False):
    """Create a node for `mover` to move; sc maps candidate -> _cell tuple."""
    node = _Node()
    node.mover = mover
    node.sc = sc
    node.total = 0
    node.wsum = 0.0
    node.proven = 0
    ia = mover - 1
    idd = 2 - mover
    iv = mover + 1
    ic = mover + 3
    lam = _LAM
    items = []
    la = []
    ld = []
    capq = []
    for q, e in sc.items():
        a = e[ia]
        d = e[idd]
        la.append(a)
        ld.append(d)
        items.append((a + lam * d - e[iv], q))
        if e[ic]:
            capq.append(q)
    if not items:
        node.moves = ()
        node.value = 0.0
        return node
    la.sort()
    ld.sort()
    maxa = la[-1]
    maxd = ld[-1]

    if allowed is not None:
        sel = [it for it in items if it[1] in allowed]
        value = 0.0
    elif maxa >= _FIVE:
        node.moves = ()
        node.value = 1.0
        node.proven = 1
        return node
    elif maxd >= _FIVE:
        sel = []
        nblock = 0
        for it in items:
            e = sc[it[1]]
            if e[idd] >= _FIVE:
                nblock += 1
                sel.append(it)
            elif e[ic]:
                sel.append(it)
        if nblock == 1:
            value = -0.3
        elif len(sel) == nblock:
            value = -0.9
        else:
            value = -0.6
    else:
        items.sort(reverse=True)
        sel = items[:K]
        if capq:
            chosen = set([it[1] for it in sel])
            for q in capq:
                if q not in chosen:
                    e = sc[q]
                    sel.append((e[ia] + lam * e[idd] - e[iv], q))
        if maxa >= _OPEN4:
            value = 0.8
        elif maxa >= _FOUR3:
            value = 0.7
        else:
            sa = 0
            for x in la[-3:]:
                sa += x if x < 3000 else 3000
            sd = 0
            for x in ld[-3:]:
                sd += x if x < 3000 else 3000
            pos = 0.5 * math.tanh((1.3 * sa - sd) / 4000.0)
            if maxd >= _OPEN4:
                value = -0.2 + 0.5 * pos
                if maxa >= 1000:
                    value += 0.08
            elif maxa >= _DBL3:
                value = 0.45
            elif maxd >= _DBL3:
                value = -0.1 + 0.5 * pos
            else:
                value = pos

    n = len(sel)
    ws = [_squash(it[0]) for it in sel]
    if noise:
        ws = [w * (0.85 + 0.3 * random.random()) for w in ws]
    tot = sum(ws)
    node.moves = [it[1] for it in sel]
    node.P = [w / tot for w in ws]
    node.N = [0] * n
    node.Wt = [0.0] * n
    node.ch = [None] * n
    node.value = value
    return node


class MctsPlayer(Player):
    """Heuristic-guided PUCT search for capture omok."""

    def __init__(self, color):
        super().__init__(color)
        self.stats = []          # (simulations, seconds) per searched move
        self.last_sims = 0

    # ------------------------------------------------------------------
    def take_turn(self, board, time):
        t0 = perf_counter()
        move = None
        try:
            move = self._think(board, time, t0)
        except Exception:
            move = None
        try:
            if (move is not None and 0 <= move[0] < _N and 0 <= move[1] < _N
                    and board[move[0]][move[1]] == -1):
                return (int(move[0]), int(move[1]))
        except Exception:
            pass
        return self._fallback(board)

    @staticmethod
    def _fallback(board):
        best = None
        for r in range(_N):
            row = board[r]
            for c in range(_N):
                if row[c] == -1:
                    near = 0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr = r + dr
                            cc = c + dc
                            if (0 <= rr < _N and 0 <= cc < _N
                                    and board[rr][cc] != -1):
                                near += 1
                    key = (near, -abs(r - 9) - abs(c - 9))
                    if best is None or key > best[0]:
                        best = (key, (r, c))
        return best[1] if best else (0, 0)

    # ------------------------------------------------------------------
    @staticmethod
    def _budget(time, stones):
        if time is None or time < 0:
            budget = 1.0
        else:
            usable = time / 1000.0 - 3.0
            if usable <= 0:
                return 0.0
            budget = usable / 25.0
            if budget > 1.3:
                budget = 1.3
        if stones <= 2:
            budget *= 0.25
        return budget

    def _think(self, board, time, t0):
        me = self.color + 1
        opp = 3 - me
        b = _TEMPLATE[:]
        stones = 0
        for r in range(_N):
            row = board[r]
            base = (r + _OFF) * _W
            for c in range(_N):
                v = row[c]
                if v == 0 or v == 1:
                    b[base + c] = v + 1
                    stones += 1
        if stones == 0:
            return (9, 9)

        sc = {}
        for p in _CELLS:
            if b[p]:
                for q in _NEIGH[p]:
                    if b[q] == 0 and q not in sc:
                        sc[q] = None
        if not sc:
            for p in _CELLS:
                if b[p] == 0:
                    sc[p] = None
            if not sc:
                return None
        for q in sc:
            sc[q] = _cell(b, q)

        def rc(p):
            return (p // _W - _OFF, p % _W)

        ia = me - 1
        idd = 2 - me
        ic = me + 3

        # --- 1. immediate win; note moves that lose or draw at once --------
        bad = []
        for q, e in sc.items():
            if e[ia] >= _FIVE or e[ic]:
                code = _outcome(b, q, me)[0]
                if code == 1:
                    return rc(q)
                if code != 0:
                    bad.append(q)
        if len(bad) < len(sc):
            for q in bad:
                del sc[q]

        # --- 2. opponent's immediate wins --------------------------------
        allowed = None
        threats = [q for q in sc if sc[q][idd] >= _FIVE
                   and _outcome(b, q, opp)[0] == 1]
        if threats:
            resp = set(threats)
            for q, e in sc.items():
                if e[ic]:
                    resp.add(q)
            safe = []
            for r_ in resp:
                nb = b[:]
                caps = _place(nb, r_, me)
                check = list(threats)
                if caps:
                    check.extend(caps)
                still = False
                for w in check:
                    if nb[w] == 0 and _outcome(nb, w, opp)[0] == 1:
                        still = True
                        break
                if not still:
                    safe.append(r_)
            if len(safe) == 1:
                return rc(safe[0])
            if not safe:
                best = max(threats, key=lambda q: sc[q][ia] + sc[q][idd])
                return rc(best)
            allowed = set(safe)
        else:
            # --- 3. verified open four / double four ----------------------
            strong = [q for q in sc if sc[q][ia] >= _OPEN4]
            strong.sort(key=lambda q: -sc[q][ia])
            for q in strong[:6]:
                if self._wins_in_three(b, q, me, sc):
                    return rc(q)

        # --- 4. search ------------------------------------------------------
        root = _make_node(me, sc, _K_ROOT, allowed=allowed, noise=True)
        if root.proven or not root.moves:
            # table said "five" but the exact check disagreed: search all
            root = _make_node(me, sc, _K_ROOT, allowed=set(sc), noise=True)
        if not root.moves:
            return None
        if len(root.moves) == 1:
            return rc(root.moves[0])

        budget = self._budget(time, stones)
        deadline = t0 + budget
        sims = 0
        if budget > 0.0:
            sims = self._search(b, root, t0, deadline)
        self.last_sims = sims
        self.stats.append((sims, perf_counter() - t0))
        return rc(self._choose(root, stones))

    # ------------------------------------------------------------------
    @staticmethod
    def _wins_in_three(b, q, me, sc):
        """True if playing q leaves me a five whatever the opponent does."""
        opp = 3 - me
        code, nb, caps = _outcome(b, q, me)
        if code != 0:
            return False
        pts = [x for x in _RAY[q] if nb[x] == 0 and _outcome(nb, x, me)[0] == 1]
        if not pts:
            return False
        area = set(sc)
        area.discard(q)
        if caps:
            area.update(caps)
        area.update(pts)
        # the opponent must not be able to win first
        for x in area:
            if nb[x] == 0 and _outcome(nb, x, opp)[0] == 1:
                return False
        replies = set(pts)
        for x in area:
            if nb[x] == 0 and _can_capture(nb, x, opp):
                replies.add(x)
        for r_ in replies:
            code2, nb2, caps2 = _outcome(nb, r_, opp)
            if code2 != 0:
                return False
            check = list(pts)
            if caps2:
                check.extend(caps2)
            ok = False
            for x in check:
                if nb2[x] == 0 and _outcome(nb2, x, me)[0] == 1:
                    ok = True
                    break
            if not ok:
                return False
        return True

    # ------------------------------------------------------------------
    def _search(self, rootb, root, t0, deadline):
        sims = 0
        sqrt = math.sqrt
        cp = _CP
        K = _K_NODE
        NEIGH = _NEIGH
        RAY = _RAY
        cell = _cell
        place = _place
        budget = deadline - t0
        early = t0 + 0.4 * budget
        while True:
            now = perf_counter()
            if now >= deadline or root.proven:
                break
            if (sims & 31) == 0 and sims and now >= early:
                if max(root.N) > 0.75 * root.total:
                    break
            sims += 1
            b = rootb[:]
            node = root
            path = []
            pathi = []
            while True:
                moves = node.moves
                n = len(moves)
                if n == 0:
                    v = 0.0
                    break
                ch = node.ch
                Ns = node.N
                Ws = node.Wt
                Ps = node.P
                tot = node.total
                sq = cp * sqrt(tot + 1)
                fpu = (node.wsum / tot - 0.1) if tot else 0.0
                best = -1
                bs = -1e18
                for i in range(n):
                    ni = Ns[i]
                    if ni:
                        if ch[i].proven:
                            continue
                        u = Ws[i] / ni + sq * Ps[i] / (1 + ni)
                    else:
                        u = fpu + sq * Ps[i]
                    if u > bs:
                        bs = u
                        best = i
                if best < 0:
                    # every child is a proven loss (should already be flagged)
                    node.proven = -1
                    v = -1.0
                    break
                mv = moves[best]
                mover = node.mover
                caps = place(b, mv, mover)
                path.append(node)
                pathi.append(best)
                c = ch[best]
                if c is None:
                    sc = dict(node.sc)
                    del sc[mv]
                    for q in NEIGH[mv]:
                        if b[q] == 0 and q not in sc:
                            sc[q] = None
                    if caps:
                        for x in caps:
                            sc[x] = None
                        for x in caps:
                            for q in RAY[x]:
                                if q in sc:
                                    sc[q] = cell(b, q)
                            sc[x] = cell(b, x)
                    for q in RAY[mv]:
                        if q in sc:
                            sc[q] = cell(b, q)
                    c = _make_node(3 - mover, sc, K)
                    ch[best] = c
                    v = c.value
                    break
                node = c
                if c.proven:
                    v = float(c.proven)
                    break
            # backup; v is from the point of view of the side to move below
            for j in range(len(path) - 1, -1, -1):
                v = -v
                nd = path[j]
                i = pathi[j]
                nd.N[i] += 1
                nd.Wt[i] += v
                nd.total += 1
                nd.wsum += v
                pv = nd.ch[i].proven
                if pv == -1:
                    nd.proven = 1
                    v = 1.0
                elif pv == 1:
                    lost = True
                    for cc in nd.ch:
                        if cc is None or cc.proven != 1:
                            lost = False
                            break
                    if lost:
                        nd.proven = -1
                        v = -1.0
        return sims

    # ------------------------------------------------------------------
    @staticmethod
    def _choose(root, stones):
        moves = root.moves
        n = len(moves)
        ch = root.ch
        for i in range(n):
            if ch[i] is not None and ch[i].proven == -1:
                return moves[i]
        live = [i for i in range(n) if ch[i] is None or ch[i].proven != 1]
        if not live:
            live = list(range(n))
        if root.total == 0:
            return moves[max(live, key=lambda i: root.P[i])]
        Ns = root.N
        best = max(live, key=lambda i: (Ns[i], root.P[i]))
        if stones < 10 and Ns[best] > 0:
            qb = root.Wt[best] / Ns[best]
            pool = [i for i in live
                    if Ns[i] >= 0.6 * Ns[best]
                    and root.Wt[i] / Ns[i] >= qb - 0.05]
            if pool:
                best = random.choice(pool)
        return moves[best]
