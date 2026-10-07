"""ThreatPlayer: a deterministic threat-hierarchy player for capture omok.

Rules it is built around:
  * exactly five in a row wins (six or more does not),
  * placing a stone captures `NEW opp opp OWN` pairs in any of 8 directions.

Decision order at the root:
  1. win now (exact five);
  2. neutralise the opponent's immediate five (block, or capture a pair out of the line);
  3. forced win by continuous fours (VCF) - the defender may answer by blocking OR by
     capturing, both are searched;
  4. if the opponent has a VCF (open three, four-three, ...), restrict to moves that kill it;
  5/6. otherwise a shallow alpha-beta over pattern-ordered moves, with capture-aware
     ordering and evaluation, followed by a safety check of the chosen move.

Everything is pure Python on a flat padded board with incrementally maintained
5-cell window counts.
"""

from player import Player
import time as _time
import random as _random

_clock = _time.perf_counter

N = 19
W = N + 2
EMPTY = 2
WALL = 3
DIR4 = (1, W, W + 1, W - 1)
DIR8 = (1, -1, W, -W, W + 1, -W - 1, W - 1, -W + 1)
CELLS = [(r + 1) * W + (c + 1) for r in range(N) for c in range(N)]
_REAL = set(CELLS)

# ---- 5-cell windows -------------------------------------------------------------------
WCELLS = []      # window -> tuple of 5 cells
WBEF = []        # cell just before the window (may be a wall)
WAFT = []        # cell just after the window (may be a wall)
WLINE = []       # id of the board line the window lies on
CW = [() for _ in range(W * W)]   # cell -> tuple of windows containing it


def _build():
    cw = [[] for _ in range(W * W)]
    for di, d in enumerate(DIR4):
        for p in CELLS:
            cells = tuple(p + k * d for k in range(5))
            if not all(q in _REAL for q in cells):
                continue
            w = len(WCELLS)
            WCELLS.append(cells)
            WBEF.append(p - d)
            WAFT.append(p + 5 * d)
            r, c = p // W - 1, p % W - 1
            lid = (r, c, r - c + 18, r + c)[di]
            WLINE.append(di * 64 + lid)
            for q in cells:
                cw[q].append(w)
    for p in CELLS:
        CW[p] = tuple(cw[p])


_build()
NW = len(WCELLS)

N2 = [() for _ in range(W * W)]   # neighbours within Chebyshev distance 2
for _p in CELLS:
    _l = []
    for _dr in range(-2, 3):
        for _dc in range(-2, 3):
            if _dr or _dc:
                _q = _p + _dr * W + _dc
                _r, _c = _p // W + _dr, _p % W + _dc
                if 1 <= _r <= N and 1 <= _c <= N:
                    _l.append(_q)
    N2[_p] = tuple(_l)

_rng = _random.Random(20240917)
ZOB = ([_rng.getrandbits(62) for _ in range(W * W)],
       [_rng.getrandbits(62) for _ in range(W * W)])

# window code = (#black) + 6 * (#white); pure black k -> k, pure white k -> 6k.
INC = (1, 6)
FIVE = (5, 30)
_V = (0, 1, 6, 30, 160, 0)
VAL = [0] * 36                    # static value from black's point of view
for _k in range(1, 6):
    VAL[_k] = _V[_k]
    VAL[6 * _k] = -_V[_k]

_ATT = (1, 5, 26, 150, 6000)      # value of adding a stone to a pure own window of k
_DEF = (0, 3, 18, 115, 2500)      # value of spoiling a pure enemy window of k
_CAPV = (0, 2, 12, 90, 700, 0)    # value of removing a stone from a pure enemy window of k
ORD = ([0] * 36, [0] * 36)        # ORD[stm][code]
CAPT = ([0] * 36, [0] * 36)       # CAPT[stm][code]: enemy stone worth
for _k in range(0, 5):
    ORD[0][_k] = _ATT[_k]
    ORD[1][6 * _k] = _ATT[_k]
    if _k:
        ORD[0][6 * _k] = _DEF[_k]
        ORD[1][_k] = _DEF[_k]
for _k in range(1, 6):
    CAPT[0][6 * _k] = _CAPV[_k]
    CAPT[1][_k] = _CAPV[_k]

WIN = 1000000
INF = 10 * WIN


class _TimeUp(Exception):
    pass


class ThreatPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        self.vtt = {}
        self.deadline = 0.0
        self.nodes = 0
        self.dirty = True
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self):
        b = [WALL] * (W * W)
        for p in CELLS:
            b[p] = EMPTY
        self.b = b
        self.code = [0] * NW
        self.score = 0
        self.hash = 0
        self.stones = (set(), set())
        self.nb = [0] * (W * W)
        ws = [None] * 36
        for k in (3, 4, 18, 24):
            ws[k] = set()
        self.ws = ws
        self.s3 = (ws[3], ws[18])
        self.s4 = (ws[4], ws[24])
        self.hist = []
        self.dirty = False

    def _load(self, board):
        if self.dirty or self.hist:
            self._reset()
        b = self.b
        for r in range(N):
            row = board[r]
            base = (r + 1) * W + 1
            for c in range(N):
                v = row[c]
                if v != 0 and v != 1:
                    v = EMPTY
                p = base + c
                cur = b[p]
                if cur != v:
                    if cur != EMPTY:
                        self._take(p, cur)
                    if v != EMPTY:
                        self._put(p, v)

    def _put(self, p, c):
        self.b[p] = c
        inc = INC[c]
        code = self.code
        ws = self.ws
        sc = self.score
        for w in CW[p]:
            old = code[w]
            new = old + inc
            code[w] = new
            sc += VAL[new] - VAL[old]
            s = ws[old]
            if s is not None:
                s.discard(w)
            s = ws[new]
            if s is not None:
                s.add(w)
        self.score = sc
        self.hash ^= ZOB[c][p]
        self.stones[c].add(p)
        nb = self.nb
        for q in N2[p]:
            nb[q] += 1

    def _take(self, p, c):
        self.b[p] = EMPTY
        inc = INC[c]
        code = self.code
        ws = self.ws
        sc = self.score
        for w in CW[p]:
            old = code[w]
            new = old - inc
            code[w] = new
            sc += VAL[new] - VAL[old]
            s = ws[old]
            if s is not None:
                s.discard(w)
            s = ws[new]
            if s is not None:
                s.add(w)
        self.score = sc
        self.hash ^= ZOB[c][p]
        self.stones[c].discard(p)
        nb = self.nb
        for q in N2[p]:
            nb[q] -= 1

    def make(self, p, c):
        self._put(p, c)
        b = self.b
        o = 1 - c
        cap = None
        for d in DIR8:
            q = p + d
            if b[q] == o:
                q2 = q + d
                if b[q2] == o and b[q2 + d] == c:
                    self._take(q, o)
                    self._take(q2, o)
                    if cap is None:
                        cap = [q, q2]
                    else:
                        cap.append(q)
                        cap.append(q2)
        self.hist.append((p, c, cap))

    def undo(self):
        p, c, cap = self.hist.pop()
        self._take(p, c)
        if cap:
            o = 1 - c
            for q in cap:
                self._put(q, o)

    def _unwind(self, base):
        while len(self.hist) > base:
            self.undo()

    # ------------------------------------------------------------- threat queries
    def fivepts(self, c):
        """Empty cells where colour c makes exactly five."""
        s4 = self.s4[c]
        if not s4:
            return []
        b = self.b
        res = []
        for w in s4:
            if b[WBEF[w]] != c and b[WAFT[w]] != c:
                for p in WCELLS[w]:
                    if b[p] == EMPTY:
                        if p not in res:
                            res.append(p)
                        break
        if len(res) > 1:
            res.sort()
        return res

    def has5(self, c):
        s4 = self.s4[c]
        if not s4:
            return False
        b = self.b
        for w in s4:
            if b[WBEF[w]] != c and b[WAFT[w]] != c:
                return True
        return False

    def fourmap(self, c):
        """move -> set of five-points it would create (from pure three-windows)."""
        s3 = self.s3[c]
        res = {}
        if not s3:
            return res
        b = self.b
        for w in s3:
            if b[WBEF[w]] != c and b[WAFT[w]] != c:
                e1 = e2 = -1
                for p in WCELLS[w]:
                    if b[p] == EMPTY:
                        if e1 < 0:
                            e1 = p
                        else:
                            e2 = p
                s = res.get(e1)
                if s is None:
                    res[e1] = {e2}
                else:
                    s.add(e2)
                s = res.get(e2)
                if s is None:
                    res[e2] = {e1}
                else:
                    s.add(e1)
        return res

    def open_lines(self, c):
        """Number of distinct lines on which colour c can make an open (double) four."""
        s3 = self.s3[c]
        if not s3:
            return 0
        b = self.b
        fm = {}
        wl = []
        for w in s3:
            if b[WBEF[w]] != c and b[WAFT[w]] != c:
                e1 = e2 = -1
                for p in WCELLS[w]:
                    if b[p] == EMPTY:
                        if e1 < 0:
                            e1 = p
                        else:
                            e2 = p
                fm.setdefault(e1, set()).add(e2)
                fm.setdefault(e2, set()).add(e1)
                wl.append((e1, e2, WLINE[w]))
        lines = set()
        for e1, e2, ln in wl:
            if len(fm[e1]) >= 2 or len(fm[e2]) >= 2:
                lines.add(ln)
        return len(lines)

    def capture_moves(self, c):
        """List of (move, s1, s2): colour c playing `move` captures enemy pair s1,s2."""
        o = 1 - c
        b = self.b
        res = []
        for s in self.stones[o]:
            for d in DIR4:
                if b[s + d] == o:
                    x = b[s - d]
                    if x == c:
                        if b[s + d + d] == EMPTY:
                            res.append((s + d + d, s, s + d))
                    elif x == EMPTY:
                        if b[s + d + d] == c:
                            res.append((s - d, s, s + d))
        if len(res) > 1:
            res.sort()
        return res

    def vuln(self, c):
        """Number of pairs of colour c that the enemy can capture right now."""
        o = 1 - c
        b = self.b
        n = 0
        for s in self.stones[c]:
            for d in DIR4:
                if b[s + d] == c:
                    x = b[s - d]
                    if x == o:
                        if b[s + d + d] == EMPTY:
                            n += 1
                    elif x == EMPTY:
                        if b[s + d + d] == o:
                            n += 1
        return n

    def replies(self, defc, A):
        """Moves of `defc` that leave the attacker with no five-point (block or capture)."""
        att = 1 - defc
        cands = list(A)
        for m, _s1, _s2 in self.capture_moves(defc):
            if m not in cands:
                cands.append(m)
        res = []
        for m in cands:
            self.make(m, defc)
            if not self.has5(att):
                res.append(m)
            self.undo()
        return res

    def outcome(self, m, c):
        """Exact result of colour c playing m: 1 win, 0 nothing, -1 loss, 2 draw."""
        self.make(m, c)
        b = self.b
        code = self.code
        t = FIVE[c]
        mine = False
        for w in CW[m]:
            if code[w] == t and b[WBEF[w]] != c and b[WAFT[w]] != c:
                mine = True
                break
        theirs = False
        if self.hist[-1][2]:
            o = 1 - c
            t = FIVE[o]
            if t in code:
                for w in range(NW):
                    if code[w] == t and b[WBEF[w]] != o and b[WAFT[w]] != o:
                        theirs = True
                        break
        self.undo()
        if mine:
            return 2 if theirs else 1
        return -1 if theirs else 0

    # ------------------------------------------------------------------ VCF
    def vcf(self, att, depth):
        """Forced win for `att` (to move) by continuous fours. Returns first move or None."""
        if _clock() > self.deadline:
            raise _TimeUp()
        self.nodes += 1
        key = (self.hash, att)
        e = self.vtt.get(key)
        if e is not None:
            if e[0] >= 0:
                return e[0]
            if e[1] >= depth:
                return None
        A = self.fivepts(att)
        if A:
            return A[0]
        if depth <= 0:
            return None
        defc = 1 - att
        D = self.fivepts(defc)
        fm = self.fourmap(att)
        caps = self.capture_moves(att)
        if not fm and not caps:
            self.vtt[key] = (-1, 99)
            return None
        cands = sorted(fm.keys(), key=lambda m: (-len(fm[m]), m))
        capset = set()
        for m, _s1, _s2 in caps:
            capset.add(m)
            if m not in fm:
                cands.append(m)
        if D:
            cands = [m for m in cands if m in D or m in capset]
        pending = []
        for m in cands:
            self.make(m, att)
            if self.has5(defc):
                self.undo()
                continue
            A2 = self.fivepts(att)
            if not A2:
                self.undo()
                continue
            R = self.replies(defc, A2)
            self.undo()
            if not R:
                self.vtt[key] = (m, 0)
                return m
            pending.append((len(R), m, R))
        if depth > 1 and pending:
            pending.sort()
            for _n, m, R in pending:
                self.make(m, att)
                ok = True
                for r in R:
                    self.make(r, defc)
                    x = self.vcf(att, depth - 1)
                    self.undo()
                    if x is None:
                        ok = False
                        break
                self.undo()
                if ok:
                    self.vtt[key] = (m, 0)
                    return m
        self.vtt[key] = (-1, depth if pending else 99)
        return None

    def vcf_id(self, att, depths):
        for d in depths:
            m = self.vcf(att, d)
            if m is not None:
                return m
        return None

    # ------------------------------------------------------------ move generation
    def gen_moves(self, stm, K):
        b = self.b
        nb = self.nb
        tget = ORD[stm].__getitem__
        cget = self.code.__getitem__
        opp = 1 - stm
        sc = {p: sum(map(tget, map(cget, CW[p])))
              for p in CELLS if nb[p] and b[p] == EMPTY}
        if not sc:
            return [p for p in CELLS if b[p] == EMPTY][:K]
        caps = self.capture_moves(stm)
        if caps:
            vget = CAPT[stm].__getitem__
            for m, s1, s2 in caps:
                sc[m] += (160 + sum(map(vget, map(cget, CW[s1])))
                          + sum(map(vget, map(cget, CW[s2]))))
        ocaps = self.capture_moves(opp)
        if ocaps:
            vget = CAPT[opp].__getitem__
            for m, s1, s2 in ocaps:
                sc[m] += (90 + (sum(map(vget, map(cget, CW[s1])))
                                + sum(map(vget, map(cget, CW[s2])))) // 2)
        top = sorted(((v, p) for p, v in sc.items()), reverse=True)[:K + 5]
        out = []
        for v, p in top:
            for d in DIR8:
                x = b[p + d]
                if x == stm:
                    # would (p, p+d) become a capturable pair `opp own own empty`?
                    y = b[p - d]
                    z = b[p + d + d]
                    if (y == opp and z == EMPTY) or (y == EMPTY and z == opp):
                        v -= 130
                elif x == opp:
                    # capture threat `NEW opp opp empty`
                    if b[p + d + d] == opp and b[p + 3 * d] == EMPTY:
                        v += 25
            out.append((v, p))
        out.sort(reverse=True)
        return [p for _v, p in out[:K]]

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, stm):
        opp = 1 - stm
        s = self.score if stm == 0 else -self.score
        if self.s3[stm]:
            fm = self.fourmap(stm)
            if fm:
                best = 0
                for v in fm.values():
                    if len(v) > best:
                        best = len(v)
                s += 2500 if best >= 2 else 45
        if self.s3[opp]:
            n = self.open_lines(opp)
            if n >= 2:
                s -= 1800
            elif n == 1:
                s -= 140
        vo = self.vuln(opp)
        if vo:
            s += 75 * (vo if vo < 3 else 3)
        vs = self.vuln(stm)
        if vs:
            s -= 45 * (vs if vs < 3 else 3)
        return s

    # ---------------------------------------------------------------- alpha-beta
    def ab(self, stm, depth, alpha, beta, ply):
        self.nodes += 1
        if (self.nodes & 31) == 0 and _clock() > self.deadline:
            raise _TimeUp()
        if self.has5(stm):
            return WIN - ply
        opp = 1 - stm
        D = self.fivepts(opp)
        if D:
            if ply >= self.maxply:
                return self.evaluate(stm) - 400
            moves = self.replies(stm, D)
            if not moves:
                return -(WIN - ply - 1)
            nd = depth - 1 if depth > 0 else 0
        else:
            if depth <= 0:
                return self.evaluate(stm)
            moves = self.gen_moves(stm, 9 if ply <= 1 else 7)
            nd = depth - 1
        best = -INF
        for m in moves:
            self.make(m, stm)
            v = -self.ab(opp, nd, -beta, -alpha, ply + 1)
            self.undo()
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best

    def root_search(self, me, rc, deadline):
        """Iterative-deepening alpha-beta over root candidates; returns ranked moves."""
        opp = 1 - me
        order = list(rc)
        base = len(self.hist)
        self.deadline = deadline
        best_val = None
        for depth in range(1, 9):
            self.maxply = depth + 6
            done = []
            alpha = -INF
            try:
                for m in order:
                    self.make(m, me)
                    v = -self.ab(opp, depth - 1, -INF, -alpha, 1)
                    self.undo()
                    done.append((v, m))
                    if v > alpha:
                        alpha = v
            except _TimeUp:
                self._unwind(base)
                if done:
                    bv, bm = max(done, key=lambda t: t[0])
                    if bm != order[0]:
                        order.remove(bm)
                        order.insert(0, bm)
                break
            # stable sort: best first, ties keep previous order
            idx = {m: i for i, m in enumerate(order)}
            done.sort(key=lambda t: (-t[0], idx[t[1]]))
            order = [m for _v, m in done]
            best_val = done[0][0]
            self.info_search = (depth, best_val)
            if best_val >= WIN - 100 or best_val <= -(WIN - 100):
                break
            if len(order) <= 1:
                break
            if _clock() > deadline:
                break
        return order

    # ------------------------------------------------------------------- safety
    def unsafe_after(self, m, me, depth=8):
        """After I play m, does the opponent have a quick forced win?"""
        opp = 1 - me
        base = len(self.hist)
        self.make(m, me)
        try:
            if self.has5(opp):
                return True
            A = self.fivepts(me)
            if A:
                R = self.replies(opp, A)
                if not R:
                    return False
                for r in R:
                    self.make(r, opp)
                    bad = False
                    if not self.has5(me) and self.vcf(opp, depth) is not None:
                        bad = self.vcf(me, 4) is None
                    self.undo()
                    if bad:
                        return True
                return False
            return self.vcf(opp, depth) is not None
        finally:
            self._unwind(base)

    def vct1(self, att, width, dwidth):
        """`att` to move: a quiet move after which att threatens a VCF and none of the
        defender's candidate answers (best `dwidth` by ordering) removes that threat."""
        defc = 1 - att
        if self.has5(defc) or self.has5(att):
            return None
        base = len(self.hist)
        for t in self.gen_moves(att, width):
            self.make(t, att)
            try:
                if self.has5(att) or self.has5(defc):
                    continue            # fours belong to the VCF search
                if self.vcf(att, 6) is None:
                    continue            # not a threat
                held = False
                for d in self.gen_moves(defc, dwidth):
                    if not self.unsafe_after(d, defc, 6):
                        held = True
                        break
                if not held:
                    return t
            finally:
                self._unwind(base)
        return None

    def vct_after(self, m, me):
        """After I play quiet move m, does the opponent have a winning threat move?"""
        base = len(self.hist)
        self.make(m, me)
        try:
            if self.has5(me):
                return False
            return self.vct1(1 - me, 10, 14) is not None
        finally:
            self._unwind(base)

    # --------------------------------------------------------------------- root
    @staticmethod
    def _rc(p):
        return (p // W - 1, p % W - 1)

    def _think(self, board, time, t0):
        self._load(board)
        self.info = ''
        self.info_search = None
        me = self.color
        opp = 1 - me
        b = self.b
        nst = len(self.stones[0]) + len(self.stones[1])
        if nst == 0:
            return (N // 2, N // 2)

        if time is None or time < 0:
            budget = 1.0
        else:
            budget = min(1.0, (time / 1000.0 - 2.5) / 28.0)
            if budget < 0.012:
                # low on clock: a few milliseconds of search, or none at all
                budget = 0.012 if time > 700 else 0.0
        if nst <= 2:
            budget = min(budget, 0.2)
        if len(self.vtt) > 300000:
            self.vtt.clear()

        # 1. win now
        for m in self.fivepts(me):
            if self.outcome(m, me) == 1:
                self.info = 'win'
                return self._rc(m)

        # 2. stop an immediate five
        D = self.fivepts(opp)
        if D:
            forced = [m for m in self.replies(me, D) if self.outcome(m, me) != -1]
            if not forced:
                self.info = 'lost-block'
                return self._rc(D[0])
            if len(forced) == 1:
                self.info = 'forced'
                return self._rc(forced[0])
            sc = dict((p, i) for i, p in enumerate(self.gen_moves(me, 400)))
            forced.sort(key=lambda p: sc.get(p, 999))
            rc = forced
        else:
            rc = self.gen_moves(me, 16)
        if not rc:
            return None
        if budget <= 0.0:
            for m in rc:
                if self.outcome(m, me) != -1:
                    return self._rc(m)
            return self._rc(rc[0])

        base = len(self.hist)

        # 3. own forced win by fours
        self.deadline = t0 + 0.25 * budget
        win = None
        try:
            win = self.vcf_id(me, (2, 4, 7, 11))
        except _TimeUp:
            self._unwind(base)
        if win is not None and b[win] == EMPTY and self.outcome(win, me) != -1:
            self.info = 'vcf'
            return self._rc(win)

        # 3b. own winning threat move (three / double threat that cannot be answered)
        if not D:
            self.deadline = t0 + 0.45 * budget
            tw = None
            try:
                tw = self.vct1(me, 12, 400)
            except _TimeUp:
                self._unwind(base)
            if tw is not None and b[tw] == EMPTY and self.outcome(tw, me) != -1:
                self.info = 'vct'
                return self._rc(tw)

        # 4. opponent's forced win if I pass -> keep only moves that kill it
        verified = False
        if not D:
            threat = None
            self.deadline = _clock() + 0.15 * budget
            try:
                threat = self.vcf_id(opp, (2, 4, 8))
            except _TimeUp:
                self._unwind(base)
            if threat is None:
                self.deadline = _clock() + 0.15 * budget
                try:
                    threat = self.vct1(opp, 12, 14)
                except _TimeUp:
                    self._unwind(base)
            if threat is not None:
                self.info = 'threat%s' % (self._rc(threat),)
                pool = self.gen_moves(me, 26)
                if threat not in pool and b[threat] == EMPTY:
                    pool.insert(0, threat)
                for m, _s1, _s2 in self.capture_moves(me):
                    if m not in pool:
                        pool.append(m)
                safe1 = []      # no VCF for the opponent afterwards
                safe2 = []      # ... and no winning threat move either
                self.deadline = max(_clock() + 0.1 * budget, t0 + 0.7 * budget)
                try:
                    for m in pool:
                        if self.unsafe_after(m, me, 6):
                            continue
                        safe1.append(m)
                        if not self.vct_after(m, me):
                            safe2.append(m)
                            if len(safe2) >= 5:
                                break
                except _TimeUp:
                    self._unwind(base)
                self.info += ' safe1=%d safe2=%d' % (len(safe1), len(safe2))
                if safe2:
                    rc = safe2
                    verified = True
                elif safe1:
                    rc = safe1[:8]
                    verified = True
                else:
                    if threat in rc:
                        rc.remove(threat)
                    if b[threat] == EMPTY:
                        rc.insert(0, threat)

        if len(rc) == 1:
            self.info = 'only %s' % self.info
            return self._rc(rc[0])

        # 5/6. shallow search
        order = self.root_search(me, rc, max(_clock() + 0.1 * budget, t0 + 0.88 * budget))

        # final safety pass
        choice = None
        self.deadline = _clock() + 0.1 * budget
        try:
            for m in order[:6]:
                if b[m] != EMPTY or self.outcome(m, me) == -1:
                    continue
                if choice is None:
                    choice = m
                if verified or not self.unsafe_after(m, me, 6):
                    choice = m
                    break
        except _TimeUp:
            self._unwind(base)
        if choice is None:
            choice = order[0]
        return self._rc(choice)

    def _fallback(self, board):
        best = None
        for r in range(N):
            for c in range(N):
                if board[r][c] == -1:
                    near = 0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < N and 0 <= cc < N and board[rr][cc] != -1:
                                near += 1
                    key = (near > 0, -(abs(r - 9) + abs(c - 9)))
                    if best is None or key > best[0]:
                        best = (key, (r, c))
        return best[1] if best else (0, 0)

    def take_turn(self, board, time):
        t0 = _clock()
        mv = None
        try:
            mv = self._think(board, time, t0)
            if self.hist:
                self._unwind(0)
        except Exception:
            self.dirty = True
            mv = None
        try:
            if mv is not None:
                r, c = mv
                if 0 <= r < N and 0 <= c < N and board[r][c] == -1:
                    return (r, c)
        except Exception:
            pass
        self.dirty = True
        try:
            # second chance: cheap static choice on a rebuilt state
            self._reset()
            self._load(board)
            for m in self.gen_moves(self.color, 8):
                r, c = self._rc(m)
                if board[r][c] == -1:
                    self.dirty = True
                    return (r, c)
        except Exception:
            pass
        self.dirty = True
        return self._fallback(board)
