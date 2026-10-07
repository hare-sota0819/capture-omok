"""Headless matches between two agents, with the same rules and clocks as the game.

    python tools/arena.py A B [--games 8] [--minutes 1] [--jobs 1] [--record DIR] [--verbose]

A and B are player specs: `path/to/file.py:ClassName`, or one of the shortcuts
`random`, `alphabeta`, `threat`, `mcts`, `smart`. Colours alternate: A is black in even-numbered
games. A player loses a game by running out of time, returning an illegal
move, or raising an exception.

`--seconds` may be used instead of `--minutes` for quick, short-clock runs.
"""

import argparse
import importlib.util
import json
import multiprocessing
import os
import random
import sys
import time as clock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "game"))

from board import BLACK, WHITE, Board, opponent_of  # noqa: E402

SHORTCUTS = {
    "random": os.path.join(ROOT, "bots", "randombot.py") + ":RandomPlayer",
    "alphabeta": os.path.join(ROOT, "bots", "alphabeta.py") + ":AlphaBetaPlayer",
    "threat": os.path.join(ROOT, "bots", "threat.py") + ":ThreatPlayer",
    "mcts": os.path.join(ROOT, "bots", "mcts.py") + ":MctsPlayer",
    "smart": os.path.join(ROOT, "agent", "smartplayer.py") + ":SmartPlayer",
}


def load_class(spec):
    spec = SHORTCUTS.get(spec, spec)
    path, class_name = spec.rsplit(":", 1)
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if directory not in sys.path:
        sys.path.append(directory)
    module_name = "arena_" + os.path.splitext(os.path.basename(path))[0] + "_%x" % (
        hash(path) & 0xFFFFFF)
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)


def random_opening(plies, generator):
    """`plies` alternating stones scattered around the centre, with no captures or lines."""
    centre = 9
    cells = []
    while len(cells) < plies:
        r = centre + generator.randint(-3, 3)
        c = centre + generator.randint(-3, 3)
        if (r, c) not in cells:
            cells.append((r, c))
    return cells


def play_game(black_class, white_class, limit_ms, record=False, opening=()):
    """One game. Returns a dict describing how it went."""
    board = Board()
    players = [black_class(BLACK), white_class(WHITE)]
    left = [float(limit_ms), float(limit_ms)]
    used = [0.0, 0.0]
    slowest = [0.0, 0.0]
    counts = [0, 0]
    moves = []
    color = BLACK
    for row, col in opening:
        board.place(row, col, color)
        moves.append((row, col))
        color = opponent_of(color)
    winner, reason = None, None
    while True:
        argument = int(left[color]) if limit_ms > 0 else -1
        start = clock.perf_counter()
        try:
            move = players[color].take_turn(board.to_array(), argument)
        except Exception as error:  # a crash loses the game
            winner, reason = opponent_of(color), "crash: %s: %s" % (type(error).__name__, error)
            break
        elapsed = (clock.perf_counter() - start) * 1000.0
        used[color] += elapsed
        slowest[color] = max(slowest[color], elapsed)
        counts[color] += 1
        if limit_ms > 0:
            left[color] -= elapsed
            if left[color] < 0:
                winner, reason = opponent_of(color), "time"
                break
        try:
            row, col = int(move[0]), int(move[1])
            legal = board.is_valid_move(row, col)
        except Exception:
            legal = False
        if not legal:
            winner, reason = opponent_of(color), "illegal move %r" % (move,)
            break
        board.place(row, col, color)
        moves.append((row, col))
        mine, theirs = board.has_five(color), board.has_five(opponent_of(color))
        if mine and theirs:
            winner, reason = None, "both five"
            break
        if mine:
            winner, reason = color, "five"
            break
        if theirs:
            winner, reason = opponent_of(color), "five after capture"
            break
        if board.is_full():
            winner, reason = None, "board full"
            break
        color = opponent_of(color)
    result = {"winner": winner, "reason": reason, "moves": len(moves), "used_ms": used,
              "slowest_ms": slowest, "turns": counts}
    if record:
        result["record"] = moves
    return result


def _run_one(job):
    index, spec_a, spec_b, limit_ms, record, seed, opening_plies, a_color = job
    a_is_black = index % 2 == 0 if a_color is None else a_color == BLACK
    class_a, class_b = load_class(spec_a), load_class(spec_b)
    black, white = (class_a, class_b) if a_is_black else (class_b, class_a)
    # Games 2k and 2k+1 share an opening, with the colours swapped.
    opening = random_opening(opening_plies, random.Random(seed * 1000 + index // 2))
    random.seed(seed * 7919 + index)
    result = play_game(black, white, limit_ms, record, opening)
    result["index"] = index
    result["a_color"] = BLACK if a_is_black else WHITE
    return result


def run_match(spec_a, spec_b, games=8, limit_ms=60000, jobs=1, record_dir=None, seed=None,
              verbose=False, opening_plies=0, a_color=None):
    seed = random.randrange(1 << 20) if seed is None else seed
    work = [(i, spec_a, spec_b, limit_ms, record_dir is not None, seed, opening_plies, a_color)
            for i in range(games)]
    if jobs > 1:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            results = list(pool.imap_unordered(_run_one, work))
    else:
        results = [_run_one(job) for job in work]
    results.sort(key=lambda r: r["index"])

    tally = {"a_wins": 0, "b_wins": 0, "draws": 0, "a_wins_black": 0, "a_wins_white": 0,
             "a_time_losses": 0, "b_time_losses": 0, "a_faults": 0, "b_faults": 0}
    a_used, b_used, a_slowest, b_slowest = [], [], 0.0, 0.0
    for result in results:
        a, b = result["a_color"], opponent_of(result["a_color"])
        a_used.append(result["used_ms"][a])
        b_used.append(result["used_ms"][b])
        a_slowest = max(a_slowest, result["slowest_ms"][a])
        b_slowest = max(b_slowest, result["slowest_ms"][b])
        if result["winner"] is None:
            tally["draws"] += 1
        elif result["winner"] == a:
            tally["a_wins"] += 1
            tally["a_wins_black" if a == BLACK else "a_wins_white"] += 1
        else:
            tally["b_wins"] += 1
        if result["reason"] != "five" and result["winner"] is not None:
            loser = "a" if result["winner"] == b else "b"
            if result["reason"] == "time":
                tally[loser + "_time_losses"] += 1
            elif result["reason"] != "five after capture":
                tally[loser + "_faults"] += 1
        if verbose:
            who = {None: "draw", a: "A", b: "B"}[result["winner"]]
            print("  game %2d: A=%s  winner=%-4s %-22s moves=%3d  A %.1fs  B %.1fs" % (
                result["index"], "black" if a == BLACK else "white", who, result["reason"],
                result["moves"], result["used_ms"][a] / 1000, result["used_ms"][b] / 1000))
        if record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)
            with open(os.path.join(record_dir, "game_%03d.json" % result["index"]), "w") as f:
                json.dump(result, f)
    tally["games"] = games
    tally["a_avg_s"] = sum(a_used) / len(a_used) / 1000.0
    tally["b_avg_s"] = sum(b_used) / len(b_used) / 1000.0
    tally["a_max_game_s"] = max(a_used) / 1000.0
    tally["b_max_game_s"] = max(b_used) / 1000.0
    tally["a_slowest_move_s"] = a_slowest / 1000.0
    tally["b_slowest_move_s"] = b_slowest / 1000.0
    tally["avg_moves"] = sum(r["moves"] for r in results) / float(games)
    return tally, results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--minutes", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--record", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--a-color", choices=("black", "white"), default=None,
                        help="A plays this colour in every game instead of alternating")
    parser.add_argument("--opening", type=int, default=0,
                        help="start each pair of games from this many random stones")
    args = parser.parse_args()
    # The agent's memory of earlier games would tie the games of a test to each
    # other (tools/matchsim.py is the tool for testing that memory).
    os.environ.setdefault("OMOK_MEMORY", "off")
    limit_ms = args.seconds * 1000 if args.seconds is not None else args.minutes * 60000
    tally, _ = run_match(args.a, args.b, args.games, limit_ms, args.jobs, args.record,
                         args.seed, args.verbose, args.opening,
                         {"black": BLACK, "white": WHITE, None: None}[args.a_color])
    print("A = %s\nB = %s" % (args.a, args.b))
    print("A wins %d (black %d, white %d)   B wins %d   draws %d   of %d games" % (
        tally["a_wins"], tally["a_wins_black"], tally["a_wins_white"], tally["b_wins"],
        tally["draws"], tally["games"]))
    print("time losses: A %d, B %d    faults (crash/illegal): A %d, B %d" % (
        tally["a_time_losses"], tally["b_time_losses"], tally["a_faults"], tally["b_faults"]))
    print("clock used per game: A avg %.1fs max %.1fs   B avg %.1fs max %.1fs" % (
        tally["a_avg_s"], tally["a_max_game_s"], tally["b_avg_s"], tally["b_max_game_s"]))
    print("slowest single move: A %.2fs   B %.2fs    average game length %.0f moves" % (
        tally["a_slowest_move_s"], tally["b_slowest_move_s"], tally["avg_moves"]))


if __name__ == "__main__":
    main()
