"""Tests for the GameManager and its clocks. Run: python tests/test_game.py"""

import os
import sys
import time as clock
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "game"))
sys.path.insert(0, os.path.join(HERE, "..", "bots"))

import misbehaving as bad  # noqa: E402
from board import BLACK, EMPTY, WHITE  # noqa: E402
from gamemanager import GameListener, GameManager  # noqa: E402
from humanplayer import HumanPlayer  # noqa: E402
from randombot import RandomPlayer  # noqa: E402
from scoreboard import DRAW, LOSS, WIN, Scoreboard  # noqa: E402


class Recorder(GameListener):
    def __init__(self):
        self.output = []
        self.moves = []
        self.prompts = []
        self.turns = []
        self.results = []

    def turn_started(self, color):
        self.turns.append(color)

    def player_output(self, color, text):
        self.output.append((color, text))

    def input_requested(self, color, prompt):
        self.prompts.append((color, prompt))

    def move_played(self, color, position, captured):
        self.moves.append((color, position, tuple(captured)))

    def game_over(self, result):
        self.results.append(result)


def all_stopped(manager):
    return not any(p._process.is_alive() for p in manager._processes)


def play(black, white, minutes=-1, time_left_ms=None):
    recorder = Recorder()
    manager = GameManager(black, white, minutes, recorder)
    if time_left_ms is not None:
        manager._time_left_ms = list(time_left_ms)
    start = clock.perf_counter()
    result = manager.run()
    return result, manager, recorder, clock.perf_counter() - start


class Timeouts(unittest.TestCase):
    def check_cut_off(self, stuck_class):
        result, manager, recorder, took = play(stuck_class, RandomPlayer, 1, [400.0, 400.0])
        self.assertEqual(result.winner, WHITE)
        self.assertIn("ran out of time", result.reason)
        self.assertLess(took, 3.0)  # start-up + 0.4s, nowhere near "forever"
        self.assertEqual(manager.move_count, 0)
        self.assertEqual(manager.time_left_ms(BLACK), 0.0)
        self.assertTrue(all_stopped(manager))
        self.assertEqual(recorder.results, [result])

    def test_sleeping_player_is_cut_off(self):
        self.check_cut_off(bad.SleepyPlayer)

    def test_busy_looping_player_is_cut_off(self):
        self.check_cut_off(bad.BusyPlayer)

    def test_idle_human_is_cut_off(self):
        result, manager, recorder, took = play(RandomPlayer, HumanPlayer, 1, [400.0, 400.0])
        self.assertEqual(result.winner, BLACK)
        self.assertIn("White", result.reason)
        self.assertIn("ran out of time", result.reason)
        self.assertEqual(len(recorder.prompts), 1)
        self.assertTrue(all_stopped(manager))

    def test_time_adds_up_over_several_turns(self):
        # 120 ms per move with a 300 ms budget: the third move runs out.
        result, manager, recorder, _ = play(bad.SlowPlayer, RandomPlayer, 1, [300.0, 60000.0])
        self.assertEqual(result.winner, WHITE)
        self.assertIn("ran out of time", result.reason)
        self.assertEqual(sum(1 for m in recorder.moves if m[0] == BLACK), 2)

    def test_no_limit_never_times_out(self):
        result, manager, recorder, _ = play(bad.SlowPlayer, bad.IllegalPlayer)
        self.assertEqual(result.winner, BLACK)
        self.assertIsNone(manager.time_left_ms(BLACK))


class Misbehaviour(unittest.TestCase):
    def test_crash(self):
        result, manager, _, _ = play(bad.CrashPlayer, RandomPlayer)
        self.assertEqual(result.winner, WHITE)
        self.assertIn("crashed (RuntimeError: oops)", result.reason)
        self.assertTrue(all_stopped(manager))

    def test_process_exit(self):
        result, manager, _, _ = play(RandomPlayer, bad.ExitPlayer)
        self.assertEqual(result.winner, BLACK)
        self.assertIn("stopped unexpectedly", result.reason)

    def test_illegal_moves(self):
        # Black: malformed value on an empty board.
        result, _, _, _ = play(bad.IllegalPlayer, RandomPlayer)
        self.assertEqual(result.winner, WHITE)
        self.assertIn("illegal move: 'nonsense'", result.reason)
        # White: occupied point (Black opens at 9,9).
        result, manager, _, _ = play(RandomPlayer, bad.IllegalPlayer)
        self.assertEqual(result.winner, BLACK)
        self.assertIn("illegal move: (9, 9)", result.reason)
        self.assertEqual(manager.board.get(9, 9), BLACK)

    def test_constructor_failure(self):
        result, manager, _, _ = play(bad.BadStartPlayer, RandomPlayer)
        self.assertEqual(result.winner, WHITE)
        self.assertIn("could not be started (ValueError: cannot start)", result.reason)
        self.assertTrue(all_stopped(manager))


class NormalPlay(unittest.TestCase):
    def test_player_state_survives_between_turns_and_output_is_forwarded(self):
        result, manager, recorder, _ = play(bad.StatefulPlayer, bad.StatefulPlayer, 1)
        black_moves = [m[1] for m in recorder.moves if m[0] == BLACK]
        self.assertEqual(black_moves[:3], [(0, 3), (2, 10), (4, 3)])
        black_lines = [text for color, text in recorder.output if color == BLACK]
        self.assertTrue(black_lines[0].startswith("turn 1, time 60000"))
        self.assertTrue(black_lines[1].startswith("turn 2, time "))
        self.assertLessEqual(int(black_lines[1].split()[-1]), 60000)
        self.assertEqual(recorder.turns[:4], [BLACK, WHITE, BLACK, WHITE])

    def test_constructor_without_arguments_and_nested_processes(self):
        result, manager, recorder, _ = play(bad.NoArgPlayer, bad.SpawnerPlayer)
        # Both simply fill the board in order; Black gets five first.
        self.assertIsNotNone(result)
        self.assertNotIn("crashed", result.reason)
        self.assertNotIn("could not", result.reason)

    def test_bot_games_end_legally(self):
        for black, white in ((RandomPlayer, RandomPlayer),):
            for _ in range(4):
                result, manager, recorder, _ = play(black, white, 1)
                self.assertNotIn("crashed", result.reason)
                self.assertNotIn("illegal", result.reason)
                self.assertNotIn("time", result.reason)
                if result.winner is not None:
                    self.assertTrue(manager.board.has_five(result.winner))
                self.assertEqual(manager.move_count, len(recorder.moves))
                self.assertLess(manager.time_left_ms(BLACK), 60000.0)
                self.assertTrue(all_stopped(manager))


class HumanBridge(unittest.TestCase):
    def test_clicks_reach_the_unmodified_humanplayer(self):
        recorder = Recorder()
        manager = GameManager(HumanPlayer, bad.StatefulPlayer, 1, recorder)
        manager.start()
        answers = ["hello", "9,9", "1,8", "9,9", "9,10"]  # 1,8 is White's first stone
        deadline = clock.perf_counter() + 30
        while manager.poll() is None and clock.perf_counter() < deadline:
            if manager.awaiting_input:
                if not answers:
                    break
                manager.submit_input(answers.pop(0))
                self.assertFalse(manager.awaiting_input)
            clock.sleep(0.002)
        manager.abort()
        self.assertEqual(answers, [])
        black_moves = [m[1] for m in recorder.moves if m[0] == BLACK]
        self.assertEqual(black_moves, [(9, 9), (9, 10)])
        text = [t for color, t in recorder.output if color == BLACK]
        self.assertTrue(text[0].startswith("Time remaining: 1:00.0"))
        self.assertEqual(sum("Invalid move" in t for t in text), 3)
        self.assertTrue(all("row,col" in p for _, p in recorder.prompts))
        self.assertTrue(all_stopped(manager))
        self.assertIsNone(manager.result)

    def test_abort_while_waiting_for_a_click(self):
        manager = GameManager(HumanPlayer, HumanPlayer, -1, Recorder())
        manager.start()
        deadline = clock.perf_counter() + 30
        while not manager.awaiting_input and clock.perf_counter() < deadline:
            manager.poll()
            clock.sleep(0.002)
        self.assertTrue(manager.awaiting_input)
        manager.abort()
        self.assertTrue(all_stopped(manager))
        self.assertIsNone(manager.poll())
        manager.submit_input("9,9")  # ignored, must not raise


class Scores(unittest.TestCase):
    def test_counts(self):
        board = Scoreboard()
        self.assertEqual(board.totals(), (0, 0, 0))
        for name, outcome in (("Smart", WIN), ("Smart", LOSS),
                              ("Random", WIN), ("Smart", WIN),
                              ("Human", DRAW)):
            board.record(name, outcome)
        self.assertEqual(board.totals(), (3, 1, 1))
        self.assertEqual(board.rows(), [("Smart", 2, 1, 0), ("Random", 1, 0, 0),
                                        ("Human", 0, 0, 1)])
        self.assertEqual(board.games_played(), 5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
