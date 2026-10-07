"""Behaviour tests for SmartPlayer. Run: python tests/test_agent.py"""

import os
import random
import sys
import time as clock
import unittest

os.environ["OMOK_MEMORY"] = "off"       # (the memory of a match has tests of its own)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "game"))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))

from board import BLACK, EMPTY, WHITE, Board  # noqa: E402
from player import Player  # noqa: E402
from smartplayer import SmartPlayer  # noqa: E402


def board_with(black=(), white=()):
    board = Board()
    for r, c in black:
        board._grid[r][c] = BLACK
    for r, c in white:
        board._grid[r][c] = WHITE
    board._stone_count = len(black) + len(white)
    return board


def move_for(color, board, time_ms=3000):
    return tuple(SmartPlayer(color).take_turn(board.to_array(), time_ms))


class Interface(unittest.TestCase):
    def test_is_a_player_and_returns_a_pair_of_ints(self):
        agent = SmartPlayer(BLACK)
        self.assertIsInstance(agent, Player)
        move = agent.take_turn(Board().to_array(), 60000)
        self.assertEqual(len(move), 2)
        self.assertTrue(all(isinstance(v, int) for v in move))

    def test_opens_in_the_centre(self):
        self.assertEqual(move_for(BLACK, Board(), 60000), (9, 9))

    def test_first_reply_is_next_to_the_stone(self):
        for r, c in ((9, 9), (3, 15), (0, 0), (18, 9)):
            row, col = move_for(WHITE, board_with(black=[(r, c)]), 60000)
            self.assertLessEqual(max(abs(row - r), abs(col - c)), 1)
            self.assertNotEqual((row, col), (r, c))

    def test_first_replies_take_turns_between_straight_and_diagonal(self):
        agent = SmartPlayer(WHITE)
        grid = board_with(black=[(9, 9)]).to_array()
        kinds = []
        seen = set()
        for _ in range(120):
            row, col = agent.take_turn(grid, 60000)
            seen.add((row - 9, col - 9))
            kinds.append(row == 9 or col == 9)
        self.assertEqual(seen, {(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)})
        self.assertTrue(all(a != b for a, b in zip(kinds, kinds[1:])))

    def test_does_not_modify_the_board(self):
        board = board_with(black=[(9, 9), (9, 10)], white=[(8, 8)])
        grid = board.to_array()
        copy = [row[:] for row in grid]
        SmartPlayer(WHITE).take_turn(grid, 2000)
        self.assertEqual(grid, copy)

    def test_accepts_tuples_and_no_time_limit(self):
        grid = tuple(tuple(row) for row in board_with(black=[(9, 9)], white=[(8, 8)]).to_array())
        start = clock.perf_counter()
        row, col = SmartPlayer(BLACK).take_turn(grid, -1)
        self.assertLess(clock.perf_counter() - start, 2.5)
        self.assertEqual(grid[row][col], EMPTY)

    def test_one_instance_can_be_reused_across_positions(self):
        agent = SmartPlayer(BLACK)
        board = Board()
        colour = BLACK
        other = SmartPlayer(WHITE)
        for _ in range(12):
            player = agent if colour == BLACK else other
            row, col = player.take_turn(board.to_array(), 2000)
            self.assertTrue(board.is_valid_move(row, col))
            board.place(row, col, colour)
            if board.has_five(BLACK) or board.has_five(WHITE):
                break
            colour = 1 - colour


class Tactics(unittest.TestCase):
    def test_completes_five(self):
        board = board_with(black=[(9, 5), (9, 6), (9, 7), (9, 8)], white=[(9, 4), (3, 3), (4, 4)])
        self.assertEqual(move_for(BLACK, board), (9, 9))

    def test_completes_five_in_a_gap(self):
        board = board_with(white=[(5, 5), (6, 6), (8, 8), (9, 9)], black=[(1, 1), (1, 3), (1, 5), (2, 9)])
        self.assertEqual(move_for(WHITE, board), (7, 7))

    def test_prefers_winning_to_blocking(self):
        board = board_with(black=[(9, 5), (9, 6), (9, 7), (9, 8)],
                           white=[(12, 5), (12, 6), (12, 7), (12, 8)])
        self.assertIn(move_for(BLACK, board), ((9, 4), (9, 9)))
        self.assertIn(move_for(WHITE, board), ((12, 4), (12, 9)))

    def test_does_not_count_six_as_a_win(self):
        # Filling 9,7 would make six for Black; the real five is at 3,7.
        black = [(9, 4), (9, 5), (9, 6), (9, 8), (9, 9), (3, 3), (3, 4), (3, 5), (3, 6)]
        board = board_with(black=black, white=[(3, 2), (10, 10), (11, 11), (5, 12)])
        self.assertEqual(move_for(BLACK, board), (3, 7))

    def test_blocks_a_four(self):
        board = board_with(white=[(9, 5), (9, 6), (9, 7), (9, 8)], black=[(9, 4), (3, 3), (4, 4), (5, 9)])
        self.assertEqual(move_for(BLACK, board), (9, 9))

    def test_breaks_an_open_four_by_capturing(self):
        # White's open four cannot be blocked at both ends, but 7,6 captures
        # the pair 8,6 / 9,6 (flanked by Black's 10,6) and takes a stone out of it.
        white = [(9, 5), (9, 6), (9, 7), (9, 8), (8, 6)]
        board = board_with(white=white, black=[(10, 6), (3, 3), (4, 12), (12, 12), (14, 3)])
        self.assertEqual(move_for(BLACK, board), (7, 6))

    def test_makes_the_open_four(self):
        board = board_with(black=[(9, 6), (9, 7), (9, 8)], white=[(3, 3), (4, 5), (12, 12)])
        self.assertIn(move_for(BLACK, board), ((9, 5), (9, 9)))

    def test_answers_an_open_three(self):
        board = board_with(white=[(9, 6), (9, 7), (9, 8)], black=[(8, 7), (3, 3), (12, 12)])
        self.assertIn(move_for(BLACK, board), ((9, 4), (9, 5), (9, 9), (9, 10)))

    def test_takes_a_free_pair_when_nothing_is_urgent(self):
        board = board_with(black=[(9, 9), (9, 12), (5, 5)], white=[(9, 10), (9, 11), (12, 12)])
        board._grid[9][9] = EMPTY          # Black can capture by playing 9,9
        board._stone_count -= 1
        self.assertEqual(move_for(BLACK, board, 20000), (9, 9))
        self.assertEqual(move_for(BLACK, board, 2600), (9, 9))      # almost out of time


class Robustness(unittest.TestCase):
    def random_position(self, generator, stones):
        board = Board()
        colour = BLACK
        cells = [(r, c) for r in range(19) for c in range(19)]
        generator.shuffle(cells)
        for r, c in cells[:stones]:
            if board.is_valid_move(r, c):
                board.place(r, c, colour)
                colour = 1 - colour
        return board, colour

    def test_always_legal_and_never_slow(self):
        generator = random.Random(7)
        for stones in (2, 5, 20, 60, 150, 300, 355, 360):
            for _ in range(3):
                board, colour = self.random_position(generator, stones)
                if board.is_full():
                    continue
                for time_ms in (30000, 900, 80, 0):
                    start = clock.perf_counter()
                    row, col = SmartPlayer(colour).take_turn(board.to_array(), time_ms)
                    elapsed = clock.perf_counter() - start
                    self.assertTrue(board.is_valid_move(row, col), (stones, time_ms, row, col))
                    self.assertLess(elapsed, max(time_ms / 1000.0, 0.6) + 0.6, (stones, time_ms))

    def test_short_clock_game_does_not_run_out(self):
        board = Board()
        players = [SmartPlayer(BLACK), SmartPlayer(WHITE)]
        left = [6000.0, 6000.0]
        colour = BLACK
        for _ in range(120):
            start = clock.perf_counter()
            row, col = players[colour].take_turn(board.to_array(), int(left[colour]))
            left[colour] -= (clock.perf_counter() - start) * 1000.0
            self.assertGreater(left[colour], 0, "ran out of time")
            self.assertTrue(board.is_valid_move(row, col))
            board.place(row, col, colour)
            if board.has_five(BLACK) or board.has_five(WHITE) or board.is_full():
                break
            colour = 1 - colour


if __name__ == "__main__":
    unittest.main(verbosity=1)
