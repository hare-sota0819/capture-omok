"""Rule tests for the board. Run: python tests/test_rules.py"""

import builtins
import io
import os
import random
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "game"))

from board import BLACK, EMPTY, WHITE, Board  # noqa: E402
from humanplayer import HumanPlayer  # noqa: E402
from player import Player  # noqa: E402

B, W = BLACK, WHITE


def board_with(black=(), white=()):
    board = Board()
    for r, c in black:
        board._grid[r][c] = BLACK
    for r, c in white:
        board._grid[r][c] = WHITE
    board._stone_count = len(black) + len(white)
    return board


class FiveDetection(unittest.TestCase):
    def test_horizontal_vertical_and_both_diagonals(self):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            board = board_with(black=[(9 + i * dr, 9 + i * dc) for i in range(4)])
            self.assertFalse(board.has_five(B))
            board.place(9 + 4 * dr, 9 + 4 * dc, B)
            self.assertTrue(board.has_five(B), (dr, dc))
            self.assertFalse(board.has_five(W))

    def test_filling_the_middle_wins(self):
        board = board_with(white=[(5, 3), (5, 4), (5, 6), (5, 7)])
        board.place(5, 5, W)
        self.assertTrue(board.has_five(W))

    def test_five_with_a_gap_is_not_a_win(self):
        board = board_with(black=[(2, 2), (3, 3), (4, 4), (6, 6)])
        board.place(7, 7, B)
        self.assertFalse(board.has_five(B))

    def test_six_seven_and_longer_do_not_win(self):
        for length in (6, 7, 9):
            stones = [(8, c) for c in range(length)]
            board = board_with(black=stones[:3] + stones[4:])
            board.place(8, 3, B)
            self.assertFalse(board.has_five(B), length)

    def test_two_fives_from_one_stone(self):
        row = [(9, c) for c in (5, 6, 7, 8)]
        col = [(r, 9) for r in (5, 6, 7, 8)]
        board = board_with(white=row + col)
        board.place(9, 9, W)
        self.assertEqual(len(board.find_fives(W)), 2)

    def test_five_and_six_crossing_still_wins_on_the_five(self):
        row = [(9, c) for c in (4, 5, 6, 7, 8, 10)][:5]  # 4..8 -> six with 9
        col = [(r, 9) for r in (5, 6, 7, 8)]
        board = board_with(black=row + col)
        board.place(9, 9, B)
        self.assertEqual(len(board.find_fives(B)), 1)

    def test_fives_touching_the_edges_and_corners(self):
        board = board_with(black=[(0, c) for c in range(4)])
        board.place(0, 4, B)
        self.assertTrue(board.has_five(B))
        board = board_with(white=[(18 - i, 18 - i) for i in range(4)])
        board.place(14, 14, W)
        self.assertTrue(board.has_five(W))
        board = board_with(white=[(i, 18 - i) for i in range(4)])
        board.place(4, 14, W)
        self.assertTrue(board.has_five(W))

    def test_lines_do_not_wrap_around_the_edge(self):
        board = board_with(black=[(3, 16), (3, 17), (3, 18), (4, 0)])
        board.place(4, 1, B)
        self.assertFalse(board.has_five(B))


class Captures(unittest.TestCase):
    def test_capture_in_all_eight_directions(self):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0):
                    continue
                pair = [(9 + dr, 9 + dc), (9 + 2 * dr, 9 + 2 * dc)]
                board = board_with(black=[(9 + 3 * dr, 9 + 3 * dc)], white=pair)
                taken = board.place(9, 9, B)
                self.assertEqual(sorted(taken), sorted(pair), (dr, dc))
                for r, c in pair:
                    self.assertEqual(board.get(r, c), EMPTY)
                    self.assertTrue(board.is_valid_move(r, c))

    def test_several_pairs_at_once(self):
        white = [(9, 10), (9, 11), (10, 9), (11, 9), (8, 8), (7, 7)]
        black = [(9, 12), (12, 9), (6, 6)]
        board = board_with(black=black, white=white)
        self.assertEqual(sorted(board.place(9, 9, B)), sorted(white))
        self.assertEqual(board._stone_count, 4)

    def test_moving_into_a_flank_is_safe(self):
        board = board_with(black=[(4, 3), (4, 6)], white=[(4, 4)])
        self.assertEqual(board.place(4, 5, W), [])
        self.assertEqual(board.get(4, 4), W)
        self.assertEqual(board.get(4, 5), W)

    def test_gap_one_stone_and_three_stones_are_not_captured(self):
        gap = board_with(black=[(5, 8)], white=[(5, 6), (5, 7)])
        self.assertEqual(gap.place(5, 4, B), [])
        single = board_with(black=[(5, 7)], white=[(5, 6)])
        self.assertEqual(single.place(5, 5, B), [])
        triple = board_with(black=[(5, 9)], white=[(5, 6), (5, 7), (5, 8)])
        self.assertEqual(triple.place(5, 5, B), [])

    def test_far_end_must_be_own_stone_or_on_board(self):
        open_end = board_with(white=[(5, 6), (5, 7)])
        self.assertEqual(open_end.place(5, 5, B), [])
        at_edge = board_with(white=[(5, 17), (5, 18)])
        self.assertEqual(at_edge.place(5, 16, B), [])
        near_edge = board_with(black=[(5, 18)], white=[(5, 16), (5, 17)])
        self.assertEqual(len(near_edge.place(5, 15, B)), 2)

    def test_own_stones_are_never_captured(self):
        board = board_with(black=[(5, 6), (5, 7), (5, 8)])
        self.assertEqual(board.place(5, 5, B), [])

    def test_invalid_moves_are_rejected(self):
        board = board_with(black=[(3, 3)])
        for move in ((3, 3), (-1, 0), (0, 19), (19, 0)):
            with self.assertRaises(ValueError):
                board.place(move[0], move[1], W)
            self.assertFalse(board.is_valid_move(*move))

    def test_to_array_is_a_copy(self):
        board = board_with(black=[(3, 3)])
        copy = board.to_array()
        copy[3][3] = WHITE
        copy[0][0] = WHITE
        self.assertEqual(board.get(3, 3), BLACK)
        self.assertEqual(board.get(0, 0), EMPTY)


class Human(unittest.TestCase):
    def ask(self, typed, board=None, time=61234):
        typed = iter(typed)
        prompts = []

        def fake_input(prompt=""):
            prompts.append(prompt)
            return next(typed)

        real_input = builtins.input
        builtins.input = fake_input
        try:
            with redirect_stdout(io.StringIO()) as out:
                move = HumanPlayer(BLACK).take_turn(board or Board().to_array(), time)
        finally:
            builtins.input = real_input
        return move, out.getvalue(), len(prompts)

    def test_valid_move(self):
        move, out, asked = self.ask(["13,4"])
        self.assertEqual(move, (13, 4))
        self.assertEqual(asked, 1)
        self.assertIn("1:01.2", out)

    def test_reasks_until_valid(self):
        board = Board()
        board.place(3, 3, WHITE)
        bad = ["", "abc", "3", "3,", ",3", "3 ,4", "3, 4", "3,4,5", "-1,4", "19,0",
               "0,19", "3.0,4", "3;4", "3,3", "99999999999999999999,1"]
        move, out, asked = self.ask(bad + ["0,18"], board.to_array())
        self.assertEqual(move, (0, 18))
        self.assertEqual(asked, len(bad) + 1)
        self.assertEqual(out.count("Invalid move"), len(bad))

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertEqual(self.ask(["  7,8 \n"])[0], (7, 8))

    def test_time_formats(self):
        fmt = HumanPlayer._format_time
        self.assertEqual(fmt(60000), "1:00.0")
        self.assertEqual(fmt(59960), "0:59.9")
        self.assertEqual(fmt(0), "0:00.0")
        self.assertEqual(fmt(-1), "no limit")
        self.assertEqual(fmt(600000), "10:00.0")

    def test_directly_inherits_player(self):
        self.assertEqual(HumanPlayer.__bases__, (Player,))


if __name__ == "__main__":
    unittest.main(verbosity=1)
