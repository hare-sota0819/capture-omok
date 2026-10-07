"""Misbehaving players used by test_game.py. Kept in their own module so the
player processes can import them."""

import multiprocessing
import os
import time as clock

from player import Player


def _first_empty(board):
    for r in range(len(board)):
        for c in range(len(board)):
            if board[r][c] == -1:
                return (r, c)


class SleepyPlayer(Player):
    def take_turn(self, board, time):
        while True:
            clock.sleep(0.01)


class BusyPlayer(Player):
    def take_turn(self, board, time):
        while True:
            pass


class CrashPlayer(Player):
    def take_turn(self, board, time):
        raise RuntimeError("oops")


class IllegalPlayer(Player):
    def take_turn(self, board, time):
        return (9, 9) if board[9][9] != -1 else "nonsense"


class ExitPlayer(Player):
    def take_turn(self, board, time):
        os._exit(3)


class BadStartPlayer(Player):
    def __init__(self, color):
        raise ValueError("cannot start")

    def take_turn(self, board, time):
        pass


class NoArgPlayer(Player):
    def __init__(self):
        super().__init__(None)

    def take_turn(self, board, time):
        return _first_empty(board)


class StatefulPlayer(Player):
    """Plays down column `color`, one row per turn, remembering the turn count."""

    def __init__(self, color):
        super().__init__(color)
        self.turns = 0

    def take_turn(self, board, time):
        move = (self.turns * 2 + self.color, 3 + 5 * self.color + (self.turns % 2) * 7)
        self.turns += 1
        print("turn %d, time %d" % (self.turns, time))
        return move


class SlowPlayer(Player):
    """Uses 120 ms per move."""

    def take_turn(self, board, time):
        clock.sleep(0.12)
        return _first_empty(board)


def _child_work(queue):
    queue.put(42)


class SpawnerPlayer(Player):
    """Starts a process of its own, which a daemon process would not be allowed to do."""

    def take_turn(self, board, time):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(target=_child_work, args=(queue,))
        child.start()
        assert queue.get(timeout=20) == 42
        child.join()
        return _first_empty(board)
