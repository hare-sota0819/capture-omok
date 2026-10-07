"""The interface every player implements."""

from abc import ABC, abstractmethod


class Player(ABC):
    """One side of a game. `color` is 0 for black (moves first) and 1 for white."""

    def __init__(self, color):
        self.color = color

    @abstractmethod
    def take_turn(self, board, time):
        """Return the point to play as (row, col).

        `board` is a 19 x 19 list of lists (-1 empty, 0 black, 1 white) and
        `time` is how many milliseconds this player has left for the game.
        """
