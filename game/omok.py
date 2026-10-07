"""Capture omok in a window.

    python omok.py                                  open the menu
    python omok.py [player-1] [player-2] [time-limit]   go straight into that game

A player is a code (0 human, 1 Random, 2 AlphaBeta, 3 Threat, 4 MCTS, 5 Smart)
or a file of your own, written as path/to/file.py:ClassName.
time-limit: minutes per player for the whole game, or -1 for no limit
"""

import argparse
import sys

from gamemanager import NO_TIME_LIMIT, load_player


def _time_limit(text):
    value = int(text)
    if value != NO_TIME_LIMIT and value <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive number of minutes, or -1 for no limit"
        )
    return value


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Play capture omok in a window. Without arguments the menu opens."
    )
    codes = ("0 human, 1 Random, 2 AlphaBeta, 3 Threat, 4 MCTS, 5 Smart, "
             "or path/to/file.py:ClassName")
    parser.add_argument("player1", nargs="?", metavar="player-1",
                        help="first player, black (%s)" % codes)
    parser.add_argument("player2", nargs="?", metavar="player-2",
                        help="second player, white (%s)" % codes)
    parser.add_argument("time_limit", nargs="?", type=_time_limit, metavar="time-limit",
                        help="minutes per player for the whole game; -1 for no limit")
    args = parser.parse_args(argv)
    given = [value is not None for value in (args.player1, args.player2, args.time_limit)]
    if any(given) and not all(given):
        parser.error("give all of player-1, player-2 and time-limit, or none of them")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    initial_game = None
    if args.player1 is not None:
        try:
            initial_game = (load_player(args.player1), load_player(args.player2),
                            args.time_limit)
        except ValueError as error:
            sys.exit("omok.py: %s" % error)

    # Imported here, not at the top: the player processes re-import this file
    # when they start and have no use for the GUI.
    import gui
    gui.run(initial_game)
    return 0


# The guard matters: each player runs in a child process that imports this file.
if __name__ == "__main__":
    sys.exit(main())
