"""GameManager: one game of capture omok, with enforced clocks.

The manager never blocks inside a player's turn. Each player runs in its own process (see playerprocess.py); the manager is
driven by calling poll() regularly, which lets a GUI stay responsive and lets
the manager end a turn the moment the player's time runs out.
"""

import importlib
import os
import sys
import time as clock
from collections import namedtuple

from board import BLACK, WHITE, STONE_CHARS, Board, opponent_of
from humanplayer import HumanPlayer
from playerprocess import PlayerProcess

HUMAN, RANDOM, ALPHABETA, THREAT, MCTS, SMART = range(6)
PLAYER_CODES = (HUMAN, RANDOM, ALPHABETA, THREAT, MCTS, SMART)

# code -> (folder next to this one, module, class, name shown to the user)
COMPUTER_PLAYERS = {
    RANDOM: ("bots", "randombot", "RandomPlayer", "Random"),
    ALPHABETA: ("bots", "alphabeta", "AlphaBetaPlayer", "AlphaBeta"),
    THREAT: ("bots", "threat", "ThreatPlayer", "Threat"),
    MCTS: ("bots", "mcts", "MctsPlayer", "MCTS"),
    SMART: ("agent", "smartplayer", "SmartPlayer", "Smart"),
}

NO_TIME_LIMIT = -1
MS_PER_MINUTE = 60 * 1000
# How long past the deadline a move may still be in the pipe before the
# player is cut off. A move that arrives is always judged by its own timing.
KILL_GRACE_MS = 50
STARTUP_LIMIT_S = 30

COLOR_NAMES = {BLACK: "Black", WHITE: "White"}

GameResult = namedtuple("GameResult", ["winner", "reason"])


def _import_from(folder, module_name):
    # Appended, not inserted: the player processes get the same search path
    # and must find the same modules under the same names.
    if folder not in sys.path:
        sys.path.append(folder)
    return importlib.import_module(module_name)


def player_class_for_code(code):
    """The Player class behind one of PLAYER_CODES."""
    if code == HUMAN:
        return HumanPlayer
    if code not in COMPUTER_PLAYERS:
        raise ValueError("unknown player code: %r" % (code,))
    folder, module_name, class_name, _ = COMPUTER_PLAYERS[code]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return getattr(_import_from(os.path.join(root, folder), module_name), class_name)
    except (ImportError, AttributeError) as error:
        raise ValueError("player %d is not available: %s" % (code, error))


def player_class_from_file(text):
    """A Player class given as `path/to/file.py:ClassName`."""
    path, _, class_name = text.rpartition(":")
    if not path or not class_name or not path.endswith(".py"):
        raise ValueError("give a player file as path/to/file.py:ClassName")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError("no such file: %s" % path)
    module_name = os.path.splitext(os.path.basename(path))[0]
    try:
        module = _import_from(os.path.dirname(path), module_name)
    except ImportError as error:
        raise ValueError("could not load %s: %s" % (path, error))
    loaded_from = os.path.abspath(getattr(module, "__file__", None) or "")
    if os.path.normcase(loaded_from) != os.path.normcase(path):
        raise ValueError("another file called %s.py is already in use (%s); "
                         "rename yours" % (module_name, loaded_from))
    if not hasattr(module, class_name):
        raise ValueError("%s has no class called %s" % (path, class_name))
    return getattr(module, class_name)


def load_player(text):
    """A Player class from a command-line argument: a code or `file.py:ClassName`."""
    text = str(text)
    if text.lstrip("-").isdigit():
        return player_class_for_code(int(text))
    return player_class_from_file(text)


def player_label(player_class):
    """What to call a player class on screen."""
    if player_class is HumanPlayer:
        return "Human"
    for _, module_name, class_name, label in COMPUTER_PLAYERS.values():
        if player_class.__module__ == module_name and player_class.__name__ == class_name:
            return label
    return player_class.__name__


class GameListener:
    """Callbacks a front end can override to follow the game."""

    def turn_started(self, color):
        pass

    def player_output(self, color, text):
        """The player printed a line (e.g. HumanPlayer's remaining-time display)."""

    def input_requested(self, color, prompt):
        """The player called input() and is waiting for submit_input()."""

    def move_played(self, color, position, captured):
        pass

    def game_over(self, result):
        pass


class GameManager:
    _STARTING, _PLAYING, _OVER = "starting", "playing", "over"

    def __init__(self, black_class, white_class, time_limit_minutes=NO_TIME_LIMIT,
                 listener=None):
        self._classes = [black_class, white_class]
        self._listener = listener or GameListener()
        self._board = Board()
        if time_limit_minutes == NO_TIME_LIMIT:
            self._time_left_ms = None
        else:
            self._time_left_ms = [float(time_limit_minutes * MS_PER_MINUTE)] * 2
        self._processes = [None, None]
        self._ready = [False, False]
        self._state = None
        self._started_at = None
        self._current = BLACK
        self._turn_started_at = None
        self._awaiting_input = False
        self._move_count = 0
        self._stones_captured_by = [0, 0]
        self._result = None

    # ------------------------------------------------------------ queries

    @property
    def board(self):
        return self._board

    @property
    def result(self):
        return self._result

    @property
    def current_color(self):
        return self._current

    @property
    def move_count(self):
        return self._move_count

    @property
    def is_playing(self):
        return self._state == self._PLAYING

    @property
    def awaiting_input(self):
        """True while the current player is blocked in input()."""
        return self._state == self._PLAYING and self._awaiting_input

    def stones_captured_by(self, color):
        return self._stones_captured_by[color]

    def player_name(self, color):
        return self._classes[color].__name__

    def describe(self, color):
        return "%s (%s, %s)" % (COLOR_NAMES[color], STONE_CHARS[color], self.player_name(color))

    def time_left_ms(self, color):
        """Remaining time right now, counting the turn in progress. None = no limit."""
        if self._time_left_ms is None:
            return None
        left = self._time_left_ms[color]
        if color == self._current and self._turn_started_at is not None:
            left -= self._turn_elapsed_ms()
        return max(left, 0.0)

    # ------------------------------------------------------------ control

    def start(self):
        """Launch both players. The clocks start only once both are ready."""
        self._state = self._STARTING
        self._started_at = clock.perf_counter()
        for color in (BLACK, WHITE):
            self._processes[color] = PlayerProcess(self._classes[color], color)

    def poll(self):
        """Advance the game as far as it can go without waiting.

        Call this every few milliseconds. Returns the GameResult once the game
        is over, None before that.
        """
        if self._state == self._STARTING:
            self._poll_startup()
        elif self._state == self._PLAYING:
            self._poll_turn()
        return self._result

    def run(self, poll_interval_s=0.002):
        """Blocking convenience for use without a GUI."""
        if self._state is None:
            self.start()
        while self.poll() is None:
            clock.sleep(poll_interval_s)
        return self._result

    def submit_input(self, text):
        """Answer the current player's input() call (used for mouse clicks)."""
        if self.awaiting_input:
            self._awaiting_input = False
            self._processes[self._current].send_input(text)

    def abort(self):
        """Stop the game without a result (e.g. the window was closed)."""
        self._state = self._OVER
        self._turn_started_at = None
        self._stop_players()

    # ----------------------------------------------------------- internals

    def _poll_startup(self):
        for color in (BLACK, WHITE):
            for message in self._processes[color].messages():
                if message[0] == "ready":
                    self._ready[color] = True
                elif message[0] == "output":
                    self._listener.player_output(color, message[1])
                else:
                    detail = message[1] if message[0] == "error" else "process ended"
                    self._finish(GameResult(
                        opponent_of(color),
                        "%s could not be started (%s)." % (self.describe(color), detail),
                    ))
                    return
        if all(self._ready):
            self._state = self._PLAYING
            self._begin_turn(BLACK)
        elif clock.perf_counter() - self._started_at > STARTUP_LIMIT_S:
            slow = BLACK if not self._ready[BLACK] else WHITE
            self._finish(GameResult(
                opponent_of(slow), "%s took too long to start." % self.describe(slow)
            ))

    def _begin_turn(self, color):
        self._current = color
        self._awaiting_input = False
        self._listener.turn_started(color)
        self._turn_started_at = clock.perf_counter()
        self._processes[color].request_move(self._board.to_array(), self._time_argument(color))

    def _poll_turn(self):
        color = self._current
        rival = opponent_of(color)
        for message in self._processes[color].messages():
            kind = message[0]
            if kind == "output":
                self._listener.player_output(color, message[1])
            elif kind == "input_request":
                self._awaiting_input = True
                self._listener.input_requested(color, message[1])
            elif kind == "move":
                self._complete_turn(color, message[1], message[2], message[3])
                return
            elif kind == "error":
                self._finish(GameResult(
                    rival, "%s crashed (%s)." % (self.describe(color), message[1])
                ))
                return
            elif kind == "died":
                self._finish(GameResult(
                    rival, "%s stopped unexpectedly." % self.describe(color)
                ))
                return

        if (self._time_left_ms is not None
                and self._turn_elapsed_ms() > self._time_left_ms[color] + KILL_GRACE_MS):
            # Out of time in the middle of the turn: cut the player off now.
            self._processes[color].stop()
            self._time_left_ms[color] = 0.0
            self._finish(GameResult(rival, "%s ran out of time." % self.describe(color)))

    def _complete_turn(self, color, position, raw_move, elapsed_ms):
        rival = opponent_of(color)
        self._turn_started_at = None  # this turn's time is settled below
        if self._time_left_ms is not None:
            self._time_left_ms[color] -= elapsed_ms
            if self._time_left_ms[color] < 0:
                self._time_left_ms[color] = 0.0
                self._finish(GameResult(rival, "%s ran out of time." % self.describe(color)))
                return

        if position is None or not self._board.is_valid_move(*position):
            self._finish(GameResult(
                rival, "%s made an illegal move: %s." % (self.describe(color), raw_move)
            ))
            return

        captured = self._board.place(position[0], position[1], color)
        self._move_count += 1
        self._stones_captured_by[color] += len(captured)
        self._listener.move_played(color, position, captured)

        result = self._judge(color)
        if result is not None:
            self._finish(result)
        else:
            self._begin_turn(rival)

    def _judge(self, mover):
        """Victory check after `mover`'s stone and its captures are on the board."""
        rival = opponent_of(mover)
        mover_has_five = self._board.has_five(mover)
        rival_has_five = self._board.has_five(rival)
        if mover_has_five and rival_has_five:
            return GameResult(None, "Both players have five in a row.")
        if mover_has_five:
            return GameResult(mover, "%s made five in a row." % self.describe(mover))
        if rival_has_five:
            return GameResult(
                rival,
                "%s's capture left %s with exactly five in a row."
                % (COLOR_NAMES[mover], self.describe(rival)),
            )
        if self._board.is_full():
            return GameResult(None, "The board is full.")
        return None

    def _finish(self, result):
        self._state = self._OVER
        self._turn_started_at = None
        self._result = result
        self._stop_players()
        self._listener.game_over(result)

    def _stop_players(self):
        for process in self._processes:
            if process is not None:
                process.stop()

    def _turn_elapsed_ms(self):
        return (clock.perf_counter() - self._turn_started_at) * 1000.0

    def _time_argument(self, color):
        if self._time_left_ms is None:
            return NO_TIME_LIMIT
        return int(self._time_left_ms[color])
