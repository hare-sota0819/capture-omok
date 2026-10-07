"""Runs one Player in its own process so that its turn can be cut off.

A thread cannot be stopped from outside, but a process can be killed. Each
player therefore lives in a child process for the whole game (so players that
remember things between turns keep working) and talks to the GameManager over a
pipe. The Player classes themselves are used exactly as they are.

Inside the child, `input()` and `print()` are redirected to the pipe. That is
what lets the unmodified HumanPlayer be driven by mouse clicks: the click is
sent as the text HumanPlayer would otherwise have read from the keyboard.

Messages, parent -> child:   ("turn", board, time_ms)  ("input", text)  ("quit",)
Messages, child -> parent:   ("ready",)  ("output", text)  ("input_request", prompt)
                             ("move", position_or_None, raw_repr, elapsed_ms)
                             ("error", description)
The parent additionally reports ("died",) if the child vanished.
"""

import inspect
import multiprocessing
import operator
import time as clock

# "spawn" behaves the same on Windows, macOS and Linux, and is safe to use
# from a process that already has a GUI open.
_CONTEXT = multiprocessing.get_context("spawn")


def create_player(player_class, color):
    """Instantiate a Player, passing the colour only if the constructor takes it."""
    parameters = inspect.signature(player_class).parameters.values()
    takes_argument = any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in parameters
    )
    return player_class(color) if takes_argument else player_class()


class PlayerProcess:
    """Parent-side handle of a player running in a child process."""

    def __init__(self, player_class, color):
        self._connection, child_connection = _CONTEXT.Pipe()
        # Not a daemon: a daemon process may not start processes of its own,
        # which a player is free to do. stop() is what cleans up.
        self._process = _CONTEXT.Process(
            target=_serve_player, args=(child_connection, player_class, color)
        )
        self._process.start()
        child_connection.close()
        self._stopped = False

    def request_move(self, board, time_ms):
        self._send(("turn", board, time_ms))

    def send_input(self, text):
        self._send(("input", text))

    def messages(self):
        """Everything the child has sent so far, without waiting."""
        received = []
        if self._stopped:
            return received
        try:
            while self._connection.poll():
                received.append(self._connection.recv())
        except (EOFError, OSError):
            received.append(("died",))
        return received

    def stop(self):
        """End the child now, whatever it is doing."""
        if self._stopped:
            return
        self._stopped = True
        self._send(("quit",))
        self._process.join(0.05)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(1.0)
        self._connection.close()

    def _send(self, message):
        try:
            self._connection.send(message)
        except (OSError, ValueError):
            pass  # the child is gone; messages() reports that as ("died",)


# --------------------------------------------------------------------------
# Everything below runs inside the child process.


class _LineSender:
    """Stand-in for sys.stdout that forwards complete lines to the parent."""

    def __init__(self, send):
        self._send = send
        self._pending = ""

    def write(self, text):
        self._pending += str(text)
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._send("output", line)
        return len(text)

    def flush(self):
        if self._pending:
            self._send("output", self._pending)
            self._pending = ""

    def isatty(self):
        return False


def _describe(error):
    return "%s: %s" % (type(error).__name__, error)


def _as_position(move):
    """(row, col) as plain ints, or None if the player returned something else."""
    try:
        row, col = move
        return operator.index(row), operator.index(col)
    except (TypeError, ValueError):
        return None


def _serve_player(connection, player_class, color):
    import builtins
    import sys

    def send(*message):
        try:
            connection.send(message)
        except (OSError, ValueError):
            raise SystemExit(0)  # the parent is gone

    def receive():
        try:
            return connection.recv()
        except (EOFError, OSError):
            raise SystemExit(0)

    def piped_input(prompt=""):
        sys.stdout.flush()
        send("input_request", str(prompt))
        while True:
            message = receive()
            if message[0] == "input":
                return message[1]
            if message[0] == "quit":
                raise SystemExit(0)

    builtins.input = piped_input
    sys.stdout = _LineSender(send)

    try:
        player = create_player(player_class, color)
    except Exception as error:
        send("error", _describe(error))
        return
    send("ready")

    while True:
        message = receive()
        if message[0] == "quit":
            return
        if message[0] != "turn":
            continue
        _, board, time_ms = message
        start = clock.perf_counter()
        try:
            move = player.take_turn(board, time_ms)
        except Exception as error:
            sys.stdout.flush()
            send("error", _describe(error))
            continue
        # Timed here, next to the player, so the pipe and the manager's
        # polling interval are not charged to the player's clock.
        elapsed_ms = (clock.perf_counter() - start) * 1000.0
        sys.stdout.flush()
        send("move", _as_position(move), repr(move), elapsed_ms)
