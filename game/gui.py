"""Tkinter front end: opponent menu, clickable board, clocks and session record."""

import base64
import math
import sys
import time as clock
import tkinter as tk
import tkinter.font as tkfont

import sprites
from board import BLACK, BOARD_SIZE, EMPTY, WHITE, opponent_of
from gamemanager import (COLOR_NAMES, COMPUTER_PLAYERS, NO_TIME_LIMIT, GameListener,
                         GameManager, player_class_for_code, player_label)
from humanplayer import HumanPlayer
from scoreboard import DRAW, LOSS, WIN, Scoreboard

POLL_MS = 15
GLIDE_MS = 40           # how quickly the preview stone catches up with the cursor
FADE_MS = 260           # how long captured stones take to fade out
LOW_TIME_MS = 10000     # the clock turns red below this

# ----- palette
BG = "#16171b"
SURFACE = "#23252c"
SURFACE_HI = "#2f323b"
SURFACE_ACTIVE = "#32353f"
TEXT = "#eceef2"
TEXT_DIM = "#c3c7cf"
MUTED = "#8b919d"
ACCENT = "#f2b84b"
ACCENT_HI = "#ffd27f"
ON_ACCENT = "#1b1a17"
DANGER = "#ff6b6b"
GOOD = "#4cd98a"
BOARD_LINE = "#5e4419"
BOARD_LABEL = "#8a6a30"
BOARD_EDGE = "#8f6a35"
BANNER = "#101115"

STONE_SHADES = {BLACK: ((120, 120, 126), (10, 10, 12)),
                WHITE: ((255, 255, 255), (198, 196, 188))}
STAR_POINTS = (3, 9, 15)
FONT_FAMILIES = ("Segoe UI", "SF Pro Display", "Helvetica Neue", "Inter", "Noto Sans",
                 "DejaVu Sans")

TIME_CHOICES = (("No limit", NO_TIME_LIMIT), ("1 min", 1), ("3 min", 3), ("5 min", 5),
                ("10 min", 10))
COLOR_CHOICES = (("Black  (moves first)", BLACK), ("White", WHITE))


def opponent_choices():
    """(label, Player class) for everything the user can choose to play against."""
    choices = [("Human", HumanPlayer)]
    for code in sorted(COMPUTER_PLAYERS):
        try:
            choices.append((COMPUTER_PLAYERS[code][3], player_class_for_code(code)))
        except ValueError:
            pass
    return choices


def format_clock(time_ms):
    if time_ms is None:
        return "∞"
    if time_ms < LOW_TIME_MS:
        tenths = int(time_ms // 100)
        return "%d:%04.1f" % (tenths // 600, (tenths % 600) / 10.0)
    seconds = int(time_ms // 1000)
    return "%d:%02d" % (seconds // 60, seconds % 60)


def rgb(colour):
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


class ImageStore:
    """Creates each image once. Also keeps them alive: Tk drops unreferenced images."""

    def __init__(self, root):
        self._root = root
        self._images = {}

    def get(self, key, make):
        image = self._images.get(key)
        if image is None:
            width, height, rgba = make()
            data = base64.b64encode(sprites.encode_png(width, height, rgba))
            image = tk.PhotoImage(master=self._root, data=data, format="png")
            self._images[key] = image
        return image

    def rounded(self, width, height, radius, colour, alpha=255):
        return self.get(
            ("rounded", width, height, radius, colour, alpha),
            lambda: sprites.rounded_rect(width, height, radius, rgb(colour) + (alpha,)),
        )


class PillButton(tk.Canvas):
    """A flat rounded button with a hover state."""

    def __init__(self, app, parent, text, command, style="secondary", width=None,
                 height=None, font=None, bg=BG):
        font = font or app.font_bold
        height = height or app.px(38)
        if width is None:
            width = tkfont.Font(root=app.root, font=font).measure(text) + app.px(34)
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0,
                         bd=0, cursor="hand2")
        self._app = app
        self._command = command
        self._size = (width, height)
        self._styles = {
            "primary": (ACCENT, ACCENT_HI, ON_ACCENT),
            "selected": (ACCENT, ACCENT, ON_ACCENT),
            "secondary": (SURFACE, SURFACE_HI, TEXT),
            "ghost": (bg, SURFACE, MUTED),
        }
        self._style = style
        self._hover = False
        self._image = self.create_image(0, 0, anchor="nw")
        self._label = self.create_text(width // 2, height // 2, text=text, font=font)
        self.bind("<Enter>", lambda event: self._set_hover(True))
        self.bind("<Leave>", lambda event: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._on_release)
        self._paint()

    def set_style(self, style):
        if style != self._style:
            self._style = style
            self._paint()

    def invoke(self):
        self._command()

    def _set_hover(self, hover):
        self._hover = hover
        self._paint()

    def _on_release(self, event):
        width, height = self._size
        if 0 <= event.x < width and 0 <= event.y < height:
            self._command()

    def _paint(self):
        fill, hover_fill, text = self._styles[self._style]
        width, height = self._size
        image = self._app.images.rounded(width, height, self._app.px(9),
                                         hover_fill if self._hover else fill)
        self.itemconfig(self._image, image=image)
        self.itemconfig(self._label, fill=text)


class Segmented(tk.Frame):
    """A row of buttons of which exactly one is selected."""

    def __init__(self, app, parent, variable, choices, bg=BG):
        super().__init__(parent, bg=bg)
        self._variable = variable
        self._buttons = []
        for label, value in choices:
            button = PillButton(app, self, label, lambda v=value: self.select(v),
                                font=app.font_body, bg=bg)
            button.pack(side="left", padx=(0, app.px(8)))
            self._buttons.append((value, button))
        self.refresh()

    def select(self, value):
        self._variable.set(value)
        self.refresh()

    def refresh(self):
        chosen = self._variable.get()
        for value, button in self._buttons:
            button.set_style("selected" if value == chosen else "secondary")


class PlayerCard(tk.Canvas):
    """Name, clock, time bar and capture count of one player."""

    def __init__(self, app, parent, color):
        px = app.px
        self._app = app
        self._width, self._height = app.panel_width, px(86)
        super().__init__(parent, width=self._width, height=self._height, bg=BG,
                         highlightthickness=0, bd=0)
        self._active = None
        self._background = self.create_image(0, 0, anchor="nw")
        self._stripe = self.create_image(
            px(8), self._height // 2, anchor="w", state="hidden",
            image=app.images.rounded(px(4), self._height - px(30), px(2), ACCENT))
        self.create_image(px(36), px(36), image=app.stone_image(color, px(13)))
        self._name = self.create_text(px(62), px(27), anchor="w", font=app.font_bold,
                                      fill=TEXT)
        self._detail = self.create_text(px(62), px(47), anchor="w", font=app.font_small,
                                        fill=MUTED)
        self._clock = self.create_text(self._width - px(16), px(36), anchor="e",
                                       font=app.font_clock, fill=TEXT)
        self._bar_box = (px(18), self._height - px(15), self._width - px(18),
                         self._height - px(11))
        self._track = self.create_rectangle(*self._bar_box, fill=SURFACE_HI, outline="")
        self._bar = self.create_rectangle(*self._bar_box, fill=ACCENT, outline="")
        self.set_active(False)

    def set_name(self, text):
        self.itemconfig(self._name, text=text)

    def set_active(self, active):
        if active == self._active:
            return
        self._active = active
        self.itemconfig(self._background, image=self._app.images.rounded(
            self._width, self._height, self._app.px(12),
            SURFACE_ACTIVE if active else SURFACE))
        self.itemconfig(self._stripe, state="normal" if active else "hidden")
        self.itemconfig(self._track, fill="#40444f" if active else SURFACE_HI)

    def show(self, captured, time_left_ms, total_ms):
        self.itemconfig(self._detail, text="captured %d" % captured)
        low = time_left_ms is not None and time_left_ms < LOW_TIME_MS
        self.itemconfig(self._clock, text=format_clock(time_left_ms),
                        fill=DANGER if low else TEXT)
        left, top, right, bottom = self._bar_box
        if time_left_ms is None or not total_ms:
            self.itemconfig(self._track, state="hidden")
            self.itemconfig(self._bar, state="hidden")
            return
        fraction = max(0.0, min(1.0, time_left_ms / float(total_ms)))
        self.itemconfig(self._track, state="normal")
        self.itemconfig(self._bar, state="normal", fill=DANGER if low else ACCENT)
        self.coords(self._bar, left, top, left + (right - left) * fraction, bottom)


class OmokApp(GameListener):
    def __init__(self, root):
        self.root = root
        self.scoreboard = Scoreboard()
        self.manager = None
        self._tick_id = None
        self._game = None            # settings of the game on screen, for "Play again"
        self._total_ms = None
        self._last_move = None
        self._last_captured = ()
        self._fade = None            # (start time, colour, positions) of stones fading out
        self._winning_lines = ()
        self._hover = None
        self._ghost_at = None        # where the preview stone is drawn right now
        self._ghost_time = 0.0
        self._input_sent_this_turn = False

        root.title("Capture Omok")
        root.configure(bg=BG)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.exit)

        self._scale = root.winfo_fpixels("1i") / 96.0
        self._cell = max(20, min(self.px(34),
                                 (root.winfo_screenheight() - self.px(150)) // 21))
        self._margin = int(self._cell * 1.3)
        self._board_px = 2 * self._margin + (BOARD_SIZE - 1) * self._cell
        self._board_edge = self.px(7)       # visible thickness of the board's front edge
        self._stone_radius = self._cell * 0.47
        self.panel_width = self.px(310)
        pad = self.px(20)
        root.geometry("%dx%d" % (3 * pad + self._board_px + self.panel_width,
                                 2 * pad + self._board_px + self._board_edge))

        available = set(tkfont.families(root))
        family = next((name for name in FONT_FAMILIES if name in available),
                      tkfont.nametofont("TkDefaultFont").actual("family"))
        self.font_body = (family, 10)
        self.font_bold = (family, 10, "bold")
        self.font_small = (family, 9)
        self.font_caption = (family, 8, "bold")
        self.font_heading = (family, 15, "bold")
        self.font_title = (family, 30, "bold")
        self.font_clock = (family, 19, "bold")

        self.images = ImageStore(root)
        self._opponents = opponent_choices()
        self._opponent_var = tk.IntVar(value=len(self._opponents) - 1)  # the strongest
        self._color_var = tk.IntVar(value=BLACK)
        self._time_var = tk.IntVar(value=NO_TIME_LIMIT)

        self.menu_frame = tk.Frame(root, bg=BG)
        self.game_frame = tk.Frame(root, bg=BG)
        self._build_menu()
        self._build_game(pad)
        self.show_menu()

    def px(self, value):
        """Scale a size given for a 96-dpi screen to this screen."""
        return int(round(value * self._scale))

    def stone_image(self, color, radius, alpha=1.0):
        light, dark = STONE_SHADES[color]
        solid = lambda: sprites.stone(radius, light, dark)  # noqa: E731
        if alpha >= 1.0:
            return self.images.get(("stone", color, radius), solid)
        return self.images.get(("stone", color, radius, alpha),
                               lambda: sprites.with_alpha(solid(), alpha))

    # ================================================================= menu

    def _build_menu(self):
        px = self.px
        columns = tk.Frame(self.menu_frame, bg=BG)
        columns.place(relx=0.5, rely=0.5, anchor="center")
        inner = tk.Frame(columns, bg=BG)
        inner.pack(side="left")

        logo = tk.Canvas(inner, width=px(96), height=px(60), bg=BG, highlightthickness=0)
        logo.create_image(px(34), px(30), image=self.stone_image(BLACK, px(21)))
        logo.create_image(px(62), px(30), image=self.stone_image(WHITE, px(21)))
        logo.pack()
        tk.Label(inner, text="Capture Omok", font=self.font_title, bg=BG, fg=TEXT).pack()
        tk.Label(inner, text="Exactly five in a row wins.  Flank two stones to capture them.",
                 font=self.font_body, bg=BG, fg=MUTED).pack(pady=(0, px(22)))

        self._segments = []
        groups = (
            ("OPPONENT", self._opponent_var,
             [(label, index) for index, (label, _) in enumerate(self._opponents)]),
            ("YOUR STONES", self._color_var, COLOR_CHOICES),
            ("TIME PER PLAYER", self._time_var, TIME_CHOICES),
        )
        for caption, variable, choices in groups:
            tk.Label(inner, text=caption, font=self.font_caption, bg=BG, fg=MUTED).pack(
                anchor="w", pady=(0, px(6)))
            segment = Segmented(self, inner, variable, choices)
            segment.pack(anchor="w", pady=(0, px(18)))
            self._segments.append(segment)

        inner.update_idletasks()
        self._menu_width = max(px(440), max(s.winfo_reqwidth() for s in self._segments))
        buttons = tk.Frame(inner, bg=BG)
        buttons.pack(anchor="w", pady=(px(4), 0))
        exit_width = px(90)
        self._start_button = PillButton(
            self, buttons, "Start game", self._start_from_menu, style="primary",
            width=self._menu_width - exit_width - px(10), height=px(44))
        self._start_button.pack(side="left", padx=(0, px(10)))
        PillButton(self, buttons, "Exit", self.exit, width=exit_width,
                   height=px(44)).pack(side="left")

        self._record_holder = tk.Frame(columns, bg=BG)
        self._record_holder.pack(side="left", padx=(px(40), 0))

    def _refresh_record(self):
        px = self.px
        for child in self._record_holder.winfo_children():
            child.destroy()
        played = self.scoreboard.games_played() > 0
        table = []
        if played:
            table = self.scoreboard.rows() + [("Total",) + self.scoreboard.totals()]
        line = px(26)
        width, height = px(300), px(54) + line * (len(table) + 1 if played else 1)
        canvas = tk.Canvas(self._record_holder, width=width, height=height, bg=BG,
                           highlightthickness=0)
        canvas.create_image(0, 0, anchor="nw",
                            image=self.images.rounded(width, height, px(12), SURFACE))
        canvas.create_text(px(18), px(22), anchor="w", text="SESSION RECORD",
                           font=self.font_caption, fill=MUTED)
        top = px(52)
        if not played:
            canvas.create_text(px(18), top, anchor="w", text="No games played yet.",
                               font=self.font_body, fill=TEXT_DIM)
        else:
            columns = [width - px(134), width - px(76), width - px(18)]
            for x, title in zip(columns, ("Won", "Lost", "Drawn")):
                canvas.create_text(x, top, anchor="e", text=title, font=self.font_small,
                                   fill=MUTED)
            for index, row in enumerate(table):
                y = top + line * (index + 1)
                total = index == len(table) - 1
                font = self.font_bold if total else self.font_body
                canvas.create_text(px(18), y, anchor="w", text=row[0], font=font,
                                   fill=TEXT if total else TEXT_DIM)
                for x, value in zip(columns, row[1:]):
                    canvas.create_text(x, y, anchor="e", text=str(value), font=font,
                                       fill=TEXT)
        canvas.pack()

    def show_menu(self):
        self._stop_game()
        self.game_frame.pack_forget()
        for segment in self._segments:
            segment.refresh()
        self._refresh_record()
        self.menu_frame.pack(fill="both", expand=True)

    def _start_from_menu(self):
        label, opponent_class = self._opponents[self._opponent_var.get()]
        you = self._color_var.get()
        classes = [None, None]
        classes[you] = HumanPlayer
        classes[opponent_of(you)] = opponent_class
        self.start_game(classes[BLACK], classes[WHITE], self._time_var.get(), you, label)

    # ================================================================= game

    def _build_game(self, pad):
        px = self.px
        frame = self.game_frame
        self.canvas = tk.Canvas(frame, width=self._board_px,
                                height=self._board_px + self._board_edge, bg=BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack(side="left", padx=(pad, 0), pady=pad, anchor="n")
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda event: self._set_hover(None))
        self.canvas.bind("<Button-1>", self._on_click)
        self._draw_board()

        panel = tk.Frame(frame, bg=BG, width=self.panel_width)
        panel.pack(side="left", fill="y", padx=pad, pady=pad)
        panel.pack_propagate(False)

        self._cards = {}
        for color in (BLACK, WHITE):
            self._cards[color] = PlayerCard(self, panel, color)
            self._cards[color].pack(pady=(0, px(10)))

        self._status = tk.Label(panel, text="", font=self.font_heading, bg=BG, fg=TEXT,
                                wraplength=self.panel_width, justify="left", anchor="w")
        self._status.pack(fill="x", pady=(px(8), 0))
        self._message = tk.Label(panel, text="", font=self.font_body, bg=BG, fg=MUTED,
                                 wraplength=self.panel_width, justify="left", anchor="w")
        self._message.pack(fill="x", pady=(px(2), px(10)))

        # Packed from the bottom up so the move list takes whatever space is left.
        self._buttons = tk.Frame(panel, bg=BG)
        self._buttons.pack(side="bottom", fill="x")
        self._tally = tk.Label(panel, text="", font=self.font_small, bg=BG, fg=MUTED,
                               anchor="w")
        self._tally.pack(side="bottom", fill="x", pady=(px(10), px(8)))

        self._log = tk.Text(panel, width=1, height=1, font=self.font_small, bg=SURFACE,
                            fg=TEXT_DIM, bd=0, relief="flat", state="disabled", wrap="word",
                            padx=px(12), pady=px(10), spacing1=px(2), cursor="arrow",
                            highlightthickness=0, insertwidth=0, selectbackground=SURFACE_HI,
                            selectforeground=TEXT)
        self._log.tag_configure("note", foreground=MUTED)
        self._log.tag_configure("result", foreground=ACCENT, font=self.font_bold)
        self._log.pack(fill="both", expand=True)

    def start_game(self, black_class, white_class, minutes, you, opponent_label):
        """Begin a game. `you` is the user's colour, or None when only watching."""
        self._stop_game()
        self._game = (black_class, white_class, minutes, you, opponent_label)
        self._total_ms = None if minutes == NO_TIME_LIMIT else minutes * 60000.0
        self._last_move = None
        self._last_captured = ()
        self._fade = None
        self._winning_lines = ()
        self._hover = None
        self._input_sent_this_turn = False

        self.manager = GameManager(black_class, white_class, minutes, listener=self)
        versus_human = black_class is HumanPlayer and white_class is HumanPlayer
        for color in (BLACK, WHITE):
            name = self.manager.player_name(color)
            if you == color and not versus_human:
                name = "You"
            elif name == "HumanPlayer":
                name = "Human"
            self._cards[color].set_name("%s  ·  %s" % (COLOR_NAMES[color], name))
        self._set_log("")
        self._set_status("Starting the players…")
        self._set_message("")
        self._show_buttons(("Quit to menu", self.show_menu, "secondary"))
        self._refresh_tally()
        self._refresh_panel()
        self.canvas.delete("banner")
        self._draw_position()

        self.menu_frame.pack_forget()
        self.game_frame.pack(fill="both", expand=True)
        self.manager.start()
        self._tick()

    def _tick(self):
        self._tick_id = None
        if self.manager is None:
            return
        self.manager.poll()
        self._refresh_panel()
        if self._fade is not None:
            self._draw_fade()
        self._draw_hover()
        if self.manager is not None and (self.manager.result is None
                                         or self._fade is not None):
            self._tick_id = self.root.after(POLL_MS, self._tick)

    def _stop_game(self):
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        if self.manager is not None:
            self.manager.abort()
            self.manager = None

    def exit(self):
        self._stop_game()
        self.root.destroy()

    # ------------------------------------------------- GameListener callbacks

    def turn_started(self, color):
        self._input_sent_this_turn = False
        self._set_message("")
        if self._game[color] is HumanPlayer:
            self._set_status("%s to move" % COLOR_NAMES[color])
        else:
            self._set_status("%s is thinking…" % self.manager.player_name(color))

    def player_output(self, color, text):
        text = text.strip()
        if not text:
            return
        self._append_log("%s: %s\n" % (COLOR_NAMES[color], text), "note")
        # Whatever a player prints after being given a click is its answer to
        # that click (HumanPlayer's "Invalid move..."), so keep it in view.
        if self._input_sent_this_turn:
            self._set_message(text, DANGER)

    def input_requested(self, color, prompt):
        who = "Your turn" if self._game[3] == color else "%s's turn" % COLOR_NAMES[color]
        self._set_status(who)
        if not self._input_sent_this_turn:
            self._set_message("Click a point to place a stone.")
        self._draw_hover()

    def move_played(self, color, position, captured):
        self._last_move = position
        self._last_captured = tuple(captured)
        self._fade = None
        if captured:
            self._fade = (clock.perf_counter(), opponent_of(color), tuple(captured))
        self._set_message("")
        line = "%d.  %s %d,%d" % (self.manager.move_count, COLOR_NAMES[color],
                                  position[0], position[1])
        if captured:
            line += "   captures " + "  ".join("%d,%d" % rc for rc in sorted(captured))
        self._append_log(line + "\n")
        self._draw_position()

    def game_over(self, result):
        board = self.manager.board
        self._winning_lines = tuple(board.find_fives(BLACK)) + tuple(board.find_fives(WHITE))
        self._hover = None
        self._draw_position()

        you, opponent_label = self._game[3], self._game[4]
        versus_human = self._game[0] is HumanPlayer and self._game[1] is HumanPlayer
        colour = TEXT
        if result.winner is None:
            headline, outcome = "Draw", DRAW
        elif you is None or versus_human:
            headline = "%s wins" % COLOR_NAMES[result.winner]
            outcome = WIN if result.winner == you else LOSS
        elif result.winner == you:
            headline, outcome, colour = "You win!", WIN, GOOD
        else:
            headline, outcome, colour = "You lose", LOSS, DANGER
        if you is not None:
            self.scoreboard.record(opponent_label, outcome)

        self._set_status(headline, colour)
        self._set_message(result.reason)
        self._refresh_tally()
        self._show_buttons(
            ("Play again", lambda: self.start_game(*self._game), "primary"),
            ("Choose opponent", self.show_menu, "secondary"),
            ("Exit", self.exit, "secondary"),
        )
        # The extra buttons shrink the log, so scroll it only once that is laid out.
        self.root.update_idletasks()
        self._append_log("%s  %s\n" % (headline, result.reason), "result")
        self._draw_banner(headline, result.reason, colour)

    # ------------------------------------------------------------ side panel

    def _set_status(self, text, colour=TEXT):
        self._status.config(text=text, fg=colour)

    def _set_message(self, text, colour=MUTED):
        self._message.config(text=text, fg=colour)

    def _set_log(self, text):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.insert("end", text)
        self._log.config(state="disabled")

    def _append_log(self, text, tag=None):
        self._log.config(state="normal")
        self._log.insert("end", text, (tag,) if tag else ())
        self._log.see("end")
        self._log.config(state="disabled")

    def _show_buttons(self, *buttons):
        for child in self._buttons.winfo_children():
            child.destroy()
        for text, command, style in buttons:
            PillButton(self, self._buttons, text, command, style=style,
                       width=self.panel_width, height=self.px(40)).pack(pady=(self.px(8), 0))

    def _refresh_tally(self):
        wins, losses, draws = self.scoreboard.totals()
        self._tally.config(
            text="SESSION     %d won  ·  %d lost  ·  %d drawn" % (wins, losses, draws))

    def _refresh_panel(self):
        manager = self.manager
        if manager is None:
            return
        active = manager.current_color if manager.is_playing else None
        for color in (BLACK, WHITE):
            card = self._cards[color]
            card.set_active(color == active)
            card.show(manager.stones_captured_by(color), manager.time_left_ms(color),
                      self._total_ms)

    # ----------------------------------------------------------------- board

    def _xy(self, row, col):
        return self._margin + col * self._cell, self._margin + row * self._cell

    def _point_at(self, event):
        col = int(round((event.x - self._margin) / float(self._cell)))
        row = int(round((event.y - self._margin) / float(self._cell)))
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return (row, col)
        return None

    def _draw_board(self):
        canvas = self.canvas
        size, radius = self._board_px, self.px(14)
        # A darker copy shifted down reads as the front edge of a thick board.
        canvas.create_image(0, self._board_edge, anchor="nw",
                            image=self.images.rounded(size, size, radius, BOARD_EDGE))
        canvas.create_image(0, 0, anchor="nw", image=self.images.get(
            ("wood", size, radius), lambda: sprites.wood(size, size, radius)))

        first, last = self._margin, self._margin + (BOARD_SIZE - 1) * self._cell
        label_font = (self.font_small[0], 8)
        label_gap = self._cell * 0.82
        for index in range(BOARD_SIZE):
            offset = self._margin + index * self._cell
            canvas.create_line(first, offset, last, offset, fill=BOARD_LINE)
            canvas.create_line(offset, first, offset, last, fill=BOARD_LINE)
            canvas.create_text(offset, first - label_gap, text=str(index), fill=BOARD_LABEL,
                               font=label_font)
            canvas.create_text(first - label_gap, offset, text=str(index), fill=BOARD_LABEL,
                               font=label_font)
        canvas.create_rectangle(first, first, last, last, outline=BOARD_LINE, width=2)
        dot_radius = max(2.0, self._cell * 0.1)
        dot = self.images.get(("star", dot_radius),
                              lambda: sprites.disc(dot_radius, rgb(BOARD_LINE)))
        for row in STAR_POINTS:
            for col in STAR_POINTS:
                canvas.create_image(*self._xy(row, col), image=dot)

    def _draw_position(self):
        canvas = self.canvas
        canvas.delete("piece")
        canvas.delete("fade")
        if self.manager is None:
            return
        board = self.manager.board
        radius = self._stone_radius
        images = {color: self.stone_image(color, radius) for color in (BLACK, WHITE)}
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stone = board.get(row, col)
                if stone != EMPTY:
                    canvas.create_image(*self._xy(row, col), image=images[stone],
                                        tags="piece")
        # Points just emptied by a capture keep a small red ring until the next move.
        mark_radius = radius * 0.3
        mark = self.images.get(
            ("captured", mark_radius),
            lambda: sprites.ring(mark_radius, max(1.6, self._scale * 1.8), rgb(DANGER)))
        for row, col in self._last_captured:
            canvas.create_image(*self._xy(row, col), image=mark, tags="piece")
        if self._last_move is not None:
            dot_radius = radius * 0.22
            dot = self.images.get(("last", dot_radius),
                                  lambda: sprites.disc(dot_radius, (255, 92, 80)))
            canvas.create_image(*self._xy(*self._last_move), image=dot, tags="piece")
        if self._winning_lines:
            glow = self.images.get(
                ("win", radius),
                lambda: sprites.ring(radius + self.px(1), self.px(3), rgb(GOOD),
                                     glow=self.px(5)))
            for line in self._winning_lines:
                for row, col in line:
                    canvas.create_image(*self._xy(row, col), image=glow, tags="piece")
        self._draw_fade()
        self._draw_hover()
        canvas.tag_raise("hover")
        canvas.tag_raise("banner")

    def _draw_fade(self):
        """Captured stones linger for a moment, getting fainter."""
        self.canvas.delete("fade")
        if self._fade is None:
            return
        started, color, positions = self._fade
        progress = (clock.perf_counter() - started) * 1000.0 / FADE_MS
        if progress >= 1.0:
            self._fade = None
            return
        alpha = (0.8, 0.6, 0.4, 0.2)[min(3, int(progress * 4))]
        image = self.stone_image(color, self._stone_radius, alpha)
        for row, col in positions:
            self.canvas.create_image(*self._xy(row, col), image=image, tags="fade")
        self.canvas.tag_raise("banner")

    def _draw_banner(self, headline, reason, colour):
        """Result strip across the board, placed away from the winning stones."""
        canvas = self.canvas
        canvas.delete("banner")
        px = self.px
        width, height = self._board_px - 2 * self._margin + self._cell, px(96)
        rows = [row for line in self._winning_lines for row, _ in line]
        winning_row = sum(rows) / float(len(rows)) if rows else 0.0
        centre_row = 14.5 if winning_row < BOARD_SIZE / 2.0 else 3.5
        x, y = self._board_px // 2, int(self._margin + centre_row * self._cell)
        canvas.create_image(x, y, tags="banner", image=self.images.rounded(
            width, height, px(14), BANNER, alpha=228))
        canvas.create_text(x, y - px(16), text=headline, font=self.font_heading, fill=colour,
                           tags="banner")
        canvas.create_text(x, y + px(18), text=reason, font=self.font_body, fill=TEXT_DIM,
                           width=width - px(40), justify="center", tags="banner")

    def _can_click(self, point):
        return (point is not None and self.manager is not None
                and self.manager.awaiting_input
                and self.manager.board.get(*point) == EMPTY)

    def _set_hover(self, point):
        if point != self._hover:
            self._hover = point
            self._draw_hover()

    def _draw_hover(self):
        """Preview stone at the point under the cursor.

        Jumping from point to point looks like stutter, so the stone glides
        towards its target a little every frame instead.
        """
        canvas = self.canvas
        if not self._can_click(self._hover):
            canvas.delete("hover")
            self._ghost_at = None
            return
        target = self._xy(*self._hover)
        now = clock.perf_counter()
        if self._ghost_at is None or not canvas.find_withtag("hover"):
            canvas.delete("hover")
            image = self.stone_image(self.manager.current_color, self._stone_radius, 0.5)
            canvas.create_image(target[0], target[1], image=image, tags="hover")
            canvas.tag_raise("banner")
            self._ghost_at, self._ghost_time = target, now
            return
        x, y = self._ghost_at
        if (x, y) != target:
            blend = 1.0 - math.exp(-(now - self._ghost_time) * 1000.0 / GLIDE_MS)
            x += (target[0] - x) * blend
            y += (target[1] - y) * blend
            if abs(target[0] - x) < 0.6 and abs(target[1] - y) < 0.6:
                x, y = target
            canvas.coords("hover", x, y)
            self._ghost_at = (x, y)
        self._ghost_time = now

    def _on_motion(self, event):
        self._set_hover(self._point_at(event))

    def _on_click(self, event):
        manager = self.manager
        if manager is not None and manager.result is not None:
            self.canvas.delete("banner")   # click to look at the final position
            return
        point = self._point_at(event)
        if point is None or manager is None or not manager.awaiting_input:
            return
        # The click is handed to the player as the text it would have read
        # from the keyboard; HumanPlayer itself decides whether it is valid.
        self._input_sent_this_turn = True
        manager.submit_input("%d,%d" % point)
        self._draw_hover()


def enable_sharp_rendering():
    """Ask Windows not to blur the window on high-DPI screens."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def run(initial_game=None):
    """Open the window. `initial_game` = (black_class, white_class, minutes) to skip the menu."""
    enable_sharp_rendering()
    root = tk.Tk()
    app = OmokApp(root)
    if initial_game is not None:
        black_class, white_class, minutes = initial_game
        if black_class is HumanPlayer:
            you = BLACK
        elif white_class is HumanPlayer:
            you = WHITE
        else:
            you = None
        rival = white_class if you in (BLACK, None) else black_class
        label = player_label(rival)
        app.start_game(black_class, white_class, minutes, you, label)
    root.mainloop()
