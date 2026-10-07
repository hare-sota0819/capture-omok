"""Win/loss/draw counts across the games of one session."""

WIN, LOSS, DRAW = "win", "loss", "draw"


class Scoreboard:
    def __init__(self):
        self._by_opponent = {}  # opponent name -> {WIN: n, LOSS: n, DRAW: n}

    def record(self, opponent_name, outcome):
        counts = self._by_opponent.setdefault(opponent_name, {WIN: 0, LOSS: 0, DRAW: 0})
        counts[outcome] += 1

    def totals(self):
        """(wins, losses, draws) over all opponents."""
        return tuple(
            sum(counts[outcome] for counts in self._by_opponent.values())
            for outcome in (WIN, LOSS, DRAW)
        )

    def rows(self):
        """[(opponent name, wins, losses, draws)] in the order first played."""
        return [
            (name, counts[WIN], counts[LOSS], counts[DRAW])
            for name, counts in self._by_opponent.items()
        ]

    def games_played(self):
        return sum(self.totals())
