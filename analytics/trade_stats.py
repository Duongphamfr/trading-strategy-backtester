"""Statistics computed from the trade log rather than from the equity curve.

Every other module in analytics measures the portfolio's value over time. This
one measures the decisions: how many positions were opened and closed, how many
of them made money, and how the winners compare with the losers. It is the only
place that connects directly to the number of trades, which is also the number
that transaction costs will multiply once Phase 4 charges them.

THE ONE THING TO UNDERSTAND ABOUT THIS GROUP
A high win rate does not mean a good strategy, and the belief that it does is
probably the most common mistake in retail trading. A strategy can win on 90% of
its trades and still lose money, if the losing 10% are large enough to swallow
all the small gains. Selling insurance against rare disasters looks exactly like
this, and so does any rule that takes profits quickly while letting losses run.

That is precisely why win_rate is never reported alone here. average_win against
average_loss shows the asymmetry, and profit_factor collapses the whole question
into one number: below 1 the strategy lost money regardless of how often it was
right. The __main__ block below contains a 90%-win-rate strategy with a profit
factor of 0.45, purely to make the point concrete.
"""

from typing import Any, Dict, List, NamedTuple

import numpy as np
import pandas as pd

from constants import BUY, SELL

ROUND_TRIP_COLUMNS = [
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "shares",
    "profit",
    "return_pct",
]

REQUIRED_TRADE_KEYS = ("date", "action", "price", "shares")


class TradeStatistics(NamedTuple):
    """Summary of a trade log, computed over completed round-trips.

    Attributes:
        number_of_trades: Count of completed round-trips, each a BUY matched
            with the SELL that closed it. An unclosed final BUY is not counted.
        win_rate: Fraction of completed round-trips with a profit strictly above
            zero, between 0 and 1. Never read this without the two below.
        average_win: Mean profit of the winning round-trips, in currency units.
            Positive, or NaN when nothing won.
        average_loss: Mean profit of the losing round-trips, in currency units.
            NEGATIVE, since it is the mean of actual losses rather than their
            magnitude, or NaN when nothing lost.
        profit_factor: Gross profit divided by gross loss. Above 1 means the
            winners outweighed the losers and the strategy was net profitable.
        gross_profit: Sum of all winning profits, positive.
        gross_loss: Sum of all losing profits as a POSITIVE magnitude, which is
            the denominator of profit_factor.
        open_trades: Positions still open when the log ends, excluded from every
            statistic above. Normally 0 or 1 under the all-in/all-out model.
    """

    number_of_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    gross_profit: float
    gross_loss: float
    open_trades: int


def round_trips(trade_log: List[Dict[str, Any]]) -> pd.DataFrame:
    """Pair each BUY with the SELL that closed it.

    The Phase 1 execution model is all-in / all-out: a BUY invests the whole
    portfolio and the next SELL liquidates it, so trades alternate and pairing is
    unambiguous. The scan below is a small state machine rather than a vectorised
    expression, because the pairing is inherently sequential and trade logs are
    short. Vectorising it would cost clarity and buy nothing.

    Cases handled defensively, so that a future strategy or a partially
    implemented one cannot produce silently wrong statistics:
        - A SELL with no position open is ignored. It cannot reach the log from
          the current backtester, which only records executed trades, but a
          strategy emitting SELL while flat is perfectly normal.
        - A BUY while a position is already open is ignored, since under this
          model there is nothing to add to.
        - A BUY still open when the log ends produces no round-trip. It is
          reported separately as an open trade instead of being closed at an
          arbitrary price, which would invent a result the backtest never had.

    Profit convention:
        profit     = (exit_price - entry_price) * shares
        return_pct = exit_price / entry_price - 1

    Profit is gross: no transaction costs, since Phase 1 charges none. When
    Phase 4 adds them inside the Broker, they will already be embedded in the
    logged execution prices and will flow through here automatically.

    Args:
        trade_log: Executed trades in chronological order, each a dict with the
            keys date, action, price and shares, as produced by the backtester.

    Returns:
        A DataFrame with one row per completed round-trip and the columns
        entry_date, exit_date, entry_price, exit_price, shares, profit and
        return_pct. Empty, with those same columns, when nothing closed.

    Raises:
        ValueError: If a trade is missing one of the required keys.
    """
    completed = []
    open_entry = None

    for trade in trade_log or []:
        missing = [key for key in REQUIRED_TRADE_KEYS if key not in trade]
        if missing:
            raise ValueError(
                f"Trade log entry is missing required keys: "
                f"{', '.join(missing)}. Got {sorted(trade)}."
            )

        if trade["action"] == BUY:
            if open_entry is None:
                open_entry = trade
        elif trade["action"] == SELL and open_entry is not None:
            entry_price = float(open_entry["price"])
            exit_price = float(trade["price"])
            shares = float(trade["shares"])

            completed.append(
                {
                    "entry_date": open_entry["date"],
                    "exit_date": trade["date"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "profit": (exit_price - entry_price) * shares,
                    "return_pct": (
                        exit_price / entry_price - 1.0
                        if entry_price > 0
                        else float("nan")
                    ),
                }
            )
            open_entry = None

    if not completed:
        return pd.DataFrame(columns=ROUND_TRIP_COLUMNS)

    return pd.DataFrame(completed)[ROUND_TRIP_COLUMNS]


def trade_statistics(trade_log: List[Dict[str, Any]]) -> TradeStatistics:
    """Every trade-level statistic, computed from completed round-trips.

    One function returning one structured result, rather than eight functions
    each re-pairing the same trades. The pairing is the expensive and
    error-prone part, so it happens once.

    HOW TO READ THE RESULT, IN ORDER
    Start with number_of_trades, because it is what transaction costs multiply:
    a strategy with an edge of 0.3% per trade and a cost of 0.2% per trade has
    almost nothing left. Then read profit_factor, which answers whether the
    strategy made money at all. Only then look at win_rate, and read it strictly
    alongside average_win and average_loss, which tell you what shape of bet
    produced that rate. A win rate of 0.9 with a profit factor of 0.45 is a
    losing strategy that feels like a winning one.

    Args:
        trade_log: Executed trades in chronological order, as produced by the
            backtester. An empty list or None is accepted.

    Returns:
        A TradeStatistics. With no completed round-trips, number_of_trades and
        open_trades are honest counts, gross_profit and gross_loss are 0.0, and
        every ratio is NaN, since a rate over zero trades is undefined rather
        than zero. This is a routine outcome, not an error: a moving-average
        strategy on a short or trendless period genuinely never closes a trade.
    """
    trips = round_trips(trade_log)
    open_trades = _count_open_trades(trade_log)

    if trips.empty:
        return TradeStatistics(
            number_of_trades=0,
            win_rate=float("nan"),
            average_win=float("nan"),
            average_loss=float("nan"),
            profit_factor=float("nan"),
            gross_profit=0.0,
            gross_loss=0.0,
            open_trades=open_trades,
        )

    profits = trips["profit"].astype(float)
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    # A trip that broke exactly even is neither a win nor a loss, so it stays
    # out of both averages. It remains in the denominator of the win rate,
    # though: it was a trade, and it was not won.
    win_rate = float(len(wins) / len(profits))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        # No losing trade at all. The ratio is genuinely infinite rather than
        # undefined, and saying so preserves the distinction from the no-trades
        # case, which returns NaN.
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    return TradeStatistics(
        number_of_trades=int(len(profits)),
        win_rate=win_rate,
        average_win=float(wins.mean()) if len(wins) else float("nan"),
        average_loss=float(losses.mean()) if len(losses) else float("nan"),
        profit_factor=profit_factor,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        open_trades=open_trades,
    )


def _count_open_trades(trade_log: List[Dict[str, Any]]) -> int:
    """Positions opened but never closed by the end of the log."""
    open_count = 0
    for trade in trade_log or []:
        if trade.get("action") == BUY and open_count == 0:
            open_count = 1
        elif trade.get("action") == SELL and open_count == 1:
            open_count = 0
    return open_count


if __name__ == "__main__":
    def trade(day: int, action: str, price: float, shares: float) -> Dict[str, Any]:
        """One trade log entry, dated for readability."""
        return {
            "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=day),
            "action": action,
            "price": price,
            "shares": shares,
        }

    def show(label: str, trade_log: List[Dict[str, Any]]) -> None:
        """Print the round-trips and then every statistic."""
        trips = round_trips(trade_log)
        result = trade_statistics(trade_log)

        print(f"\n{label}")
        if trips.empty:
            print("  (no completed round-trips)")
        else:
            for _, trip in trips.iterrows():
                print(f"  {trip['entry_price']:>7.2f} -> {trip['exit_price']:>7.2f}"
                      f"  x {trip['shares']:>6.2f} shares"
                      f"  profit {trip['profit']:>+9.2f}"
                      f"  ({trip['return_pct']:>+7.2%})")

        def number(value: float, spec: str) -> str:
            """Render a figure, or n/a when the statistic does not apply."""
            return "n/a" if np.isnan(value) else format(value, spec)

        print(f"  number_of_trades  {result.number_of_trades}")
        print(f"  open_trades       {result.open_trades}")
        print(f"  win_rate          {number(result.win_rate, '.4f')}")
        print(f"  average_win       {number(result.average_win, '+.2f')}")
        print(f"  average_loss      {number(result.average_loss, '+.2f')}")
        print(f"  gross_profit      {result.gross_profit:+.2f}")
        print(f"  gross_loss        {result.gross_loss:.2f}")
        print(f"  profit_factor     {number(result.profit_factor, '.4f')}")

    # Four closed round-trips and one position left open at the end.
    # By hand: profits are +100, -50, +600, -600.
    #   wins  = 2, gross profit 700, average win  350
    #   losses = 2, gross loss  650, average loss -325
    #   win_rate = 2/4 = 0.50
    #   profit_factor = 700 / 650 = 1.0769
    # The final BUY has no matching SELL, so it is reported as 1 open trade and
    # contributes to nothing else.
    show(
        "Mixed log: 2 winners, 2 losers, 1 position still open",
        [
            trade(0, BUY, 100.0, 10.0), trade(10, SELL, 110.0, 10.0),
            trade(20, BUY, 110.0, 10.0), trade(30, SELL, 105.0, 10.0),
            trade(40, BUY, 100.0, 20.0), trade(50, SELL, 130.0, 20.0),
            trade(60, BUY, 120.0, 10.0), trade(70, SELL, 60.0, 10.0),
            trade(80, BUY, 90.0, 5.0),
        ],
    )

    # THE POINT OF THIS WHOLE MODULE. Nine wins of +10 against a single loss of
    # -200. By hand: win_rate = 9/10 = 0.90, gross profit 90, gross loss 200,
    # profit_factor = 90 / 200 = 0.45. Nine times out of ten this strategy was
    # right, and it still lost 110. Anyone shown only the win rate would call it
    # excellent.
    high_win_rate = []
    for index in range(9):
        high_win_rate += [
            trade(index * 2, BUY, 100.0, 1.0),
            trade(index * 2 + 1, SELL, 110.0, 1.0),
        ]
    # The tenth trip: 4 shares bought at 100 and sold at 50, losing 200.
    high_win_rate += [
        trade(100, BUY, 100.0, 4.0),
        trade(101, SELL, 50.0, 4.0),
    ]
    show("90% win rate, and still a losing strategy", high_win_rate)

    # A strategy that never closed anything: the routine case for a slow
    # moving-average crossover on a short window of data.
    show("Never closed a trade", [trade(0, BUY, 100.0, 10.0)])
    show("Empty log", [])
