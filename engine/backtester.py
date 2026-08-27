"""Core backtest loop: walks the price history one bar at a time."""

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from constants import BUY, HOLD, SELL, VALID_SIGNALS
from engine.broker import Broker
from engine.portfolio import Portfolio


@dataclass
class BacktestResult:
    """Everything a completed backtest produces.

    Attributes:
        equity_curve: Per-bar portfolio state, indexed by date, with the columns
            price, cash, shares, position_value and total_value.
        trade_log: One entry per executed trade, each a dict with the keys date,
            action, price and shares. Empty if the strategy never traded.
        benchmark_curve: Value over time of a buy-and-hold position opened on the
            first bar with the same initial cash, indexed by the same dates.
    """

    equity_curve: pd.DataFrame
    trade_log: List[Dict[str, Any]]
    benchmark_curve: pd.Series

    @property
    def final_value(self) -> float:
        """Portfolio value on the last bar of the backtest."""
        return float(self.equity_curve["total_value"].iloc[-1])

    @property
    def benchmark_final_value(self) -> float:
        """Buy-and-hold value on the last bar of the backtest."""
        return float(self.benchmark_curve.iloc[-1])


class Backtester:
    """Runs a strategy over historical prices, bar by bar.

    The Backtester owns the simulation loop. It asks the strategy for signals,
    hands execution decisions to the Broker, and lets the Portfolio keep the
    books. It knows nothing about where the data came from or how the results
    will be displayed.

    Simplifying assumptions, stated explicitly rather than hidden:
        - Trades fill at the same bar's closing price. A more conservative model
          would fill at the next bar's open, since in practice a signal computed
          from today's close cannot be traded at that same close.
        - The portfolio is either fully invested or fully in cash.
        - Transaction costs are proportional and symmetric, and the whole
          position is turned over at once. Fixed per-trade fees and the fact
          that a large order moves the market more than a small one are not
          modelled.

    THE BENCHMARK IS NOT CHARGED COSTS
    Buy-and-hold pays the spread and commission twice in its life, once getting
    in and once getting out, which on any horizon of years is negligible against
    the difference costs make to a strategy trading dozens of times. Leaving the
    benchmark frictionless keeps it a fixed reference line across every cost
    scenario, so a scenario comparison isolates the effect on the strategy. It
    does flatter the benchmark very slightly, and a strategy that loses to it by
    a hair is really a tie.

    Attributes:
        data: The OHLCV price history driving the simulation.
        initial_cash: Starting capital, used for both the strategy and benchmark.
        strategy: Any object exposing generate_signals(data) -> pandas Series.
        commission: Proportional commission per trade, passed to the Broker.
        spread: Bid-ask spread as a fraction of price, passed to the Broker.
        slippage: Adverse price move as a fraction of price, passed to the Broker.
        portfolio: The Portfolio holding cash and the position.
        broker: The Broker executing orders against that Portfolio.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float,
        strategy: Any,
        commission: float = 0.0,
        spread: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Set up a backtest.

        The three cost arguments are passed straight through to the Broker,
        which owns the entire cost model; the Backtester only forwards them. All
        default to zero, so a call written before Phase 4 produces exactly the
        result it always did.

        Args:
            data: OHLCV DataFrame indexed by date, as returned by
                market_data.get_price_data. Must contain a Close column and be
                sorted chronologically.
            initial_cash: Starting capital. Must be strictly positive.
            strategy: Object exposing generate_signals(data), returning a pandas
                Series of "BUY" / "SELL" / "HOLD" aligned to the data's index.
            commission: Proportional commission charged on each trade's value.
            spread: Bid-ask spread as a fraction of price; half is paid per side.
            slippage: Adverse price move as a fraction of price, per side.

        Raises:
            ValueError: If the data is empty, lacks a Close column, is not
                sorted chronologically, or if a cost parameter is out of range.
            TypeError: If the strategy does not expose generate_signals.
        """
        if data is None or data.empty:
            raise ValueError("Cannot run a backtest on an empty price DataFrame.")
        if "Close" not in data.columns:
            raise ValueError("The price DataFrame must contain a 'Close' column.")
        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "The price DataFrame must be sorted chronologically, oldest bar "
                "first, otherwise the simulation would not run forward in time."
            )
        if not callable(getattr(strategy, "generate_signals", None)):
            raise TypeError(
                "The strategy must expose a callable generate_signals(data) method."
            )

        self.data = data
        self.initial_cash = float(initial_cash)
        self.strategy = strategy
        self.commission = float(commission)
        self.spread = float(spread)
        self.slippage = float(slippage)

        self.portfolio = Portfolio(self.initial_cash)
        self.broker = self._new_broker(self.portfolio)

    def run(self) -> BacktestResult:
        """Execute the backtest and return its results.

        Returns:
            A BacktestResult holding the equity curve, the trade log and the
            buy-and-hold benchmark curve.

        Raises:
            ValueError: If the strategy returns signals that are not one of
                "BUY", "SELL" or "HOLD".
        """
        # Rebuild the books so the same Backtester can be run repeatedly and
        # always produce identical results.
        self.portfolio = Portfolio(self.initial_cash)
        self.broker = self._new_broker(self.portfolio)
        trade_log: List[Dict[str, Any]] = []

        signals = self._prepare_signals()

        dates = self.data.index
        prices = self.data["Close"].to_numpy(dtype=float)
        signal_values = signals.to_numpy()

        # HOW LOOK-AHEAD BIAS IS AVOIDED
        # 1. The loop moves strictly forward through the index, one bar at a
        #    time. Bar T is never indexed with i + 1, and no negative shift is
        #    ever applied, so a future row cannot influence a past decision.
        # 2. The action taken on bar T uses only the signal for bar T, and fills
        #    at bar T's own close. Tomorrow's price is unknown to the loop when
        #    the order is placed.
        # 3. Signals are computed in one vectorised call before the loop purely
        #    for speed. That is safe only under a contract the strategies must
        #    honour: the signal at bar T may depend on data up to and including
        #    bar T, never beyond. Any strategy using a forward-looking transform
        #    must shift it back before returning, and Phase 2 strategies will be
        #    written and unit-tested against that rule.
        for i, date in enumerate(dates):
            price = prices[i]
            signal = signal_values[i]
            holding = self.portfolio.shares > 0

            if signal == BUY and not holding:
                shares = self.broker.buy_all(price)
                if shares > 0:
                    trade_log.append(
                        self._log_entry(
                            date, BUY, price, shares,
                            self.broker.buy_fill_price(price),
                        )
                    )
            elif signal == SELL and holding:
                shares = self.broker.sell_all(price)
                if shares > 0:
                    trade_log.append(
                        self._log_entry(
                            date, SELL, price, shares,
                            self.broker.sell_fill_price(price),
                        )
                    )

            # Mark to market after any trade, so the bar's recorded value
            # already reflects what was executed on it.
            self.portfolio.record(date, price)

        return BacktestResult(
            equity_curve=self.portfolio.to_dataframe(),
            trade_log=trade_log,
            benchmark_curve=self._buy_and_hold_curve(),
        )

    def _new_broker(self, portfolio: Portfolio) -> Broker:
        """Build a Broker carrying this backtest's cost model.

        Kept in one place because run() rebuilds the books on every call, and a
        Broker created without the costs would silently run a frictionless
        backtest while reporting a priced one.

        Args:
            portfolio: The Portfolio the new Broker should execute against.

        Returns:
            A Broker configured with this Backtester's costs.
        """
        return Broker(
            portfolio,
            commission=self.commission,
            spread=self.spread,
            slippage=self.slippage,
        )

    @staticmethod
    def _log_entry(
        date: pd.Timestamp,
        action: str,
        quoted_price: float,
        shares: float,
        fill_price: float,
    ) -> Dict[str, Any]:
        """Build one trade log entry.

        WHY "price" IS THE FILL PRICE AND NOT THE QUOTE
        analytics.trade_stats computes round-trip profit as
        (exit_price - entry_price) * shares, reading the "price" key. Recording
        the quoted price there would make the Trades section of every report
        show gross profits while the equity curve beside it showed net ones, and
        the two would disagree by precisely the amount Phase 4 exists to
        measure. Recording the fill price instead makes that profit net of every
        cost, automatically and with no change to the analytics, which is what
        trade_stats' own docstring already promised would happen.

        The quoted price is kept alongside under "quoted_price" for anyone
        plotting trades against a price chart, together with the cash cost of
        the friction on that trade. At zero costs the two prices coincide and the
        cost is zero, so nothing about earlier results changes.

        Args:
            date: Bar the trade executed on.
            action: BUY or SELL.
            quoted_price: Market price on that bar.
            shares: Quantity traded.
            fill_price: All-in per-share price, costs included.

        Returns:
            A dict with the keys date, action, price, shares, quoted_price and
            cost.
        """
        return {
            "date": date,
            "action": action,
            "price": fill_price,
            "shares": shares,
            "quoted_price": quoted_price,
            "cost": abs(fill_price - quoted_price) * shares,
        }

    def _prepare_signals(self) -> pd.Series:
        """Fetch, align and validate the strategy's signals.

        Bars the strategy left unlabelled, such as the warm-up period of an
        indicator, are treated as HOLD rather than as an error.

        Returns:
            A Series of signals aligned one-to-one with the data's index.

        Raises:
            ValueError: If any signal is outside the accepted vocabulary.
        """
        signals = self.strategy.generate_signals(self.data)

        if not isinstance(signals, pd.Series):
            raise TypeError(
                f"generate_signals must return a pandas Series, got "
                f"{type(signals).__name__}."
            )

        signals = signals.reindex(self.data.index).fillna(HOLD)

        unknown = set(signals.unique()) - VALID_SIGNALS
        if unknown:
            raise ValueError(
                f"The strategy produced unsupported signals: "
                f"{', '.join(sorted(str(value) for value in unknown))}. "
                f"Allowed values are {BUY}, {SELL} and {HOLD}."
            )

        return signals

    def _buy_and_hold_curve(self) -> pd.Series:
        """Value over time of investing all the initial cash on the first bar.

        This is computed directly from the price series rather than by running a
        second simulation through the Portfolio and Broker. Deriving it
        independently is what makes it a genuine check: if the engine had an
        accounting bug, a benchmark built on that same engine would hide it.

        Returns:
            A Series named benchmark_value, indexed by the data's dates.
        """
        close = self.data["Close"].astype(float)
        curve = self.initial_cash * (close / close.iloc[0])
        curve.name = "benchmark_value"
        return curve

    def __repr__(self) -> str:
        return (
            f"Backtester(bars={len(self.data)}, "
            f"initial_cash={self.initial_cash:.2f}, "
            f"strategy={type(self.strategy).__name__}, "
            f"commission={self.commission}, spread={self.spread}, "
            f"slippage={self.slippage})"
        )


if __name__ == "__main__":
    # Imported here rather than at module level: the engine must stay
    # independent of where the data comes from.
    from data.market_data import get_price_data

    class DummyBuyAndHoldStrategy:
        """Buys on the first bar and holds to the end.

        Phase 1 smoke check only. Real strategies arrive in Phase 2, in the
        strategies package, behind the shared base_strategy interface.
        """

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            signals = pd.Series(HOLD, index=data.index, dtype=object)
            signals.iloc[0] = BUY
            return signals

    class PeriodicFlipStrategy:
        """Alternates between fully invested and fully in cash every N bars.

        Not a strategy anyone would trade. Its purpose is to make transaction
        costs measurable: it turns the portfolio over a known number of times
        regardless of what prices do, so the cost drag can be predicted in
        closed form and checked against what the engine actually produces. A
        real strategy would confound the two, since costs also change which
        trades happen next.
        """

        def __init__(self, period: int = 10) -> None:
            self.period = period

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            signals = pd.Series(HOLD, index=data.index, dtype=object)
            positions = range(0, len(data), self.period)
            for count, position in enumerate(positions):
                signals.iloc[position] = BUY if count % 2 == 0 else SELL
            return signals

    prices = get_price_data("AAPL", "2022-01-01", "2023-01-01")
    CAPITAL = 10_000.0

    backtester = Backtester(
        prices,
        initial_cash=CAPITAL,
        strategy=DummyBuyAndHoldStrategy(),
    )
    result = backtester.run()

    print(f"Bars simulated:   {len(result.equity_curve)}")
    print(f"Trades executed:  {len(result.trade_log)}")
    print(f"Final value:      {result.final_value:,.2f}")
    print(f"Benchmark value:  {result.benchmark_final_value:,.2f}")

    difference = abs(result.final_value - result.benchmark_final_value)
    print(f"Difference:       {difference:,.10f}")

    if difference < 1e-6:
        print("\nPHASE 1 CHECK PASSED: buy-and-hold strategy matches the benchmark.")
    else:
        print("\nPHASE 1 CHECK FAILED: the engine has an accounting bug.")

    print("\n\nPHASE 4 — HOW COSTS ERODE A FREQUENTLY TRADING STRATEGY")
    print(f"Flip in and out every 10 bars over {len(prices)} bars of AAPL.\n")

    flip = PeriodicFlipStrategy(period=10)
    free = Backtester(prices, initial_cash=CAPITAL, strategy=flip).run()

    print(f"  {'commission':>10} {'trades':>7} {'final value':>13} "
          f"{'return':>9} {'cost drag':>10} {'predicted':>10} {'costs paid':>11}")

    for commission in (0.0, 0.0005, 0.0010, 0.0025):
        priced = Backtester(
            prices,
            initial_cash=CAPITAL,
            strategy=flip,
            commission=commission,
        ).run()

        trades = len(priced.trade_log)
        drag = priced.final_value / free.final_value - 1.0

        # Each trade multiplies the portfolio by (1 - commission), so after n
        # trades the value is scaled by (1 - commission) ** n. Matching this is
        # the real check on the cost model: it confirms the charge lands once per
        # trade, on the full position value, and compounds rather than being
        # levied on the starting capital.
        predicted = (1.0 - commission) ** trades - 1.0
        paid = sum(trade["cost"] for trade in priced.trade_log)

        print(f"  {commission:>10.4%} {trades:>7} {priced.final_value:>13,.2f} "
              f"{priced.final_value / CAPITAL - 1:>8.2%} {drag:>9.2%} "
              f"{predicted:>9.2%} {paid:>11,.2f}")

    print("\n  Costs default to zero, so the priced and unpriced runs agree "
          "exactly:")
    zero = Backtester(prices, initial_cash=CAPITAL, strategy=flip,
                      commission=0.0, spread=0.0, slippage=0.0).run()
    print(f"    difference between default and explicit zero costs: "
          f"{abs(zero.final_value - free.final_value):.2e}")

    print("\n  The three frictions at a realistic retail combination:")
    combined = Backtester(
        prices,
        initial_cash=CAPITAL,
        strategy=flip,
        commission=0.0005,
        spread=0.0005,
        slippage=0.0005,
    ).run()
    print(f"    commission 0.05% + spread 0.05% + slippage 0.05% -> "
          f"{combined.final_value:,.2f} "
          f"({combined.final_value / free.final_value - 1:+.2%} versus free)")

    print("\n  Round-trip profits in the trade log are already net of costs:")
    print(f"    {'quoted':>9} {'fill':>9} {'action':>7} {'cost':>9}")
    for trade in combined.trade_log[:4]:
        print(f"    {trade['quoted_price']:>9.2f} {trade['price']:>9.2f} "
              f"{trade['action']:>7} {trade['cost']:>9.2f}")
