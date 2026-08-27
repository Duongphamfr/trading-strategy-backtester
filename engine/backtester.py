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

    Simplifying assumptions in Phase 1, stated explicitly rather than hidden:
        - Trades fill at the same bar's closing price. A more conservative model
          would fill at the next bar's open, since in practice a signal computed
          from today's close cannot be traded at that same close.
        - The portfolio is either fully invested or fully in cash.
        - No transaction costs. Those arrive in Phase 4, inside the Broker.

    Attributes:
        data: The OHLCV price history driving the simulation.
        initial_cash: Starting capital, used for both the strategy and benchmark.
        strategy: Any object exposing generate_signals(data) -> pandas Series.
        portfolio: The Portfolio holding cash and the position.
        broker: The Broker executing orders against that Portfolio.
    """

    def __init__(self, data: pd.DataFrame, initial_cash: float, strategy: Any) -> None:
        """Set up a backtest.

        Args:
            data: OHLCV DataFrame indexed by date, as returned by
                market_data.get_price_data. Must contain a Close column and be
                sorted chronologically.
            initial_cash: Starting capital. Must be strictly positive.
            strategy: Object exposing generate_signals(data), returning a pandas
                Series of "BUY" / "SELL" / "HOLD" aligned to the data's index.

        Raises:
            ValueError: If the data is empty, lacks a Close column, or is not
                sorted chronologically.
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

        self.portfolio = Portfolio(self.initial_cash)
        self.broker = Broker(self.portfolio)

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
        self.broker = Broker(self.portfolio)
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
                        {
                            "date": date,
                            "action": BUY,
                            "price": price,
                            "shares": shares,
                        }
                    )
            elif signal == SELL and holding:
                shares = self.broker.sell_all(price)
                if shares > 0:
                    trade_log.append(
                        {
                            "date": date,
                            "action": SELL,
                            "price": price,
                            "shares": shares,
                        }
                    )

            # Mark to market after any trade, so the bar's recorded value
            # already reflects what was executed on it.
            self.portfolio.record(date, price)

        return BacktestResult(
            equity_curve=self.portfolio.to_dataframe(),
            trade_log=trade_log,
            benchmark_curve=self._buy_and_hold_curve(),
        )

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
            f"strategy={type(self.strategy).__name__})"
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

    prices = get_price_data("AAPL", "2022-01-01", "2023-01-01")
    backtester = Backtester(
        prices,
        initial_cash=10_000.0,
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
