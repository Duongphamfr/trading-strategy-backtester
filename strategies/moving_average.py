"""Moving average crossover: the classic trend-following strategy."""

import pandas as pd

from constants import BUY, HOLD, SELL
from strategies.base_strategy import BaseStrategy


class MovingAverageCrossover(BaseStrategy):
    """Trades the crossings of a fast and a slow simple moving average.

    The premise is trend-following: when the short-term average rises above the
    long-term average, recent prices are stronger than the established trend and
    the move is assumed to continue. That upward crossing is the golden cross, a
    buy. The mirror image, the death cross, is the exit.

    The strategy is intentionally naive. It has no stop loss, no position sizing
    and no filter for choppy markets, where an average crossover whipsaws back
    and forth and generates losses through repeated trading. Measuring how badly
    that hurts, especially once Phase 4 charges transaction costs, is exactly
    what the backtest is for.

    Attributes:
        fast_window: Lookback of the short moving average, in bars.
        slow_window: Lookback of the long moving average, in bars.
    """

    def __init__(self, fast_window: int = 50, slow_window: int = 200) -> None:
        """Configure the two moving averages.

        Args:
            fast_window: Number of bars in the short average. Must be positive
                and strictly smaller than slow_window.
            slow_window: Number of bars in the long average. Must be positive.

        Raises:
            ValueError: If either window is not a positive integer, or if
                fast_window is not strictly smaller than slow_window.
        """
        if int(fast_window) != fast_window or int(slow_window) != slow_window:
            raise ValueError(
                f"Moving average windows must be whole numbers of bars, got "
                f"fast_window={fast_window} and slow_window={slow_window}."
            )

        fast_window = int(fast_window)
        slow_window = int(slow_window)

        if fast_window <= 0 or slow_window <= 0:
            raise ValueError(
                f"Moving average windows must be strictly positive, got "
                f"fast_window={fast_window} and slow_window={slow_window}."
            )
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window must be strictly smaller than slow_window, got "
                f"fast_window={fast_window} and slow_window={slow_window}. "
                f"Equal or inverted windows cannot produce a crossover."
            )

        super().__init__(fast_window=fast_window, slow_window=slow_window)
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Emit BUY on golden crosses, SELL on death crosses, HOLD elsewhere.

        No-look-ahead compliance: both averages come from rolling(), whose window
        at bar T covers only bars at or before T, and the only shift applied is
        .shift(1), which looks one bar into the past. Nothing reaches forward.

        If the history is shorter than slow_window the long average is never
        defined, so the result is all HOLD and the strategy simply never trades.

        Args:
            data: OHLCV price history indexed by date, sorted chronologically.

        Returns:
            A Series of BUY / SELL / HOLD values aligned to data.index.
        """
        close = data["Close"].astype(float)

        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()

        # Both averages need their full window before they mean anything. Until
        # the slow one is defined, the comparison below is meaningless, so those
        # warm-up bars are excluded and stay HOLD.
        defined = fast_ma.notna() & slow_ma.notna()
        above = fast_ma > slow_ma

        # Comparing each bar with the previous one turns a continuous state
        # ("fast is above slow") into discrete events ("fast just moved above
        # slow"). Signalling only on those transitions keeps the trade log to
        # one entry per actual position change; emitting BUY on every bar of an
        # uptrend would flood it with orders the broker would ignore anyway,
        # since the portfolio is already fully invested.
        previous_above = above.shift(1, fill_value=False)
        previous_defined = defined.shift(1, fill_value=False)

        # Requiring the previous bar to be defined too means the first bar after
        # the warm-up cannot fire: at that point the averages have no history to
        # be compared against, so no crossing can be established there.
        crossed_up = defined & previous_defined & above & ~previous_above
        crossed_down = defined & previous_defined & ~above & previous_above

        signals = self.hold_signals(data)
        signals[crossed_up] = BUY
        signals[crossed_down] = SELL
        return signals


if __name__ == "__main__":
    from data.market_data import get_price_data

    prices = get_price_data("AAPL", "2020-01-01", "2023-01-01")
    strategy = MovingAverageCrossover(fast_window=50, slow_window=200)
    signals = strategy.generate_signals(prices)

    print(f"Strategy:        {strategy!r}")
    print(f"Bars:            {len(signals)}")
    print(f"BUY signals:     {(signals == BUY).sum()}")
    print(f"SELL signals:    {(signals == SELL).sum()}")

    crosses = signals[signals != HOLD]
    print("\nFirst crosses:")
    for date, action in crosses.head(5).items():
        print(f"  {date.date()}  {action}  close={prices.loc[date, 'Close']:.2f}")
