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
        enter_on_existing_trend: Whether to open a position when the warm-up
            ends with the fast average already above the slow one.
    """

    def __init__(
        self,
        fast_window: int = 50,
        slow_window: int = 200,
        enter_on_existing_trend: bool = False,
    ) -> None:
        """Configure the two moving averages.

        THE WARM-UP BOUNDARY DECISION
        Both averages need their full window before they exist, so the first
        bars of any history carry no signal. When they finally become defined,
        the fast average may already be above the slow one: the trend is under
        way, but the crossing that started it happened before the data begins
        and is therefore invisible to the strategy.

        enter_on_existing_trend decides what to do at that boundary, and the
        decision is far from cosmetic. On AAPL over 2020-2022 the averages
        become defined in October 2020 with the fast one on top, and the two
        settings differ by roughly twenty percentage points of total return.

        False, the default, waits for a golden cross to actually occur within
        the observed data. It is the conservative reading, and it is the default
        precisely because it refuses to credit the strategy with a trend it
        never saw form. Buying into a pre-existing trend means the entry rests
        on information from before the sample, which quietly makes the result a
        function of the arbitrary start date of the download rather than of the
        rule being tested. Keeping the default conservative means an optimistic
        number can only ever be produced deliberately, never by accident.

        The cost of that honesty is real and should be stated rather than
        hidden: the strategy may sit in cash through a large move it would have
        ridden, as it does through the 2020-2021 rally. That is a limitation of
        the measurement, not evidence that trend-following does not work.

        True buys in on the first bar where both averages exist and the fast one
        is above. It measures the strategy as a *state* ("be invested while in
        an uptrend") rather than as a set of *events*, which is closer to how
        crossover rules are usually framed in the literature. It is a legitimate
        choice, available explicitly, and worth reporting alongside the default
        rather than instead of it.

        Args:
            fast_window: Number of bars in the short average. Must be positive
                and strictly smaller than slow_window.
            slow_window: Number of bars in the long average. Must be positive.
            enter_on_existing_trend: Whether to buy into a trend that was
                already established when the averages first became computable.
                Defaults to False, the conservative behaviour described above.

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

        super().__init__(
            fast_window=fast_window,
            slow_window=slow_window,
            enter_on_existing_trend=bool(enter_on_existing_trend),
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.enter_on_existing_trend = bool(enter_on_existing_trend)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Emit BUY on golden crosses, SELL on death crosses, HOLD elsewhere.

        No-look-ahead compliance: both averages come from rolling(), whose window
        at bar T covers only bars at or before T, and the only shift applied is
        .shift(1), which looks one bar into the past. Nothing reaches forward.

        On the warm-up boundary, see enter_on_existing_trend. The choice is not
        cosmetic: on AAPL over 2020-2022 the averages become defined in October
        2020 with the fast one already on top, so waiting for a visible crossing
        leaves the strategy in cash through the entire 2020-2021 rally.

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

        # The trend may already be under way when the averages become defined.
        # The crossing that started it is simply not visible in this window, so
        # the transition rules above cannot fire and the strategy would sit out
        # a trend it is designed to ride. This buys into it instead.
        if self.enter_on_existing_trend and defined.any():
            warm_up_end = defined.idxmax()
            if above[warm_up_end]:
                signals[warm_up_end] = BUY

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
