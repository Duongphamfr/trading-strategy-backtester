"""RSI mean reversion: betting that an overshoot comes back."""

import numpy as np
import pandas as pd

from constants import BUY, HOLD, SELL
from strategies.base_strategy import BaseStrategy


class RSIMeanReversion(BaseStrategy):
    """Buys oversold conditions and sells overbought ones, measured by RSI.

    This is the opposite wager to the moving average crossover. Trend-following
    assumes a move that has started will continue; mean reversion assumes the
    move has gone too far and will snap back. Both cannot be right about the
    same market at the same time, which is precisely why running them through
    the same engine on the same data is worth doing.

    The Relative Strength Index compresses the recent balance of gains against
    losses into a 0-100 reading. Near 0 the window has been almost all losses,
    which the rule reads as capitulation and therefore as a buying opportunity;
    near 100 it has been almost all gains, read as exhaustion and an exit.

    A STRUCTURAL ASYMMETRY WORTH UNDERSTANDING BEFORE READING ANY RESULT
    In the crossover strategy, buy and sell are mirror images of one state
    variable: whenever the fast average is not above the slow one it is below,
    so an entry is always eventually followed by an exit. RSI has no such
    symmetry. Entry and exit are governed by two different thresholds with a
    wide dead zone between them, and that has two consequences the backtest will
    show but the concept does not advertise:

    First, a position can be held indefinitely. Buy at RSI 29, watch it recover
    to 55, and if it never reaches 70 there is simply no exit rule. The strategy
    holds through whatever comes next, including a crash. It has no stop loss.

    Second, consecutive buys with no intervening sell are normal: RSI can dip
    below 30, recover to 40, and dip again. Under the Phase 1 all-in model the
    second buy is a no-op because the portfolio is already fully invested, so
    the trade log stays clean, but the signal series will contain both.

    Neither is a bug in this implementation. They are properties of the rule,
    and the fact that a naive reading of "buy low, sell high" hides them is a
    good argument for measuring rather than assuming.

    Attributes:
        rsi_period: Lookback of the RSI, in bars.
        oversold: Threshold below which the asset is considered oversold.
        overbought: Threshold above which the asset is considered overbought.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        """Configure the indicator and the two thresholds.

        The defaults, 14 bars with bands at 30 and 70, are Wilder's own and are
        what every charting platform ships. They are conventions rather than
        optimised values, which is an advantage here: a strategy tested on the
        settings everybody already uses cannot be accused of having been fitted
        to the sample.

        Args:
            rsi_period: Number of bars the RSI averages over. Must be a positive
                whole number.
            oversold: Buy threshold, strictly between 0 and overbought.
            overbought: Sell threshold, strictly between oversold and 100.

        Raises:
            ValueError: If the period is not a positive whole number, or if the
                thresholds do not satisfy 0 < oversold < overbought < 100.
        """
        if int(rsi_period) != rsi_period:
            raise ValueError(
                f"rsi_period must be a whole number of bars, got {rsi_period}."
            )

        rsi_period = int(rsi_period)
        oversold = float(oversold)
        overbought = float(overbought)

        if rsi_period <= 0:
            raise ValueError(
                f"rsi_period must be strictly positive, got {rsi_period}."
            )
        if not 0.0 < oversold < overbought < 100.0:
            raise ValueError(
                f"Thresholds must satisfy 0 < oversold < overbought < 100, got "
                f"oversold={oversold} and overbought={overbought}. RSI is bounded "
                f"to [0, 100], so thresholds outside that range, or inverted, can "
                f"never trigger or would fire on every bar."
            )

        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def rsi(self, data: pd.DataFrame) -> pd.Series:
        """Compute the RSI of the closing prices.

        WHICH SMOOTHING, AND WHY
        Wilder's smoothing, an exponential average with alpha = 1 / period,
        seeded by the simple mean of the first `period` changes. A plain rolling
        mean of gains and losses would have been acceptable and marginally
        simpler, but Wilder's is what the indicator actually is: it is the
        definition in his 1978 book and the one implemented by TradingView,
        StockCharts and TA-Lib. Matching it means an RSI(14) printed here can be
        checked against any chart, which turns an untestable internal number
        into a verifiable one. A rolling-mean variant would be close but never
        exactly equal, and every discrepancy would then need explaining away.

        The seed matters for the same reason. Pandas' ewm on its own starts the
        recursion at the very first change, which is a different indicator for
        the first few dozen bars. Placing Wilder's simple-mean seed at bar
        `period` and letting the recursion run from there reproduces the
        canonical series exactly.

        NUMERICAL FORM
        The textbook writes RSI = 100 - 100 / (1 + RS) with RS = average gain
        over average loss. This uses the algebraically identical rearrangement

            RSI = 100 * average gain / (average gain + average loss)

        which avoids dividing by zero when a window contains no losses at all.
        That case is not hypothetical for an asset in a strong run, and the
        rearranged form simply returns 100, as it should.

        A window in which the price never moved leaves both averages at zero and
        the RSI genuinely undefined. It stays NaN, which generate_signals treats
        as HOLD. Refusing to invent a reading for a market that did nothing is
        the honest choice, and it only arises for halted or untraded instruments.

        No-look-ahead compliance: diff() reaches one bar back, ewm() averages
        only bars at or before the current one, and no shift is forward. The
        value at bar T is computable on the evening of bar T.

        Args:
            data: OHLCV price history indexed by date, sorted chronologically.

        Returns:
            A Series of RSI values in [0, 100] aligned to data.index, NaN
            through the warm-up period. All NaN if the history is too short to
            seed, that is if it holds no more than rsi_period bars.
        """
        close = data["Close"].astype(float)
        change = close.diff()

        gain = change.clip(lower=0.0)
        loss = (-change).clip(lower=0.0)

        # A seed needs rsi_period changes, and diff() consumes the first bar, so
        # anything shorter than rsi_period + 1 bars cannot produce a reading.
        if len(close) <= self.rsi_period:
            return pd.Series(np.nan, index=data.index, dtype=float)

        average_gain = self._wilder_average(gain)
        average_loss = self._wilder_average(loss)

        total = average_gain + average_loss
        return 100.0 * average_gain.where(total > 0.0) / total.where(total > 0.0)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Emit BUY on entering oversold, SELL on entering overbought.

        Signals fire on transitions, not on states: BUY on the bar where the RSI
        crosses from at-or-above the oversold threshold down through it, SELL on
        the bar where it crosses up through the overbought threshold. This
        matches the crossover strategy's convention and for the same reason. A
        state-based rule would emit BUY on every bar of a long oversold stretch,
        which the broker would ignore once fully invested while the signal series
        filled up with orders that never happened.

        The consequence is that only the moment of entering an extreme counts.
        An asset already oversold when the warm-up ends is not bought, because
        the crossing that took it there happened outside the observable window,
        exactly as with the crossover's pre-existing trend.

        No-look-ahead compliance: the RSI is causal (see rsi), and the only
        shift is .shift(1), one bar into the past. During the warm-up the RSI is
        NaN and the bar stays HOLD.

        Args:
            data: OHLCV price history indexed by date, sorted chronologically.

        Returns:
            A Series of BUY / SELL / HOLD values aligned to data.index.
        """
        rsi = self.rsi(data)

        defined = rsi.notna()
        below = rsi < self.oversold
        above = rsi > self.overbought

        previous_defined = defined.shift(1, fill_value=False)
        previous_below = below.shift(1, fill_value=False)
        previous_above = above.shift(1, fill_value=False)

        # Requiring the previous bar to be defined as well means the first bar
        # after the warm-up cannot fire: there is no earlier reading to have
        # crossed from, so no crossing can be established there.
        entered_oversold = defined & previous_defined & below & ~previous_below
        entered_overbought = defined & previous_defined & above & ~previous_above

        signals = self.hold_signals(data)
        signals[entered_oversold] = BUY
        signals[entered_overbought] = SELL

        return signals

    def _wilder_average(self, values: pd.Series) -> pd.Series:
        """Wilder's smoothed average of a gain or loss series.

        The recursion is average[T] = average[T-1] + (value[T] - average[T-1]) /
        period, which is an exponential average with alpha = 1 / period. It is
        seeded at bar `period` with the simple mean of the first `period`
        changes, following Wilder.

        Implementation note: rather than looping, the seed is written into the
        series at bar `period` and everything before it is blanked. Pandas' ewm
        skips leading NaNs and takes its first observation as the starting
        value, so the recursion begins from the seed on its own.

        Args:
            values: Per-bar gains or losses, with NaN at bar 0 from diff().

        Returns:
            The smoothed series, NaN before bar `period`.
        """
        period = self.rsi_period

        seeded = values.copy()
        # Changes live at positions 1..period; position 0 is the NaN from diff().
        seeded.iloc[:period] = np.nan
        seeded.iloc[period] = values.iloc[1:period + 1].mean()

        return seeded.ewm(alpha=1.0 / period, adjust=False).mean()


if __name__ == "__main__":
    from data.market_data import get_price_data

    prices = get_price_data("AAPL", "2020-01-01", "2023-01-01")
    strategy = RSIMeanReversion()
    signals = strategy.generate_signals(prices)
    rsi = strategy.rsi(prices)

    print(f"Strategy:        {strategy!r}")
    print(f"Bars:            {len(signals)}")
    print(f"Warm-up bars:    {rsi.isna().sum()}  (RSI undefined, forced HOLD)")
    print(f"BUY signals:     {(signals == BUY).sum()}")
    print(f"SELL signals:    {(signals == SELL).sum()}")
    print(f"RSI range:       {rsi.min():.2f} to {rsi.max():.2f}")
    print(f"Bars oversold:   {(rsi < strategy.oversold).sum()}")
    print(f"Bars overbought: {(rsi > strategy.overbought).sum()}")

    events = signals[signals != HOLD]
    print("\nFirst signals:")
    for date, action in events.head(6).items():
        print(f"  {date.date()}  {action}  close="
              f"{prices.loc[date, 'Close']:7.2f}  RSI={rsi.loc[date]:5.2f}")

    # The count of bars spent inside an extreme is far larger than the count of
    # signals, which is the transition rule doing its job: one signal per visit
    # to an extreme rather than one per bar of the visit.
    print("\nEntering an extreme fires once per visit, not once per bar:")
    print(f"  oversold bars {(rsi < strategy.oversold).sum():4d}"
          f"  ->  {(signals == BUY).sum():3d} BUY")
    print(f"  overbought    {(rsi > strategy.overbought).sum():4d}"
          f"  ->  {(signals == SELL).sum():3d} SELL")

    # THE CAUSALITY CHECK
    # If a signal at bar T depended on anything after T, then deleting the tail
    # of the history would change signals in the part that remains. Recomputing
    # on a truncated history and comparing the overlap is therefore a direct
    # empirical test of the no-look-ahead contract, not a proxy for it. Wilder's
    # smoothing is recursive, so a leak would show up here immediately.
    print("\nNo-look-ahead check (recompute on truncated history)")
    for fraction in (0.4, 0.6, 0.8):
        cut = int(len(prices) * fraction)
        truncated = strategy.generate_signals(prices.iloc[:cut])
        identical = truncated.equals(signals.iloc[:cut])
        drift = (rsi.iloc[:cut] - strategy.rsi(prices.iloc[:cut])).abs().max()
        print(f"  first {cut:4d} bars ({fraction:.0%})  signals identical: "
              f"{identical}   max RSI difference: {drift:.2e}")

    print("\nA hand-checkable RSI, on a monotonic series:")
    rising = pd.DataFrame({"Close": [100.0 + i for i in range(20)]})
    falling = pd.DataFrame({"Close": [100.0 - i for i in range(20)]})
    flat = pd.DataFrame({"Close": [100.0] * 20})
    print(f"  20 straight gains -> RSI {strategy.rsi(rising).iloc[-1]:.2f}"
          f"   (no losses at all, so 100)")
    print(f"  20 straight losses -> RSI {strategy.rsi(falling).iloc[-1]:.2f}"
          f"   (no gains at all, so 0)")
    print(f"  20 unchanged bars -> RSI {strategy.rsi(flat).iloc[-1]}"
          f"   (undefined, treated as HOLD)")

    print("\nToo-short history and rejected parameters:")
    short = prices.iloc[:10]
    print(f"  10 bars, period 14 -> all HOLD: "
          f"{(strategy.generate_signals(short) == HOLD).all()}")
    for arguments in ({"rsi_period": 0}, {"oversold": 70, "overbought": 30},
                      {"overbought": 120}):
        try:
            RSIMeanReversion(**arguments)
        except ValueError as error:
            print(f"  {arguments} -> ValueError: {str(error)[:58]}...")
