"""Time-series momentum: staying invested only while the asset is rising."""

import numpy as np
import pandas as pd

from constants import BUY, HOLD, SELL
from strategies.base_strategy import BaseStrategy


class Momentum(BaseStrategy):
    """Holds the asset while its own trailing return is positive, else sits out.

    The rule is as blunt as it sounds: look back over the lookback window, and
    if the asset is higher than it was, stay invested. If it is not, go to cash.
    Check only every rebalance_freq bars, so the position persists between
    checks rather than reacting to every wobble.

    WHICH MOMENTUM THIS IS, AND WHICH IT IS NOT
    Worth stating plainly, because the two are routinely conflated and only one
    of them is the famous result. This is *time-series* momentum, also called
    absolute momentum: one asset judged against its own past. The heavily cited
    anomaly of the academic literature is *cross-sectional* momentum, ranking
    many assets against each other and buying the winners, which is what
    Jegadeesh and Titman documented in 1993. Time-series momentum has its own
    respectable evidence, notably Moskowitz, Ooi and Pedersen in 2012, but it is
    a different claim. Since the Phase 1 engine holds a single asset, this is the
    version that can honestly be implemented here, and any write-up should say
    so rather than borrowing the credibility of the cross-sectional result.

    IT WILL LOOK A LOT LIKE THE MOVING AVERAGE CROSSOVER, AND THAT MATTERS
    Both are trend-following: both are long when the recent past has been up and
    flat when it has been down. So the three strategies in this project are not
    three independent bets on market behaviour, they are two. Expect momentum
    and the crossover to be correlated, to win and lose in the same regimes, and
    to disagree with the RSI strategy at the same times. Treating their results
    as independent evidence would overstate what the backtest shows.

    ON HOW THE TRAILING RETURN IS MEASURED
    Close[T] / Close[T - lookback] - 1 depends on exactly two prices. Everything
    that happened between them is discarded, so a single unusual print at the
    far end of the window can flip the sign of the signal. That is the standard
    definition and it is kept here, but it is fragile in a way that an average
    over the window would not be. Practitioners often also skip the most recent
    month, the so-called 12-2 formulation, because the very latest move tends to
    reverse. Neither refinement is implemented; both are honest extensions.

    Attributes:
        lookback: Number of bars the trailing return spans.
        rebalance_freq: Number of bars between decisions. 1 checks every bar.
    """

    def __init__(self, lookback: int = 126, rebalance_freq: int = 21) -> None:
        """Configure the lookback window and the decision schedule.

        The defaults are roughly six months of trailing return, reviewed about
        monthly, on a 252-bar trading year. Both are conventions from the
        literature rather than fitted values, which is deliberate: a strategy
        run on the settings everybody already quotes cannot be accused of having
        been tuned to this particular sample.

        Args:
            lookback: Bars in the trailing return window. Must be a positive
                whole number.
            rebalance_freq: Bars between decisions. Must be a whole number of at
                least 1, where 1 means the rule is evaluated on every bar.

        Raises:
            ValueError: If either argument is not a whole number, if lookback is
                not strictly positive, or if rebalance_freq is below 1.
        """
        if int(lookback) != lookback or int(rebalance_freq) != rebalance_freq:
            raise ValueError(
                f"lookback and rebalance_freq must be whole numbers of bars, got "
                f"lookback={lookback} and rebalance_freq={rebalance_freq}."
            )

        lookback = int(lookback)
        rebalance_freq = int(rebalance_freq)

        if lookback <= 0:
            raise ValueError(
                f"lookback must be strictly positive, got {lookback}. A trailing "
                f"return needs at least one bar of distance to measure."
            )
        if rebalance_freq < 1:
            raise ValueError(
                f"rebalance_freq must be at least 1, got {rebalance_freq}. Use 1 "
                f"to evaluate the rule on every bar."
            )

        super().__init__(lookback=lookback, rebalance_freq=rebalance_freq)
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq

    def momentum(self, data: pd.DataFrame) -> pd.Series:
        """Compute the trailing return over the lookback window.

        WHY THIS IS FREE OF LOOK-AHEAD BIAS
        The value at bar T is Close[T] / Close[T - lookback] - 1. Both prices
        carry timestamps at or before T, so the whole quantity is known on the
        evening of bar T. The shift is positive, which in pandas moves values
        forward in time, meaning each row is paired with an *older* price. A
        negative shift would do the opposite and pull a future price onto today,
        which is the violation the base class warns about.

        Args:
            data: OHLCV price history indexed by date, sorted chronologically.

        Returns:
            A Series of trailing returns aligned to data.index, NaN for the
            first lookback bars where no earlier price exists to compare with.
        """
        close = data["Close"].astype(float)
        return close / close.shift(self.lookback) - 1.0

    def target_position(self, data: pd.DataFrame) -> pd.Series:
        """Whether the strategy intends to be invested on each bar.

        This is where the strategy's statefulness lives, and it is what sets the
        rule apart from the other two. "Buy if momentum is positive and we are
        currently flat" refers to the strategy's own past decisions, not only to
        prices, because between two rebalance dates the position simply persists.
        The crossover and the RSI rules have no such memory: their state is
        recomputable from the price series at any single bar.

        The schedule is anchored at the end of the warm-up, so the first decision
        falls on bar lookback and every rebalance_freq bars thereafter. Anchoring
        forward from a fixed early bar is not a stylistic preference, it is what
        keeps the strategy causal: anchoring backward from the last bar of the
        sample would make every rebalance date depend on where the data happens
        to end, which is future information leaking into the past.

        The honest cost of forward anchoring is that the schedule depends on the
        sample's start date. Download the same history beginning a week earlier
        and every decision date shifts, which can change the result even though
        the rule did not. It is a genuine sensitivity, and one worth reporting
        alongside any number this strategy produces.

        Args:
            data: OHLCV price history indexed by date, sorted chronologically.

        Returns:
            A boolean Series aligned to data.index: True on bars the strategy
            means to hold the asset, False on bars it means to be in cash.
        """
        trailing = self.momentum(data)
        defined = trailing.notna()

        # Bar numbers, not dates, since the schedule counts bars. Using dates
        # would drag calendar gaps and holidays into a rule that has no view
        # about them.
        bar = np.arange(len(trailing))
        first_decision = self.lookback
        due = (bar >= first_decision) & ((bar - first_decision) % self.rebalance_freq == 0)
        decides = pd.Series(due, index=trailing.index) & defined

        # Only rebalance bars express an opinion; every other bar is left blank
        # and inherits the last decision below.
        target = pd.Series(np.nan, index=trailing.index, dtype=float)
        target[decides] = (trailing[decides] > 0.0).astype(float)

        # ffill carries a past decision forward, which is exactly the position
        # persisting until the next review. Note that this direction is the safe
        # one: bfill would carry a later decision backwards and would be a
        # textbook look-ahead violation. Bars before the first decision stay
        # flat, since the strategy has not yet had anything to act on.
        return target.ffill().fillna(0.0) > 0.0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Emit BUY when entering the market, SELL when leaving it, else HOLD.

        The rule is evaluated on rebalance bars only, and a signal is emitted
        just on the bars where the intended position actually changes. That is
        the same transition convention as the other two strategies, and here it
        does more work than elsewhere: a positive-momentum regime lasting a year
        contains a dozen rebalance dates that all decide to stay invested, and
        only the first of them is a trade.

        A COMPARABILITY WARNING AGAINST THE CROSSOVER STRATEGY
        Momentum is a rule about a *state*, so it enters at the first rebalance
        date where the trailing return happens to be positive, whether or not
        that condition arose within the observed window. The crossover strategy,
        with its default enter_on_existing_trend=False, deliberately refuses that
        entry and waits for a crossing it can actually see. The difference is not
        a matter of one being better implemented: it is inherent to phrasing one
        rule as a state and the other as an event. When the two are compared,
        part of any gap is this convention rather than the signal, and the fair
        comparison sets the crossover's flag to True.

        No-look-ahead compliance: the trailing return is causal (see momentum),
        the position is carried forward and never backward (see target_position),
        and the only shift here is .shift(1), one bar into the past. During the
        warm-up the return is undefined, the target is flat, and the bar is HOLD.

        Args:
            data: OHLCV price history indexed by date, sorted chronologically.

        Returns:
            A Series of BUY / SELL / HOLD values aligned to data.index.
        """
        holding = self.target_position(data)
        previously_holding = holding.shift(1, fill_value=False)

        signals = self.hold_signals(data)
        signals[holding & ~previously_holding] = BUY
        signals[~holding & previously_holding] = SELL

        return signals


if __name__ == "__main__":
    from data.market_data import get_price_data

    prices = get_price_data("AAPL", "2020-01-01", "2023-01-01")
    strategy = Momentum()
    signals = strategy.generate_signals(prices)
    trailing = strategy.momentum(prices)
    holding = strategy.target_position(prices)

    reviews = (len(prices) - strategy.lookback + strategy.rebalance_freq - 1)
    reviews = max(reviews // strategy.rebalance_freq, 0)

    print(f"Strategy:        {strategy!r}")
    print(f"Bars:            {len(signals)}")
    print(f"Warm-up bars:    {trailing.isna().sum()}  (no earlier price to compare)")
    print(f"Rebalance dates: {reviews}")
    print(f"BUY signals:     {(signals == BUY).sum()}")
    print(f"SELL signals:    {(signals == SELL).sum()}")
    print(f"Bars invested:   {holding.sum()} of {len(holding)} "
          f"({holding.mean():.2%})")
    print(f"Momentum range:  {trailing.min():+.2%} to {trailing.max():+.2%}")

    events = signals[signals != HOLD]
    print("\nFirst signals:")
    for date, action in events.head(6).items():
        print(f"  {date.date()}  {action}  close="
              f"{prices.loc[date, 'Close']:7.2f}  "
              f"trailing return={trailing.loc[date]:+8.2%}")

    # Many reviews, few trades. The gap is the transition rule: a regime that
    # stays positive across a dozen rebalance dates is one entry, not a dozen.
    print(f"\n{reviews} reviews produced {(signals != HOLD).sum()} signals, "
          f"because only changes of mind are trades.")

    # Checking more often reacts sooner but trades more, which is free here and
    # will not be once Phase 4 charges for it.
    print("\nEffect of the review frequency:")
    for frequency in (1, 5, 21, 63):
        variant = Momentum(lookback=126, rebalance_freq=frequency)
        variant_signals = variant.generate_signals(prices)
        print(f"  every {frequency:3d} bars ->"
              f" {(variant_signals == BUY).sum():2d} BUY,"
              f" {(variant_signals == SELL).sum():2d} SELL,"
              f" invested {variant.target_position(prices).mean():6.2%}")

    # THE CAUSALITY CHECK
    # A signal at bar T that depended on anything after T would change when the
    # tail of the history is deleted. Recomputing on truncated data and
    # comparing the overlap is a direct test of the no-look-ahead contract. It
    # also checks the rebalance schedule specifically: had it been anchored to
    # the last bar of the sample rather than to the warm-up, truncation would
    # move every decision date and this would fail loudly.
    print("\nNo-look-ahead check (recompute on truncated history)")
    for fraction in (0.5, 0.7, 0.9):
        cut = int(len(prices) * fraction)
        truncated = strategy.generate_signals(prices.iloc[:cut])
        identical = truncated.equals(signals.iloc[:cut])
        drift = (trailing.iloc[:cut] - strategy.momentum(prices.iloc[:cut])).abs().max()
        print(f"  first {cut:4d} bars ({fraction:.0%})  signals identical: "
              f"{identical}   max momentum difference: {drift:.2e}")

    print("\nHand-checkable cases (lookback 5, checked every bar):")
    simple = Momentum(lookback=5, rebalance_freq=1)
    for label, closes in [
        ("rising throughout", [100.0 + i for i in range(12)]),
        ("falling throughout", [100.0 - i for i in range(12)]),
        ("flat throughout", [100.0] * 12),
        ("up then down", [100.0 + i for i in range(8)] + [107.0 - 3 * i for i in range(4)]),
    ]:
        frame = pd.DataFrame({"Close": closes})
        result = simple.generate_signals(frame)
        marks = "".join({BUY: "B", SELL: "S", HOLD: "."}[s] for s in result)
        print(f"  {label:<19} {marks}   "
              f"invested {simple.target_position(frame).mean():.0%}")

    print("\nToo-short history and rejected parameters:")
    short = prices.iloc[:50]
    print(f"  50 bars, lookback 126 -> all HOLD: "
          f"{(strategy.generate_signals(short) == HOLD).all()}")
    for arguments in ({"lookback": 0}, {"rebalance_freq": 0}, {"lookback": 12.5}):
        try:
            Momentum(**arguments)
        except ValueError as error:
            print(f"  {arguments} -> ValueError: {str(error)[:56]}...")
