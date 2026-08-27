"""Risk measures computed from an equity curve.

This module answers "how much did it hurt", separately from "how much did it
earn". It covers dispersion, through annualized volatility, and loss, through the
drawdown family. The risk-adjusted ratios that combine the two live in
analytics.metrics, which imports from here.

Like metrics, these functions know nothing about strategies, brokers or data
sources: hand them a Series of portfolio values indexed by date and they will
measure it, whatever produced it.

Sign conventions used throughout:
    - Returns are decimal fractions, not percentages. 0.25 means +25%.
    - Drawdowns are negative or zero. -0.25 means the portfolio was 25% below
      its previous peak.
    - Volatility is non-negative.
    - Measures that cannot be defined for the input return NaN rather than
      raising, so that a parameter sweep is never interrupted by a degenerate
      case. Durations, being counts, return 0 instead.
"""

from typing import Union

import numpy as np
import pandas as pd

from constants import TRADING_DAYS_PER_YEAR


def volatility(
    equity: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized standard deviation of the curve's periodic returns.

    Formula:
        volatility = std(returns, ddof=1) * sqrt(periods_per_year)

    The sample standard deviation is used, dividing by n - 1, which is pandas'
    default. The square root of time scaling assumes returns are independent
    across periods; serial correlation, which real strategies often have, makes
    this figure understate the true annual dispersion.

    Args:
        equity: Portfolio value over time.
        periods_per_year: Number of bars in a year, 252 for daily data.

    Returns:
        Annualized volatility as a decimal fraction, so 0.20 means 20% per year.
        Always non-negative. Returns NaN when fewer than two returns can be
        computed or periods_per_year is not positive.
    """
    returns = periodic_returns(equity)
    if len(returns) < 2 or periods_per_year <= 0:
        return float("nan")

    deviation = float(returns.std(ddof=1))
    if not np.isfinite(deviation):
        return float("nan")

    return deviation * float(np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown at every point of the curve.

    Formula:
        drawdown[t] = equity[t] / max(equity[0..t]) - 1

    The running maximum is the high-water mark: the best the portfolio had ever
    been worth up to that bar. The drawdown measures how far below that mark the
    portfolio currently sits, so it is 0 at every new peak and negative in
    between. It is a purely backward-looking quantity, which is what makes it
    usable as a live risk measure and not only as a post-mortem one.

    Args:
        equity: Portfolio value over time.

    Returns:
        A Series named drawdown, aligned to the input index, with values in
        (-1, 0]. Bars where the running peak is non-positive yield NaN, since a
        percentage decline from zero has no meaning. An empty input gives an
        empty Series.
    """
    equity = as_float_series(equity)
    if equity.empty:
        return pd.Series(dtype=float, name="drawdown")

    peak = equity.cummax()

    # Mathematically the ratio can never exceed 1, since the peak is by
    # definition at least the current value. Clipping only removes the floating
    # point dust that could otherwise show a drawdown of +1e-16.
    drawdown = (equity / peak - 1).where(peak > 0).clip(upper=0.0)
    drawdown.name = "drawdown"
    return drawdown


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline over the whole curve.

    This is the single number most investors react to first, because it answers
    "how much would I have been down at the worst moment", which is felt far
    more directly than volatility.

    Args:
        equity: Portfolio value over time.

    Returns:
        The most negative value of the drawdown series, as a decimal fraction,
        so -0.25 means a 25% peak-to-trough decline. Returns 0.0 for a curve
        that never declines, and NaN for an empty curve.
    """
    drawdown = drawdown_series(equity)
    if drawdown.empty:
        return float("nan")

    worst = drawdown.min()
    return float(worst) if pd.notna(worst) else float("nan")


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch spent below a previous peak, counted in bars.

    Depth is only half of what a drawdown costs. A 20% decline recovered in a
    month and a 20% decline that takes four years to recover are entirely
    different experiences, and only the second one gets investors to give up.
    This measures that second dimension.

    Definition used: the length of the longest run of consecutive bars whose
    drawdown is strictly negative. Counting starts on the first bar below the
    peak and ends on the bar before the peak is regained, so the peak bar itself
    and the recovery bar are excluded.

    Caveat worth knowing: if the curve ends while still below its peak, that
    final stretch is counted as it stands. Its real duration is unknown, since
    the recovery would happen after the data ends, so the figure is a lower
    bound in that case rather than a completed duration.

    Args:
        equity: Portfolio value over time.

    Returns:
        The longest underwater stretch in number of bars. Returns 0 for a curve
        that is empty or never below its peak.
    """
    drawdown = drawdown_series(equity)
    if drawdown.empty:
        return 0

    underwater = drawdown.fillna(0.0) < 0
    if not underwater.any():
        return 0

    # Each bar at or above the peak increments the counter, so every consecutive
    # underwater stretch shares one group id and the sum of the flags inside a
    # group is that stretch's length. Vectorised, no explicit loop over bars.
    stretch_ids = (~underwater).cumsum()
    return int(underwater.groupby(stretch_ids).sum().max())


def periodic_returns(equity: pd.Series) -> pd.Series:
    """Simple percentage returns from one bar to the next.

    Shared by the risk measures here and by the risk-adjusted ratios in
    analytics.metrics, so that both groups are always measuring the same return
    stream.

    The first bar has no predecessor and so no return, which is why a curve of
    n bars yields n - 1 returns. Non-finite results, which a zero or negative
    portfolio value would produce, are dropped rather than propagated.

    Args:
        equity: Portfolio value over time.

    Returns:
        A Series of periodic returns as decimal fractions, one shorter than the
        input. An input of fewer than two bars gives an empty Series.
    """
    equity = as_float_series(equity)
    if len(equity) < 2:
        return pd.Series(dtype=float)

    returns = equity.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def as_float_series(equity: Union[pd.Series, None]) -> pd.Series:
    """Coerce the input into a float Series, treating None as empty."""
    if equity is None:
        return pd.Series(dtype=float)
    if not isinstance(equity, pd.Series):
        equity = pd.Series(equity)
    return equity.astype(float)


if __name__ == "__main__":
    def as_equity(values: list) -> pd.Series:
        """Turn a list of portfolio values into a dated Series."""
        return pd.Series(
            values,
            index=pd.date_range("2022-01-03", periods=len(values), freq="D"),
            dtype=float,
        )

    def show(label: str, values: list) -> None:
        """Print every risk measure for a hand-checkable curve."""
        equity = as_equity(values)
        bars = len(values)
        print(f"\n{label}: {values}")
        print(f"  max_drawdown            {max_drawdown(equity):+.4f}")
        print(f"  max_drawdown_duration   {max_drawdown_duration(equity)} bars")
        print(f"  volatility ({bars}/yr)      {volatility(equity, bars):.4f}")
        print(f"  drawdown_series         "
              f"{[round(value, 4) for value in drawdown_series(equity)]}")

    # 100 -> 120 -> 90 -> 130. By hand: the peak is 120 and the trough 90, so
    # the worst drawdown is 90/120 - 1 = -0.25, lasting a single bar before the
    # new high at 130.
    show("Simple curve", [100, 120, 90, 130])

    # Same peak and trough, so the same max drawdown, but the recovery takes
    # three bars instead of one. That is exactly what the duration metric is
    # there to distinguish, and no other measure here can tell them apart.
    show("Slow recovery", [100, 120, 90, 95, 110, 130])

    print("\nEdge cases")
    empty = pd.Series(dtype=float)
    single = pd.Series([100.0])
    rising = as_equity([100.0, 101.0, 102.0, 103.0])
    print(f"  empty    -> max_dd={max_drawdown(empty)} "
          f"duration={max_drawdown_duration(empty)} "
          f"volatility={volatility(empty)}")
    print(f"  single   -> max_dd={max_drawdown(single)} "
          f"duration={max_drawdown_duration(single)} "
          f"volatility={volatility(single)}")
    print(f"  no loss  -> max_dd={max_drawdown(rising)} "
          f"duration={max_drawdown_duration(rising)} "
          f"(never below its peak)")
