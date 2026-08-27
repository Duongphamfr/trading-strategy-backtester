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
from scipy.stats import norm

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


def historical_var(equity: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk read straight off the observed return distribution.

    Formula:
        var = -quantile(returns, 1 - confidence)

    At 95% confidence this is the 5th percentile of the periodic returns, sign
    flipped. The reading is: on the worst 5% of bars, the loss was at least this
    large. Note what it does not say, which is where VaR gets its bad
    reputation: it puts a floor under the bad cases and says nothing whatsoever
    about how far past that floor they went. That is CVaR's job.

    PER-PERIOD FIGURE, NOT ANNUALIZED. On daily data this is a one-day VaR.
    Scaling it by sqrt(252) to get an annual figure would import the very
    normality assumption this function was chosen to avoid.

    SIGN CONVENTION: returned as a positive loss magnitude, so 0.021 means a
    2.1% loss. This is the usual presentation and it makes the three tail
    measures directly comparable. A negative result is possible and meaningful:
    it says the quantile itself was a gain, so the curve did not lose even in
    its worst bars.

    Args:
        equity: Portfolio value over time.
        confidence: Confidence level in (0, 1), 0.95 for the 5th percentile.

    Returns:
        The per-period VaR as a decimal fraction. Returns NaN when fewer than
        two returns exist or confidence is outside (0, 1).
    """
    returns = periodic_returns(equity)
    if len(returns) < 2 or not 0.0 < confidence < 1.0:
        return float("nan")

    quantile = float(returns.quantile(1.0 - confidence))
    if not np.isfinite(quantile):
        return float("nan")

    return -quantile


def parametric_var(equity: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk under the assumption that returns are normal.

    Formula:
        z   = normal quantile at (1 - confidence), about -1.645 at 95%
        var = -(mean(returns) + z * std(returns, ddof=1))

    Only two numbers from the sample are used, its mean and its standard
    deviation. Everything else about the shape of the distribution is discarded
    and replaced by the assumption of a bell curve.

    PER-PERIOD FIGURE, NOT ANNUALIZED, exactly as for the historical version.

    SIGN CONVENTION: positive means a loss, identical to historical_var, so the
    two can be printed side by side and subtracted.

    THE LIMITATION THAT MATTERS
    Real return distributions have fat tails: extreme moves happen far more
    often than a normal distribution allows. Because this estimator knows only
    the standard deviation, it cannot see them, and it systematically
    understates how bad the extremes get. That is precisely what the skewness
    and kurtosis metrics are for, and why they belong next to this number
    rather than in a separate curiosity section.

    One subtlety worth stating, because it is where a careless demonstration
    goes wrong: the understatement is not uniform across confidence levels. A
    handful of violent outliers inflates the standard deviation, which can push
    parametric VaR *above* the historical figure at 95%, while still falling
    badly short at 99% and beyond, where the true tail lives. Comparing the two
    at a single confidence level therefore proves very little; comparing them
    across levels, and against CVaR, is what exposes the problem.

    Args:
        equity: Portfolio value over time.
        confidence: Confidence level in (0, 1), 0.95 for a 1.645-sigma move.

    Returns:
        The per-period parametric VaR as a decimal fraction. Returns NaN when
        fewer than two returns exist or confidence is outside (0, 1).
    """
    returns = periodic_returns(equity)
    if len(returns) < 2 or not 0.0 < confidence < 1.0:
        return float("nan")

    mean = float(returns.mean())
    deviation = float(returns.std(ddof=1))
    if not np.isfinite(mean) or not np.isfinite(deviation):
        return float("nan")

    z_score = float(norm.ppf(1.0 - confidence))
    return -(mean + z_score * deviation)


def conditional_var(equity: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall: the average loss once the VaR threshold is breached.

    Formula:
        threshold = quantile(returns, 1 - confidence)
        cvar      = -mean(returns where returns <= threshold)

    Where VaR asks "how bad does it get before the worst 5% of days", CVaR asks
    "and how bad are those 5% on average". It therefore answers the question VaR
    dodges, and it is the reason CVaR is preferred in modern risk work: two
    portfolios can share a VaR while one loses 5% in its bad tail and the other
    loses 40%.

    By construction CVaR is always at least as large as the historical VaR at
    the same confidence, since it averages values that are all at or below the
    threshold. If the two are close, the tail stops right at the threshold; a
    wide gap means the tail keeps going, which is exactly the fat-tail
    signature that parametric VaR cannot represent.

    PER-PERIOD FIGURE, NOT ANNUALIZED.

    SIGN CONVENTION: positive means a loss, as for both VaR functions.

    Args:
        equity: Portfolio value over time.
        confidence: Confidence level in (0, 1), 0.95 to average the worst 5%.

    Returns:
        The per-period CVaR as a decimal fraction. Returns NaN when fewer than
        two returns exist or confidence is outside (0, 1).
    """
    returns = periodic_returns(equity)
    if len(returns) < 2 or not 0.0 < confidence < 1.0:
        return float("nan")

    threshold = returns.quantile(1.0 - confidence)
    if not np.isfinite(threshold):
        return float("nan")

    # The sample minimum is always at or below the quantile, so the tail is
    # never empty. Selecting by value rather than by count means the tail may
    # hold slightly more bars than (1 - confidence) * n when values repeat,
    # which is the standard treatment of ties.
    tail = returns[returns <= threshold]
    if tail.empty:
        return float("nan")

    return -float(tail.mean())


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

    def equity_from_returns(returns, start: float = 100.0) -> pd.Series:
        """Build the equity curve implied by a sequence of periodic returns."""
        values = [start]
        for periodic_return in returns:
            values.append(values[-1] * (1.0 + float(periodic_return)))
        return as_equity(values)

    def show_tail_risk(label: str, returns) -> None:
        """Print the three tail measures at two confidence levels."""
        equity = equity_from_returns(returns)
        sample = np.asarray(returns, dtype=float)

        print(f"\n{label}")
        print(f"  {len(sample)} returns, mean {sample.mean():+.5f}, "
              f"std {sample.std(ddof=1):.5f}")

        for confidence in (0.95, 0.99):
            historical = historical_var(equity, confidence)
            parametric = parametric_var(equity, confidence)
            expected_shortfall = conditional_var(equity, confidence)

            print(f"  at {confidence:.0%} confidence")
            print(f"    historical VaR      {historical:8.4%}")
            print(f"    parametric VaR      {parametric:8.4%}")
            print(f"    CVaR                {expected_shortfall:8.4%}")
            print(f"    historical / param  {historical / parametric:8.2f}")
            print(f"    CVaR / historical   "
                  f"{expected_shortfall / historical:8.2f}"
                  f"   (CVaR >= VaR: "
                  f"{expected_shortfall >= historical - 1e-12})")

    # Drawn from an actual normal distribution, so the assumption behind
    # parametric VaR holds by construction. The two estimates should land close
    # together, which is the control case: any gap seen elsewhere is a property
    # of the data, not an artefact of the code.
    generator = np.random.default_rng(42)
    show_tail_risk(
        "Near-normal returns (the assumption holds)",
        generator.normal(0.0005, 0.01, 500),
    )

    # 93 quiet gains and a left tail that keeps going: -3% down to -10%. Seven
    # percent of the mass sits in the tail, so the 5% quantile lands inside it
    # and historical VaR already exceeds the parametric estimate. The real
    # verdict is the 99% row, where the gap widens sharply, and the CVaR/VaR
    # multiple, which stays far above what a bell curve would produce.
    show_tail_risk(
        "Fat-tailed returns (the assumption fails)",
        [0.005] * 93 + [-0.03, -0.035, -0.04, -0.05, -0.06, -0.08, -0.10],
    )

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
    print(f"  no loss  -> historical_var={historical_var(rising):.4%} "
          f"cvar={conditional_var(rising):.4%} "
          f"(negative: even the worst bars were gains)")
    print(f"  bad conf -> historical_var={historical_var(rising, 1.5)} "
          f"parametric_var={parametric_var(rising, 0.0)}")
