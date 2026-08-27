"""Performance metrics computed from an equity curve.

This module answers "how much did it earn", and then "was the earning worth the
risk taken". The pure risk measures it needs for that second question, volatility
and the drawdown family, live in analytics.risk and are imported from there.

Every function here takes the portfolio value over time, as produced by the
backtester, and answers one question about it. They are deliberately free of any
notion of strategy, broker or data source: give them a Series of values indexed
by date and they will measure it, whatever produced it. That is what lets the
same functions grade a strategy and its benchmark on identical terms.

Sign conventions used throughout:
    - Returns are decimal fractions, not percentages. 0.25 means +25%.
    - Ratios are positive when the strategy is rewarded for its risk, negative
      when it is not.
    - Metrics that cannot be defined for the input return NaN rather than
      raising, so that a parameter sweep is never interrupted by a degenerate
      case.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from analytics.risk import as_float_series, max_drawdown, periodic_returns
from constants import TRADING_DAYS_PER_YEAR


def total_return(equity: pd.Series) -> float:
    """Total percentage change between the first and last value of the curve.

    Formula:
        total_return = equity[-1] / equity[0] - 1

    Args:
        equity: Portfolio value over time.

    Returns:
        The total return as a decimal fraction, so 0.30 means +30%. Returns 0.0
        for a single-value curve, since nothing has changed, and NaN if the
        curve is empty or starts at a non-positive value.
    """
    equity = as_float_series(equity)
    if equity.empty:
        return float("nan")

    start = float(equity.iloc[0])
    if start <= 0:
        return float("nan")

    return float(equity.iloc[-1] / start - 1)


def annualized_return(
    equity: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annual growth rate of the curve.

    Formula:
        years = (number of bars - 1) / periods_per_year
        cagr  = (equity[-1] / equity[0]) ** (1 / years) - 1

    Note that the exponent uses the number of *intervals*, one fewer than the
    number of bars: a curve of 253 daily values spans 252 days, that is one year.

    CAGR is what makes periods of different lengths comparable, but it is only
    meaningful over a reasonable span. Annualising a handful of bars produces
    arithmetically correct and practically absurd figures, because it
    extrapolates a short move over a full year.

    Args:
        equity: Portfolio value over time.
        periods_per_year: Number of bars in a year. 252 for daily trading data,
            52 for weekly, 12 for monthly.

    Returns:
        The annualized return as a decimal fraction. Returns NaN when the curve
        holds fewer than two values, when the first or last value is
        non-positive, or when periods_per_year is not positive, since in each
        case the growth rate is undefined rather than zero.
    """
    equity = as_float_series(equity)
    if len(equity) < 2 or periods_per_year <= 0:
        return float("nan")

    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return float("nan")

    years = (len(equity) - 1) / periods_per_year
    return float((end / start) ** (1 / years) - 1)


def sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized excess return per unit of total volatility.

    Formula:
        excess[t] = returns[t] - periodic risk-free rate
        sharpe    = mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)

    Conventions, all of which matter when comparing figures with another source:
        - risk_free_rate is an ANNUAL rate, de-annualized internally by
          compounding, (1 + rate) ** (1 / periods_per_year) - 1, rather than by
          simple division. The two differ negligibly at realistic rates, but
          compounding is consistent with how the returns themselves accumulate.
        - The numerator annualizes the arithmetic mean by multiplying by
          periods_per_year, which is the standard definition. It is not the same
          as the geometric annualized_return above and will read slightly higher.
        - Subtracting a constant from every return leaves the dispersion
          untouched, so the denominator is identical whether computed on excess
          or on raw returns.

    The metric assumes returns are normally distributed. Once the distribution
    group of Phase 3 shows skewness and fat tails, that assumption is measurably
    false, and Sharpe will be flattering precisely the strategies whose risk sits
    in the left tail it cannot see.

    Args:
        equity: Portfolio value over time.
        risk_free_rate: Annual risk-free rate as a decimal fraction, 0.04 for 4%.
        periods_per_year: Number of bars in a year, 252 for daily data.

    Returns:
        The annualized Sharpe ratio, negative when the strategy fails to beat
        the risk-free rate. Returns NaN when fewer than two returns exist, when
        volatility is zero, or when the inputs are out of range.
    """
    returns = periodic_returns(equity)
    if len(returns) < 2 or periods_per_year <= 0:
        return float("nan")

    periodic_risk_free = _deannualize(risk_free_rate, periods_per_year)
    if not np.isfinite(periodic_risk_free):
        return float("nan")

    excess = returns - periodic_risk_free
    deviation = float(excess.std(ddof=1))
    if not np.isfinite(deviation) or deviation <= 0:
        return float("nan")

    return float(excess.mean() / deviation * np.sqrt(periods_per_year))


def sortino_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized excess return per unit of downside deviation.

    The idea behind Sortino is that Sharpe punishes a strategy for its upside.
    An unexpectedly large gain raises the standard deviation and lowers Sharpe,
    even though no investor complains about it. Sortino keeps the same numerator
    and replaces the denominator with a measure of harmful movement only.

    HOW DOWNSIDE DEVIATION IS COMPUTED HERE
    Target semideviation about the risk-free rate:

        excess[t]          = returns[t] - periodic risk-free rate
        shortfall[t]       = min(excess[t], 0)
        downside_deviation = sqrt(mean(shortfall ** 2)) * sqrt(periods_per_year)
        sortino            = mean(excess) * periods_per_year / downside_deviation

    Two details in that definition are choices, not accidents:
        - Squared shortfalls are averaged over EVERY period, not only over the
          losing ones. Dividing by the number of losses instead would reward a
          strategy for losing rarely twice over, once through the smaller sum
          and once through the smaller divisor, and would make the figure
          incomparable between strategies that trade at different frequencies.
        - Deviation is measured from the target, the risk-free rate, not from
          the mean of the losses. This is what makes it a shortfall measure
          rather than the standard deviation of a filtered subset. Taking the
          plain std of the negative returns alone, a common shortcut, measures
          how much the losses differ from each other, which is a different and
          far less useful question. It would also call an asset that loses
          exactly 5% every single day riskless.

    Because the denominator ignores upside dispersion, Sortino is normally the
    higher of the two ratios. How much higher is the interesting part: a wide gap
    means the volatility was mostly favourable, while a narrow gap means the
    volatility was the losses themselves.

    Args:
        equity: Portfolio value over time.
        risk_free_rate: Annual risk-free rate as a decimal fraction.
        periods_per_year: Number of bars in a year, 252 for daily data.

    Returns:
        The annualized Sortino ratio. Returns NaN when fewer than two returns
        exist or when no period fell short of the target, since a strategy with
        no downside has no downside risk to divide by.
    """
    returns = periodic_returns(equity)
    if len(returns) < 2 or periods_per_year <= 0:
        return float("nan")

    periodic_risk_free = _deannualize(risk_free_rate, periods_per_year)
    if not np.isfinite(periodic_risk_free):
        return float("nan")

    excess = returns - periodic_risk_free
    shortfall = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt((shortfall**2).mean()))
    if not np.isfinite(downside_deviation) or downside_deviation <= 0:
        return float("nan")

    annualized_excess = float(excess.mean()) * periods_per_year
    return annualized_excess / (downside_deviation * float(np.sqrt(periods_per_year)))


def calmar_ratio(
    equity: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized return per unit of worst peak-to-trough decline.

    Formula:
        calmar = annualized_return / abs(max_drawdown)

    Where Sharpe divides by the average size of the wobbles, Calmar divides by
    the single worst one. That makes it far more sensitive to one bad episode,
    and much closer to how a real allocator thinks about a strategy: not "how
    noisy was it" but "how much did it hurt at the worst point, and was the
    return worth that".

    Args:
        equity: Portfolio value over time.
        periods_per_year: Number of bars in a year, 252 for daily data.

    Returns:
        The Calmar ratio, higher being better, negative when the annualized
        return is negative. Returns NaN when the curve never declined, since
        dividing by a zero drawdown is undefined, and when either input metric
        is itself undefined.
    """
    annual_return = annualized_return(equity, periods_per_year)
    worst_drawdown = max_drawdown(equity)

    if not np.isfinite(annual_return) or not np.isfinite(worst_drawdown):
        return float("nan")
    if worst_drawdown == 0:
        return float("nan")

    return float(annual_return / abs(worst_drawdown))


class CapmResult(NamedTuple):
    """Outcome of regressing strategy excess returns on market excess returns.

    Attributes:
        alpha: Annualized intercept, the return not explained by market exposure.
        beta: Slope, the strategy's sensitivity to market moves.
        r_squared: Fraction of the strategy's variance explained by the market,
            between 0 and 1.
    """

    alpha: float
    beta: float
    r_squared: float


def capm_regression(
    equity: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> CapmResult:
    """Regress the strategy's excess returns on the market's excess returns.

    Model:
        strategy_excess[t] = alpha_per_period + beta * market_excess[t] + error
        where excess means the return minus the per-period risk-free rate.

    HOW TO READ THE THREE NUMBERS
    Beta is market sensitivity. A beta of 1 means the strategy moves with the
    market one for one; 0.5 means it captures half of each move, up and down;
    above 1 means it amplifies them. Beta near 0 means the returns are unrelated
    to the market, which for a long-only equity strategy almost always means it
    was not invested rather than that it found something genuinely uncorrelated.

    Alpha is the part of the return that market exposure does not account for.
    It is the closest thing here to "value added": a strategy that returned 20%
    with a beta of 1 in a market that returned 20% has added nothing, and its
    alpha will be near zero. Positive alpha is the interesting result, and also
    the one to be most suspicious of, because it is exactly what overfitting
    manufactures.

    R squared is how much of the strategy's variance the market explains. High
    R squared means the strategy is essentially a leveraged or dampened version
    of the index, whatever it calls itself. Low R squared means its ups and downs
    came from somewhere else, and it also means the alpha and beta estimates are
    resting on a weak relationship and should be trusted less.

    THE CASH-HEAVY CASE, WHICH YOU WILL MEET IMMEDIATELY
    A strategy that sits in cash most of the time posts a return of exactly zero
    on every bar it is flat. Those zeros are uncorrelated with the market by
    construction, so beta collapses toward zero and R squared with it. That is
    not evidence of market neutrality or of skill at avoiding the index; it is
    just an absence of exposure, and the honest way to report it is alongside the
    fraction of time the strategy was actually invested. Beta measured over a
    period the strategy mostly sat out describes the cash, not the strategy.

    ALPHA IS REPORTED ANNUALIZED
    The regression yields a per-period intercept, which on daily data is a
    number around 1e-4 and impossible to read. It is multiplied by
    periods_per_year to give an annual figure, so 0.03 means three percentage
    points of annual return unexplained by the market. This is arithmetic
    scaling, not compounding, consistent with how sharpe_ratio annualizes its
    numerator. Beta and R squared are dimensionless and need no scaling.

    IMPLEMENTATION CHOICE
    scipy.stats.linregress is used. It is already a project dependency, and it
    returns precisely the three quantities needed, with the numerics handled by
    a well-tested library rather than by a hand-rolled covariance ratio.

    statsmodels is deliberately NOT added. It would give standard errors,
    t-statistics and p-values, and that is a real limitation worth being honest
    about: an alpha reported without a standard error is a point estimate, not a
    claim of statistical significance. Alpha is estimated far less precisely than
    beta, and on a few years of daily data its confidence interval is typically
    wide enough to contain zero. If this project ever needs to assert that an
    alpha is significant rather than merely positive, statsmodels earns its place
    then. Until that claim is made, it would be a dependency carried for nothing.

    Args:
        equity: Strategy portfolio value over time.
        benchmark: Market portfolio value over time, typically buy-and-hold.
        risk_free_rate: ANNUAL risk-free rate as a decimal fraction, de-annualized
            internally and subtracted from both return series.
        periods_per_year: Number of bars in a year, 252 for daily data. Used to
            annualize alpha and to de-annualize the risk-free rate.

    Returns:
        A CapmResult with annualized alpha, beta and R squared. All three are
        NaN when the two curves share fewer than three common dates, when the
        market's excess returns do not vary, or when the inputs are out of range.
        The two series are aligned on their common dates, so curves of different
        lengths are handled by intersection rather than by error.
    """
    undefined = CapmResult(float("nan"), float("nan"), float("nan"))

    if periods_per_year <= 0:
        return undefined

    periodic_risk_free = _deannualize(risk_free_rate, periods_per_year)
    if not np.isfinite(periodic_risk_free):
        return undefined

    aligned = pd.DataFrame(
        {
            "strategy": periodic_returns(equity),
            "market": periodic_returns(benchmark),
        }
    ).dropna()

    # Three points is the practical minimum: two would fit a line through both
    # observations exactly and report a meaningless R squared of 1.
    if len(aligned) < 3:
        return undefined

    strategy_excess = aligned["strategy"] - periodic_risk_free
    market_excess = aligned["market"] - periodic_risk_free

    if float(market_excess.std(ddof=0)) <= 0:
        return undefined

    regression = stats.linregress(market_excess, strategy_excess)

    return CapmResult(
        alpha=float(regression.intercept) * periods_per_year,
        beta=float(regression.slope),
        r_squared=float(regression.rvalue) ** 2,
    )


def _deannualize(annual_rate: float, periods_per_year: int) -> float:
    """Convert an annual rate into the equivalent rate for one bar."""
    if annual_rate <= -1.0 or periods_per_year <= 0:
        return float("nan")
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


if __name__ == "__main__":
    def as_equity(values: list) -> pd.Series:
        """Turn a list of portfolio values into a dated Series."""
        return pd.Series(
            values,
            index=pd.date_range("2022-01-03", periods=len(values), freq="D"),
            dtype=float,
        )

    def equity_from_returns(returns: list, start: float = 100.0) -> pd.Series:
        """Build the equity curve implied by a list of periodic returns."""
        values = [start]
        for periodic_return in returns:
            values.append(values[-1] * (1.0 + periodic_return))
        return as_equity(values)

    def show_returns(label: str, values: list) -> None:
        """Print the return metrics for a hand-checkable curve."""
        equity = as_equity(values)
        print(f"\n{label}: {values}")
        print(f"  total_return            {total_return(equity):+.4f}")
        print(f"  annualized (252/yr)     {annualized_return(equity):+.4f}")
        print(f"  annualized (4/yr)       {annualized_return(equity, 4):+.4f}")

    def show_risk_adjusted(label: str, returns: list) -> None:
        """Print the risk-adjusted metrics for a curve built from returns."""
        equity = equity_from_returns(returns)
        periods = len(returns)

        # Recomputed here straight from the input list, independently of the
        # module, so the printed ratios can be checked rather than trusted.
        as_array = np.array(returns, dtype=float)
        total_deviation = as_array.std(ddof=1)
        downside_deviation = np.sqrt(np.mean(np.minimum(as_array, 0.0) ** 2))

        sharpe = sharpe_ratio(equity, periods_per_year=periods)
        sortino = sortino_ratio(equity, periods_per_year=periods)

        print(f"\n{label}: {periods} returns, "
              f"{int((as_array < 0).sum())} of them negative")
        print(f"  mean return / period    {as_array.mean():+.6f}")
        print(f"  std / period            {total_deviation:.6f}")
        print(f"  downside dev / period   {downside_deviation:.6f}")
        print(f"  sharpe     ({periods}/yr)     {sharpe:+.4f}")
        print(f"  sortino    ({periods}/yr)     {sortino:+.4f}")
        print(f"  calmar     ({periods}/yr)     {calmar_ratio(equity, periods):+.4f}")
        print(f"  sortino / sharpe        {sortino / sharpe:.4f}"
              f"   (equals std / downside dev = "
              f"{total_deviation / downside_deviation:.4f})")

    # 100 -> 120 -> 90 -> 130. By hand: total return 130/100 - 1 = +0.30.
    show_returns("Simple curve", [100, 120, 90, 130])

    # Same first and last value, so the same total return: this pair shows that
    # return metrics alone cannot distinguish two very different journeys. The
    # drawdown measures in analytics.risk are what separate them.
    show_returns("Slow recovery", [100, 120, 90, 95, 110, 130])

    # Eight gains of 2% against two small losses. The dispersion is dominated by
    # the upside, which the downside deviation ignores entirely, so Sortino
    # should come out several times larger than Sharpe. The exact multiple is
    # not arbitrary: since both ratios share the same numerator, their quotient
    # must equal std / downside deviation, which the output verifies.
    show_risk_adjusted(
        "Positive skew, gains dominate the dispersion",
        [0.02, 0.02, -0.01, 0.02, 0.02, 0.02, -0.005, 0.02, 0.02, 0.02],
    )

    # Nine steady gains of 1% wiped out by a single 5% loss. Here the volatility
    # IS the loss, so removing the upside from the denominator barely changes it
    # and Sortino ends up only slightly above Sharpe. Same numerator, same
    # ordering, but the gap between the two ratios has almost vanished, which is
    # the honest signal that this return stream carries genuine tail risk.
    show_risk_adjusted(
        "Negative skew, one loss dominates the dispersion",
        [0.01] * 9 + [-0.05],
    )

    # CAPM recovery test. Strategy returns are built from a known relationship,
    # so the regression has a right answer to be checked against rather than
    # merely a plausible-looking one.
    BETA_TRUE = 0.60
    ALPHA_TRUE = 0.03
    PERIODS = 252
    BARS = 2000

    generator = np.random.default_rng(7)
    market_returns = generator.normal(0.0004, 0.011, BARS)
    noise = generator.normal(0.0, 0.001, BARS)
    built_returns = ALPHA_TRUE / PERIODS + BETA_TRUE * market_returns + noise

    market_equity = equity_from_returns(market_returns)
    built_equity = equity_from_returns(built_returns)
    recovered = capm_regression(
        built_equity, market_equity, periods_per_year=PERIODS
    )

    print(f"\nCAPM recovery test: {BARS} bars of "
          f"alpha={ALPHA_TRUE:.2%}/yr + beta={BETA_TRUE} * market + noise")
    print(f"  beta      true {BETA_TRUE:8.4f}   recovered {recovered.beta:8.4f}")
    print(f"  alpha     true {ALPHA_TRUE:8.4f}   recovered {recovered.alpha:8.4f}")
    print(f"  r_squared                  recovered {recovered.r_squared:8.4f}")

    # The cash-heavy case the docstring warns about: the same market, but the
    # strategy is only exposed for the first 40 of 2000 bars and flat otherwise.
    # Beta and R squared both collapse, and neither figure says anything about
    # skill. They describe a portfolio that was almost never in the market.
    cash_heavy_returns = np.zeros(BARS)
    cash_heavy_returns[:40] = market_returns[:40]
    cash_heavy = capm_regression(
        equity_from_returns(cash_heavy_returns),
        market_equity,
        periods_per_year=PERIODS,
    )

    print(f"\nCash-heavy strategy: fully exposed on 40 of {BARS} bars, "
          f"flat on the rest")
    print(f"  beta      {cash_heavy.beta:8.4f}   (exposure was 1.0 while invested)")
    print(f"  alpha     {cash_heavy.alpha:8.4f}")
    print(f"  r_squared {cash_heavy.r_squared:8.4f}")

    print("\nEdge cases")
    empty = pd.Series(dtype=float)
    single = pd.Series([100.0])
    rising = as_equity([100.0, 101.0, 102.0, 103.0])
    print(f"  empty    -> total_return={total_return(empty)} "
          f"sharpe={sharpe_ratio(empty)}")
    print(f"  single   -> total_return={total_return(single)} "
          f"annualized={annualized_return(single)} "
          f"sharpe={sharpe_ratio(single)}")
    print(f"  no loss  -> sortino={sortino_ratio(rising)} "
          f"calmar={calmar_ratio(rising)} "
          f"(both undefined: no downside, no drawdown)")
    print(f"  no overlap -> capm={capm_regression(rising, empty)}")
    print(f"  flat market -> capm="
          f"{capm_regression(rising, as_equity([100.0] * 5))}")
