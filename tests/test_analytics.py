"""Automated tests for the analytics layer: metrics, risk, trade statistics.

These convert the self-checks from the __main__ blocks of the analytics modules
into assertions, and add the degenerate-input coverage those blocks only sampled.

WHERE THE EXPECTED VALUES COME FROM
Almost every number below is either derived on paper or an identity that must
hold whatever the data. Both are stronger than a value read off a previous run,
which can only ever confirm that the code still does what it did.

The paper derivations lean on one trick: choosing periods_per_year to make the
arithmetic come out round. The curve 100 -> 120 -> 90 -> 130 has four bars, so
with three periods per year it spans exactly one year, and the annualized return
must equal the total return of +30% precisely. That is also the cleanest possible
test of the (n - 1) exponent, since a version counting bars instead of intervals
would return +22.1% here.

The identities are the more interesting half, because they constrain the code
without anyone having to know the right answer:

    Sortino divided by Sharpe must equal total deviation over downside
    deviation, since the two ratios share a numerator.

    A constant-loss series must give a Sortino of exactly -sqrt(periods_per_year),
    which is the counterexample that settled the definition: reading "downside
    deviation" as the standard deviation of the losing periods alone would make
    an asset that loses 5% every single day look riskless, and Sortino infinite.

    CVaR can never be smaller than historical VaR, since it averages values that
    all sit at or below the VaR threshold.

    The Jarque-Bera statistic must equal n / 6 * (skew^2 + excess_kurtosis^2 / 4)
    computed from this module's own skewness and kurtosis, which ties the three
    distribution functions together and would catch one of them drifting onto a
    different estimator convention.

    Regressing the benchmark on itself must give beta 1, alpha 0 and R-squared 1.

WHY THE DEGENERATE CASES GET THIS MUCH ATTENTION
A parameter sweep runs hundreds of backtests, and some combinations genuinely
never trade, never lose, or never move. Every metric here is documented to return
NaN rather than raise in those cases, and that promise is what keeps a sweep from
dying two thirds of the way through on a combination nobody cared about. It is
also invisible in normal use, so it is exactly the kind of behaviour that rots
without a test watching it.
"""

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics.metrics import (
    annualized_return,
    calmar_ratio,
    capm_regression,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from analytics.risk import (
    conditional_var,
    drawdown_series,
    exposure,
    historical_var,
    jarque_bera,
    kurtosis,
    max_drawdown,
    max_drawdown_duration,
    parametric_var,
    periodic_returns,
    skewness,
    volatility,
)
from analytics.trade_stats import round_trips, trade_statistics
from constants import BUY, SELL, TRADING_DAYS_PER_YEAR


def as_equity(values: List[float]) -> pd.Series:
    """Turn a list of portfolio values into a dated Series."""
    return pd.Series(
        values,
        index=pd.date_range("2022-01-03", periods=len(values), freq="D"),
        dtype=float,
    )


def equity_from_returns(returns, start: float = 100.0) -> pd.Series:
    """Build the equity curve implied by a sequence of periodic returns."""
    values = [start]
    for periodic_return in returns:
        values.append(values[-1] * (1.0 + float(periodic_return)))
    return as_equity(values)


def trade(day: int, action: str, price: float, shares: float) -> Dict[str, Any]:
    """One trade log entry, dated for readability."""
    return {
        "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=day),
        "action": action,
        "price": float(price),
        "shares": float(shares),
    }


# The reference curve, used by every drawdown-based expectation below.
#   peak:     100, 120, 120, 130
#   drawdown:   0,   0, 90 / 120 - 1 = -0.25, 0
# So the worst decline is 25%, and it lasts a single bar before the new high.
SIMPLE_CURVE = [100.0, 120.0, 90.0, 130.0]

# Same peak and same trough, so the same depth, but the climb back takes three
# bars. Depth alone cannot tell these two apart, which is the whole reason the
# duration metric exists.
SLOW_RECOVERY_CURVE = [100.0, 120.0, 90.0, 95.0, 110.0, 130.0]

# Ninety-three quiet gains and a left tail that keeps going. Reused verbatim from
# the risk module's own demonstration so the two describe the same sample.
FAT_TAILED_RETURNS = [0.005] * 93 + [
    -0.03, -0.035, -0.04, -0.05, -0.06, -0.08, -0.10,
]


@pytest.fixture
def near_normal_returns() -> np.ndarray:
    """Five hundred draws from an actual normal distribution.

    The control case: the assumption behind parametric VaR holds here by
    construction, so any gap seen on the fat-tailed sample is a property of that
    data rather than an artefact of the code. Reused from the risk module's
    demonstration, seed included. numpy guarantees a stable stream for a seeded
    default_rng, so this is reproducible rather than merely probable.
    """
    return np.random.default_rng(42).normal(0.0005, 0.01, 500)


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def test_total_return_is_the_ratio_of_last_to_first():
    assert total_return(as_equity(SIMPLE_CURVE)) == pytest.approx(0.30, rel=1e-12)


def test_total_return_cannot_distinguish_two_different_journeys():
    """Both curves start at 100 and end at 130, so both return +30%."""
    assert total_return(as_equity(SIMPLE_CURVE)) == pytest.approx(
        total_return(as_equity(SLOW_RECOVERY_CURVE)), rel=1e-12)


def test_total_return_is_negative_when_the_curve_falls():
    assert total_return(as_equity([100.0, 75.0])) == pytest.approx(-0.25,
                                                                   rel=1e-12)


def test_annualized_return_counts_intervals_not_bars():
    """The (n - 1) exponent, isolated.

    Four bars at three periods per year span exactly one year, so the annualized
    figure must equal the total return of +30% to the last digit. Dividing by the
    bar count instead would give (1.3 ** (3 / 4)) - 1, about +22.1%, which is why
    this case pins the off-by-one rather than merely being consistent with it.
    """
    equity = as_equity(SIMPLE_CURVE)

    assert annualized_return(equity, periods_per_year=3) == pytest.approx(
        0.30, rel=1e-12)
    assert annualized_return(equity, periods_per_year=3) != pytest.approx(
        1.3 ** 0.75 - 1, rel=1e-6)


def test_annualized_return_of_a_curve_spanning_exactly_one_trading_year():
    """253 daily values span 252 intervals, that is one year."""
    equity = as_equity([100.0] * 252 + [130.0])

    assert len(equity) == TRADING_DAYS_PER_YEAR + 1
    assert annualized_return(equity) == pytest.approx(0.30, rel=1e-12)


def test_annualized_return_compounds_over_multiple_years():
    """Two years of doubling: 4x overall is 100% a year."""
    equity = as_equity([100.0] * 8 + [400.0])

    assert annualized_return(equity, periods_per_year=4) == pytest.approx(
        1.0, rel=1e-12)


def test_a_flat_curve_neither_returns_nor_grows():
    equity = as_equity([100.0] * 10)

    assert total_return(equity) == pytest.approx(0.0, abs=1e-15)
    assert annualized_return(equity, periods_per_year=9) == pytest.approx(
        0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Risk-adjusted ratios
# ---------------------------------------------------------------------------

# Two returns of +1% and +3%, at two periods per year. By hand:
#   mean            = 0.02
#   std (ddof = 1)  = sqrt(0.0002) = 0.01 * sqrt(2)
#   sharpe          = 0.02 / (0.01 * sqrt(2)) * sqrt(2) = 2 exactly
#   volatility      = 0.01 * sqrt(2) * sqrt(2) = 0.02 exactly
TWO_GAIN_CURVE = [100.0, 101.0, 104.03]


def test_sharpe_ratio_on_a_hand_worked_pair_of_returns():
    assert sharpe_ratio(as_equity(TWO_GAIN_CURVE),
                        periods_per_year=2) == pytest.approx(2.0, rel=1e-9)


def test_volatility_on_a_hand_worked_pair_of_returns():
    assert volatility(as_equity(TWO_GAIN_CURVE),
                      periods_per_year=2) == pytest.approx(0.02, rel=1e-9)


def test_a_positive_risk_free_rate_lowers_the_sharpe_ratio():
    equity = as_equity(TWO_GAIN_CURVE)

    assert sharpe_ratio(equity, risk_free_rate=0.10, periods_per_year=2) < \
        sharpe_ratio(equity, risk_free_rate=0.0, periods_per_year=2)


def test_sharpe_is_negative_when_the_curve_loses():
    equity = equity_from_returns([-0.01, -0.03])

    assert sharpe_ratio(equity, periods_per_year=2) == pytest.approx(-2.0,
                                                                     rel=1e-9)


@pytest.mark.parametrize("periods_per_year", [4, 12, 52, 252])
def test_constant_losses_give_a_finite_sortino_of_minus_root_periods(
        periods_per_year):
    """The counterexample that settled the downside-deviation definition.

    An asset losing exactly 5% every period has zero dispersion among its
    losses. Reading downside deviation as the standard deviation of the losing
    periods alone would therefore call it riskless and report an infinite
    Sortino. Target semideviation measures the shortfall from the target
    instead, so the denominator is the 5% itself.

    Working it out, with r the constant loss and P the periods per year:
        downside deviation = sqrt(mean(r ** 2)) = |r|
        sortino = r * P / (|r| * sqrt(P)) = -sqrt(P)

    The loss size cancels entirely, which makes this an exact expectation with
    nothing fitted to it.
    """
    equity = equity_from_returns([-0.05] * periods_per_year)

    assert sortino_ratio(equity, periods_per_year=periods_per_year) == \
        pytest.approx(-np.sqrt(periods_per_year), rel=1e-9)


# Halving three times. Powers of two are exact in binary, so every pct_change
# here is exactly -0.5 and the sample standard deviation is exactly zero. Built
# by compounding instead, as equity_from_returns does, the ratios differ in their
# last bits and the deviation lands around 1e-17 rather than at 0.
EXACTLY_CONSTANT_LOSS_CURVE = [8.0, 4.0, 2.0, 1.0]


def test_exactly_constant_losses_leave_sharpe_undefined_but_sortino_defined():
    """The two ratios disagree here, and both are right to.

    Sharpe divides by total dispersion, which is exactly zero on this curve, so
    it has nothing to say. Sortino divides by the shortfall from the target,
    which is emphatically not zero, and reports -sqrt(3) as the identity above
    requires.
    """
    equity = as_equity(EXACTLY_CONSTANT_LOSS_CURVE)

    assert np.isnan(sharpe_ratio(equity, periods_per_year=3))
    assert sortino_ratio(equity, periods_per_year=3) == pytest.approx(
        -np.sqrt(3.0), rel=1e-12)


def test_sortino_survives_constant_losses_whichever_way_the_curve_was_built():
    """Sortino is robust to the arithmetic; Sharpe has nothing left to divide by.

    The two curves describe the same idea, a constant loss every period, and
    differ only in floating point dust. Sortino is unmoved, since its
    denominator is the size of the loss. Sharpe's denominator IS that dust, so
    it correctly declines to answer. This is the case that decided the
    downside-deviation definition.
    """
    for equity, periods in ((as_equity(EXACTLY_CONSTANT_LOSS_CURVE), 3),
                            (equity_from_returns([-0.05] * 10), 10)):
        assert np.isnan(sharpe_ratio(equity, periods_per_year=periods))
        assert sortino_ratio(equity, periods_per_year=periods) == pytest.approx(
            -np.sqrt(periods), rel=1e-6)


def test_near_constant_returns_do_not_produce_a_giant_spurious_sharpe():
    """The regression lock on the relative near-zero-deviation guard.

    Compounding a constant loss into an equity curve and differencing it back
    into returns leaves a dispersion of about 1e-17 where the arithmetic says
    zero. An absolute deviation <= 0 test waves that residue through, and Sharpe
    then divides a real mean by numerical dust: this curve used to report
    -2.5e15.

    The sign made that instance obvious, but the danger is the positive one. A
    parameter sweep ranks combinations by Sharpe, so a spurious fifteen-digit
    value wins outright, and the combination earning it is the least interesting
    on the grid, having merely managed to be numerically constant.
    """
    equity = equity_from_returns([-0.05] * 10)
    returns = periodic_returns(equity)

    # The premise: there really is a positive residue here, so the test is
    # exercising the guard rather than a trivially exact zero.
    dispersion = float(returns.std(ddof=1))
    assert 0.0 < dispersion < 1e-15

    assert np.isnan(sharpe_ratio(equity, periods_per_year=10))


def test_the_deviation_guard_is_relative_so_low_volatility_still_counts():
    """The guard must reject dust without rejecting genuinely quiet returns.

    This is what a relative test buys over simply raising the absolute
    threshold. The dispersion here is tiny in absolute terms, far below any
    fixed cutoff that would have caught the 1e-17 residue, but it is a real
    feature of the data rather than rounding error, and Sharpe must still answer.

    Asserted against the formula recomputed from the returns rather than against
    a magnitude, since the magnitude of a nearly riskless return stream is
    legitimately enormous and there is nothing to be learned from pinning it.
    """
    equity = equity_from_returns([0.02] * 40 + [0.02 + 1e-8] + [0.02] * 40)
    returns = periodic_returns(equity)

    dispersion = float(returns.std(ddof=1))
    scale = float(returns.abs().mean())

    # Real but small: well above the noise floor, well below ordinary dispersion.
    assert 1e-12 < dispersion / scale < 1e-3

    sharpe = sharpe_ratio(equity, periods_per_year=252)

    assert np.isfinite(sharpe)
    assert sharpe == pytest.approx(
        float(returns.mean()) / dispersion * np.sqrt(252), rel=1e-9)


def test_the_guard_leaves_an_ordinary_return_series_untouched():
    """Nothing about a realistic curve should come near the guard."""
    equity = equity_from_returns(
        np.random.default_rng(1).normal(0.0005, 0.01, 500))
    returns = periodic_returns(equity)

    # Ordinary dispersion is of the same order as the returns themselves, twelve
    # orders of magnitude clear of the threshold.
    assert float(returns.std(ddof=1)) / float(returns.abs().mean()) > 0.1

    sharpe = sharpe_ratio(equity)

    assert np.isfinite(sharpe)
    assert sharpe == pytest.approx(
        float(returns.mean()) / float(returns.std(ddof=1))
        * np.sqrt(TRADING_DAYS_PER_YEAR), rel=1e-9)


def test_sortino_needs_no_equivalent_guard():
    """Records why the fix was applied to Sharpe alone.

    Sortino's denominator is the root mean squared shortfall from the target. It
    can only collapse to dust when almost every excess return is non-negative,
    and in exactly that situation the mean excess in the numerator is itself
    near zero. The two ends collapse together, so the quotient stays bounded
    instead of exploding the way Sharpe's did.

    Constructed here by setting the risk-free rate so the excess returns
    straddle zero, which is the only way to get a dust-sized shortfall out of a
    constant-return curve.
    """
    periods = 252
    periodic = 0.05
    annual = (1.0 + periodic) ** periods - 1.0
    equity = equity_from_returns([periodic] * 300)

    sortino = sortino_ratio(equity, risk_free_rate=annual,
                            periods_per_year=periods)

    assert np.isnan(sortino) or abs(sortino) < 10.0


def test_sortino_over_sharpe_equals_total_over_downside_deviation():
    """An identity, since the two ratios share their numerator exactly."""
    returns = [0.02, 0.02, -0.01, 0.02, 0.02, 0.02, -0.005, 0.02, 0.02, 0.02]
    equity = equity_from_returns(returns)
    periods = len(returns)

    sample = np.array(returns, dtype=float)
    total_deviation = sample.std(ddof=1)
    downside_deviation = np.sqrt(np.mean(np.minimum(sample, 0.0) ** 2))

    ratio = (sortino_ratio(equity, periods_per_year=periods)
             / sharpe_ratio(equity, periods_per_year=periods))

    assert ratio == pytest.approx(total_deviation / downside_deviation, rel=1e-6)


def test_sortino_exceeds_sharpe_when_the_dispersion_is_mostly_upside():
    """Eight gains of 2% against two small losses."""
    equity = equity_from_returns(
        [0.02, 0.02, -0.01, 0.02, 0.02, 0.02, -0.005, 0.02, 0.02, 0.02])

    sharpe = sharpe_ratio(equity, periods_per_year=10)
    sortino = sortino_ratio(equity, periods_per_year=10)

    assert sortino > 2.0 * sharpe


def test_the_two_ratios_nearly_agree_when_the_dispersion_is_the_loss():
    """Nine gains of 1% wiped out by a single 5% loss.

    Here the volatility essentially is the loss, so removing the upside from the
    denominator barely moves it. The narrow gap is the honest signal that this
    return stream carries real tail risk, and it is the opposite reading to the
    test above on the same scale.
    """
    equity = equity_from_returns([0.01] * 9 + [-0.05])

    sharpe = sharpe_ratio(equity, periods_per_year=10)
    sortino = sortino_ratio(equity, periods_per_year=10)

    assert 1.0 < sortino / sharpe < 1.5


def test_calmar_is_the_annualized_return_over_the_worst_decline():
    """On the reference curve: 0.30 / 0.25 = 1.2 exactly."""
    equity = as_equity(SIMPLE_CURVE)

    assert calmar_ratio(equity, periods_per_year=3) == pytest.approx(1.2,
                                                                     rel=1e-12)


def test_calmar_punishes_the_slower_recovery_less_than_duration_does():
    """Both curves share a depth, so Calmar cannot separate them either.

    Only the annualization differs, because the second curve took more bars to
    reach the same place. Worth asserting so the division of labour between
    Calmar and max_drawdown_duration stays explicit.
    """
    fast = calmar_ratio(as_equity(SIMPLE_CURVE), periods_per_year=3)
    slow = calmar_ratio(as_equity(SLOW_RECOVERY_CURVE), periods_per_year=5)

    assert max_drawdown(as_equity(SIMPLE_CURVE)) == pytest.approx(
        max_drawdown(as_equity(SLOW_RECOVERY_CURVE)), rel=1e-12)
    assert fast == pytest.approx(slow, rel=1e-12)


# ---------------------------------------------------------------------------
# Drawdown family
# ---------------------------------------------------------------------------

def test_drawdown_series_is_zero_at_peaks_and_negative_between():
    drawdown = drawdown_series(as_equity(SIMPLE_CURVE))

    assert list(drawdown.round(10)) == [0.0, 0.0, -0.25, 0.0]
    assert drawdown.name == "drawdown"


def test_drawdown_series_never_exceeds_zero_or_falls_below_minus_one():
    drawdown = drawdown_series(as_equity(SLOW_RECOVERY_CURVE))

    assert (drawdown <= 0.0).all()
    assert (drawdown > -1.0).all()


def test_max_drawdown_is_minus_twenty_five_percent_on_the_reference_curve():
    assert max_drawdown(as_equity(SIMPLE_CURVE)) == pytest.approx(-0.25,
                                                                  rel=1e-12)


def test_max_drawdown_ignores_the_shape_of_the_recovery():
    assert max_drawdown(as_equity(SLOW_RECOVERY_CURVE)) == pytest.approx(
        -0.25, rel=1e-12)


def test_max_drawdown_duration_is_one_bar_on_the_reference_curve():
    """Only the bar at 90 sits below the peak of 120."""
    assert max_drawdown_duration(as_equity(SIMPLE_CURVE)) == 1


def test_max_drawdown_duration_is_three_bars_on_the_slow_recovery():
    """The bars at 90, 95 and 110 are all below the peak of 120."""
    assert max_drawdown_duration(as_equity(SLOW_RECOVERY_CURVE)) == 3


def test_max_drawdown_duration_takes_the_longest_stretch_not_the_last():
    # Two separate dips: a four-bar one, then a one-bar one.
    equity = as_equity([100.0, 90.0, 91.0, 92.0, 93.0, 110.0, 105.0, 120.0])

    assert max_drawdown_duration(equity) == 4


def test_an_unrecovered_final_drawdown_is_counted_as_it_stands():
    """Documented as a lower bound: the recovery is after the data ends."""
    equity = as_equity([100.0, 120.0, 90.0, 95.0])

    assert max_drawdown_duration(equity) == 2


def test_a_curve_that_only_rises_has_no_drawdown_at_all():
    equity = as_equity([100.0, 101.0, 102.0, 103.0])

    assert max_drawdown(equity) == 0.0
    assert max_drawdown_duration(equity) == 0
    assert (drawdown_series(equity) == 0.0).all()


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------

def test_historical_var_is_the_interpolated_quantile_sign_flipped():
    """Hand-derived from the tail of the fat-tailed sample.

    Sorted ascending the seven tail values are -0.10, -0.08, -0.06, -0.05,
    -0.04, -0.035, -0.03, and then ninety-three gains. Pandas places the 1%
    quantile of a hundred observations at position 0.01 * 99 = 0.99, that is 99%
    of the way from -0.10 to -0.08, giving -0.0802.
    """
    equity = equity_from_returns(FAT_TAILED_RETURNS)

    assert historical_var(equity, confidence=0.99) == pytest.approx(0.0802,
                                                                    rel=1e-6)


def test_conditional_var_averages_only_the_tail_beyond_the_threshold():
    """At 99% the threshold is -0.0802, so only the -0.10 bar qualifies."""
    equity = equity_from_returns(FAT_TAILED_RETURNS)

    assert conditional_var(equity, confidence=0.99) == pytest.approx(0.10,
                                                                     rel=1e-6)


def test_parametric_var_is_the_normal_quantile_of_the_sample_moments():
    """An identity check rather than a magic number.

    The expectation is recomputed from the sample's own mean and standard
    deviation, so it verifies the formula and the sign convention without anyone
    having to hand-compute a standard deviation to six digits.
    """
    equity = equity_from_returns(FAT_TAILED_RETURNS)
    returns = periodic_returns(equity)

    expected = -(float(returns.mean())
                 + float(stats.norm.ppf(0.01)) * float(returns.std(ddof=1)))

    assert parametric_var(equity, confidence=0.99) == pytest.approx(expected,
                                                                    rel=1e-12)


# Nine returns: one of -50%, three of -25% and five of +25%. Every ratio here is
# a dyadic rational, 0.5, 0.75 and 1.25, so the curve and the returns recovered
# from it are bit-exact and the three -25% values stay identical to the last bit.
#
# WHY THIS CURVE AND THIS CONFIDENCE LEVEL, PRECISELY
# The tail boundary has to land exactly ON an observation for the choice between
# including and excluding it to matter at all. Two things have to line up for
# that, and both are easy to get wrong:
#
#   1. 1 - confidence must be exactly representable. At the conventional 0.95 and
#      0.99 it is not, and even 0.90 gives 0.09999999999999998, which drags the
#      interpolated quantile a hair below the observation and quietly dissolves
#      the tie. 0.75 leaves exactly 0.25.
#   2. The quantile position, (1 - confidence) * (n - 1), must be a whole number.
#      With nine returns that is 0.25 * 8 = 2, the third-smallest observation.
#
# Both hold here, so the quantile is exactly -0.25 and three bars share it:
#     selecting returns <= -0.25 gives (-0.5, -0.25, -0.25, -0.25), mean -0.3125
#     selecting returns <  -0.25 gives (-0.5,) alone,               mean -0.5
#
# This case exists because a mutation flipping that comparison survived every
# other test in the file: an interpolated quantile never touches an observation,
# so the two versions always selected the same tail.
TIE_RATIOS = [0.5, 0.75, 0.75, 0.75] + [1.25] * 5
TIE_CONFIDENCE = 0.75


def tie_boundary_curve() -> pd.Series:
    """A curve whose tail quantile lands exactly on a repeated observation."""
    values = [1024.0]
    for ratio in TIE_RATIOS:
        values.append(values[-1] * ratio)
    return as_equity(values)


def test_the_tie_boundary_curve_really_does_produce_a_tie():
    """Guards the premise of the test below rather than the code itself.

    Three separate things have to stay exact for the next test to test anything:
    the returns, the confidence complement, and the quantile landing on an
    observation rather than between two. If any of them drifts, the tie
    dissolves and the tail-selection test would keep passing while measuring
    nothing. Asserting the premise means that failure mode is loud.
    """
    returns = periodic_returns(tie_boundary_curve())

    assert list(returns) == [-0.5, -0.25, -0.25, -0.25] + [0.25] * 5
    assert 1.0 - TIE_CONFIDENCE == 0.25
    assert float(returns.quantile(1.0 - TIE_CONFIDENCE)) == -0.25
    assert int((returns == -0.25).sum()) == 3


def test_conditional_var_includes_the_bars_sitting_on_the_threshold():
    """Tie handling: the tail is selected by value, so equal bars all count.

    Hand-derived. The quantile is exactly -0.25 and three bars share it, so the
    tail is (-0.5, -0.25, -0.25, -0.25) and its mean is -0.3125. Excluding the
    boundary would leave the -0.5 bar alone and report 0.5, overstating the
    expected shortfall by sixty percent.
    """
    equity = tie_boundary_curve()

    assert historical_var(equity, confidence=TIE_CONFIDENCE) == pytest.approx(
        0.25, rel=1e-12)
    assert conditional_var(equity, confidence=TIE_CONFIDENCE) == pytest.approx(
        0.3125, rel=1e-12)


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_conditional_var_is_never_below_historical_var(confidence,
                                                       near_normal_returns):
    """True by construction: CVaR averages values at or below the threshold."""
    for returns in (FAT_TAILED_RETURNS, list(near_normal_returns)):
        equity = equity_from_returns(returns)

        assert conditional_var(equity, confidence) >= \
            historical_var(equity, confidence) - 1e-12


def test_on_fat_tails_the_observed_measures_exceed_the_normal_estimate():
    """The key relationship, and the reason parametric VaR is not enough.

    At 99% the real tail is where the normal assumption breaks down: a handful of
    violent outliers inflates the standard deviation enough to keep parametric
    VaR respectable at 95%, while it falls badly short further out.
    """
    equity = equity_from_returns(FAT_TAILED_RETURNS)

    historical = historical_var(equity, confidence=0.99)
    parametric = parametric_var(equity, confidence=0.99)
    expected_shortfall = conditional_var(equity, confidence=0.99)

    assert historical > parametric
    assert expected_shortfall > parametric
    # The shortfall is not marginally worse than the bell curve allows, it is
    # roughly double, which is the point of measuring rather than assuming.
    assert expected_shortfall > 2.0 * parametric


def test_on_near_normal_returns_the_two_var_estimates_agree(near_normal_returns):
    """The control: with the assumption satisfied, the two methods converge."""
    equity = equity_from_returns(near_normal_returns)

    historical = historical_var(equity, confidence=0.95)
    parametric = parametric_var(equity, confidence=0.95)

    assert historical == pytest.approx(parametric, rel=0.15)


@pytest.mark.parametrize("measure", [historical_var, parametric_var,
                                     conditional_var])
@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
def test_a_confidence_outside_the_open_unit_interval_is_undefined(measure,
                                                                  confidence):
    equity = equity_from_returns(FAT_TAILED_RETURNS)

    assert np.isnan(measure(equity, confidence))


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------

def test_the_jarque_bera_statistic_matches_its_own_components():
    """The self-consistency identity the three functions must satisfy.

    jb = n / 6 * (skew ** 2 + excess_kurtosis ** 2 / 4). Asserting it ties the
    test to this module's own skewness and kurtosis, so one of the three
    silently switching to a different estimator convention, biased against
    bias-corrected for instance, would break it.
    """
    for returns in (FAT_TAILED_RETURNS, [0.01, -0.02, 0.03, -0.01, 0.05, 0.0]):
        equity = equity_from_returns(returns)
        bars = len(periodic_returns(equity))

        expected = bars / 6.0 * (skewness(equity) ** 2
                                 + kurtosis(equity, excess=True) ** 2 / 4.0)

        assert jarque_bera(equity).statistic == pytest.approx(expected,
                                                              rel=1e-9)


def test_raw_kurtosis_is_excess_kurtosis_plus_three():
    """The normal distribution's raw kurtosis of 3 is the whole convention."""
    equity = equity_from_returns(FAT_TAILED_RETURNS)

    assert kurtosis(equity, excess=False) == pytest.approx(
        kurtosis(equity, excess=True) + 3.0, rel=1e-12)


def test_a_long_left_tail_reads_as_negative_skew_and_fat_tails():
    equity = equity_from_returns(FAT_TAILED_RETURNS)

    assert skewness(equity) < -1.0
    assert kurtosis(equity, excess=True) > 1.0


def test_jarque_bera_rejects_normality_for_the_fat_tailed_sample():
    result = jarque_bera(equity_from_returns(FAT_TAILED_RETURNS))

    assert result.statistic > 0.0
    assert result.p_value < 0.05


def test_jarque_bera_cannot_reject_normality_for_a_normal_sample(
        near_normal_returns):
    equity = equity_from_returns(near_normal_returns)
    result = jarque_bera(equity)

    assert abs(skewness(equity)) < 0.5
    assert abs(kurtosis(equity, excess=True)) < 0.5
    assert result.p_value > 0.05


def test_a_symmetric_sample_has_no_skew():
    """Mirror every return and the asymmetry has to vanish."""
    returns = [0.01, -0.01, 0.03, -0.03, 0.02, -0.02]

    assert skewness(equity_from_returns(returns)) == pytest.approx(0.0,
                                                                   abs=1e-9)


# ---------------------------------------------------------------------------
# CAPM regression
# ---------------------------------------------------------------------------

def test_regressing_the_benchmark_on_itself_gives_beta_one_and_no_alpha():
    """An exact identity, and the strongest single check on the regression."""
    market = equity_from_returns(
        np.random.default_rng(3).normal(0.0004, 0.011, 300))

    result = capm_regression(market, market, periods_per_year=252)

    assert result.beta == pytest.approx(1.0, rel=1e-9)
    assert result.alpha == pytest.approx(0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, rel=1e-9)


def test_a_noiseless_construction_recovers_its_parameters_exactly():
    """No noise, so there is nothing for the fit to trade off: it must be exact."""
    beta_true, alpha_true, periods = 0.60, 0.03, 252
    market_returns = np.random.default_rng(11).normal(0.0004, 0.011, 400)
    built = alpha_true / periods + beta_true * market_returns

    result = capm_regression(equity_from_returns(built),
                             equity_from_returns(market_returns),
                             periods_per_year=periods)

    assert result.beta == pytest.approx(beta_true, rel=1e-6)
    assert result.alpha == pytest.approx(alpha_true, rel=1e-4)
    assert result.r_squared == pytest.approx(1.0, rel=1e-9)


def test_the_regression_recovers_a_known_beta_and_alpha_through_noise():
    """The recovery test, on the same construction verified earlier."""
    beta_true, alpha_true, periods, bars = 0.60, 0.03, 252, 2000

    generator = np.random.default_rng(7)
    market_returns = generator.normal(0.0004, 0.011, bars)
    noise = generator.normal(0.0, 0.001, bars)
    built = alpha_true / periods + beta_true * market_returns + noise

    result = capm_regression(equity_from_returns(built),
                             equity_from_returns(market_returns),
                             periods_per_year=periods)

    assert result.beta == pytest.approx(beta_true, abs=0.01)
    assert result.alpha == pytest.approx(alpha_true, abs=0.01)
    assert result.r_squared > 0.95


def test_alpha_scales_with_the_periods_per_year_but_beta_does_not():
    """Alpha is annualized by arithmetic scaling; beta is dimensionless."""
    market_returns = np.random.default_rng(5).normal(0.0004, 0.011, 300)
    built = 0.02 / 252 + 0.8 * market_returns

    single = capm_regression(equity_from_returns(built),
                             equity_from_returns(market_returns),
                             periods_per_year=252)
    doubled = capm_regression(equity_from_returns(built),
                              equity_from_returns(market_returns),
                              periods_per_year=504)

    assert doubled.alpha == pytest.approx(2.0 * single.alpha, rel=1e-9)
    assert doubled.beta == pytest.approx(single.beta, rel=1e-12)


def test_a_strategy_mostly_in_cash_has_a_beta_near_zero():
    """Not market neutrality, just absence, as the docstring insists."""
    bars = 2000
    market_returns = np.random.default_rng(7).normal(0.0004, 0.011, bars)
    cash_heavy = np.zeros(bars)
    cash_heavy[:40] = market_returns[:40]

    result = capm_regression(equity_from_returns(cash_heavy),
                             equity_from_returns(market_returns),
                             periods_per_year=252)

    assert abs(result.beta) < 0.10
    assert result.r_squared < 0.10


def test_capm_aligns_two_curves_of_different_lengths_on_their_overlap():
    market_returns = np.random.default_rng(13).normal(0.0004, 0.011, 200)
    market = equity_from_returns(market_returns)

    full = capm_regression(market, market, periods_per_year=252)
    partial = capm_regression(market.iloc[:120], market, periods_per_year=252)

    assert np.isfinite(partial.beta)
    assert partial.beta == pytest.approx(1.0, rel=1e-9)
    assert np.isfinite(full.beta)


def test_capm_is_undefined_without_a_varying_market():
    rising = as_equity([100.0, 101.0, 102.0, 103.0, 104.0])
    flat_market = as_equity([100.0] * 5)

    result = capm_regression(rising, flat_market)

    assert np.isnan(result.beta)
    assert np.isnan(result.alpha)
    assert np.isnan(result.r_squared)


@pytest.mark.parametrize("bars", [0, 1, 2, 3])
def test_capm_needs_three_common_returns_to_fit_a_line(bars):
    """Two points fit a line exactly and would report a meaningless R-squared."""
    market_returns = np.random.default_rng(17).normal(0.0004, 0.011, 40)
    market = equity_from_returns(market_returns)

    result = capm_regression(market.iloc[:bars], market)

    # bars values give bars - 1 returns, so four bars is the first fittable case.
    if bars < 4:
        assert np.isnan(result.beta)
    else:
        assert np.isfinite(result.beta)


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------

# Three completed round-trips and one position left open. By hand:
#   (110 - 100) * 100 = +1000
#   (100 - 110) *  90 =  -900
#   (130 - 100) *  50 = +1500
# So two winners averaging 1250, one loser of -900, gross profit 2500 against
# gross loss 900, and a profit factor of 2500 / 900.
MIXED_LOG = [
    trade(1, BUY, 100.0, 100.0), trade(2, SELL, 110.0, 100.0),
    trade(3, BUY, 110.0, 90.0), trade(4, SELL, 100.0, 90.0),
    trade(5, BUY, 100.0, 50.0), trade(6, SELL, 130.0, 50.0),
    trade(7, BUY, 130.0, 40.0),
]


def test_round_trips_pairs_each_buy_with_the_sell_that_closed_it():
    trips = round_trips(MIXED_LOG)

    assert len(trips) == 3
    assert list(trips["profit"]) == [1000.0, -900.0, 1500.0]
    assert list(trips["entry_price"]) == [100.0, 110.0, 100.0]
    assert list(trips["exit_price"]) == [110.0, 100.0, 130.0]


def test_round_trip_return_is_the_price_ratio():
    trips = round_trips(MIXED_LOG)

    assert trips["return_pct"].iloc[0] == pytest.approx(0.10, rel=1e-12)
    assert trips["return_pct"].iloc[1] == pytest.approx(-10.0 / 110.0, rel=1e-12)


def test_trade_statistics_on_the_hand_built_log():
    stats = trade_statistics(MIXED_LOG)

    assert stats.number_of_trades == 3
    assert stats.win_rate == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert stats.average_win == pytest.approx(1250.0, rel=1e-12)
    assert stats.average_loss == pytest.approx(-900.0, rel=1e-12)
    assert stats.gross_profit == pytest.approx(2500.0, rel=1e-12)
    assert stats.gross_loss == pytest.approx(900.0, rel=1e-12)
    assert stats.profit_factor == pytest.approx(2500.0 / 900.0, rel=1e-12)
    assert stats.open_trades == 1


def test_average_loss_is_signed_negative_not_a_magnitude():
    """A convention worth pinning: the report prints it as it comes."""
    assert trade_statistics(MIXED_LOG).average_loss < 0.0


def test_a_ninety_percent_win_rate_can_still_be_a_losing_strategy():
    """The caveat the trade statistics exist to expose.

    Nine winners of +10 against one loser of -500. The win rate says 90% and the
    profit factor says 0.18, and only one of the two is about making money.
    """
    log: List[Dict[str, Any]] = []
    for index in range(9):
        log += [trade(2 * index, BUY, 100.0, 1.0),
                trade(2 * index + 1, SELL, 110.0, 1.0)]
    log += [trade(100, BUY, 100.0, 5.0), trade(101, SELL, 0.01, 5.0)]

    stats = trade_statistics(log)

    assert stats.number_of_trades == 10
    assert stats.win_rate == pytest.approx(0.9, rel=1e-12)
    assert stats.average_win == pytest.approx(10.0, rel=1e-12)
    assert stats.average_loss == pytest.approx(-499.95, rel=1e-12)
    assert stats.profit_factor < 1.0
    assert stats.gross_loss > stats.gross_profit


def test_profit_factor_is_nan_when_nothing_closed():
    """A routine outcome, not an error: a rate over zero trades is undefined."""
    stats = trade_statistics([])

    assert stats.number_of_trades == 0
    assert np.isnan(stats.profit_factor)
    assert np.isnan(stats.win_rate)
    assert np.isnan(stats.average_win)
    assert np.isnan(stats.average_loss)
    assert stats.gross_profit == 0.0
    assert stats.gross_loss == 0.0


def test_profit_factor_is_zero_when_every_trade_lost():
    log = [trade(1, BUY, 100.0, 10.0), trade(2, SELL, 90.0, 10.0),
           trade(3, BUY, 90.0, 10.0), trade(4, SELL, 80.0, 10.0)]

    stats = trade_statistics(log)

    assert stats.profit_factor == 0.0
    assert stats.win_rate == 0.0
    assert np.isnan(stats.average_win)
    assert stats.average_loss == pytest.approx(-100.0, rel=1e-12)


def test_profit_factor_is_infinite_when_nothing_ever_lost():
    """Genuinely infinite rather than undefined, which the no-trades case is."""
    log = [trade(1, BUY, 100.0, 10.0), trade(2, SELL, 110.0, 10.0),
           trade(3, BUY, 110.0, 10.0), trade(4, SELL, 130.0, 10.0)]

    stats = trade_statistics(log)

    assert stats.profit_factor == float("inf")
    assert stats.win_rate == 1.0
    assert np.isnan(stats.average_loss)


def test_a_break_even_trip_counts_as_a_trade_but_wins_nothing():
    log = [trade(1, BUY, 100.0, 10.0), trade(2, SELL, 100.0, 10.0)]

    stats = trade_statistics(log)

    assert stats.number_of_trades == 1
    assert stats.win_rate == 0.0
    assert np.isnan(stats.profit_factor)
    assert stats.gross_profit == 0.0
    assert stats.gross_loss == 0.0


def test_an_unclosed_final_buy_is_reported_but_not_scored():
    log = [trade(1, BUY, 100.0, 10.0)]

    stats = trade_statistics(log)

    assert stats.number_of_trades == 0
    assert stats.open_trades == 1
    assert round_trips(log).empty


def test_a_sell_with_nothing_open_is_ignored():
    log = [trade(1, SELL, 100.0, 10.0), trade(2, BUY, 100.0, 10.0),
           trade(3, SELL, 110.0, 10.0)]

    stats = trade_statistics(log)

    assert stats.number_of_trades == 1
    assert stats.open_trades == 0


def test_a_second_buy_while_already_invested_is_ignored():
    """Normal for the RSI rule, which can dip below its band twice in a row."""
    log = [trade(1, BUY, 100.0, 10.0), trade(2, BUY, 105.0, 10.0),
           trade(3, SELL, 110.0, 10.0)]

    trips = round_trips(log)

    assert len(trips) == 1
    assert trips["entry_price"].iloc[0] == 100.0


def test_a_trade_missing_a_required_key_is_rejected():
    with pytest.raises(ValueError, match="missing required keys"):
        round_trips([{"date": pd.Timestamp("2022-01-01"), "action": BUY}])


def test_a_none_trade_log_is_treated_as_empty():
    assert trade_statistics(None).number_of_trades == 0
    assert round_trips(None).empty


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def test_exposure_is_the_fraction_of_bars_holding_a_position():
    assert exposure(pd.Series([0.0, 0.0, 10.0, 10.0, 0.0])) == pytest.approx(
        0.4, rel=1e-12)


def test_exposure_counts_a_bar_whose_price_happened_not_to_move():
    """Why the metric reads positions and not nonzero returns.

    The position is held on three of five bars, and on the middle one the price
    closed unchanged. A return-based proxy scores that quiet bar as flat and
    reports 25% instead of 60%, understating exposure precisely on the calm days.
    """
    positions = pd.Series([0.0, 4.0, 4.0, 4.0, 0.0])
    equity = as_equity([100.0, 100.0, 100.0, 120.0, 120.0])

    returns = periodic_returns(equity)
    return_based_proxy = float((returns != 0.0).mean())

    assert exposure(positions) == pytest.approx(0.6, rel=1e-12)
    assert return_based_proxy == pytest.approx(0.25, rel=1e-12)


def test_exposure_is_one_when_always_invested_and_zero_when_never():
    assert exposure(pd.Series([5.0] * 10)) == 1.0
    assert exposure(pd.Series([0.0] * 10)) == 0.0


def test_exposure_matches_the_known_cash_heavy_reading():
    """The moving-average figure verified earlier: nine held bars in 755."""
    positions = pd.Series([0.0] * 746 + [1.0] * 9)

    assert exposure(positions) == pytest.approx(9 / 755, rel=1e-12)
    assert exposure(positions) == pytest.approx(0.0119, abs=5e-5)


def test_a_short_position_counts_as_exposure():
    """Nothing shorts today, but absolute value is the right convention."""
    assert exposure(pd.Series([-5.0, 0.0])) == pytest.approx(0.5, rel=1e-12)


def test_floating_point_dust_does_not_count_as_a_position():
    assert exposure(pd.Series([1e-15, 0.0, 0.0, 0.0])) == 0.0


def test_exposure_of_nothing_is_undefined():
    assert np.isnan(exposure(pd.Series(dtype=float)))


# ---------------------------------------------------------------------------
# Degenerate inputs
#
# The group that keeps a parameter sweep alive. Every metric is documented to
# return NaN or a sensible constant rather than raise, and none of that is
# exercised by ordinary use.
# ---------------------------------------------------------------------------

# Metrics undefined on both an empty and a single-value curve, since all of them
# need at least two bars to see a change.
NEEDS_TWO_BARS: List[Callable[[pd.Series], float]] = [
    annualized_return, volatility, sharpe_ratio, sortino_ratio, calmar_ratio,
    historical_var, parametric_var, conditional_var, skewness, kurtosis,
]


@pytest.mark.parametrize("measure", NEEDS_TWO_BARS,
                         ids=lambda f: f.__name__)
@pytest.mark.parametrize("curve", [[], [100.0]], ids=["empty", "single-value"])
def test_metrics_needing_a_change_return_nan_without_one(measure, curve):
    assert np.isnan(measure(as_equity(curve)))


def test_the_metrics_with_a_defined_answer_on_a_single_bar_say_so():
    """Two deliberate exceptions to the rule above, both documented.

    A one-bar curve has not changed, so its total return is zero rather than
    unknown, and it has never declined, so its drawdown is zero. Reporting NaN
    there would be less informative, not more careful.
    """
    single = as_equity([100.0])

    assert total_return(single) == 0.0
    assert max_drawdown(single) == 0.0
    assert max_drawdown_duration(single) == 0


def test_an_empty_curve_is_undefined_everywhere_it_should_be():
    empty = as_equity([])

    assert np.isnan(total_return(empty))
    assert np.isnan(max_drawdown(empty))
    assert max_drawdown_duration(empty) == 0
    assert drawdown_series(empty).empty
    assert periodic_returns(empty).empty
    assert np.isnan(jarque_bera(empty).statistic)
    assert np.isnan(jarque_bera(empty).p_value)


def test_a_curve_that_never_moves_is_flat_not_broken():
    """Zero volatility, and the ratios that divide by it stand down."""
    flat = as_equity([100.0] * 10)

    assert volatility(flat, periods_per_year=9) == 0.0
    assert max_drawdown(flat) == 0.0
    assert np.isnan(sharpe_ratio(flat, periods_per_year=9))
    assert np.isnan(sortino_ratio(flat, periods_per_year=9))
    assert np.isnan(calmar_ratio(flat, periods_per_year=9))
    assert np.isnan(skewness(flat))
    assert np.isnan(kurtosis(flat))
    assert np.isnan(jarque_bera(flat).p_value)


def test_a_curve_that_never_declines_leaves_sortino_and_calmar_undefined():
    """No downside to divide by, and no drawdown either."""
    rising = as_equity([100.0, 101.0, 102.0, 103.0])

    assert np.isnan(sortino_ratio(rising, periods_per_year=3))
    assert np.isnan(calmar_ratio(rising, periods_per_year=3))
    assert np.isfinite(sharpe_ratio(rising, periods_per_year=3))


@pytest.mark.parametrize("periods_per_year", [0, -1, -252])
def test_a_non_positive_period_count_is_undefined(periods_per_year):
    equity = as_equity(SIMPLE_CURVE)

    assert np.isnan(annualized_return(equity, periods_per_year))
    assert np.isnan(volatility(equity, periods_per_year))
    assert np.isnan(sharpe_ratio(equity, periods_per_year=periods_per_year))
    assert np.isnan(sortino_ratio(equity, periods_per_year=periods_per_year))
    assert np.isnan(calmar_ratio(equity, periods_per_year))


def test_a_curve_starting_at_zero_is_undefined_rather_than_infinite():
    assert np.isnan(total_return(as_equity([0.0, 100.0])))
    assert np.isnan(annualized_return(as_equity([0.0, 100.0]),
                                      periods_per_year=1))


# Every pathological curve that a degenerate parameter combination could
# plausibly produce during a sweep.
PATHOLOGICAL_CURVES = {
    "empty": [],
    "single-value": [100.0],
    "two-bars": [100.0, 100.0],
    "flat": [100.0] * 10,
    "monotonic-rise": [100.0, 101.0, 102.0, 103.0],
    "monotonic-fall": [100.0, 99.0, 98.0, 97.0],
    "starts-at-zero": [0.0, 100.0, 120.0],
    "hits-zero": [100.0, 50.0, 0.0, 0.0],
    "negative": [100.0, -50.0, 20.0],
    "with-nan": [100.0, float("nan"), 120.0],
    "huge": [1e300, 1e-300, 1e300],
}


@pytest.mark.parametrize("name", sorted(PATHOLOGICAL_CURVES),
                         ids=sorted(PATHOLOGICAL_CURVES))
def test_no_metric_raises_on_any_pathological_curve(name):
    """The promise a parameter sweep depends on, stated as bluntly as possible.

    Hundreds of backtests run unattended, and some combinations genuinely never
    trade or never move. Any metric raising here would kill the sweep partway
    through over a combination nobody was interested in.
    """
    equity = as_equity(PATHOLOGICAL_CURVES[name])

    measures: List[Callable[[], Any]] = [
        lambda: total_return(equity),
        lambda: annualized_return(equity),
        lambda: volatility(equity),
        lambda: sharpe_ratio(equity),
        lambda: sortino_ratio(equity),
        lambda: calmar_ratio(equity),
        lambda: drawdown_series(equity),
        lambda: max_drawdown(equity),
        lambda: max_drawdown_duration(equity),
        lambda: exposure(equity),
        lambda: historical_var(equity),
        lambda: parametric_var(equity),
        lambda: conditional_var(equity),
        lambda: skewness(equity),
        lambda: kurtosis(equity),
        lambda: jarque_bera(equity),
        lambda: capm_regression(equity, as_equity(SIMPLE_CURVE)),
        lambda: capm_regression(as_equity(SIMPLE_CURVE), equity),
    ]

    for measure in measures:
        measure()


@pytest.mark.parametrize("name", sorted(PATHOLOGICAL_CURVES),
                         ids=sorted(PATHOLOGICAL_CURVES))
def test_every_scalar_metric_returns_a_real_number_or_nan(name):
    """Never a complex number, an infinity from a stray division, or a string."""
    equity = as_equity(PATHOLOGICAL_CURVES[name])

    for value in (total_return(equity), annualized_return(equity),
                  volatility(equity), sharpe_ratio(equity),
                  sortino_ratio(equity), calmar_ratio(equity),
                  max_drawdown(equity), skewness(equity), kurtosis(equity)):
        assert isinstance(value, float)
        assert np.isnan(value) or np.isfinite(value)
