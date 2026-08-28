"""Automated tests for the validation layer: parameter sweeps and robustness.

WHY THIS FILE IS SEPARATE FROM test_analytics.py
The other analytics modules are pure arithmetic on a finished series, and their
tests need nothing but a handful of numbers. Validation drives the engine: a
sweep runs a real backtest per cell, so its tests need prices, a strategy and a
Backtester. Keeping the two apart means the arithmetic suite stays instant and
dependency-free, and it makes the one place in analytics that simulates rather
than measures visible in the test layout as well as in the source.

WHAT IS LOCKED HERE
Two claims, both of which were broken before these tests existed.

    A sweep describes the strategy it was asked to describe. Every cell must
    apply the caller's warm-up setting, because a grid is only ever read against
    a particular run: the dashboard outlines one cell and labels it as the
    user's own. A grid computed under a different setting puts a stranger's
    score in that cell, and on real AAPL data the two settings differ by enough
    to flip the sign of the verdict.

    The plateau band has a usable width whatever the peak's level. The band is a
    fraction of the best score while that score is positive, and a fraction of
    the grid's own spread once it is not. A band measured against the best score
    in both cases collapses to nothing at exactly zero, which is the one point
    the second case exists to cover.
"""

from typing import List

import numpy as np
import pandas as pd
import pytest

from analytics.report import (
    NUMBER_OF_TRADES,
    SHARPE,
    STRATEGY_COLUMN,
    performance_report,
)
from analytics.validation import (
    ENTER_ON_EXISTING_TREND,
    PLATEAU_TOLERANCE,
    assess,
    metric_grid,
    sweep,
)
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageCrossover

# Short windows, so a few dozen synthetic bars are enough to leave the warm-up
# and still produce crossings in both directions.
FAST, SLOW = 3, 5
INITIAL_CASH = 10_000.0


def ramp(start: float, end: float, bars: int) -> List[float]:
    """A straight line of closes from start to end, end included."""
    return list(np.linspace(start, end, bars))


# Rising, then falling, then rising again. The opening rise is what makes the
# warm-up setting matter: by the bar where the slow average first exists the
# fast one is already above it, and the crossing that put it there is off the
# left edge of the data. The later legs give the run that declined to enter a
# crossing it can observe, so neither setting is left with a flat curve.
TREND_CLOSES: List[float] = (
    ramp(100.0, 130.0, 12)
    + ramp(129.0, 90.0, 12)
    + ramp(91.0, 150.0, 16)
)


def prices(closes: List[float]) -> pd.DataFrame:
    """An OHLCV frame on business days, flat within each bar."""
    index = pd.date_range("2020-01-01", periods=len(closes), freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=index,
    )


def standalone(closes: List[float], enter_on_existing_trend: bool) -> pd.Series:
    """The performance report of one backtest, run without the sweep.

    The sweep's own machinery is deliberately not reused: the point is to have
    an independent path to the same number, so that a cell agreeing with it
    means something.
    """
    frame = prices(closes)
    result = Backtester(
        frame,
        initial_cash=INITIAL_CASH,
        strategy=MovingAverageCrossover(
            fast_window=FAST,
            slow_window=SLOW,
            enter_on_existing_trend=enter_on_existing_trend,
        ),
    ).run()

    return performance_report(
        equity=result.equity_curve["total_value"],
        benchmark=result.benchmark_curve,
        trade_log=result.trade_log,
        positions=result.equity_curve["shares"],
    )[STRATEGY_COLUMN]


def one_cell_sweep(closes: List[float], enter_on_existing_trend: bool) -> pd.Series:
    """A sweep of the single (FAST, SLOW) cell, as a metric column."""
    results = sweep(
        prices(closes),
        INITIAL_CASH,
        fast_windows=(FAST,),
        slow_windows=(SLOW,),
        enter_on_existing_trend=enter_on_existing_trend,
    )
    return results[(FAST, SLOW)]


def grid(rows: List[List[float]], fast: List[int], slow: List[int]) -> pd.DataFrame:
    """A metric grid built by hand, shaped the way metric_grid returns one."""
    return pd.DataFrame(rows, index=pd.Index(fast, name="fast"),
                        columns=pd.Index(slow, name="slow"))


# --------------------------------------------------------------------------
# The sweep honours the warm-up setting it was given
# --------------------------------------------------------------------------

def test_the_two_warm_up_settings_really_do_differ_on_this_series():
    """Guard the guard: without this, the next two tests could pass vacuously.

    If the synthetic series were one where the setting made no difference, a
    sweep that ignored the argument entirely would still agree with both
    standalone runs, and the comparison below would prove nothing.
    """
    entering = standalone(TREND_CLOSES, True)
    waiting = standalone(TREND_CLOSES, False)

    assert entering[NUMBER_OF_TRADES] != waiting[NUMBER_OF_TRADES]


@pytest.mark.parametrize("enter_on_existing_trend", [False, True])
def test_every_swept_cell_uses_the_warm_up_setting_it_was_passed(
    enter_on_existing_trend: bool,
):
    """A cell must match a backtest run independently under the same setting.

    This is the assertion the dashboard's outlined cell rests on. Before the
    setting was a parameter the sweep hard-wired it, so on the dashboard's own
    default the cell labelled as the user's carried the other variant's score:
    +0.297 against a true -0.502 on AAPL over 2020-2022.
    """
    swept = one_cell_sweep(TREND_CLOSES, enter_on_existing_trend)
    direct = standalone(TREND_CLOSES, enter_on_existing_trend)

    pd.testing.assert_series_equal(
        swept, direct, check_names=False,
    )


def test_the_sweep_default_is_the_setting_the_command_line_scripts_expect():
    """run_parameter_sweep and run_walk_forward rely on the default.

    Both call sweep without naming the setting, and run_walk_forward pairs the
    grid with an evaluation run that names the constant explicitly. If the
    default drifted away from the constant, selection and evaluation inside the
    walk-forward would silently describe two different strategies.
    """
    defaulted = sweep(
        prices(TREND_CLOSES),
        INITIAL_CASH,
        fast_windows=(FAST,),
        slow_windows=(SLOW,),
    )

    pd.testing.assert_series_equal(
        defaulted[(FAST, SLOW)],
        one_cell_sweep(TREND_CLOSES, ENTER_ON_EXISTING_TREND),
    )


def test_the_warm_up_setting_does_not_disturb_the_shape_of_the_grid():
    """Only the scores may change: the axes and the invalid cells may not.

    The setting is a property of the strategy, not of the search, so switching
    it must not turn a valid combination invalid or move a window.
    """
    axes = {"fast_windows": (2, 3, 4), "slow_windows": (3, 5)}
    frame = prices(TREND_CLOSES)

    waiting = metric_grid(sweep(frame, INITIAL_CASH, **axes,
                                enter_on_existing_trend=False), SHARPE)
    entering = metric_grid(sweep(frame, INITIAL_CASH, **axes,
                                 enter_on_existing_trend=True), SHARPE)

    assert waiting.index.equals(entering.index)
    assert waiting.columns.equals(entering.columns)
    # fast >= slow is skipped by the strategy's own rule, whatever the setting.
    assert waiting.notna().equals(entering.notna())


# --------------------------------------------------------------------------
# The plateau band keeps a usable width wherever the peak sits
# --------------------------------------------------------------------------

def test_a_peak_of_exactly_zero_still_admits_its_close_neighbours():
    """The case the fallback branch exists for, and used to fail on.

    Scores spread over a range of 1.0 with a best of exactly 0.0. A band of
    PLATEAU_TOLERANCE of that range is 0.1 wide, so the cell at -0.05 belongs to
    the plateau and the one at -1.0 does not: two valid cells of three. Taking
    the band as a fraction of the best score instead gives a width of exactly
    zero, admitting only the peak itself and reporting a third.
    """
    summary = assess(grid([[-1.0, -0.05, 0.0]], [10], [50, 100, 150]),
                     reference=-2.0)

    assert summary.best_score == 0.0
    assert summary.plateau_share == pytest.approx(2 / 3)


def test_a_positive_peak_keeps_the_proportional_band_untouched():
    """The branch every real sweep takes must be unchanged by the fix.

    Best 1.0, so the band runs down to 0.9 and holds two of the three cells.
    """
    summary = assess(grid([[0.5, 0.95, 1.0]], [10], [50, 100, 150]),
                     reference=0.0)

    assert summary.best_score == 1.0
    assert summary.plateau_share == pytest.approx(2 / 3)


def test_a_uniformly_negative_grid_takes_its_band_from_its_own_level():
    """With no spread to measure, the peak's magnitude is the scale left.

    Every cell is -0.5, so the spread is zero and a band derived from it alone
    would be zero too. The larger of the two scales is used, giving a width of
    0.05 and a plateau holding the whole grid, which is the truthful reading of
    a perfectly flat surface.
    """
    summary = assess(grid([[-0.5, -0.5], [-0.5, -0.5]], [10, 20], [50, 100]),
                     reference=0.0)

    assert summary.plateau_share == pytest.approx(1.0)


def test_an_entirely_zero_grid_reports_a_complete_plateau():
    """The degenerate corner where both scales vanish at once.

    Nothing is left to measure a band with, and nothing needs one: every cell
    already sits at the peak, so the whole grid is the plateau. The guard is
    that this returns 1.0 rather than raising on a zero-width comparison.
    """
    summary = assess(grid([[0.0, 0.0, 0.0]], [10], [50, 100, 150]),
                     reference=-1.0)

    assert summary.plateau_share == pytest.approx(1.0)


def test_the_band_is_exactly_the_documented_fraction_of_the_spread():
    """Pin the width itself, not just its consequences.

    A grid spanning 2.0 gives a band of 2.0 * PLATEAU_TOLERANCE below the peak.
    The two cells placed just inside and just outside that edge are what make
    the assertion about the width rather than about the ordering.
    """
    edge = 2.0 * PLATEAU_TOLERANCE
    inside, outside = -edge * 0.99, -edge * 1.01

    summary = assess(grid([[-2.0, outside, inside, 0.0]], [10],
                          [50, 100, 150, 200]), reference=-3.0)

    assert summary.plateau_share == pytest.approx(2 / 4)


def test_nan_cells_are_excluded_from_the_spread_that_sets_the_band():
    """An invalid combination must not widen the band it plays no part in.

    The NaN stands for a fast >= slow pair the strategy refuses. Were it read as
    a value the spread would be undefined and the band with it.
    """
    summary = assess(grid([[np.nan, -1.0, -0.05, 0.0]], [10],
                          [50, 100, 150, 200]), reference=-2.0)

    assert summary.plateau_share == pytest.approx(2 / 3)
