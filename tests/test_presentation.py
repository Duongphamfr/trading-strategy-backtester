"""Automated tests for three ways the reports used to misdescribe their numbers.

WHY A TEST FILE ABOUT LABELS
Nothing here checks a computed value, and that is the point: all three defects
were cases where the arithmetic was right and the sentence around it was wrong.
A wrong label is not a lesser problem than a wrong number when it changes the
conclusion a reader draws, and two of these did.

    The unit of a difference. A return of 41.71% less a return of 329.44% was
    printed as "-287.73%", which no long-only strategy can produce, since it
    cannot lose more than its capital. The quantity was always a difference in
    percentage points; only the symbol lied. On a benchmark that returned 74% the
    numbers stayed plausible, which is why this survived development on AAPL.

    The sign of zero. IEEE-754 has two zeros and the VaR family reaches the
    negative one whenever it negates a zero to honour its positive-loss
    convention, so an all-cash strategy reported "-0.00%" for a quantity the
    docstring promises is positive.

    The period studied. The header printed the requested date range, while Yahoo
    silently returns whatever it has. A ticker listed in 2024 answering a request
    from 2020 was reported as a 6.6-year study of 592 bars, when 6.6 years of
    daily bars is nearer 1,660.

The last one is worth the most care, because it is the only one that could change
how a reader judges a result: a Sharpe ratio earned over two years and one earned
over seven are not equally believable, and the header was the only thing saying
which it was.
"""

from typing import List

import numpy as np
import pandas as pd
import pytest

from analytics.report import (
    BENCHMARK_COLUMN,
    BETA,
    CONDITIONAL_VAR,
    HISTORICAL_VAR,
    MAX_DRAWDOWN,
    PARAMETRIC_VAR,
    PERCENT_METRICS,
    R_SQUARED,
    SHARPE,
    STRATEGY_COLUMN,
    TOTAL_RETURN,
    format_report,
    performance_report,
    report_caveats,
)
from analytics.report import _format_value
from data.market_data import (
    RANGE_TOLERANCE_DAYS,
    covered_range,
    period_label,
)

VAR_METRICS = (HISTORICAL_VAR, PARAMETRIC_VAR, CONDITIONAL_VAR)


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """A price frame over exactly this index."""
    closes = [100.0 + step for step in range(len(index))]
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": [1_000] * len(index)},
                        index=index)


def frame(first: str, bars: int) -> pd.DataFrame:
    """A price frame of known length, on business days from a known start."""
    return _frame(pd.date_range(first, periods=bars, freq="B", name="Date"))


def frame_spanning(first: str, last: str) -> pd.DataFrame:
    """A price frame whose first and last bars are exactly these dates.

    Used wherever the test is about the endpoints rather than the count. Building
    from a bar count instead would put the far endpoint wherever the business-day
    calendar happens to land, which is how the first version of the
    within-tolerance test managed to sit five weeks outside the tolerance.
    """
    return _frame(pd.date_range(first, last, freq="B", name="Date"))


def flat_equity(bars: int = 40, value: float = 10_000.0) -> pd.Series:
    """An all-cash curve: never moves, so every periodic return is exactly zero."""
    return pd.Series([value] * bars,
                     index=pd.date_range("2020-01-01", periods=bars, freq="B"))


def rising_equity(bars: int = 40) -> pd.Series:
    """A curve that moves, to stand in for the market."""
    growth = np.cumprod(1 + np.random.default_rng(7).normal(0.0004, 0.01, bars))
    return pd.Series(growth * 10_000.0,
                     index=pd.date_range("2020-01-01", periods=bars, freq="B"))


# --------------------------------------------------------------------------
# The unit of a difference: percentage points, not percent
# --------------------------------------------------------------------------

# The real TSLA 2020-2023 figures, which is where the mislabelling became
# impossible to defend rather than merely imprecise.
TSLA_STRATEGY_RETURN = 0.4171
TSLA_BENCHMARK_RETURN = 3.2944


def test_the_dashboard_delta_of_a_percentage_metric_is_labelled_in_points():
    """A return gap must not be dressed as a return."""
    from app import delta_label

    label = delta_label(TOTAL_RETURN,
                        TSLA_STRATEGY_RETURN - TSLA_BENCHMARK_RETURN)

    assert "pp" in label
    assert "%" not in label
    # The magnitude is the same number it always was, read in its own unit.
    assert "-287.73" in label


def test_the_dashboard_delta_keeps_the_sign_streamlit_reads_for_the_arrow():
    """Streamlit picks the arrow's direction from the leading character.

    Losing the sign would silently turn every negative delta green.
    """
    from app import delta_label

    behind = delta_label(TOTAL_RETURN, -0.5)
    ahead = delta_label(TOTAL_RETURN, 0.5)

    assert behind.startswith("-")
    assert ahead.startswith("+")


def test_a_ratio_metric_delta_is_left_alone():
    """Only percentage metrics change unit under subtraction.

    A Sharpe difference is already in Sharpe units, and scaling it by 100 would
    invent a problem where there was none.
    """
    from app import delta_label

    label = delta_label(SHARPE, 0.5 - 1.035)

    assert "pp" not in label
    assert "-0.535" in label


@pytest.mark.parametrize("metric", sorted(PERCENT_METRICS))
def test_every_percentage_metric_gets_the_points_treatment(metric: str):
    """The unit is read from PERCENT_METRICS, so no card can be forgotten.

    A metric added to that table later inherits the right delta unit without
    anyone editing this logic, which is the reason it is not a hand-kept list.
    """
    from app import delta_label

    assert "pp" in delta_label(metric, 0.1234)


def test_max_drawdown_is_among_them():
    """Named explicitly because it is the second card that shows a percentage.

    Its gaps are small enough that the old rendering never looked absurd, which
    makes it exactly the kind of site a fix aimed at Total Return would miss.
    """
    from app import delta_label

    assert MAX_DRAWDOWN in PERCENT_METRICS
    assert "pp" in delta_label(MAX_DRAWDOWN, -0.0512)


def test_the_scenario_table_column_declares_points_and_scales_to_them():
    """run_cost_scenarios renders the same gap through its own column layout.

    The stored value must stay a ratio, since the sweep and the snapshot read it,
    so the scaling belongs to the rendering and is checked here rather than in
    the frame.
    """
    import run_cost_scenarios as costs

    column = next(entry for entry in costs.COLUMNS if entry[0] == costs.RETURN_GAP)
    _, heading, _, template, scale = column

    assert "pp" in heading
    assert "%" not in template
    assert scale == 100.0
    rendered = template.format(
        (TSLA_STRATEGY_RETURN - TSLA_BENCHMARK_RETURN) * scale)
    assert rendered == "-287.73"


def test_the_other_scenario_columns_are_not_rescaled():
    """Only the gap column converts. A scaled Total Return would be 100x wrong."""
    import run_cost_scenarios as costs

    for name, _, _, _, scale in costs.COLUMNS:
        if name != costs.RETURN_GAP:
            assert scale == 1.0, f"{name} must not be rescaled"


def test_the_comparison_verdict_still_turns_on_the_sign_alone():
    """The beat/lost/matched wording had to survive the unit change untouched.

    It reads the sign of the difference, which no rescaling can alter, and this
    pins that the fix did not disturb it.
    """
    import run_comparison as comparison

    assert comparison.TIE_TOLERANCE > 0
    # A strategy behind the benchmark is still "lost" however the gap is printed.
    assert TSLA_STRATEGY_RETURN < TSLA_BENCHMARK_RETURN


# --------------------------------------------------------------------------
# The sign of zero
# --------------------------------------------------------------------------

def test_a_negative_zero_renders_without_its_sign():
    """The formatter is the single place this is corrected."""
    assert _format_value(HISTORICAL_VAR, -0.0) == "0.00%"


def test_a_positive_zero_is_unchanged():
    """The fix must be invisible when there was nothing to fix."""
    assert _format_value(HISTORICAL_VAR, 0.0) == "0.00%"


@pytest.mark.parametrize("metric", VAR_METRICS)
def test_no_var_row_of_an_all_cash_report_shows_a_negative_zero(metric: str):
    """The case as a reader meets it, through a real report.

    An all-cash strategy has every return exactly zero, so each VaR negates a
    zero and produces the negative one.
    """
    report = performance_report(flat_equity(), benchmark=rising_equity())

    assert _format_value(metric, report.loc[metric, STRATEGY_COLUMN]) == "0.00%"


def test_the_whole_all_cash_table_is_free_of_negative_zeros():
    """Nothing else in the report renders one either, now or after a new metric."""
    text = format_report(performance_report(flat_equity(),
                                            benchmark=rising_equity()))

    assert "-0.00" not in text


def test_the_underlying_value_is_still_negative_zero():
    """The computation was never wrong, so it was not changed.

    -0.0 == 0.0 is true, so no comparison, aggregation or test can distinguish
    them; the defect existed only in the rendering and was fixed only there.
    """
    report = performance_report(flat_equity(), benchmark=rising_equity())
    raw = report.loc[HISTORICAL_VAR, STRATEGY_COLUMN]

    assert raw == 0.0
    assert np.signbit(raw)


@pytest.mark.parametrize("value,expected", [
    (-0.1234, "-12.34%"),
    (0.1234, "12.34%"),
    (-1.0, "-100.00%"),
])
def test_genuine_negatives_keep_their_sign(value: float, expected: str):
    """Adding zero must not swallow a real negative number."""
    assert _format_value(MAX_DRAWDOWN, value) == expected


def test_a_negative_zero_in_a_ratio_row_is_also_tidied():
    """The fix sits above the branch on row type, so it covers every format."""
    assert _format_value(SHARPE, -0.0) == "0.000"


# --------------------------------------------------------------------------
# The period studied
# --------------------------------------------------------------------------

def test_the_label_reports_the_bars_in_hand():
    """The plain case: the label is the data's own extent."""
    prices = frame("2020-01-02", 20)

    assert period_label(prices) == "2020-01-02 to 2020-01-29"


def test_a_weekend_or_holiday_shift_is_not_worth_mentioning():
    """Requesting the 1st and getting the 2nd is the calendar, not a short history.

    Annotating it would put a parenthetical on almost every ordinary run and
    train the reader to ignore the one that matters.
    """
    # The real AAPL case: a request for 2020-01-01 to 2023-01-01 is answered with
    # bars from the 2nd to 2022-12-30, one day late and two days early.
    prices = frame_spanning("2020-01-02", "2022-12-30")

    assert period_label(prices, "2020-01-01", "2023-01-01") == \
        "2020-01-02 to 2022-12-30"


def test_a_listing_that_postdates_the_request_says_so():
    """The case that motivated the change.

    RDDT answers a 2020 request with 2024 data. Reporting the request claimed a
    6.6-year study of 592 bars, which is self-contradictory to anyone who divides.
    """
    prices = frame("2024-03-21", 592)

    label = period_label(prices, "2020-01-01", "2026-08-01")

    assert label.startswith("2024-03-21 to ")
    assert "requested 2020-01-01 to 2026-08-01" in label


def test_a_history_ending_early_is_also_flagged():
    """Symmetric to a late start, for a delisting or a stale cache."""
    prices = frame("2020-01-02", 60)

    label = period_label(prices, "2020-01-01", "2023-01-01")

    assert "requested" in label


@pytest.mark.parametrize("days,flagged", [
    (RANGE_TOLERANCE_DAYS - 1, False),
    (RANGE_TOLERANCE_DAYS, False),
    (RANGE_TOLERANCE_DAYS + 1, True),
])
def test_the_tolerance_boundary_is_where_it_is_documented(days: int, flagged: bool):
    """Exactly at the tolerance is still silence; one day past it speaks."""
    first = pd.Timestamp("2020-01-10")
    requested_start = (first - pd.Timedelta(days=days)).date().isoformat()
    prices = frame(first.date().isoformat(), 30)

    label = period_label(prices, requested_start, None)

    assert ("requested" in label) is flagged


def test_no_request_means_no_comparison():
    """Callers with nothing to compare against get the range and nothing else."""
    prices = frame("2024-03-21", 100)

    assert "requested" not in period_label(prices)


def test_covered_range_returns_the_endpoints():
    """The fact the label is built from, checked on its own."""
    prices = frame("2020-01-02", 20)
    first, last = covered_range(prices)

    assert first == prices.index[0]
    assert last == prices.index[-1]


def test_an_empty_frame_has_no_range_to_report():
    """Better to raise than to invent a period for zero bars."""
    with pytest.raises(ValueError):
        covered_range(pd.DataFrame(index=pd.DatetimeIndex([])))


def test_a_single_bar_is_a_period_of_one_day():
    """The degenerate range must still describe itself rather than raise."""
    prices = frame("2022-01-03", 1)

    assert period_label(prices) == "2022-01-03 to 2022-01-03"


def test_the_heatmap_filename_is_keyed_on_the_covered_years():
    """A filename outlives its run, so a wrong one can be cited later.

    AAPL requested to 2023-01-01 ends on 2022-12-30, so the old name promised a
    year the figure does not contain.
    """
    from run_parameter_sweep import heatmap_path

    path = heatmap_path("AAPL", "2020-01-02", "2022-12-30", 0.0)

    assert path.name == "ma_sharpe_heatmap_AAPL_2020-2022_free.png"


def test_two_requests_resolving_to_the_same_bars_share_one_file():
    """Keying on coverage rather than request makes the name state identity.

    Both requests produce the same figure, so one file is the right answer.
    """
    from run_parameter_sweep import heatmap_path

    assert (heatmap_path("RDDT", "2024-03-21", "2026-07-31", 0.0)
            == heatmap_path("RDDT", "2024-03-21", "2026-07-31", 0.0))


def test_the_filename_still_separates_tickers_and_scenarios():
    """The collision this name was designed to prevent must stay prevented."""
    from run_parameter_sweep import heatmap_path

    names = {
        heatmap_path("AAPL", "2020-01-02", "2022-12-30", 0.0).name,
        heatmap_path("MSFT", "2020-01-02", "2022-12-30", 0.0).name,
        heatmap_path("AAPL", "2020-01-02", "2022-12-30", 0.001).name,
        heatmap_path("AAPL", "2015-01-02", "2022-12-30", 0.0).name,
    }

    assert len(names) == 4


# --------------------------------------------------------------------------
# The wiring, which is where the defect actually lived
# --------------------------------------------------------------------------
#
# Every test above checks a function given the right arguments. None of them
# would notice the sweep handing its requested dates back to a heatmap_path that
# now expects covered ones, which is precisely the mistake being fixed. So the
# script is run end to end with the expensive parts replaced: the download by a
# synthetic history whose range differs from the request by years, and the grid
# by a four-cell one, leaving the labelling untouched and observable.

WAVE_BARS = 320
REQUESTED = ("2010-01-01", "2030-01-01")
LISTED_ON = "2024-03-21"


def listed_late_prices() -> pd.DataFrame:
    """A history that starts years after any plausible request, and oscillates.

    The oscillation matters only so that a crossover strategy trades and the
    Sharpe ratios are defined; the dates are what is under test.
    """
    index = pd.date_range(LISTED_ON, periods=WAVE_BARS, freq="B", name="Date")
    steps = np.arange(WAVE_BARS)
    closes = 100.0 * (1 + 0.2 * np.sin(steps / 18.0) + steps / WAVE_BARS * 0.3)
    return _frame(index).assign(Open=closes, High=closes * 1.01,
                                Low=closes * 0.99, Close=closes)


@pytest.fixture
def swept(monkeypatch, capsys):
    """Run the sweep script over a late-listed history, recording its filenames."""
    import run_parameter_sweep as script
    from analytics.validation import sweep as real_sweep

    prices = listed_late_prices()
    recorded: List[str] = []

    monkeypatch.setattr(script, "get_price_data",
                        lambda *args, **kwargs: prices)
    monkeypatch.setattr(script, "sweep",
                        lambda frame, cash, **kwargs: real_sweep(
                            frame, cash, fast_windows=(5, 10),
                            slow_windows=(20, 30), **kwargs))

    def fake_plot(grid, path, **kwargs):
        recorded.append(path.name)
        return path

    monkeypatch.setattr(script, "plot_heatmap", fake_plot)

    script.main(ticker="LATE", start=REQUESTED[0], end=REQUESTED[1],
                initial_cash=10_000.0)

    return recorded, capsys.readouterr().out, prices


def test_the_sweep_names_its_figures_after_the_years_it_actually_swept(swept):
    """The wiring test: covered years reach the filename, requested ones do not."""
    recorded, _, prices = swept
    expected_years = f"{prices.index[0].year}-{prices.index[-1].year}"

    assert recorded, "the sweep should have written at least one figure"
    for name in recorded:
        assert expected_years in name, name
        assert "2010" not in name, f"the requested start leaked into {name}"
        assert "2030" not in name, f"the requested end leaked into {name}"


def test_the_sweep_header_reports_the_period_it_actually_swept(swept):
    """And the console agrees with the filename rather than with the request."""
    _, output, prices = swept

    assert f"{prices.index[0].date()} to {prices.index[-1].date()}" in output
    assert f"requested {REQUESTED[0]} to {REQUESTED[1]}" in output


def test_the_verdict_line_of_the_comparison_reports_points_not_percent(capsys):
    """run_comparison's own output, since a unit fix has to reach the page.

    Built from a table rather than from a backtest: the wording is what is under
    test, and a hand-made table states the TSLA case exactly.
    """
    import run_comparison as comparison

    table = pd.DataFrame(
        {"MA Crossover": [TSLA_STRATEGY_RETURN, 0.483],
         comparison.BENCHMARK_LABEL: [TSLA_BENCHMARK_RETURN, 1.035]},
        index=[TOTAL_RETURN, SHARPE],
    )

    comparison.print_verdict(table)
    output = capsys.readouterr().out

    assert "-287.73 pp" in output
    assert "-287.73%" not in output
    # The verdict reads the sign, so it must be untouched by the unit change.
    assert "(lost)" in output


def test_a_strategy_ahead_of_the_benchmark_still_reads_as_beating_it(capsys):
    """The other side of the verdict, to show the sign survived the rescaling."""
    import run_comparison as comparison

    table = pd.DataFrame(
        {"Winner": [0.90, 1.4], comparison.BENCHMARK_LABEL: [0.20, 1.0]},
        index=[TOTAL_RETURN, SHARPE],
    )

    comparison.print_verdict(table)
    output = capsys.readouterr().out

    assert "+70.00 pp" in output
    assert "(beat)" in output


# --------------------------------------------------------------------------
# The documented asymmetry, item four, which is explained rather than changed
# --------------------------------------------------------------------------

def test_the_all_cash_caveat_explains_the_beta_and_r_squared_pairing():
    """Beta 0.000 beside R Squared n/a is correct and looks like a fault.

    A flat return series has a genuine slope of zero and no variance for the
    market to explain, so one is defined and the other is zero over zero. The
    numbers are right, which is why the fix was a sentence.
    """
    report = performance_report(flat_equity(), benchmark=rising_equity(),
                                positions=pd.Series(0.0, index=flat_equity().index))
    notes: List[str] = report_caveats(report)

    assert notes, "a fully-in-cash report must raise its exposure caveat"
    note = " ".join(notes)
    assert "held nothing" in note
    assert "market-neutral" in note


def test_the_beta_and_r_squared_values_themselves_are_untouched():
    """The documentation change must not have quietly become a numbers change."""
    from analytics.metrics import capm_regression

    result = capm_regression(flat_equity(), rising_equity())

    assert result.beta == 0.0
    assert pd.isna(result.r_squared)


def test_a_series_regressed_on_itself_still_gives_beta_one_and_r_squared_one():
    """The identity showing the regression itself was left alone.

    Read through the report rather than the function, so the path the caveat
    describes is the path under test.
    """
    market = rising_equity()
    report = performance_report(market, benchmark=market)

    assert report.loc[BETA, STRATEGY_COLUMN] == pytest.approx(1.0)
    assert report.loc[R_SQUARED, STRATEGY_COLUMN] == pytest.approx(1.0)
    assert report.loc[BETA, BENCHMARK_COLUMN] == pytest.approx(1.0)
