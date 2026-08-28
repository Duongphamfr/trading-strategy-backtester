"""Automated tests for the walk-forward summary's denominators.

WHY A SCRIPT GETS A TEST FILE AT ALL
run_walk_forward.py is a script, and the suite otherwise tests the library. It
earns an exception because its summary block is where the project's headline
finding is computed: the gap between in-sample and out-of-sample Sharpe, and how
often honestly chosen parameters beat buy-and-hold. Those are read as results
rather than as output, so the arithmetic behind them deserves the same treatment
as a metric.

WHAT IS LOCKED HERE
A roll whose out-of-sample Sharpe is undefined must not be silently scored as a
failure. It happens when the chosen windows leave the strategy in cash for the
whole window: there are no returns, so there is no ratio. A comparison against
NaN is False in pandas, so such a roll used to land in the numerator of every
success rate while the mean two lines above dropped it entirely, which put two
figures in the same printed block on two different denominators.

The rolls below are constructed rather than run: the summary takes a list of
Roll records and nothing else, so a synthetic list is both sufficient and the
only way to reach the NaN case, which the default history never produces.
"""

from typing import List

import pandas as pd
import pytest

from run_walk_forward import Roll, print_summary

DEFINED = 0.8
BENCHMARK = 0.2


def roll(out_sharpe: float, out_return: float = 0.10) -> Roll:
    """One synthetic roll, with only the summarised fields made meaningful.

    The dates and window choices are placeholders: print_summary reads them for
    the stability count, not for any of the rates under test.
    """
    return Roll(
        in_sample_start=pd.Timestamp("2015-01-01"),
        in_sample_end=pd.Timestamp("2016-12-31"),
        out_start=pd.Timestamp("2017-01-01"),
        out_end=pd.Timestamp("2017-06-30"),
        fast=50,
        slow=200,
        in_sample_sharpe=1.0,
        out_sharpe=out_sharpe,
        out_benchmark_sharpe=BENCHMARK,
        out_return=out_return,
        out_benchmark_return=0.05,
        out_orders=1,
        out_exposure=0.5,
    )


def rate(printed: str, label: str) -> int:
    """The percentage the summary printed on the line carrying this label."""
    for line in printed.splitlines():
        if label in line:
            return int(line.rsplit("%", 1)[0].rsplit(None, 1)[-1])
    raise AssertionError(f"No line mentioning {label!r} in:\n{printed}")


def summarise(rolls: List[Roll], capsys) -> str:
    """Run the summary and hand back what it printed."""
    print_summary(rolls)
    return capsys.readouterr().out


def test_an_unscorable_roll_is_excluded_rather_than_counted_as_a_failure(capsys):
    """Two rolls beat the benchmark, one could not be scored at all.

    The honest reading is that both scorable rolls succeeded: 100%. Counting the
    undefined one as a loss gives 67%, understating the strategy on a window
    where it made no claim either way.
    """
    printed = summarise([roll(DEFINED), roll(DEFINED), roll(float("nan"))],
                        capsys)

    assert rate(printed, "Out-of-sample Sharpe positive") == 100
    assert rate(printed, "Beat buy-and-hold on Sharpe") == 100


def test_the_exclusion_is_reported_and_not_absorbed(capsys):
    """A rate on a reduced denominator has to say so.

    Silently narrowing the sample would trade one misreading for another: the
    reader would take 100% to mean every window, when one window is missing.
    """
    printed = summarise([roll(DEFINED), roll(float("nan"))], capsys)

    assert "1 of 2 rolls" in printed
    assert "entirely in" in printed


def test_nothing_is_reported_as_excluded_when_every_roll_scored(capsys):
    """The note must stay out of the way on the history the project reports.

    All eleven rolls of the default AAPL run have a defined Sharpe, so the
    summary has to read exactly as it did before the denominators were fixed.
    """
    printed = summarise([roll(DEFINED), roll(-0.3)], capsys)

    assert "rolls whose" not in printed
    assert rate(printed, "Out-of-sample Sharpe positive") == 50


def test_the_means_and_the_rates_agree_on_which_rolls_exist(capsys):
    """The specific inconsistency that motivated the fix.

    The mean has always dropped NaN rolls. With one defined roll and one
    undefined, the mean describes a single roll, so a rate reported beside it
    must describe that same roll: a positive Sharpe means 100%, not 50%.
    """
    printed = summarise([roll(DEFINED), roll(float("nan"))], capsys)

    mean_line = next(line for line in printed.splitlines()
                     if "Mean out-of-sample Sharpe" in line)

    assert float(mean_line.split()[-1]) == pytest.approx(DEFINED)
    assert rate(printed, "Out-of-sample Sharpe positive") == 100
