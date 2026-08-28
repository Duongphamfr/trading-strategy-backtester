"""Run every Phase 2 strategy over one history and lay the results side by side.

The script only wires existing modules together. All configuration lives here at
the edge, while the engine, the strategies and the analytics stay unaware of who
called them, which is the architecture principle the project is built on.

WHY ONE COMBINED TABLE RATHER THAN THREE REPORTS IN SEQUENCE
Three strategies plus the benchmark is four columns, which fits comfortably in a
terminal, and a single table is the only layout in which a reader can actually
compare a row. Printing the reports one after another would force them to hold
numbers in their head across screens of output.

The table is produced by the same analytics.report.format_report used for a
single backtest, with no new formatting code. That works because format_report
decides how to render a cell from its row label, not from its column, so it does
not care whether the columns are Strategy-and-Benchmark or four strategies.
"""

import textwrap
from typing import Dict, List, Tuple

import pandas as pd

from analytics.report import (
    ANNUALIZED_RETURN,
    BENCHMARK_COLUMN,
    EXPOSURE,
    MAX_DRAWDOWN,
    NUMBER_OF_TRADES,
    SHARPE,
    SORTINO,
    STRATEGY_COLUMN,
    TOTAL_RETURN,
    format_report,
    performance_report,
    report_caveats,
)
from data.market_data import get_price_data, period_label
from engine.backtester import Backtester
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import RSIMeanReversion
from strategies.momentum import Momentum
from strategies.moving_average import MovingAverageCrossover

TICKER = "AAPL"
START = "2020-01-01"
END = "2023-01-01"
INITIAL_CASH = 10_000.0

# Short enough to sit in a table column. The full parameterisation is printed
# as a legend above the results, so nothing is hidden by the abbreviation.
BENCHMARK_LABEL = "Buy & Hold"

# One basis point of total return. Below this a strategy is reported as having
# matched the benchmark rather than as having narrowly won or lost, which is the
# honest reading for a rule that simply stayed invested the whole time.
TIE_TOLERANCE = 1e-4

# The rows worth seeing before the detail. Deliberately a subset of the same
# labels, so the summary and the full table can never disagree.
HEADLINE = (
    TOTAL_RETURN,
    ANNUALIZED_RETURN,
    SHARPE,
    SORTINO,
    MAX_DRAWDOWN,
    EXPOSURE,
    NUMBER_OF_TRADES,
)


def build_strategies() -> List[Tuple[str, BaseStrategy]]:
    """The three strategies to compare, with their display labels.

    ON THE CROSSOVER'S enter_on_existing_trend, SET TO True HERE
    This is the one parameter choice in this script that changes what is being
    compared rather than merely how it is configured, so it deserves stating.

    Momentum is phrased as a rule about a state: be invested while the trailing
    return is positive. It therefore enters at its first review date if the
    condition already holds, without needing to have witnessed the trend begin.
    The crossover is phrased as a rule about events, and its default refuses that
    entry, waiting for a golden cross to occur inside the observed window. On
    this sample the averages become defined in October 2020 with the fast one
    already on top, so the default leaves the crossover in cash through the whole
    2020-2021 rally while momentum rides it. Comparing them that way measures the
    difference between two reporting conventions, not between two signals.

    Setting the flag to True puts both rules on state-based footing and makes the
    comparison about the signals. It is the fair setting for this table and not
    the more flattering one in general: it credits the crossover with a trend it
    never saw form, which is exactly why it is not the library default. Flip it
    back to False to see the conservative reading, and expect a large gap.
    """
    return [
        ("MA Crossover", MovingAverageCrossover(
            fast_window=50,
            slow_window=200,
            enter_on_existing_trend=True,
        )),
        ("RSI Reversion", RSIMeanReversion(
            rsi_period=14,
            oversold=30.0,
            overbought=70.0,
        )),
        ("Momentum", Momentum(lookback=126, rebalance_freq=21)),
    ]


def run_all(
    prices: pd.DataFrame,
    initial_cash: float,
    strategies: List[Tuple[str, BaseStrategy]],
) -> Dict[str, pd.DataFrame]:
    """Backtest each strategy and return its full performance report by label.

    Every report is computed with the benchmark, the trade log and the position
    history, so the Exposure, Market and Trades groups all appear and the columns
    of the combined table are guaranteed to share one set of rows.

    Each backtest rebuilds the buy-and-hold curve from the same prices and the
    same starting cash, so all three must agree exactly. That is checked rather
    than assumed: a comparison against a benchmark that drifted between runs
    would be quietly meaningless, and this is the cheapest place to catch it.

    Args:
        prices: OHLCV history every strategy is run over.
        initial_cash: Starting capital, identical for all strategies.
        strategies: Display label and strategy instance pairs.

    Returns:
        A mapping from display label to that strategy's report DataFrame.

    Raises:
        AssertionError: If two runs produced different benchmark curves.
    """
    reports: Dict[str, pd.DataFrame] = {}
    benchmark: pd.Series = None

    for label, strategy in strategies:
        result = Backtester(
            prices,
            initial_cash=initial_cash,
            strategy=strategy,
        ).run()

        if benchmark is None:
            benchmark = result.benchmark_curve
        elif not benchmark.equals(result.benchmark_curve):
            raise AssertionError(
                f"The buy-and-hold curve differs between runs, first seen at "
                f"'{label}'. Every strategy is measured against the same "
                f"benchmark, so the comparison cannot proceed."
            )

        reports[label] = performance_report(
            equity=result.equity_curve["total_value"],
            benchmark=result.benchmark_curve,
            trade_log=result.trade_log,
            positions=result.equity_curve["shares"],
        )

    return reports


def side_by_side(reports: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge per-strategy reports into one table, benchmark in the last column.

    The benchmark is taken from the first report only. All reports contain an
    identical Benchmark column by construction, verified in run_all, and one
    copy is what a reader needs.

    Values stay numeric, consistent with performance_report, so the result can
    be fed to format_report or exported without going through a string stage.

    Args:
        reports: Mapping from display label to report DataFrame.

    Returns:
        A numeric DataFrame with one column per strategy plus the benchmark, and
        the same metric rows, in the same order, as an individual report.
    """
    first = next(iter(reports.values()))

    columns = {label: report[STRATEGY_COLUMN] for label, report in reports.items()}
    columns[BENCHMARK_LABEL] = first[BENCHMARK_COLUMN]

    combined = pd.DataFrame(columns).reindex(first.index)
    combined.index.name = first.index.name
    return combined


def print_verdict(combined: pd.DataFrame) -> None:
    """State each strategy's result relative to buy-and-hold.

    The combined table holds every strategy's absolute numbers but not the one
    quantity the comparison is actually about: the gap against simply holding
    the asset. Beating the benchmark on return while losing on Sharpe, or the
    reverse, is the common and interesting case, so both gaps are shown and the
    one-word verdict refers to total return only.

    The verdict is three-way. A strategy that stays invested throughout is a
    buy-and-hold clone and lands exactly on the benchmark, so calling that a
    loss would be wrong; anything inside a basis point is reported as a match.

    Args:
        combined: A table as returned by side_by_side.
    """
    reference_return = combined.loc[TOTAL_RETURN, BENCHMARK_LABEL]
    reference_sharpe = combined.loc[SHARPE, BENCHMARK_LABEL]

    print(f"\nAgainst {BENCHMARK_LABEL.lower()} "
          f"({reference_return:+.2%} total, Sharpe {reference_sharpe:.3f})")

    for label in combined.columns:
        if label == BENCHMARK_LABEL:
            continue

        excess = combined.loc[TOTAL_RETURN, label] - reference_return
        sharpe_gap = combined.loc[SHARPE, label] - reference_sharpe

        if abs(excess) < TIE_TOLERANCE:
            verdict = "matched"
        else:
            verdict = "beat" if excess > 0 else "lost"

        # The excess is a difference of two returns, which makes it percentage
        # points and not a percentage. Printed with a "%" it claims the
        # impossible on a strong benchmark: trailing a stock that tripled reads
        # as "-287.73%" beside a strategy that actually gained 41.71%, and a
        # long-only position cannot lose more than everything. The number is
        # untouched, only its unit is now stated correctly. The verdict above
        # reads the sign alone, so it is unaffected either way.
        print(f"  {label:<14} return {excess * 100:+8.2f} pp ({verdict})"
              f"   Sharpe {sharpe_gap:+.3f}")


def print_caveats(reports: Dict[str, pd.DataFrame]) -> None:
    """Print the interpretation notes each report raises about itself.

    The notes come from analytics.report.report_caveats, which reads them off
    the report's own contents. They are collected per strategy rather than from
    the combined table because report_caveats inspects the Strategy column that
    an individual report has and a merged one does not.

    Args:
        reports: Mapping from display label to report DataFrame.
    """
    notes = [
        (label, note)
        for label, report in reports.items()
        for note in report_caveats(report)
    ]
    if not notes:
        return

    print("\nHow to read the rows above")
    for label, note in notes:
        print(textwrap.fill(
            note,
            width=78,
            initial_indent=f"  {label}: ",
            subsequent_indent="    ",
        ))


def main(
    ticker: str = TICKER,
    start: str = START,
    end: str = END,
    initial_cash: float = INITIAL_CASH,
) -> None:
    """Run every strategy over one history and print the comparison.

    Args:
        ticker: Yahoo Finance symbol to download.
        start: First date of the history, as YYYY-MM-DD.
        end: Last date of the history, as YYYY-MM-DD.
        initial_cash: Starting capital, identical for every strategy.
    """
    prices = get_price_data(ticker, start, end)
    strategies = build_strategies()
    reports = run_all(prices, initial_cash, strategies)
    combined = side_by_side(reports)

    # The label describes the bars in hand, not the request that fetched them.
    # Yahoo answers a request that predates a listing with a shorter history and
    # no complaint, so printing the request would overstate the study period.
    title = (f"{ticker}  {period_label(prices, start, end)}  "
             f"({len(prices)} bars)")
    print(title)
    print("=" * len(title))
    print(f"Initial cash: {initial_cash:,.2f}")
    for label, strategy in strategies:
        print(f"  {label:<14} {strategy!r}")

    # Summary before detail: the headline rows are a subset of the same labels,
    # sliced from the same table, so the two views cannot contradict each other.
    print("\nHeadline")
    print(format_report(combined.loc[list(HEADLINE)]))

    print_verdict(combined)

    print("\nFull report")
    print(format_report(combined))

    print_caveats(reports)


if __name__ == "__main__":
    main()
