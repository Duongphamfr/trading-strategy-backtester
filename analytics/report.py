"""Assembles the individual metrics into one comparable report.

The point of grading a strategy next to buy-and-hold is that the two must be
measured by exactly the same yardstick. This module does nothing but apply the
same functions to two equity curves and lay the answers side by side, so the
comparison cannot drift.

One interpretive caveat worth carrying into any reading of the output. A strategy
that moves in and out of the market posts a zero return on every bar it spends in
cash. Those flat bars enter the volatility and the Sharpe denominator like any
other, which mechanically lowers measured risk. Part of any volatility advantage
over buy-and-hold is therefore simply time spent not invested, not superior risk
management. The number of trades and the equity curve shape are what tell the two
apart.
"""

from typing import Optional

import pandas as pd

from analytics.metrics import (
    annualized_return,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from analytics.risk import (
    max_drawdown,
    max_drawdown_duration,
    volatility,
)
from constants import TRADING_DAYS_PER_YEAR

STRATEGY_COLUMN = "Strategy"
BENCHMARK_COLUMN = "Benchmark"

TOTAL_RETURN = "Total Return"
ANNUALIZED_RETURN = "Annualized Return"
VOLATILITY = "Volatility"
MAX_DRAWDOWN = "Max Drawdown"
MAX_DRAWDOWN_DURATION = "Max Drawdown Duration (bars)"
SHARPE = "Sharpe Ratio"
SORTINO = "Sortino Ratio"
CALMAR = "Calmar Ratio"

# How each row should be rendered. Anything not listed is shown as a ratio.
PERCENT_METRICS = frozenset(
    {TOTAL_RETURN, ANNUALIZED_RETURN, VOLATILITY, MAX_DRAWDOWN}
)
COUNT_METRICS = frozenset({MAX_DRAWDOWN_DURATION})


def performance_report(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compute every metric for a strategy, and for a benchmark if given.

    Args:
        equity: Strategy portfolio value over time.
        benchmark: Optional buy-and-hold value over time, measured on identical
            terms. Omit it, or pass None, to report the strategy alone.
        periods_per_year: Number of bars in a year, 252 for daily data. Applied
            to both columns, so the two remain comparable.

    Returns:
        A DataFrame indexed by metric name, with a Strategy column and, when a
        benchmark was supplied, a Benchmark column. Values are raw numbers, not
        formatted strings: decimal fractions for returns, volatility and
        drawdown, a bar count for the drawdown duration, and dimensionless
        ratios for Sharpe, Sortino and Calmar. Undefined metrics hold NaN.
        Use format_report to render it for reading.
    """
    columns = {STRATEGY_COLUMN: _metric_column(equity, periods_per_year)}
    if benchmark is not None:
        columns[BENCHMARK_COLUMN] = _metric_column(benchmark, periods_per_year)

    report = pd.DataFrame(columns)
    report.index.name = "Metric"
    return report


def format_report(report: pd.DataFrame, value_width: int = 14) -> str:
    """Render a report as an aligned plain-text table.

    Percentages are shown with two decimals, ratios with three, and the
    drawdown duration as a whole number of bars. Undefined values read "n/a"
    rather than "nan", since a metric that does not apply is not a failure.

    Args:
        report: A DataFrame as returned by performance_report.
        value_width: Column width for the numbers, in characters.

    Returns:
        The table as a single string, without a trailing newline.
    """
    if report.empty:
        return "(empty report)"

    label_width = max(len(str(label)) for label in report.index)

    header = " " * label_width + "".join(
        f"{str(column):>{value_width}}" for column in report.columns
    )
    lines = [header, "-" * len(header)]

    for label, row in report.iterrows():
        values = "".join(
            f"{_format_value(label, value):>{value_width}}" for value in row
        )
        lines.append(f"{str(label):<{label_width}}{values}")

    return "\n".join(lines)


def print_report(report: pd.DataFrame, title: Optional[str] = None) -> None:
    """Print a report as an aligned table, with an optional title above it.

    Args:
        report: A DataFrame as returned by performance_report.
        title: Optional heading, underlined to separate successive reports.
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    print(format_report(report))


def _metric_column(equity: pd.Series, periods_per_year: int) -> pd.Series:
    """Every metric for one equity curve, in display order."""
    return pd.Series(
        {
            TOTAL_RETURN: total_return(equity),
            ANNUALIZED_RETURN: annualized_return(equity, periods_per_year),
            VOLATILITY: volatility(equity, periods_per_year),
            MAX_DRAWDOWN: max_drawdown(equity),
            MAX_DRAWDOWN_DURATION: float(max_drawdown_duration(equity)),
            SHARPE: sharpe_ratio(equity, periods_per_year=periods_per_year),
            SORTINO: sortino_ratio(equity, periods_per_year=periods_per_year),
            CALMAR: calmar_ratio(equity, periods_per_year),
        },
        dtype=float,
    )


def _format_value(label: str, value: float) -> str:
    """Render one cell according to what its row measures."""
    if pd.isna(value):
        return "n/a"
    if label in PERCENT_METRICS:
        return f"{value:.2%}"
    if label in COUNT_METRICS:
        return f"{int(round(value))}"
    return f"{value:.3f}"


if __name__ == "__main__":
    def equity_from_returns(returns: list, start: float = 100.0) -> pd.Series:
        """Build the equity curve implied by a list of periodic returns."""
        values = [start]
        for periodic_return in returns:
            values.append(values[-1] * (1.0 + periodic_return))
        return pd.Series(
            values,
            index=pd.date_range("2022-01-03", periods=len(values), freq="D"),
            dtype=float,
        )

    # A steady climber against a jumpier curve that ends up further ahead. The
    # jumpy one wins on return and loses on every risk row, which is exactly the
    # trade-off the risk-adjusted ratios exist to price.
    steady = equity_from_returns([0.005, 0.003, -0.001, 0.006, 0.004] * 4)
    jumpy = equity_from_returns(
        [0.03, -0.02, 0.04, -0.03, 0.02, 0.03, -0.04, 0.05, -0.02, 0.03] * 2
    )

    print_report(
        performance_report(jumpy, steady, periods_per_year=20),
        title="Jumpy curve vs steady curve (20 bars/yr)",
    )

    print_report(
        performance_report(steady, periods_per_year=20),
        title="Steady curve alone, no benchmark",
    )

    print_report(
        performance_report(pd.Series(dtype=float), pd.Series(dtype=float)),
        title="Empty curves, every metric undefined",
    )
