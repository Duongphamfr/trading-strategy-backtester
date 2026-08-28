"""Assembles the individual metrics into one comparable report.

The point of grading a strategy next to buy-and-hold is that the two must be
measured by exactly the same yardstick. This module does nothing but apply the
same functions to two equity curves and lay the answers side by side, so the
comparison cannot drift.

Groups are included only when their inputs exist. The CAPM rows need a market to
regress against, so they appear only with a benchmark; the trade rows need a
trade log; the Exposure row needs the position history. Asking for a report
without any of them is not an error, it simply produces a shorter table.

TWO INTERPRETIVE CAVEATS TO CARRY INTO ANY READING OF THE OUTPUT

A strategy that moves in and out of the market posts a zero return on every bar
it spends in cash. Those flat bars enter the volatility and the Sharpe
denominator like any other, which mechanically lowers measured risk, and they
are uncorrelated with the market, which drags beta and R squared toward zero.
Part of any apparent risk advantage over buy-and-hold is therefore simply time
spent not invested, not superior risk management. The Exposure row measures
exactly that, which is why it opens the Risk group; when it is very low,
print_report says so beneath the table via report_caveats.

The Distribution rows exist to tell you when to distrust the rows above them. If
excess kurtosis is high and Jarque-Bera rejects normality, then Sharpe and
parametric VaR are both resting on an assumption the data has just refused, and
the historical and conditional tail measures are the ones to believe.
"""

import textwrap
from typing import Any, Dict, List, Optional

import pandas as pd

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
    exposure,
    historical_var,
    jarque_bera,
    kurtosis,
    max_drawdown,
    max_drawdown_duration,
    parametric_var,
    skewness,
    volatility,
)
from analytics.trade_stats import trade_statistics
from constants import TRADING_DAYS_PER_YEAR

STRATEGY_COLUMN = "Strategy"
BENCHMARK_COLUMN = "Benchmark"

# Confidence level for the tail-risk rows. Kept at one level so the table stays
# readable; call the risk functions directly for a fuller tail profile.
VAR_CONFIDENCE = 0.95

# Significance level at which the normality verdict is decided.
NORMALITY_ALPHA = 0.05

# Below this exposure, the printed report carries a note explaining that the
# distribution, tail and CAPM rows are describing cash rather than the strategy.
# The value is a judgement call and deliberately overridable: the distortion
# grows smoothly as exposure falls, so no threshold is the true one. A quarter of
# the bars is where it stops being a nuance and starts inverting conclusions.
LOW_EXPOSURE_THRESHOLD = 0.25

TOTAL_RETURN = "Total Return"
ANNUALIZED_RETURN = "Annualized Return"

EXPOSURE = "Exposure"
VOLATILITY = "Volatility"
MAX_DRAWDOWN = "Max Drawdown"
MAX_DRAWDOWN_DURATION = "Max Drawdown Duration (bars)"

SHARPE = "Sharpe Ratio"
SORTINO = "Sortino Ratio"
CALMAR = "Calmar Ratio"

SKEWNESS = "Skewness"
EXCESS_KURTOSIS = "Excess Kurtosis"
JARQUE_BERA_P = "Jarque-Bera p-value"
NORMALITY_REJECTED = "Normality Rejected (5%)"

HISTORICAL_VAR = "Historical VaR (95%)"
PARAMETRIC_VAR = "Parametric VaR (95%)"
CONDITIONAL_VAR = "CVaR (95%)"

ALPHA = "Alpha (annualized)"
BETA = "Beta"
R_SQUARED = "R Squared"

NUMBER_OF_TRADES = "Number of Trades"
WIN_RATE = "Win Rate"
AVERAGE_WIN = "Average Win"
AVERAGE_LOSS = "Average Loss"
PROFIT_FACTOR = "Profit Factor"

# Display order and grouping of the whole table. Also the single source of truth
# for which metric belongs to which group, used both to order the rows and to
# print the section headings.
SECTIONS = (
    ("Return", (TOTAL_RETURN, ANNUALIZED_RETURN)),
    # Exposure leads the Risk group on purpose: it decides whether the three
    # measures beneath it, and the Distribution and CAPM groups further down,
    # are describing the strategy or the cash it was sitting in.
    ("Risk", (EXPOSURE, VOLATILITY, MAX_DRAWDOWN, MAX_DRAWDOWN_DURATION)),
    ("Risk-Adjusted", (SHARPE, SORTINO, CALMAR)),
    (
        "Distribution",
        (SKEWNESS, EXCESS_KURTOSIS, JARQUE_BERA_P, NORMALITY_REJECTED),
    ),
    ("Tail Risk", (HISTORICAL_VAR, PARAMETRIC_VAR, CONDITIONAL_VAR)),
    ("Market (CAPM)", (ALPHA, BETA, R_SQUARED)),
    (
        "Trades",
        (NUMBER_OF_TRADES, WIN_RATE, AVERAGE_WIN, AVERAGE_LOSS, PROFIT_FACTOR),
    ),
)

# Rows that only exist when the corresponding input was supplied.
EXPOSURE_METRICS = frozenset({EXPOSURE})
CAPM_METRICS = frozenset({ALPHA, BETA, R_SQUARED})
TRADE_METRICS = frozenset(
    {NUMBER_OF_TRADES, WIN_RATE, AVERAGE_WIN, AVERAGE_LOSS, PROFIT_FACTOR}
)

# How each row should be rendered. Anything unlisted is shown as a plain ratio.
PERCENT_METRICS = frozenset(
    {
        TOTAL_RETURN,
        ANNUALIZED_RETURN,
        EXPOSURE,
        VOLATILITY,
        MAX_DRAWDOWN,
        HISTORICAL_VAR,
        PARAMETRIC_VAR,
        CONDITIONAL_VAR,
        ALPHA,
        WIN_RATE,
    }
)
COUNT_METRICS = frozenset({MAX_DRAWDOWN_DURATION, NUMBER_OF_TRADES})
CURRENCY_METRICS = frozenset({AVERAGE_WIN, AVERAGE_LOSS})
PROBABILITY_METRICS = frozenset({JARQUE_BERA_P})
FLAG_METRICS = frozenset({NORMALITY_REJECTED})


def performance_report(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    trade_log: Optional[List[Dict[str, Any]]] = None,
    positions: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute every applicable metric for a strategy, and for a benchmark if given.

    Args:
        equity: Strategy portfolio value over time.
        benchmark: Optional buy-and-hold value over time, measured on identical
            terms. Omit it to report the strategy alone; the Market (CAPM) rows
            are dropped with it, since there is nothing to regress against.
        periods_per_year: Number of bars in a year, 252 for daily data. Applied
            to both columns, so the two remain comparable.
        risk_free_rate: ANNUAL risk-free rate as a decimal fraction, so 0.04
            means 4% per year. De-annualized internally. It affects Sharpe,
            Sortino and the CAPM rows, which measure return in excess of it;
            total return, volatility, drawdown, tail risk and Calmar do not use
            it. Defaults to 0.0, which treats cash as earning nothing. That is
            close enough to reality for 2020-2021 but clearly wrong from 2023
            onward, and leaving it at zero flatters every strategy equally by
            crediting it with the return it could have had for free.
        trade_log: Optional list of executed trades from the backtester. Omit it
            to drop the Trades rows.
        positions: Optional per-bar position size, the shares column of the
            backtester's equity curve. Omit it to drop the Exposure row. Passing
            it is strongly recommended: exposure is what tells a reader whether
            the risk and distribution rows describe the strategy or the cash.

    Returns:
        A DataFrame indexed by metric name, in the order given by SECTIONS, with
        a Strategy column and, when a benchmark was supplied, a Benchmark column.

        Values are raw numbers, not formatted strings, so the frame stays usable
        for charts and parameter sweeps: decimal fractions for returns,
        volatility, drawdown, VaR, alpha and win rate; bar counts for the
        drawdown duration and trade count; currency for average win and loss;
        and dimensionless ratios elsewhere. The normality verdict is stored as
        1.0 or 0.0 so the frame remains numeric throughout. Metrics that do not
        apply hold NaN. Use format_report to render it for reading.

        The Trades rows describe the supplied trade log, which belongs to the
        strategy, so they are NaN in the Benchmark column rather than pretending
        buy-and-hold produced them.

        The CAPM rows in the Benchmark column regress the benchmark on itself,
        which must give a beta of exactly 1, an alpha of 0 and an R squared of 1.
        That is not filler: it is a free arithmetic check on the regression, and
        anything else in those cells means something is wrong.
    """
    labels = _active_labels(
        has_benchmark=benchmark is not None,
        has_trade_log=trade_log is not None,
        has_positions=positions is not None,
    )

    columns = {
        STRATEGY_COLUMN: _metric_column(
            equity,
            market=benchmark,
            trade_log=trade_log,
            positions=positions,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ).reindex(labels)
    }

    if benchmark is not None:
        # The benchmark gets no positions argument, so its Exposure cell reads
        # n/a. Buy-and-hold is invested on every bar, but the benchmark is only
        # a curve here, and asserting 100% would be assuming what it holds.
        columns[BENCHMARK_COLUMN] = _metric_column(
            benchmark,
            market=benchmark,
            trade_log=None,
            positions=None,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ).reindex(labels)

    report = pd.DataFrame(columns, index=labels)
    report.index.name = "Metric"
    return report


def format_report(report: pd.DataFrame, value_width: int = 14) -> str:
    """Render a report as an aligned plain-text table with section headings.

    Each row is formatted according to what it measures: percentages with two
    decimals, currency with thousands separators, counts as whole numbers, the
    Jarque-Bera p-value in general notation, the normality verdict as yes or no,
    and everything else as a three-decimal ratio. Undefined values read "n/a"
    rather than "nan", since a metric that does not apply is not a failure.

    Args:
        report: A DataFrame as returned by performance_report.
        value_width: Column width for the numbers, in characters.

    Returns:
        The table as a single string, without a trailing newline.
    """
    if report.empty:
        return "(empty report)"

    indent = "  "
    label_width = max(len(indent + str(label)) for label in report.index) + 2

    header = " " * label_width + "".join(
        f"{str(column):>{value_width}}" for column in report.columns
    )
    lines = [header, "-" * len(header)]

    for section, members in _sections_present(report):
        lines.append(section)
        for label in members:
            values = "".join(
                f"{_format_value(label, value):>{value_width}}"
                for value in report.loc[label]
            )
            lines.append(f"{indent + str(label):<{label_width}}{values}")

    return "\n".join(lines)


def report_caveats(
    report: pd.DataFrame,
    low_exposure_threshold: float = LOW_EXPOSURE_THRESHOLD,
) -> List[str]:
    """Warnings about how this particular report should be read.

    A number can be arithmetically correct and still invite the wrong
    conclusion. This function collects the cases where the table's own contents
    reveal that risk, so the warning is derived from the data rather than
    guessed at by the reader.

    WHY THIS IS SEPARATE FROM THE TABLE
    Three reasons. The table stays purely numeric and machine-readable, so
    sweeps and exports are unaffected. Callers who want only the table keep
    using format_report and see nothing new. And the Phase 5 UI can render these
    same strings as proper warning banners instead of scraping them out of
    preformatted text.

    Today only low exposure triggers a note. The return type is a list so
    further checks can be added without changing any call site.

    Args:
        report: A DataFrame as returned by performance_report.
        low_exposure_threshold: Strategy exposure below which the note fires.
            Raise it to be warned more often, or set it to 0 to silence.

    Returns:
        Zero or more unwrapped sentences, in display order. An empty list means
        nothing about this report needs qualifying.
    """
    notes: List[str] = []

    if EXPOSURE in report.index and STRATEGY_COLUMN in report.columns:
        value = report.loc[EXPOSURE, STRATEGY_COLUMN]
        if not pd.isna(value) and value < low_exposure_threshold:
            notes.append(
                f"Exposure is {value:.2%}, so the strategy was in cash on almost "
                "every bar. Those flat bars return exactly zero, which lowers "
                "Volatility and both VaR figures, inflates Skewness and Excess "
                "Kurtosis, and pulls Beta and R Squared toward zero. Read those "
                "rows as a description of cash, not of the strategy. The "
                "drawdown rows need a different qualification rather than the "
                "same one: a drawdown depth is a path extremum, not an average, "
                "so flat bars cannot dilute it and the figure is exactly what "
                "the portfolio suffered. What does not follow is the credit. "
                "Falling less than a fully invested benchmark, while holding "
                "cash on nine bars in ten, measures absence from the market and "
                "not better risk control. Total Return and the Trades section "
                "are unaffected on both counts. In the limit of no position at "
                "all, Beta reads a defined 0.000 while R Squared reads n/a, "
                "which looks inconsistent and is not: a flat return series has a "
                "genuine regression slope of zero, but no variance of its own for "
                "the market to explain, so the ratio behind R Squared is zero "
                "over zero. Read that Beta as held nothing rather than as "
                "market-neutral, which is a position taken and this is not."
            )

    return notes


def print_report(
    report: pd.DataFrame,
    title: Optional[str] = None,
    caveats: bool = True,
) -> None:
    """Print a report as an aligned table, with an optional title above it.

    Args:
        report: A DataFrame as returned by performance_report.
        title: Optional heading, underlined to separate successive reports.
        caveats: Whether to print the notes from report_caveats beneath the
            table. Wrapping happens here, since it is a terminal concern.
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    print(format_report(report))

    if caveats:
        for note in report_caveats(report):
            print()
            print(textwrap.fill(note, width=76, initial_indent="Note: ",
                                subsequent_indent="      "))


def _active_labels(
    has_benchmark: bool,
    has_trade_log: bool,
    has_positions: bool,
) -> List[str]:
    """Metric labels to include, in display order, given the available inputs."""
    excluded = set()
    if not has_benchmark:
        excluded |= CAPM_METRICS
    if not has_trade_log:
        excluded |= TRADE_METRICS
    if not has_positions:
        excluded |= EXPOSURE_METRICS

    return [
        label
        for _, members in SECTIONS
        for label in members
        if label not in excluded
    ]


def _sections_present(report: pd.DataFrame):
    """Yield (section, labels) pairs for the sections this report contains.

    Any row not accounted for by SECTIONS is gathered under "Other" rather than
    silently dropped, so a metric added upstream cannot disappear from the table.
    """
    accounted = set()
    for section, members in SECTIONS:
        present = [label for label in members if label in report.index]
        accounted.update(present)
        if present:
            yield section, present

    leftover = [label for label in report.index if label not in accounted]
    if leftover:
        yield "Other", leftover


def _metric_column(
    equity: pd.Series,
    market: Optional[pd.Series],
    trade_log: Optional[List[Dict[str, Any]]],
    positions: Optional[pd.Series],
    periods_per_year: int,
    risk_free_rate: float,
) -> pd.Series:
    """Every metric computable for one equity curve, keyed by label."""
    normality = jarque_bera(equity)

    values: Dict[str, float] = {
        TOTAL_RETURN: total_return(equity),
        ANNUALIZED_RETURN: annualized_return(equity, periods_per_year),
        VOLATILITY: volatility(equity, periods_per_year),
        MAX_DRAWDOWN: max_drawdown(equity),
        MAX_DRAWDOWN_DURATION: float(max_drawdown_duration(equity)),
        SHARPE: sharpe_ratio(
            equity,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        SORTINO: sortino_ratio(
            equity,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        CALMAR: calmar_ratio(equity, periods_per_year),
        SKEWNESS: skewness(equity),
        EXCESS_KURTOSIS: kurtosis(equity, excess=True),
        JARQUE_BERA_P: normality.p_value,
        NORMALITY_REJECTED: _as_flag(normality.p_value),
        HISTORICAL_VAR: historical_var(equity, VAR_CONFIDENCE),
        PARAMETRIC_VAR: parametric_var(equity, VAR_CONFIDENCE),
        CONDITIONAL_VAR: conditional_var(equity, VAR_CONFIDENCE),
    }

    if positions is not None:
        values[EXPOSURE] = exposure(positions)

    if market is not None:
        capm = capm_regression(
            equity,
            market,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        values[ALPHA] = capm.alpha
        values[BETA] = capm.beta
        values[R_SQUARED] = capm.r_squared

    if trade_log is not None:
        trades = trade_statistics(trade_log)
        values[NUMBER_OF_TRADES] = float(trades.number_of_trades)
        values[WIN_RATE] = trades.win_rate
        values[AVERAGE_WIN] = trades.average_win
        values[AVERAGE_LOSS] = trades.average_loss
        values[PROFIT_FACTOR] = trades.profit_factor

    return pd.Series(values, dtype=float)


def _as_flag(p_value: float) -> float:
    """Turn a p-value into a numeric yes/no flag, keeping the frame numeric."""
    if pd.isna(p_value):
        return float("nan")
    return 1.0 if p_value < NORMALITY_ALPHA else 0.0


def _format_value(label: str, value: float) -> str:
    """Render one cell according to what its row measures.

    ON THE ADDITION OF ZERO
    IEEE-754 has two zeros, and the negative one reaches this function whenever a
    metric negates a zero to obey a sign convention. The VaR family is where it
    shows: the module reports a loss as a positive magnitude by returning
    -quantile, so an all-cash strategy, whose every return is exactly zero, hands
    over -0.0 and the table reads "-0.00%" for a quantity documented as positive.

    Adding zero collapses the two, since -0.0 + 0.0 is +0.0 by the standard, and
    leaves every other double untouched: the addition is exact for all finite
    values and for the infinities. NaN never arrives here, having returned above.

    Doing it once here rather than at each computation site is deliberate. The
    sign of a zero is a rendering concern and nothing else, -0.0 == 0.0 is true so
    no comparison or test anywhere can distinguish them, and a metric added later
    that negates its own zero is covered without anyone remembering to.
    """
    if pd.isna(value):
        return "n/a"
    value = value + 0.0
    if label in FLAG_METRICS:
        return "yes" if value >= 0.5 else "no"
    if label in PERCENT_METRICS:
        return f"{value:.2%}"
    if label in COUNT_METRICS:
        return f"{int(round(value))}"
    if label in CURRENCY_METRICS:
        return f"{value:,.2f}"
    if label in PROBABILITY_METRICS:
        # A p-value of 1e-300 and one of exactly 0 mean the same thing in
        # practice, and both are unreadable in general notation.
        return "<0.0001" if value < 1e-4 else f"{value:.4g}"
    return f"{value:.3f}"


if __name__ == "__main__":
    from constants import BUY, SELL

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
    #
    # Note that the two curves here are constructed independently, so the CAPM
    # row for the Strategy column is meaningless by design: regressing one
    # arbitrary series on an unrelated one yields a wild beta and a low R
    # squared, which is the correct answer to a badly posed question. The
    # Benchmark column is the one to read, since regressing the benchmark on
    # itself must return beta 1.000 and R squared 1.000 exactly.
    steady = equity_from_returns([0.005, 0.003, -0.001, 0.006, 0.004] * 4)
    jumpy = equity_from_returns(
        [0.03, -0.02, 0.04, -0.03, 0.02, 0.03, -0.04, 0.05, -0.02, 0.03] * 2
    )

    demo_trades = [
        {"date": pd.Timestamp("2022-01-04"), "action": BUY,
         "price": 100.0, "shares": 10.0},
        {"date": pd.Timestamp("2022-01-08"), "action": SELL,
         "price": 112.0, "shares": 10.0},
        {"date": pd.Timestamp("2022-01-12"), "action": BUY,
         "price": 115.0, "shares": 10.0},
        {"date": pd.Timestamp("2022-01-16"), "action": SELL,
         "price": 108.0, "shares": 10.0},
    ]

    print_report(
        performance_report(
            jumpy,
            steady,
            periods_per_year=20,
            trade_log=demo_trades,
        ),
        title="Full report: every group present",
    )

    print_report(
        performance_report(steady, periods_per_year=20),
        title="No benchmark and no trade log: CAPM and Trades dropped",
    )
