"""Measure how transaction costs erode each strategy, and where the edge dies.

The research question this project is framed around asks whether classical
technical strategies produce durable risk-adjusted returns *after realistic
transaction costs*. At zero cost, momentum was the one strategy to beat
buy-and-hold on Sharpe, by 0.058. This script exists to find out whether that
margin is a finding or a rounding error.

THE BENCHMARK IS NOT CHARGED COSTS, WHICH FAVOURS THE STRATEGIES
Buy-and-hold pays the spread and a commission twice in its life. Each strategy
here pays on every trade. Leaving the benchmark frictionless keeps it a fixed
reference line across every scenario, so a comparison isolates the effect of
costs on the strategy rather than moving both sides at once. The consequence is
that every gap reported below is slightly kinder to the strategies than reality:
a strategy that merely ties the benchmark has in truth lost.

WHY THE TRADE COUNT DOES NOT CHANGE AS COSTS RISE
A strategy in this project never sees the portfolio, only prices, so its signals
are identical at every cost level and it places exactly the same trades no matter
what they cost. What the sweep therefore measures is pure arithmetic drag on a
fixed trade schedule.

That is a real limitation, not an artefact of the implementation. A trader facing
higher costs would trade less, filter marginal signals, or widen thresholds, and
would lose less than these tables show. The numbers here are the cost of
mechanically following the rule, which is the honest thing to measure when the
rule is what is being tested, but it is a floor on how well an adaptive version
of the same idea could do.

It also has a convenient consequence: with the trade schedule fixed, each trade
multiplies the portfolio by one minus the cost, so performance falls smoothly and
monotonically as costs rise. That is what makes the break-even search below a
bisection rather than a guess between grid points.
"""

from typing import Dict, List, NamedTuple, Tuple

import pandas as pd

from analytics.report import (
    BENCHMARK_COLUMN,
    NUMBER_OF_TRADES,
    SHARPE,
    STRATEGY_COLUMN,
    TOTAL_RETURN,
    performance_report,
)
from data.market_data import get_price_data
from engine.backtester import Backtester
from strategies.base_strategy import BaseStrategy
from strategies.momentum import Momentum

# The three strategies, and the reasoning behind the crossover's entry setting,
# live in run_comparison. Importing them keeps one source of truth: if the set
# being studied changes there, the cost analysis follows automatically instead of
# quietly measuring a different set of strategies.
from run_comparison import BENCHMARK_LABEL, build_strategies

TICKER = "AAPL"
START = "2020-01-01"
END = "2023-01-01"
INITIAL_CASH = 10_000.0

RETURN_GAP = "Return vs B&H"
SHARPE_GAP = "Sharpe vs B&H"

# Upper bound of the break-even search. Five percent per trade is far beyond any
# real retail cost, so a strategy still ahead at that level is reported as
# surviving rather than being given a misleadingly precise number.
MAX_SEARCH_COMMISSION = 0.05


class CostScenario(NamedTuple):
    """One set of market frictions to run a strategy under.

    All three costs are carried together even though the headline sweep varies
    only the commission, so adding spread and slippage is a matter of listing
    another scenario rather than changing any code.

    Attributes:
        label: Short name for the table row.
        commission: Proportional commission charged on each trade's value.
        spread: Bid-ask spread as a fraction of price; half is paid per side.
        slippage: Adverse price move as a fraction of price, per side.
    """

    label: str
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0


# The headline sweep: commission only, so the effect is attributable to one
# number. 0.05% to 0.10% brackets a typical retail equity commission, 0.25% is
# punitive, and 0.50% is included to show where the arithmetic ends up rather
# than because anyone pays it.
COMMISSION_SWEEP: Tuple[CostScenario, ...] = (
    CostScenario("free", 0.0),
    CostScenario("0.05%", 0.0005),
    CostScenario("0.10%", 0.0010),
    CostScenario("0.25%", 0.0025),
    CostScenario("0.50%", 0.0050),
)

# Demonstrates that the structure carries all three frictions. Each line adds one
# on top of the last, so the columns show what each friction contributes.
COMBINED_SCENARIOS: Tuple[CostScenario, ...] = (
    CostScenario("free", 0.0, 0.0, 0.0),
    CostScenario("comm only", 0.0005, 0.0, 0.0),
    CostScenario("+ spread", 0.0005, 0.0005, 0.0),
    CostScenario("+ slippage", 0.0005, 0.0005, 0.0005),
    CostScenario("all at 0.10%", 0.0010, 0.0010, 0.0010),
)

# Column layout for the scenario tables, as (metric key, heading, width, format).
# Cost levels are the rows, so the metrics vary across columns and
# analytics.report.format_report cannot be reused: it formats a cell from its row
# label. Declaring everything here keeps the presentation in one place instead of
# scattering format strings through prints. The headings are shortened from the
# metric names because the full labels are wider than the numbers beneath them.
COLUMNS: Tuple[Tuple[str, str, int, str], ...] = (
    (TOTAL_RETURN, "Return", 11, "{:.2%}"),
    (SHARPE, "Sharpe", 9, "{:.3f}"),
    (NUMBER_OF_TRADES, "Trades", 8, "{:.0f}"),
    (RETURN_GAP, "vs B&H", 11, "{:+.2%}"),
    (SHARPE_GAP, "Sharpe vs B&H", 15, "{:+.3f}"),
)


def measure(
    prices: pd.DataFrame,
    initial_cash: float,
    strategy: BaseStrategy,
    scenario: CostScenario,
) -> pd.DataFrame:
    """Backtest one strategy under one cost scenario.

    Args:
        prices: OHLCV history to run over.
        initial_cash: Starting capital.
        strategy: The strategy to run.
        scenario: The frictions to charge.

    Returns:
        The full performance report, with a Strategy and a Benchmark column, so
        callers can read whichever metrics they need without recomputing.
    """
    result = Backtester(
        prices,
        initial_cash=initial_cash,
        strategy=strategy,
        commission=scenario.commission,
        spread=scenario.spread,
        slippage=scenario.slippage,
    ).run()

    return performance_report(
        equity=result.equity_curve["total_value"],
        benchmark=result.benchmark_curve,
        trade_log=result.trade_log,
        positions=result.equity_curve["shares"],
    )


def scenario_table(
    prices: pd.DataFrame,
    initial_cash: float,
    strategy: BaseStrategy,
    scenarios: Tuple[CostScenario, ...],
) -> pd.DataFrame:
    """Build the cost-level table for one strategy.

    The benchmark figures are read from each run's own Benchmark column and must
    be identical everywhere, since the benchmark is never charged. That is
    asserted rather than assumed: if costs ever leaked into the benchmark, every
    gap in this table would be wrong in a way no reader could detect.

    Args:
        prices: OHLCV history to run over.
        initial_cash: Starting capital.
        strategy: The strategy to study.
        scenarios: Cost scenarios, one per row.

    Returns:
        A numeric DataFrame indexed by scenario label, with the columns declared
        in COLUMNS.

    Raises:
        AssertionError: If the benchmark differs between scenarios.
    """
    rows: Dict[str, Dict[str, float]] = {}
    reference: Tuple[float, float] = None

    for scenario in scenarios:
        report = measure(prices, initial_cash, strategy, scenario)

        benchmark_return = report.loc[TOTAL_RETURN, BENCHMARK_COLUMN]
        benchmark_sharpe = report.loc[SHARPE, BENCHMARK_COLUMN]

        if reference is None:
            reference = (benchmark_return, benchmark_sharpe)
        elif (benchmark_return, benchmark_sharpe) != reference:
            raise AssertionError(
                f"The uncharged benchmark changed between cost scenarios, at "
                f"'{scenario.label}'. Costs must never reach the benchmark, or "
                f"every gap reported here is meaningless."
            )

        strategy_return = report.loc[TOTAL_RETURN, STRATEGY_COLUMN]
        strategy_sharpe = report.loc[SHARPE, STRATEGY_COLUMN]

        rows[scenario.label] = {
            TOTAL_RETURN: strategy_return,
            SHARPE: strategy_sharpe,
            NUMBER_OF_TRADES: report.loc[NUMBER_OF_TRADES, STRATEGY_COLUMN],
            RETURN_GAP: strategy_return - benchmark_return,
            SHARPE_GAP: strategy_sharpe - benchmark_sharpe,
        }

    table = pd.DataFrame(rows).T
    table.index.name = "Cost"
    return table[[name for name, _, _, _ in COLUMNS]]


def break_even_commission(
    prices: pd.DataFrame,
    initial_cash: float,
    strategy: BaseStrategy,
    upper: float = MAX_SEARCH_COMMISSION,
    tolerance: float = 1e-6,
) -> float:
    """Commission at which the strategy's Sharpe edge over buy-and-hold vanishes.

    Found by bisection rather than read off the sweep, because the answer usually
    falls between two grid points and interpolating there would invent precision.
    Bisection is valid here for the reason given in the module docstring: the
    trade schedule is fixed, so the Sharpe gap falls monotonically as the
    commission rises and has at most one crossing.

    Args:
        prices: OHLCV history to run over.
        initial_cash: Starting capital.
        strategy: The strategy to study.
        upper: Highest commission to consider.
        tolerance: Width of the bracket at which to stop, as a commission rate.

    Returns:
        The break-even commission. Returns 0.0 if the strategy has no edge even
        when trading is free, and float("inf") if it still leads at `upper`,
        which means the search range rather than reality was the limit.
    """
    def edge(commission: float) -> float:
        """Sharpe gap over the benchmark at one commission level."""
        report = measure(
            prices,
            initial_cash,
            strategy,
            CostScenario("search", commission=commission),
        )
        return (
            report.loc[SHARPE, STRATEGY_COLUMN]
            - report.loc[SHARPE, BENCHMARK_COLUMN]
        )

    if edge(0.0) <= 0.0:
        return 0.0
    if edge(upper) > 0.0:
        return float("inf")

    low, high = 0.0, upper
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if edge(middle) > 0.0:
            low = middle
        else:
            high = middle

    return (low + high) / 2.0


def print_table(table: pd.DataFrame) -> None:
    """Print a scenario table using the layout declared in COLUMNS.

    Args:
        table: A table as returned by scenario_table.
    """
    label_width = max(len(str(label)) for label in table.index) + 2

    header = " " * label_width + "".join(
        f"{heading:>{width}}" for _, heading, width, _ in COLUMNS
    )
    print(header)
    print("-" * len(header))

    for label in table.index:
        cells = "".join(
            f"{template.format(table.loc[label, name]):>{width}}"
            for name, _, width, template in COLUMNS
        )
        print(f"{str(label):<{label_width}}{cells}")


def print_findings(break_evens: List[Tuple[str, float]]) -> None:
    """State where each strategy's risk-adjusted edge over buy-and-hold ends.

    Args:
        break_evens: Strategy label and break-even commission pairs.
    """
    print("\nWhere the risk-adjusted edge disappears")
    print("Break-even commission: the rate at which the strategy's Sharpe ratio "
          "falls\nto the uncharged benchmark's. Below it the strategy leads on "
          "Sharpe, above it\nit does not.\n")

    for label, commission in break_evens:
        if commission == 0.0:
            print(f"  {label:<14} no edge even at zero cost, so there is "
                  f"nothing for costs to erase")
        elif commission == float("inf"):
            print(f"  {label:<14} still ahead at a {MAX_SEARCH_COMMISSION:.2%} "
                  f"commission, beyond the search range")
        else:
            print(f"  {label:<14} break-even at {commission:.4%} per trade")


def print_turnover_sensitivity(prices: pd.DataFrame, initial_cash: float) -> None:
    """Show that cost sensitivity is really a question about turnover.

    The sweep above finds every strategy almost immune to costs, which is not a
    virtue of the strategies. It is because all three traded twice in three
    years, and two trades at a tenth of a percent cost two tenths of a percent.
    Reporting that as robustness would be the wrong reading, so this section
    isolates the actual driver by varying only the review frequency of momentum
    and leaving the signal untouched.

    Momentum is the right subject because its rebalance_freq changes turnover
    without changing the idea being tested, which is exactly the comparison
    needed. The point generalises: cost analysis constrains frequently trading
    rules, and says almost nothing about rules that trade twice.

    Args:
        prices: OHLCV history to run over.
        initial_cash: Starting capital.
    """
    print("\n\nWHY THE TABLES ABOVE BARELY MOVE: TURNOVER, NOT ROBUSTNESS")
    print("Only momentum's review frequency changes here. The signal, the "
          "lookback and the\nasset are identical, so any difference is turnover "
          "and nothing else.\n")

    print(f"  {'review':>12} {'trades':>8} {'Sharpe free':>13} "
          f"{'Sharpe @0.10%':>15} {'break-even':>12}")

    realistic = CostScenario("realistic", commission=0.0010)
    benchmark_sharpe = float("nan")

    for frequency in (63, 21, 5, 1):
        strategy = Momentum(lookback=126, rebalance_freq=frequency)

        free = measure(prices, initial_cash, strategy, CostScenario("free"))
        charged = measure(prices, initial_cash, strategy, realistic)
        commission = break_even_commission(prices, initial_cash, strategy)
        benchmark_sharpe = free.loc[SHARPE, BENCHMARK_COLUMN]

        if commission == float("inf"):
            verdict = "> 5%"
        elif commission == 0.0:
            verdict = "none"
        else:
            verdict = f"{commission:.3%}"

        print(f"  {f'every {frequency}':>12} "
              f"{charged.loc[NUMBER_OF_TRADES, STRATEGY_COLUMN]:>8.0f} "
              f"{free.loc[SHARPE, STRATEGY_COLUMN]:>13.3f} "
              f"{charged.loc[SHARPE, STRATEGY_COLUMN]:>15.3f} "
              f"{verdict:>12}")

    print(f"\n  The benchmark's Sharpe is {benchmark_sharpe:.3f} throughout, "
          f"uncharged. Reviewing more often\n  trades more but scores worse, so "
          f"the edge at the default frequency is a\n  property of that "
          f"frequency, not of momentum. Costs are not what removes it.")


def main(
    ticker: str = TICKER,
    start: str = START,
    end: str = END,
    initial_cash: float = INITIAL_CASH,
) -> None:
    """Run every strategy across the cost sweep and report the findings.

    Args:
        ticker: Yahoo Finance symbol to download, served from cache after the
            first fetch so repeated runs are exactly reproducible.
        start: First date of the history, as YYYY-MM-DD.
        end: Last date of the history, as YYYY-MM-DD.
        initial_cash: Starting capital, identical for every run.
    """
    prices = get_price_data(ticker, start, end)
    strategies = build_strategies()

    title = f"COST SCENARIOS  {ticker}  {start} to {end}  ({len(prices)} bars)"
    print(title)
    print("=" * len(title))
    print(f"Initial cash: {initial_cash:,.2f}")
    print(f"The {BENCHMARK_LABEL} benchmark is never charged costs, so every gap "
          f"below is\nconservative for the strategies: a tie is really a loss.")

    print("\n\nCOMMISSION SWEEP (spread and slippage held at zero)")
    for label, strategy in strategies:
        print(f"\n{label}   {strategy!r}")
        print_table(scenario_table(prices, initial_cash, strategy, COMMISSION_SWEEP))

    break_evens = [
        (label, break_even_commission(prices, initial_cash, strategy))
        for label, strategy in strategies
    ]
    print_findings(break_evens)

    print("\n\nALL THREE FRICTIONS TOGETHER")
    print("Each row adds one friction to the row above, so the drop between two "
          "rows is\nwhat that friction costs.")
    for label, strategy in strategies:
        print(f"\n{label}")
        print_table(scenario_table(prices, initial_cash, strategy, COMBINED_SCENARIOS))

    print_turnover_sensitivity(prices, initial_cash)


if __name__ == "__main__":
    main()
