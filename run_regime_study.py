"""Test whether market-exit strategies earn their keep in the regimes that should favour them.

The rest of this project's comparisons live in one window, 2020-2022, which is a
crash, a violent rebound and a decline stacked into a net bull. That is one
economic story. Trend-following and momentum are sold on a different one: they
give up upside in a grind higher in exchange for being in cash when the market
falls. Judging them only on a window that ends well is asking a fire door how
it performed during a year without a fire.

This script runs the same three strategies, through the same engine, on four
windows chosen to isolate that trade-off:

    2008 crisis (bear)     the fire: does anything protect capital?
    2015-16 chop (sideways) the typical trend-following graveyard
    2017 grind (bull)       the cost of sitting out a quiet rally
    2020-22 mixed           the project's default window, for continuity

SPY is the market the hypothesis is about. AAPL is the single-name check: a
strategy that only "protects" on one mega-cap is not a regime result.

COSTS ARE LEFT AT ZERO ON PURPOSE
run_comparison and the 2020-22 numbers this script is meant to sit beside are
frictionless. Adding a commission here would mix two questions — "does the
regime change the ranking" and "do costs erase the ranking" — and make the
2020-22 row disagree with a file the reader already has. The cost question is
already answered for the default window by run_cost_scenarios.py. What this
file asks is narrower: before frictions, is the advertised bear-market
protection even there?

THE BENCHMARK IS UNCHARGED, AS EVERYWHERE ELSE
Buy-and-hold is the asset's own path. A shallower strategy drawdown in 2008 is
therefore "was in cash while the asset fell", which is exactly the claim being
tested, not the cash-as-artefact warning that applies when a strategy sits out
a bull and then claims a risk win.

Rows of the summary are one regime on one asset, not a regime collapsed across
assets. Averaging SPY and AAPL would hide the case the hypothesis cares about:
agreement. If they disagree, that is the finding.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.report import (
    BENCHMARK_COLUMN,
    EXPOSURE,
    MAX_DRAWDOWN,
    NUMBER_OF_TRADES,
    SHARPE,
    STRATEGY_COLUMN,
    TOTAL_RETURN,
    performance_report,
)
from data.market_data import DataSourceUnavailable, get_price_data, period_label
from engine.backtester import Backtester
from strategies.base_strategy import BaseStrategy

from run_comparison import BENCHMARK_LABEL, TIE_TOLERANCE, build_strategies

ASSETS: Tuple[str, ...] = ("SPY", "AAPL")
INITIAL_CASH = 10_000.0

# A drawdown difference inside this band is a match, not protection. Same order
# of magnitude as the return-tie band: one basis point of the equity path.
DRAWDOWN_TIE = 1e-4


class Regime(NamedTuple):
    """One market window, named by the story it is supposed to tell.

    `end` is exclusive, matching yfinance and the other scripts. The labels
    describe the inclusive span a reader has in mind.

    Attributes:
        name: Short label used as a table row, including the regime character.
        start: First date of the history, YYYY-MM-DD.
        end: Last date of the history, exclusive, YYYY-MM-DD.
        story: What this window is meant to test, printed above its detail.
    """

    name: str
    start: str
    end: str
    story: str


REGIMES: Tuple[Regime, ...] = (
    Regime(
        "2008 crisis (bear)",
        "2008-01-01",
        "2009-07-01",
        "Sharp bear. The real test of whether an exit rule protects capital.",
    ),
    Regime(
        "2015-16 chop (sideways)",
        "2015-01-01",
        "2017-01-01",
        "Choppy, little net progress. Trend-following typically whipsaws here.",
    ),
    Regime(
        "2017 grind (bull)",
        "2017-01-01",
        "2018-01-01",
        "Calm, almost uninterrupted uptrend. Sitting in cash is a cost, not a hedge. "
        "One year of bars is only just enough for a 200-day average to exist.",
    ),
    Regime(
        "2020-22 mixed",
        "2020-01-01",
        "2023-01-01",
        "COVID crash, rally, then decline. The project's default window.",
    ),
)


class StrategyScore(NamedTuple):
    """One strategy's headline numbers on one regime × asset cell."""

    label: str
    total_return: float
    sharpe: float
    max_drawdown: float
    exposure: float
    trades: float
    return_gap: float
    sharpe_gap: float
    shallower_drawdown: bool


class Cell(NamedTuple):
    """Everything measured on one regime × asset pair."""

    regime: Regime
    ticker: str
    n_bars: int
    period: str
    benchmark_return: float
    benchmark_sharpe: float
    benchmark_drawdown: float
    scores: Tuple[StrategyScore, ...]


def is_shallower(strategy_dd: float, benchmark_dd: float) -> bool:
    """True when the strategy's peak-to-trough decline is meaningfully smaller.

    Max drawdown is stored as a negative fraction, so "shallower" is the less
    negative number: −15% is shallower than −45%. A difference inside
    DRAWDOWN_TIE is a match, not a win, matching how return ties are treated.

    Args:
        strategy_dd: Strategy max drawdown, as a negative fraction.
        benchmark_dd: Buy-and-hold max drawdown, same convention.

    Returns:
        True only when both numbers are finite and the strategy's is better by
        more than the tie band.
    """
    if not (np.isfinite(strategy_dd) and np.isfinite(benchmark_dd)):
        return False
    return strategy_dd > benchmark_dd + DRAWDOWN_TIE


def best_by(scores: Tuple[StrategyScore, ...], field: str) -> Optional[StrategyScore]:
    """The score with the highest finite value of `field`, or None.

    Args:
        scores: The strategies measured on one cell.
        field: A StrategyScore attribute name. Used for the two gap fields.

    Returns:
        The winning score, or None if every value of that field is NaN.
    """
    valid = [score for score in scores if np.isfinite(getattr(score, field))]
    if not valid:
        return None
    return max(valid, key=lambda score: getattr(score, field))


def evaluate(
    prices: pd.DataFrame,
    initial_cash: float,
    strategies: List[Tuple[str, BaseStrategy]],
) -> Tuple[float, float, float, Tuple[StrategyScore, ...]]:
    """Run every strategy on one price history and score them against buy-and-hold.

    Rebuilds the benchmark from the first run and checks the rest against it,
    the same guard run_comparison uses: a drifting reference would make every
    gap in this file meaningless.

    Args:
        prices: OHLCV history for this cell.
        initial_cash: Starting capital, identical for every strategy.
        strategies: Display label and strategy instance pairs.

    Returns:
        Benchmark return, Sharpe and max drawdown, then one score per strategy.

    Raises:
        AssertionError: If two runs produced different benchmark curves.
    """
    scores: List[StrategyScore] = []
    benchmark: Optional[pd.Series] = None
    bench_return = bench_sharpe = bench_dd = float("nan")

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

        report = performance_report(
            equity=result.equity_curve["total_value"],
            benchmark=result.benchmark_curve,
            trade_log=result.trade_log,
            positions=result.equity_curve["shares"],
        )

        bench_return = float(report.loc[TOTAL_RETURN, BENCHMARK_COLUMN])
        bench_sharpe = float(report.loc[SHARPE, BENCHMARK_COLUMN])
        bench_dd = float(report.loc[MAX_DRAWDOWN, BENCHMARK_COLUMN])

        total = float(report.loc[TOTAL_RETURN, STRATEGY_COLUMN])
        sharpe = float(report.loc[SHARPE, STRATEGY_COLUMN])
        drawdown = float(report.loc[MAX_DRAWDOWN, STRATEGY_COLUMN])
        scores.append(StrategyScore(
            label=label,
            total_return=total,
            sharpe=sharpe,
            max_drawdown=drawdown,
            exposure=float(report.loc[EXPOSURE, STRATEGY_COLUMN]),
            trades=float(report.loc[NUMBER_OF_TRADES, STRATEGY_COLUMN]),
            return_gap=total - bench_return,
            sharpe_gap=sharpe - bench_sharpe,
            shallower_drawdown=is_shallower(drawdown, bench_dd),
        ))

    return bench_return, bench_sharpe, bench_dd, tuple(scores)


def load_cell(regime: Regime, ticker: str, initial_cash: float) -> Cell:
    """Download (or cache-load) one history and score every strategy on it.

    Args:
        regime: The window to fetch.
        ticker: Yahoo Finance symbol.
        initial_cash: Starting capital.

    Returns:
        A filled Cell.

    Raises:
        DataSourceUnavailable: The price source could not be reached.
        ValueError: Unknown ticker, empty range, or a history the engine rejects.
    """
    prices = get_price_data(ticker, regime.start, regime.end)
    bench_return, bench_sharpe, bench_dd, scores = evaluate(
        prices, initial_cash, build_strategies(),
    )
    return Cell(
        regime=regime,
        ticker=ticker,
        n_bars=len(prices),
        period=period_label(prices, regime.start, regime.end),
        benchmark_return=bench_return,
        benchmark_sharpe=bench_sharpe,
        benchmark_drawdown=bench_dd,
        scores=scores,
    )


def _pct(value: float) -> str:
    """Format a ratio as a signed percentage, or n/a."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:+.2%}"


def _pp(value: float) -> str:
    """Format a return gap in percentage points, or n/a."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100:+.2f} pp"


def _ratio(value: float) -> str:
    """Format a dimensionless ratio (Sharpe), or n/a."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:+.3f}"


def _gap_cell(score: Optional[StrategyScore], field: str) -> str:
    """'Momentum +12.30 pp' / 'MA Crossover +0.450', or an em dash."""
    if score is None:
        return "—"
    value = getattr(score, field)
    rendered = _pp(value) if field == "return_gap" else _ratio(value)
    return f"{score.label} {rendered}"


def print_cell(cell: Cell) -> None:
    """Print the per-strategy detail for one regime × asset."""
    heading = f"{cell.regime.name}  ·  {cell.ticker}  ·  {cell.period}  ({cell.n_bars} bars)"
    print(heading)
    print("-" * len(heading))
    print(cell.regime.story)
    print(
        f"Buy & hold  return {_pct(cell.benchmark_return)}   "
        f"Sharpe {_ratio(cell.benchmark_sharpe).lstrip('+')}   "
        f"max DD {_pct(cell.benchmark_drawdown)}"
    )
    print()

    columns = (
        ("label", "", 14, None),
        ("total_return", "Return", 10, _pct),
        ("sharpe", "Sharpe", 8, lambda v: _ratio(v).lstrip("+") if np.isfinite(v) else "n/a"),
        ("max_drawdown", "Max DD", 10, _pct),
        ("return_gap", "vs B&H", 12, _pp),
        ("sharpe_gap", "Sharpe vs", 11, _ratio),
        ("shallower_drawdown", "Shallower DD", 13, None),
        ("exposure", "Exposure", 10, lambda v: f"{v:.1%}" if np.isfinite(v) else "n/a"),
        ("trades", "Trades", 7, lambda v: f"{v:.0f}" if np.isfinite(v) else "n/a"),
    )

    header = "".join(
        f"{heading:>{width}}" if name != "label" else f"{heading:<{width}}"
        for name, heading, width, _ in columns
    )
    print(header)
    print("-" * len(header))

    for score in cell.scores:
        cells = []
        for name, _, width, formatter in columns:
            raw = getattr(score, name)
            if name == "label":
                text = f"{raw:<{width}}"
            elif name == "shallower_drawdown":
                text = f"{('yes' if raw else 'no'):>{width}}"
            else:
                text = f"{formatter(raw):>{width}}"
            cells.append(text)
        print("".join(cells))
    print()


def print_summary(cells: List[Cell]) -> None:
    """Print the compact table the rest of the file exists to produce.

    One row per regime × asset, so a disagreement between SPY and AAPL stays
    visible. Columns are the four quantities the prompt asked for: the
    benchmark's return (to characterise the regime), the best return gap, the
    best Sharpe gap, and whether anyone had a shallower drawdown.

    Args:
        cells: Successfully measured cells, in the order they were run.
    """
    title = "SUMMARY  —  best of three strategies against buy-and-hold"
    print(title)
    print("=" * len(title))

    rows: List[Tuple[str, str, str, str, str]] = []
    label_width = 0
    for cell in cells:
        label = f"{cell.regime.name} · {cell.ticker}"
        label_width = max(label_width, len(label))
        best_return = best_by(cell.scores, "return_gap")
        best_sharpe = best_by(cell.scores, "sharpe_gap")
        n_protect = sum(1 for score in cell.scores if score.shallower_drawdown)
        n_all = len(cell.scores)
        protection = f"yes ({n_protect}/{n_all})" if n_protect else "no"
        rows.append((
            label,
            _pct(cell.benchmark_return),
            _gap_cell(best_return, "return_gap"),
            _gap_cell(best_sharpe, "sharpe_gap"),
            protection,
        ))

    label_width += 2
    headings = (
        ("Regime", label_width, "<"),
        ("B&H return", 12, ">"),
        ("Best return vs B&H", 28, ">"),
        ("Best Sharpe vs B&H", 26, ">"),
        ("Shallower DD", 14, ">"),
    )

    header = "".join(f"{name:{align}{width}}" for name, width, align in headings)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("".join(
            f"{value:{align}{width}}"
            for value, (_, width, align) in zip(row, headings)
        ))


def print_reading(cells: List[Cell]) -> None:
    """State, regime by regime, whether the advertised trade-off showed up.

    Written from the numbers, not from the hypothesis. A bear-market shallower
    drawdown is reported as protection even when the same strategy lost on
    return and Sharpe: that is the trade-off, and burying it would be the
    one-sided reading this script exists to prevent.

    Args:
        cells: Successfully measured cells.
    """
    print("\nHow to read this")
    print("A strategy 'protects capital' when its max drawdown is shallower "
          "than the asset's,\nnot when it makes money. In a bear that usually "
          "means it was in cash. Exposure is\nin the detail tables for that "
          "reason: a shallower drawdown at 5% exposure is the\nexit working, "
          "not a risk-model artefact.\n")

    by_regime: Dict[str, List[Cell]] = {}
    for cell in cells:
        by_regime.setdefault(cell.regime.name, []).append(cell)

    for regime in REGIMES:
        group = by_regime.get(regime.name, [])
        if not group:
            print(f"  {regime.name}: no data.")
            continue

        protected = [
            f"{cell.ticker} ({', '.join(s.label for s in cell.scores if s.shallower_drawdown)})"
            for cell in group
            if any(score.shallower_drawdown for score in cell.scores)
        ]
        beat_return = [
            f"{cell.ticker}/{score.label}"
            for cell in group
            for score in cell.scores
            if np.isfinite(score.return_gap) and score.return_gap > TIE_TOLERANCE
        ]
        beat_sharpe = [
            f"{cell.ticker}/{score.label}"
            for cell in group
            for score in cell.scores
            if np.isfinite(score.sharpe_gap) and score.sharpe_gap > TIE_TOLERANCE
        ]

        # In a quiet bull the asset barely draws down, so any cash holding
        # "wins" on depth. That is the artefact the exposure caveat exists for,
        # and it is not the protection the 2008 row is allowed to claim.
        quiet_bull = all(
            np.isfinite(cell.benchmark_drawdown)
            and cell.benchmark_drawdown > -0.10
            for cell in group
        )

        bits = []
        if quiet_bull:
            bits.append(
                "shallower drawdowns here are sitting out a 2–9% dip, not "
                "protection — the finding is the missed rally"
            )
        elif protected:
            bits.append("shallower drawdown on " + "; ".join(protected))
        else:
            bits.append("no shallower drawdown than the asset")
        if beat_return:
            bits.append("beat B&H on return: " + ", ".join(beat_return))
        else:
            bits.append("nobody beat B&H on return")
        if beat_sharpe:
            bits.append("beat B&H on Sharpe: " + ", ".join(beat_sharpe))
        else:
            bits.append("nobody beat B&H on Sharpe")
        print(f"  {regime.name}: {'; '.join(bits)}.")

    print()
    print(
        "The honest close is the trade-off, not a winner. If the 2008 row shows "
        "shallower\ndrawdowns and the bull rows show given-up return, the "
        "strategies did what they\nare advertised to do. Whether that nets to a "
        "durable edge after costs is a\ndifferent question, and the one "
        "run_cost_scenarios.py already answered for the\ndefault window: a thin "
        "Sharpe lead at zero cost did not survive a realistic\ncommission. "
        "Nothing here reopens that. It only asks whether the other side of\nthe "
        "trade — the protection — is visible when the market actually falls."
    )


def main(initial_cash: float = INITIAL_CASH) -> List[Cell]:
    """Run every regime × asset cell and print the study.

    A cell that cannot be loaded is reported and skipped so one missing history
    does not bury the rest. Strategies are rebuilt per cell (build_strategies
    is cheap and stateless) so a later strategy that grew memory would not leak
    across windows.

    Args:
        initial_cash: Starting capital, identical for every backtest.

    Returns:
        The cells that produced numbers, in run order.
    """
    title = "REGIME STUDY  —  same strategies, four economic stories"
    print(title)
    print("=" * len(title))
    print(
        "Frictionless, matching run_comparison.py. Buy-and-hold is uncharged. "
        "SPY is the\nmarket the hypothesis is about; AAPL is the single-name "
        "check.\n"
    )
    for label, strategy in build_strategies():
        print(f"  {label:<14} {strategy!r}")
    print()

    cells: List[Cell] = []
    for regime in REGIMES:
        for ticker in ASSETS:
            try:
                cell = load_cell(regime, ticker, initial_cash)
            except (DataSourceUnavailable, ValueError) as error:
                print(f"{regime.name}  ·  {ticker}: skipped ({error})\n")
                continue
            cells.append(cell)
            print_cell(cell)

    if not cells:
        print("No cell produced numbers. Check the connection and the tickers.")
        return cells

    print_summary(cells)
    print_reading(cells)
    return cells


if __name__ == "__main__":
    main()
