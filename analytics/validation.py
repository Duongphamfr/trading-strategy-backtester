"""Robustness validation: does a result survive a change of parameters?

WHAT THIS MODULE IS FOR
Every other module in analytics measures one backtest. This one re-runs the
backtest many times to ask a different question: whether the figure the others
reported was a property of the strategy or an accident of the two numbers it was
handed. That is validation rather than measurement, and it needs its own home
because it works by orchestration, not by arithmetic.

A rule that captures something real about markets should work over a broad
neighbourhood of settings, because nothing about market dynamics knows the
difference between a 48-day and a 52-day average. That shows up as a plateau: a
wide region of adjacent cells that all score well. A rule that has merely been
fitted to a sample shows up as an isolated spike, where one cell scores well and
its immediate neighbours do not, which means the result rests on an arbitrary
choice rather than on a mechanism.

WHY IT LIVES IN analytics AND NOT AT THE PROJECT ROOT
The project layout reserves analytics/validation.py for walk-forward and
in-sample versus out-of-sample work, and a parameter sweep is the same kind of
question asked a different way: both are about whether a number generalises.
Filling that slot also gives the walk-forward primitives an obvious place to move
to later, since they already depend on the sweep implemented here.

One consequence is worth stating plainly rather than discovering later. This
module imports engine and strategies, which metrics.py, risk.py and
trade_stats.py deliberately do not: they take series and logs and know nothing
about how those were produced. That asymmetry is intentional. Validation cannot
be done on a finished equity curve, because the whole point is to generate
curves that were never run. The pure measurement modules stay pure; this is the
one place in the package that drives a simulation.

HOW ROBUSTNESS IS QUANTIFIED
Four readings, because no single number captures it, all carried on the
Robustness summary that assess returns:

    The gap between the best cell and the mean of its immediate neighbours. A
    small gap means the peak sits on a plateau; a large one means it is a spike.

    The share of valid cells scoring within PLATEAU_TOLERANCE of the best, which
    measures how wide the good region is.

    Contiguity of the top decile: what fraction of the best cells touch another
    of the best cells. Scattered good cells are noise, adjacent ones are signal.

    The share of the whole grid that beats uncharged buy-and-hold. This is the
    honest denominator. If the peak beats the benchmark but the median cell does
    not, then beating the benchmark was a property of the parameter search, not
    of the strategy, and a researcher choosing windows in advance would most
    likely have lost.

NO PLOTTING HERE
The functions return plain pandas and plain tuples. Callers draw them however
they like: run_parameter_sweep.py renders static matplotlib images for the
write-up, and visualization.charts renders an interactive Plotly heatmap for the
dashboard. Keeping this module free of any plotting dependency is what allows the
dashboard to run a sweep without importing matplotlib at all.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analytics.report import (
    BENCHMARK_COLUMN,
    STRATEGY_COLUMN,
    performance_report,
)
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageCrossover

FAST_WINDOWS: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
SLOW_WINDOWS: Tuple[int, ...] = (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300)

# True throughout, matching run_comparison so every script describes the same
# strategy. The setting applies uniformly to every cell, so it shifts the whole
# surface rather than distorting its shape, and the robustness read is unaffected.
ENTER_ON_EXISTING_TREND = True

# Cells within this fraction of the best count as part of the good region, and
# the top decile is the set considered for contiguity. Both are conventions, and
# both are arguments rather than constants buried in the logic.
PLATEAU_TOLERANCE = 0.10
TOP_QUANTILE = 0.90


class Robustness(NamedTuple):
    """Quantified answer to "is the peak on a plateau or alone?".

    Attributes:
        best_fast: Fast window of the highest scoring cell.
        best_slow: Slow window of the highest scoring cell.
        best_score: Score of that cell.
        neighbour_mean: Mean score of its immediate neighbours, up to eight.
        neighbour_count: How many neighbours existed and were valid.
        plateau_share: Fraction of valid cells within PLATEAU_TOLERANCE of best.
        contiguity: Fraction of top-decile cells touching another top-decile
            cell. NaN when the top decile holds a single cell.
        beats_reference: Fraction of valid cells scoring above the reference.
        median_score: Median score across valid cells.
        reference: The comparison score, normally the benchmark's.
    """

    best_fast: int
    best_slow: int
    best_score: float
    neighbour_mean: float
    neighbour_count: int
    plateau_share: float
    contiguity: float
    beats_reference: float
    median_score: float
    reference: float


def sweep(
    prices: pd.DataFrame,
    initial_cash: float,
    fast_windows: Sequence[int] = FAST_WINDOWS,
    slow_windows: Sequence[int] = SLOW_WINDOWS,
    commission: float = 0.0,
    spread: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """Backtest every valid window combination and keep all of its metrics.

    The full performance report is retained for each cell rather than only the
    Sharpe ratio. That costs a little time and almost no memory, and it means
    asking the same question about drawdown, exposure or profit factor later is
    a change of one string rather than a re-run of the sweep.

    Combinations where fast >= slow are skipped rather than caught as errors,
    since the strategy rejects them by design: two averages that never cross
    cannot produce a crossover signal.

    Args:
        prices: OHLCV history to run over.
        initial_cash: Starting capital for every cell.
        fast_windows: Fast window values to try.
        slow_windows: Slow window values to try.
        commission: Proportional commission per trade, applied to every cell.
        spread: Bid-ask spread as a fraction of price.
        slippage: Adverse price move as a fraction of price.

    Returns:
        A DataFrame whose rows are metric labels and whose columns are a
        MultiIndex of (fast, slow) pairs, holding the strategy's figures. The
        benchmark's column is appended once under ("benchmark", "benchmark"),
        since it is identical for every cell and is the reference the grid is
        judged against.

    Raises:
        ValueError: If no combination in the grid was valid.
    """
    columns: Dict[Tuple, pd.Series] = {}
    benchmark: Optional[pd.Series] = None

    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue

            strategy = MovingAverageCrossover(
                fast_window=fast,
                slow_window=slow,
                enter_on_existing_trend=ENTER_ON_EXISTING_TREND,
            )
            result = Backtester(
                prices,
                initial_cash=initial_cash,
                strategy=strategy,
                commission=commission,
                spread=spread,
                slippage=slippage,
            ).run()

            report = performance_report(
                equity=result.equity_curve["total_value"],
                benchmark=result.benchmark_curve,
                trade_log=result.trade_log,
                positions=result.equity_curve["shares"],
            )

            columns[(fast, slow)] = report[STRATEGY_COLUMN]
            if benchmark is None:
                benchmark = report[BENCHMARK_COLUMN]

    if not columns:
        raise ValueError(
            "No valid window combination in the grid: every pair had "
            "fast >= slow, so nothing could be tested."
        )

    columns[("benchmark", "benchmark")] = benchmark

    results = pd.DataFrame(columns)
    results.columns = pd.MultiIndex.from_tuples(
        results.columns,
        names=["fast", "slow"],
    )
    return results


def metric_grid(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot one metric out of a sweep into a fast by slow grid.

    Args:
        results: A sweep result as returned by sweep.
        metric: Row label to extract, for example analytics.report.SHARPE.

    Returns:
        A DataFrame indexed by fast window with slow windows as columns. Cells
        for skipped combinations are NaN, which is what marks them invalid for
        both the statistics and the plot.

    Raises:
        KeyError: If the metric is not present in the sweep.
    """
    if metric not in results.index:
        raise KeyError(
            f"'{metric}' is not one of the swept metrics. Available: "
            f"{', '.join(str(label) for label in results.index)}."
        )

    row = results.loc[metric].drop(index="benchmark", level="fast", errors="ignore")
    grid = row.unstack("slow")
    grid.index.name = "fast"
    grid.columns.name = "slow"
    return grid.sort_index().sort_index(axis=1)


def reference_value(results: pd.DataFrame, metric: str) -> float:
    """Read the benchmark's value of one metric from a sweep.

    Args:
        results: A sweep result as returned by sweep.
        metric: Row label to extract.

    Returns:
        The benchmark's figure, or NaN if the sweep carried no benchmark.
    """
    try:
        return float(results.loc[metric, ("benchmark", "benchmark")])
    except KeyError:
        return float("nan")


def assess(grid: pd.DataFrame, reference: float) -> Robustness:
    """Measure whether the grid's high scores are clustered or isolated.

    Args:
        grid: A metric grid as returned by metric_grid, higher being better.
        reference: Score to compare the grid against, normally the benchmark's.

    Returns:
        A Robustness summary.

    Raises:
        ValueError: If the grid holds no valid cell.
    """
    values = grid.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError("The grid holds no valid cell to assess.")

    flat_best = int(np.nanargmax(values))
    row, column = np.unravel_index(flat_best, values.shape)
    best_score = float(values[row, column])

    # The mean of the surrounding cells is the height of the ground the peak
    # stands on.
    neighbours = cell_neighbours(grid, int(grid.index[row]),
                                 int(grid.columns[column]))

    valid = values[np.isfinite(values)]

    # Proportional tolerance for positive scores, and absolute for the rest: a
    # Sharpe near zero would make a percentage band meaninglessly narrow.
    if best_score > 0:
        threshold = best_score * (1.0 - PLATEAU_TOLERANCE)
    else:
        threshold = best_score - abs(best_score) * PLATEAU_TOLERANCE - 1e-12

    top_cutoff = float(np.quantile(valid, TOP_QUANTILE))
    is_top = np.isfinite(values) & (values >= top_cutoff)
    contiguity = _contiguity(is_top)

    return Robustness(
        best_fast=int(grid.index[row]),
        best_slow=int(grid.columns[column]),
        best_score=best_score,
        neighbour_mean=float(np.mean(neighbours)) if neighbours else float("nan"),
        neighbour_count=len(neighbours),
        plateau_share=float(np.mean(valid >= threshold)),
        contiguity=contiguity,
        beats_reference=float(np.mean(valid > reference)) if np.isfinite(reference)
        else float("nan"),
        median_score=float(np.median(valid)),
        reference=reference,
    )


def cell_neighbours(grid: pd.DataFrame, fast: int, slow: int) -> List[float]:
    """Scores of the cells immediately surrounding one cell of the grid.

    Up to eight, diagonals included, minus those falling off the edge of the
    grid and those that were never valid. Their mean is the height of the ground
    a cell stands on, which is the whole question a sweep is run to answer: a
    score that collapses the moment either window is nudged was a coincidence,
    not a setting.

    Separate from assess so the same neighbourhood can be measured around any
    cell rather than only the best one. The dashboard needs it for whichever
    combination the user has selected, which is rarely the best.

    Args:
        grid: A metric grid as returned by metric_grid.
        fast: Fast window, an index label of the grid.
        slow: Slow window, a column label of the grid.

    Returns:
        The valid neighbouring scores, in no meaningful order. Empty when the
        cell sits alone among invalid combinations.

    Raises:
        KeyError: If the cell is not on the grid at all.
    """
    row = grid.index.get_loc(fast)
    column = grid.columns.get_loc(slow)
    values = grid.to_numpy(dtype=float)

    neighbours: List[float] = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            neighbour_row = row + row_offset
            neighbour_column = column + column_offset
            if not (0 <= neighbour_row < values.shape[0]):
                continue
            if not (0 <= neighbour_column < values.shape[1]):
                continue
            value = values[neighbour_row, neighbour_column]
            if np.isfinite(value):
                neighbours.append(float(value))

    return neighbours


def _contiguity(mask: np.ndarray) -> float:
    """Fraction of flagged cells that touch at least one other flagged cell.

    Adjacency includes diagonals, matching the neighbourhood used for the peak.
    A value near 1 means the good cells form a connected region; near 0 means
    they are scattered, which is what parameter noise looks like.

    Args:
        mask: Boolean array flagging the cells of interest.

    Returns:
        The fraction with at least one flagged neighbour, or NaN if fewer than
        two cells are flagged and the question is therefore meaningless.
    """
    flagged = int(mask.sum())
    if flagged < 2:
        return float("nan")

    padded = np.pad(mask, 1, constant_values=False)
    neighbour_count = np.zeros_like(mask, dtype=int)
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            neighbour_count += padded[
                1 + row_offset: 1 + row_offset + mask.shape[0],
                1 + column_offset: 1 + column_offset + mask.shape[1],
            ].astype(int)

    return float((neighbour_count[mask] > 0).mean())


if __name__ == "__main__":
    from analytics.report import SHARPE
    from data.market_data import get_price_data

    print("Sweeping a small grid to exercise the primitives without plotting.")
    prices = get_price_data("AAPL", "2020-01-01", "2023-01-01")

    results = sweep(prices, 10_000.0, fast_windows=(20, 50, 80),
                    slow_windows=(100, 200, 300))
    grid = metric_grid(results, SHARPE)
    reference = reference_value(results, SHARPE)

    print()
    print(f"Grid of {grid.notna().to_numpy().sum()} valid cells, benchmark "
          f"Sharpe {reference:.4f}")
    print(grid.round(3).to_string())

    summary = assess(grid, reference)
    print()
    print(f"Best cell        fast={summary.best_fast}, slow={summary.best_slow} "
          f"-> {summary.best_score:.4f}")
    print(f"Its neighbours   {summary.neighbour_count}, mean "
          f"{summary.neighbour_mean:.4f}")
    print(f"Median cell      {summary.median_score:.4f}")
    print(f"Beat benchmark   {summary.beats_reference:.1%} of valid cells")

    # assess describes only the best cell, so cell_neighbours has to agree with
    # it there. Anywhere else it is the only way to ask the question at all.
    around_best = cell_neighbours(grid, summary.best_fast, summary.best_slow)
    assert len(around_best) == summary.neighbour_count
    assert abs(float(np.mean(around_best)) - summary.neighbour_mean) < 1e-12
    print()
    print("cell_neighbours reproduces assess around the best cell, and works "
          "around any other.")

    for fast, slow in ((20, 100), (80, 300)):
        neighbours = cell_neighbours(grid, fast, slow)
        print(f"  {fast}/{slow}: own {grid.loc[fast, slow]:+.4f}, "
              f"{len(neighbours)} neighbours averaging "
              f"{float(np.mean(neighbours)):+.4f}")

    print()
    print("This module imported no plotting library. Verify with:")
    print("  python -c \"import sys, analytics.validation; "
          "print('matplotlib' in sys.modules)\"")
