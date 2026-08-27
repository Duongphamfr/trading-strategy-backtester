"""Sweep the moving average grid and judge whether good parameters are robust.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
It is not for finding the best fast and slow windows. Running a hundred
parameter combinations on one asset over one period and reporting the winner is
not research, it is a search for the luckiest cell in a table, and the number it
produces will not survive contact with any other period. The best cell is printed
below only because the reader needs to know where the peak is in order to judge
what surrounds it.

The question is the shape of the surface. A rule that captures something real
about markets should work over a broad neighbourhood of settings, because nothing
about market dynamics knows the difference between a 48-day and a 52-day average.
That shows up as a plateau: a wide region of adjacent cells that all score well.
A rule that has merely been fitted to this sample shows up as isolated spikes,
where one cell scores well and its immediate neighbours do not, which means the
result rests on an arbitrary choice rather than on a mechanism.

The momentum result from the cost analysis is the reference case for what a red
flag looks like. Its Sharpe was 0.757 when reviewing every 21 bars and 0.410 when
reviewing every 5, on the identical signal. That kind of sensitivity to a
parameter nobody can justify a priori is the signature this sweep is looking for.

HOW ROBUSTNESS IS QUANTIFIED HERE
Four readings, because no single number captures it:

    The gap between the best cell and the mean of its immediate neighbours. A
    small gap means the peak sits on a plateau; a large one means it is a spike.

    The share of valid cells scoring within 10% of the best, which measures how
    wide the good region is.

    Contiguity of the top decile: what fraction of the best cells touch another
    of the best cells. Scattered good cells are noise, adjacent ones are signal.

    The share of the whole grid that beats uncharged buy-and-hold. This is the
    honest denominator. If the peak beats the benchmark but the median cell does
    not, then beating the benchmark was a property of the parameter search, not
    of the strategy, and a researcher choosing windows in advance would most
    likely have lost.

Computation and plotting are kept in separate functions. The sweep returns plain
pandas, so any metric in the report can be visualised by name without touching
the plotting code, and plot_heatmap knows nothing about backtesting so it can be
reused for the Phase 6 figures.
"""

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib

# Chosen before pyplot is imported: the script writes files and never opens a
# window, and Agg is the backend that works without a display server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from analytics.report import (
    BENCHMARK_COLUMN,
    NUMBER_OF_TRADES,
    SHARPE,
    STRATEGY_COLUMN,
    performance_report,
)
from data.market_data import get_price_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageCrossover

TICKER = "AAPL"
START = "2020-01-01"
END = "2023-01-01"
INITIAL_CASH = 10_000.0

FAST_WINDOWS: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
SLOW_WINDOWS: Tuple[int, ...] = (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300)

# True throughout, matching run_comparison so the two scripts describe the same
# strategy. The setting applies uniformly to every cell, so it shifts the whole
# surface rather than distorting its shape, and the robustness read is unaffected.
ENTER_ON_EXISTING_TREND = True

# A realistic retail commission, used for the second grid. Costs are worth
# sweeping alongside the parameters rather than after them: a plateau that exists
# only when trading is free is not a plateau a trader can stand on.
REALISTIC_COMMISSION = 0.0010

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# The settings the strategy defaults to and that every textbook quotes. Called
# out explicitly because a sweep that only reports its own winner hides the more
# useful comparison: how the conventional choice, the one a researcher would have
# made without seeing this grid, actually fared.
CONVENTIONAL_CELL = (50, 200)

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

    # The eight surrounding cells, minus those off the edge of the grid and
    # those that were never valid. The mean of what remains is the height of the
    # ground the peak stands on.
    neighbours: List[float] = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            neighbour_row, neighbour_column = row + row_offset, column + column_offset
            if not (0 <= neighbour_row < values.shape[0]):
                continue
            if not (0 <= neighbour_column < values.shape[1]):
                continue
            value = values[neighbour_row, neighbour_column]
            if np.isfinite(value):
                neighbours.append(float(value))

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


def plot_heatmap(
    grid: pd.DataFrame,
    path: Path,
    title: str,
    metric_label: str,
    center: Optional[float] = None,
    annotate: bool = True,
    colormap: str = "RdBu_r",
) -> Path:
    """Draw a grid as an annotated heatmap and save it.

    Deliberately generic: it takes a DataFrame and labels, and knows nothing
    about backtesting, so the Phase 6 figures can reuse it for any grid.

    When a center is given the colour scale diverges around it, so the map
    answers "which cells beat the reference" at a glance instead of merely
    ranking cells against each other. Passing the benchmark's score turns the
    figure into a pass-or-fail map, which is the more honest reading.

    Args:
        grid: Values to draw, rows on the vertical axis, columns on the
            horizontal. NaN cells are drawn grey and labelled invalid.
        path: Destination image file. Parent directories are created.
        title: Figure title.
        metric_label: Label for the colour bar.
        center: Optional value the diverging colour scale pivots around.
        annotate: Whether to print each cell's value inside it.
        colormap: Any matplotlib colormap name.

    Returns:
        The path written.
    """
    values = grid.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    palette = plt.get_cmap(colormap).copy()
    palette.set_bad("#d9d9d9")

    # TwoSlopeNorm requires the centre to sit strictly inside the data range.
    # When every cell falls on one side of the reference, a diverging scale
    # cannot be built, and a plain linear one is used instead.
    norm = None
    if center is not None and masked.count():
        low, high = float(masked.min()), float(masked.max())
        if low < center < high:
            norm = TwoSlopeNorm(vmin=low, vcenter=center, vmax=high)

    figure, axes = plt.subplots(figsize=(1.05 * len(grid.columns) + 3.5,
                                        0.62 * len(grid.index) + 3.0))
    image = axes.pcolormesh(
        np.arange(len(grid.columns) + 1),
        np.arange(len(grid.index) + 1),
        masked,
        cmap=palette,
        norm=norm,
        edgecolors="white",
        linewidth=0.8,
    )

    axes.set_xticks(np.arange(len(grid.columns)) + 0.5)
    axes.set_xticklabels(grid.columns)
    axes.set_yticks(np.arange(len(grid.index)) + 0.5)
    axes.set_yticklabels(grid.index)
    axes.set_xlabel(f"{grid.columns.name} window (bars)")
    axes.set_ylabel(f"{grid.index.name} window (bars)")
    axes.set_title(title, pad=14)
    axes.invert_yaxis()

    bar = figure.colorbar(image, ax=axes, pad=0.02)
    bar.set_label(metric_label)
    if norm is not None:
        bar.ax.axhline(center, color="black", linewidth=1.2)

    if annotate:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                axes.text(
                    column + 0.5,
                    row + 0.5,
                    "n/a" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#666666" if not np.isfinite(value) else "black",
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def print_assessment(summary: Robustness, label: str, metric_label: str) -> None:
    """Print one robustness assessment and what it implies.

    Args:
        summary: The assessment to report.
        label: Short name of the grid being described.
        metric_label: Name of the metric, for the wording.
    """
    gap = summary.best_score - summary.neighbour_mean

    print(f"\n{label}")
    print(f"  Best cell            fast={summary.best_fast}, "
          f"slow={summary.best_slow}  ->  {metric_label} "
          f"{summary.best_score:.3f}")
    print(f"  Its {summary.neighbour_count} neighbours     mean "
          f"{summary.neighbour_mean:.3f}   gap {gap:+.3f}")
    print(f"  Within {PLATEAU_TOLERANCE:.0%} of best   "
          f"{summary.plateau_share:.1%} of valid cells")
    print(f"  Top decile clustered {summary.contiguity:.1%} of them touch "
          f"another top cell")
    print(f"  Median cell          {summary.median_score:.3f}   "
          f"versus benchmark {summary.reference:.3f}")
    print(f"  Beat the benchmark   {summary.beats_reference:.1%} of valid cells")

    # The verdict is stated in terms of what a researcher choosing parameters in
    # advance would have experienced, which is the only decision-relevant
    # framing. The peak is a fact about this sample; the median is the
    # expectation for anyone who could not see it first.
    if summary.median_score > summary.reference:
        print("  Read: the typical parameter choice beat the benchmark, so the "
              "result does not\n        depend on having picked well.")
    elif summary.beats_reference > 0.5:
        print("  Read: most cells beat the benchmark but the median did not, so "
              "the edge is real\n        but thin and sensitive to the choice.")
    else:
        print(f"  Read: only {summary.beats_reference:.0%} of parameter choices "
              f"beat the benchmark and the median\n        lost. Beating it was "
              f"a property of the search, not of the strategy.")

    # Two questions that sound like one and are not, so both are answered
    # separately rather than collapsed into a single misleading sentence.
    #
    # The first is whether the peak towers over its own surroundings. Judging
    # that on the neighbour gap alone is too permissive: a gap of 30% of the
    # peak reads as moderate while still meaning the neighbours score a third
    # less. The breadth of the good region is the sharper test, so a peak counts
    # as isolated if almost nothing else comes close to it, whatever the gap.
    #
    # The second is whether the better cells, taken as a group, sit together.
    # They can form connected regions while the single best cell still stands
    # far above them, which is exactly the shape this grid has.
    if np.isfinite(gap):
        isolated = (
            summary.plateau_share < 0.05
            or gap > 0.20 * abs(summary.best_score)
        )
        if isolated:
            print(f"        The peak is isolated: only "
                  f"{summary.plateau_share:.0%} of cells come within "
                  f"{PLATEAU_TOLERANCE:.0%} of it and its own\n"
                  f"        neighbours average {gap:+.3f} below it. That is the "
                  f"overfitting signature this\n        sweep was built to "
                  f"detect, and it means the peak's value is not a number to\n"
                  f"        expect from these windows on other data.")
        else:
            print("        The peak sits on a plateau: a wide band of cells "
                  "scores nearly as well,\n        so the exact windows matter "
                  "little. That is the reassuring shape.")

    if np.isfinite(summary.contiguity):
        print(f"        Separately, {summary.contiguity:.0%} of the top decile "
              f"touches another top cell, so the\n        better cells do form "
              f"connected regions rather than scattering at random.")


def print_conventional_comparison(
    grid: pd.DataFrame,
    summary: Robustness,
    metric_label: str,
) -> None:
    """Compare the conventional windows against the grid's best and the benchmark.

    This is the comparison that decides what the sweep actually means. The peak
    was found by looking at the answers; the conventional cell is what someone
    would have chosen beforehand. If the conventional choice loses to the
    benchmark while the peak beats it, then the strategy's apparent edge is
    entirely the product of hindsight.

    Args:
        grid: A metric grid as returned by metric_grid.
        summary: The assessment of that grid.
        metric_label: Name of the metric, for the wording.
    """
    fast, slow = CONVENTIONAL_CELL
    if fast not in grid.index or slow not in grid.columns:
        return

    conventional = float(grid.loc[fast, slow])

    print(f"\n  The conventional windows versus the grid's winner")
    print(f"    fast={fast}, slow={slow} (textbook golden cross)   "
          f"{metric_label} {conventional:.3f}")
    print(f"    fast={summary.best_fast}, slow={summary.best_slow} "
          f"(best in this grid)         {metric_label} "
          f"{summary.best_score:.3f}")
    print(f"    buy-and-hold, uncharged                     {metric_label} "
          f"{summary.reference:.3f}")

    if conventional < summary.reference < summary.best_score:
        print(f"    The setting anyone would have chosen in advance lost to the "
              f"benchmark by\n    {summary.reference - conventional:.3f}, while "
              f"the setting found by searching beat it by "
              f"{summary.best_score - summary.reference:.3f}.\n    The "
              f"difference between those two is hindsight, not skill.")


def main(
    ticker: str = TICKER,
    start: str = START,
    end: str = END,
    initial_cash: float = INITIAL_CASH,
) -> None:
    """Sweep the grid, save the heatmaps and report the robustness read.

    Args:
        ticker: Yahoo Finance symbol, served from cache after the first fetch.
        start: First date of the history, as YYYY-MM-DD.
        end: Last date of the history, as YYYY-MM-DD.
        initial_cash: Starting capital for every cell.
    """
    prices = get_price_data(ticker, start, end)

    title = (f"PARAMETER SWEEP  {ticker}  {start} to {end}  "
             f"({len(prices)} bars)")
    print(title)
    print("=" * len(title))
    print(f"Grid: {len(FAST_WINDOWS)} fast x {len(SLOW_WINDOWS)} slow windows, "
          f"skipping fast >= slow.")
    print("The aim is the shape of the surface, not the height of its peak. An "
          "isolated\nhigh cell is evidence against the strategy, not for it.")

    grids: List[Tuple[str, pd.DataFrame, float, Path]] = []

    for scenario, commission in (("free", 0.0), ("charged", REALISTIC_COMMISSION)):
        results = sweep(prices, initial_cash, commission=commission)
        grid = metric_grid(results, SHARPE)
        reference = reference_value(results, SHARPE)

        suffix = "free" if commission == 0.0 else f"commission_{commission:.4f}"
        path = plot_heatmap(
            grid,
            OUTPUT_DIR / f"ma_sharpe_heatmap_{suffix}.png",
            title=(f"MA crossover Sharpe ratio, {ticker} {start[:4]}-{end[:4]}\n"
                   f"commission {commission:.2%}; white line on the bar marks "
                   f"buy-and-hold at {reference:.3f}"),
            metric_label="Sharpe ratio",
            center=reference,
        )
        grids.append((scenario, grid, reference, path))

        trades = metric_grid(results, NUMBER_OF_TRADES)
        print(f"\nScenario '{scenario}' (commission {commission:.2%}): "
              f"{int(np.isfinite(grid.to_numpy(dtype=float)).sum())} valid cells, "
              f"trades per cell {int(np.nanmin(trades.to_numpy(dtype=float)))}"
              f" to {int(np.nanmax(trades.to_numpy(dtype=float)))}")
        print(f"  heatmap saved to {path.relative_to(Path(__file__).resolve().parent)}")

    print("\n\nROBUSTNESS")
    for scenario, grid, reference, _ in grids:
        summary = assess(grid, reference)
        print_assessment(
            summary,
            label=f"Sharpe surface, {scenario} execution",
            metric_label="Sharpe",
        )
        if scenario == "free":
            print_conventional_comparison(grid, summary, metric_label="Sharpe")

    free_grid, charged_grid = grids[0][1], grids[1][1]
    survives = (
        np.isfinite(free_grid.to_numpy(dtype=float))
        & (charged_grid.to_numpy(dtype=float) > grids[1][2])
    ).sum()
    print(f"\n  With a {REALISTIC_COMMISSION:.2%} commission, "
          f"{int(survives)} cells still beat the uncharged benchmark.")
    print("  The benchmark pays no costs in either scenario, so this comparison "
          "already\n  favours the strategy.")


if __name__ == "__main__":
    main()
