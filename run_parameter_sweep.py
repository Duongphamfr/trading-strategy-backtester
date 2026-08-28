"""Sweep the moving average grid and report whether good parameters are robust.

WHAT LIVES HERE AND WHAT DOES NOT
The primitives, sweep, metric_grid, reference_value, assess and cell_neighbours,
live in analytics.validation. This file is the command-line front end for them:
it chooses the grid to run, renders static heatmaps with matplotlib, and prints
the interpretation.

The split exists because the dashboard needs the same primitives and should not
have to import a plotting library to get them. Anything reusable belongs in
analytics.validation; anything that draws or prints belongs here.

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

HOW ROBUSTNESS IS QUANTIFIED
Four readings, all computed by analytics.validation.assess and documented on the
Robustness summary it returns: the gap between the best cell and its neighbours,
the width of the good region, the contiguity of the top decile, and the share of
the whole grid that beats uncharged buy-and-hold. That last one is the honest
denominator. If the peak beats the benchmark but the median cell does not, then
beating the benchmark was a property of the parameter search, not of the
strategy, and a researcher choosing windows in advance would most likely have
lost.

plot_heatmap knows nothing about backtesting, taking only a DataFrame and
labels, so it can be reused for any grid in the Phase 6 figures.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

# Chosen before pyplot is imported: the script writes files and never opens a
# window, and Agg is the backend that works without a display server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from analytics.report import NUMBER_OF_TRADES, SHARPE
from analytics.validation import (
    FAST_WINDOWS,
    PLATEAU_TOLERANCE,
    SLOW_WINDOWS,
    Robustness,
    assess,
    metric_grid,
    reference_value,
    sweep,
)
from data.market_data import covered_range, get_price_data, period_label

TICKER = "AAPL"
START = "2020-01-01"
END = "2023-01-01"
INITIAL_CASH = 10_000.0

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


def heatmap_path(ticker: str, first: str, last: str, commission: float) -> Path:
    """Where the heatmap for one ticker, period and cost scenario is written.

    WHY THE TICKER AND THE PERIOD ARE IN THE NAME
    They were not, and the omission cost a figure. The file used to be named
    after the cost scenario alone, so sweeping any second ticker overwrote the
    tracked image of the first under a name that still implied the first. The
    figure inside always carried the right title, which is exactly what made the
    substitution hard to notice: the file looked current and was simply about a
    different asset than its filename suggested.

    A saved figure is an output keyed on its inputs, the same way a cache entry
    is, and naming it after only one of three inputs guarantees a collision as
    soon as the others vary. Since the sweep is meant to be run across assets,
    that is not a corner case.

    WHY THE DATES ARE THE COVERED ONES AND NOT THE REQUESTED ONES
    A filename outlives the run that produced it, which makes a wrong one worse
    than a wrong console line: it can be cited from a write-up months later. The
    requested range is not what the image shows. Asking for AAPL to 2023-01-01
    yields bars ending 2022-12-30, so the old name said 2020-2023 of a figure
    covering 2020-2022; on a recently listed ticker the overstatement runs to
    years. Keying on the covered range also makes the name honest about identity:
    two different requests that resolve to the same bars produce the same figure
    and now share the one file, correctly.

    The ticker is sanitised because Yahoo notation admits characters a filesystem
    reads as structure: "BRK/B" would name a directory that does not exist. Only
    legibility is at stake here, unlike in the data cache where a collision would
    serve one asset's prices for another, so a substitution is enough and no hash
    is needed.

    Args:
        ticker: Yahoo Finance symbol the sweep ran on.
        first: First date actually present in the data, as YYYY-MM-DD. Pass what
            the bars say, not what was requested.
        last: Last date actually present in the data, as YYYY-MM-DD.
        commission: Proportional commission the grid was swept under.

    Returns:
        The path to write to, inside OUTPUT_DIR.
    """
    scenario = "free" if commission == 0.0 else f"commission_{commission:.4f}"
    symbol = re.sub(r"[^A-Za-z0-9._-]", "-", ticker)
    return OUTPUT_DIR / (
        f"ma_sharpe_heatmap_{symbol}_{first[:4]}-{last[:4]}_{scenario}.png"
    )


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

    # Every label below, the printed title and the one drawn into each figure,
    # comes from the bars rather than the request, so the filename, the image and
    # the console agree on what period was studied.
    first, last = covered_range(prices)
    covered = (str(first.date()), str(last.date()))

    title = (f"PARAMETER SWEEP  {ticker}  {period_label(prices, start, end)}  "
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

        path = plot_heatmap(
            grid,
            heatmap_path(ticker, *covered, commission),
            title=(f"MA crossover Sharpe ratio, {ticker} "
                   f"{first.year}-{last.year}\n"
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
