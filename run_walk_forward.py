"""Walk-forward validation: optimise on the past, then face the unseen future.

The heatmap showed that optimising the moving average windows over the whole
history lands on an isolated peak, and that the conventional 50/200 setting lost
to buy-and-hold on the same data. Both are statements about one fixed sample. The
question they leave open is the only one that matters to someone actually
trading: if you honestly pick the best parameters using only what you could have
known at the time, do they work afterwards?

Walk-forward answers it. Optimise on an in-sample window, apply the winner to the
immediately following out-of-sample window, roll forward, repeat. What comes out
is a sequence of decisions made without hindsight, and the gap between in-sample
and out-of-sample performance is a direct measurement of how much of the
in-sample result was fitting rather than signal.

HOW THE BOUNDARY IS PROTECTED, STRUCTURALLY AND NOT BY DISCIPLINE
Parameter selection happens in `select`, which receives a slice of the price
frame covering the in-sample window and nothing else. The out-of-sample bars are
not merely left unused, they are absent from the call: there is no argument
through which they could reach the selection, so no future leak is possible even
by mistake. That is a stronger guarantee than a rule a maintainer has to remember.

WHY THE OUT-OF-SAMPLE RUN STILL STARTS AT THE IN-SAMPLE DATE
A moving average needs its full window before it exists. Backtesting the chosen
combo on the out-of-sample bars alone would spend most of that window in warm-up,
emitting no signals, and would measure nothing. So the run starts at the
in-sample start date and the strategy is simply followed through into the
out-of-sample period, exactly as a trader would; only the out-of-sample slice of
the resulting equity curve is scored.

This is not a look-ahead violation. All the extra data is *older* than the
out-of-sample window, and it is used only to compute indicator values and to
carry a position forward, never to choose parameters. It also means the portfolio
enters each out-of-sample window holding whatever the in-sample period left it
holding, which is the realistic continuation rather than a convenient reset.

A KNOWN BIAS IN THE SELECTION, WORTH STATING
Each candidate is scored on the whole in-sample window, warm-up included, and a
300-bar slow average sits flat for longer than a 50-bar one before it can trade.
Long windows are therefore mildly penalised during selection. That bias is not a
defect in the measurement: a trader optimising on a fixed window faces exactly
the same arithmetic. It is part of the procedure being tested, and it is applied
identically in every roll, so the in-sample to out-of-sample comparison remains
sound. The slow grid is nonetheless capped below the sweep's 300 to keep the
warm-up a minority of the in-sample window.

OUT-OF-SAMPLE WINDOWS DO NOT OVERLAP
The roll step equals the out-of-sample length, so each bar is scored out-of-sample
at most once. Overlapping windows would recycle the same days across several rolls
and make a handful of observations look like many, which would understate the
uncertainty in every average reported below.
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analytics.report import (
    BENCHMARK_COLUMN,
    EXPOSURE,
    SHARPE,
    STRATEGY_COLUMN,
    TOTAL_RETURN,
    performance_report,
)
from data.market_data import get_price_data, period_label
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageCrossover

# The grid search and its helpers are reused wholesale from the sweep script, so
# the parameters are chosen here by exactly the code that drew the heatmap. If
# that search changes, this validation follows it instead of quietly testing a
# different procedure.
from analytics.validation import ENTER_ON_EXISTING_TREND, metric_grid, sweep

TICKER = "AAPL"
START = "2015-01-01"
END = "2023-01-01"
INITIAL_CASH = 10_000.0

# Roughly two years in-sample, six months out-of-sample, rolled by the
# out-of-sample length so the scored windows tile the history without overlap.
IN_SAMPLE_BARS = 504
OUT_OF_SAMPLE_BARS = 126
STEP_BARS = OUT_OF_SAMPLE_BARS

# Capped at 250 rather than the sweep's 300 so that the longest warm-up stays
# under half the in-sample window. See the module docstring.
FAST_GRID: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80)
SLOW_GRID: Tuple[int, ...] = (50, 75, 100, 125, 150, 175, 200, 225, 250)


class Roll(NamedTuple):
    """One in-sample optimisation and the out-of-sample result it produced.

    Attributes:
        in_sample_start: First date used for parameter selection.
        in_sample_end: Last date used for parameter selection.
        out_start: First out-of-sample date, never seen during selection.
        out_end: Last out-of-sample date.
        fast: Fast window chosen in-sample.
        slow: Slow window chosen in-sample.
        in_sample_sharpe: Sharpe of the winning combo on the in-sample window.
        out_sharpe: Sharpe of that same combo on the out-of-sample window.
        out_benchmark_sharpe: Buy-and-hold Sharpe over the out-of-sample window.
        out_return: Total return of the combo out-of-sample.
        out_benchmark_return: Buy-and-hold total return out-of-sample.
        out_orders: Orders filled inside the out-of-sample window. Counted as
            individual fills, not as completed round-trips: a position opened
            in-sample and closed out-of-sample leaves one fill in the window and
            no round-trip at all, and reporting zero there would hide the single
            decision that drove the window's result.
        out_exposure: Fraction of out-of-sample bars holding a position. This is
            what explains a result rather than the order count, since a window
            with no orders and full exposure simply reproduces buy-and-hold.
    """

    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    out_start: pd.Timestamp
    out_end: pd.Timestamp
    fast: int
    slow: int
    in_sample_sharpe: float
    out_sharpe: float
    out_benchmark_sharpe: float
    out_return: float
    out_benchmark_return: float
    out_orders: int
    out_exposure: float


def roll_boundaries(
    bars: int,
    in_sample: int = IN_SAMPLE_BARS,
    out_of_sample: int = OUT_OF_SAMPLE_BARS,
    step: int = STEP_BARS,
) -> List[Tuple[int, int, int]]:
    """Bar-index boundaries of every complete roll that fits in the history.

    A roll is only emitted when a full out-of-sample window follows the
    in-sample one. A truncated final window is dropped rather than scored on
    fewer bars, since a short window's Sharpe is not comparable to the others
    and would distort the averages.

    Args:
        bars: Number of bars available.
        in_sample: Length of the optimisation window, in bars.
        out_of_sample: Length of the evaluation window, in bars.
        step: Bars to advance between rolls.

    Returns:
        A list of (in_sample_start, out_of_sample_start, out_of_sample_end)
        index triples, where each range is half-open at the end.

    Raises:
        ValueError: If the lengths are not positive, or if the history is too
            short for even one roll.
    """
    if min(in_sample, out_of_sample, step) < 1:
        raise ValueError(
            f"Window lengths must be at least one bar, got in_sample="
            f"{in_sample}, out_of_sample={out_of_sample}, step={step}."
        )

    boundaries: List[Tuple[int, int, int]] = []
    start = 0
    while start + in_sample + out_of_sample <= bars:
        boundaries.append((start, start + in_sample, start + in_sample + out_of_sample))
        start += step

    if not boundaries:
        raise ValueError(
            f"{bars} bars is too short for a {in_sample}-bar in-sample window "
            f"followed by a {out_of_sample}-bar out-of-sample window. Extend the "
            f"date range or shorten the windows."
        )

    return boundaries


def select(
    in_sample_prices: pd.DataFrame,
    initial_cash: float,
    fast_windows: Sequence[int] = FAST_GRID,
    slow_windows: Sequence[int] = SLOW_GRID,
    commission: float = 0.0,
) -> Tuple[int, int, float]:
    """Choose the window pair with the best Sharpe ratio on the given prices.

    THE FUNCTION THAT CANNOT CHEAT
    Its only view of the world is the frame it is handed. Passing the in-sample
    slice therefore makes an out-of-sample leak impossible rather than merely
    forbidden, which is the whole architectural point of doing selection here
    instead of inside the walk-forward loop.

    Ties are broken by the first cell in row-major order over the grid, which is
    arbitrary but deterministic, so a repeated run makes identical choices.

    Args:
        in_sample_prices: Price history for the optimisation window only.
        initial_cash: Starting capital for each candidate.
        fast_windows: Fast window values to consider.
        slow_windows: Slow window values to consider.
        commission: Proportional commission applied to every candidate.

    Returns:
        The chosen fast window, slow window, and its in-sample Sharpe ratio.

    Raises:
        ValueError: If no candidate produced a usable Sharpe ratio.
    """
    results = sweep(
        in_sample_prices,
        initial_cash,
        fast_windows=fast_windows,
        slow_windows=slow_windows,
        commission=commission,
    )
    grid = metric_grid(results, SHARPE)
    values = grid.to_numpy(dtype=float)

    if not np.isfinite(values).any():
        raise ValueError(
            "No window pair produced a usable Sharpe ratio on the in-sample "
            "window, so there is nothing to select."
        )

    row, column = np.unravel_index(int(np.nanargmax(values)), values.shape)
    return int(grid.index[row]), int(grid.columns[column]), float(values[row, column])


def evaluate(
    prices: pd.DataFrame,
    initial_cash: float,
    fast: int,
    slow: int,
    in_sample_start: int,
    out_start: int,
    out_end: int,
    commission: float = 0.0,
) -> Tuple[pd.DataFrame, int]:
    """Score one window pair on the out-of-sample bars only.

    The backtest spans the in-sample window as well, because the averages need
    history before they exist and because a real position carries across the
    boundary. Every curve is then sliced to the out-of-sample dates before any
    metric is computed, so the in-sample period contributes warm-up and a
    starting position but no performance.

    Slicing rather than re-running is what makes the out-of-sample figures
    comparable to the benchmark's: both come from the same simulation over the
    same dates. Every metric used is return-based and therefore unaffected by
    the equity curve's absolute level at the slice boundary.

    Args:
        prices: The full price history.
        initial_cash: Starting capital at the in-sample start date.
        fast: Fast window to run.
        slow: Slow window to run.
        in_sample_start: Bar index the simulation starts at.
        out_start: First out-of-sample bar index.
        out_end: One past the last out-of-sample bar index.
        commission: Proportional commission per trade.

    Returns:
        The performance report covering the out-of-sample window, with a
        Strategy and a Benchmark column, and the number of orders filled inside
        the window. The order count is returned separately because the report's
        trade statistics count completed round-trips, and a position opened
        before the window and closed inside it is neither a round-trip nor
        nothing.
    """
    strategy = MovingAverageCrossover(
        fast_window=fast,
        slow_window=slow,
        enter_on_existing_trend=ENTER_ON_EXISTING_TREND,
    )
    result = Backtester(
        prices.iloc[in_sample_start:out_end],
        initial_cash=initial_cash,
        strategy=strategy,
        commission=commission,
    ).run()

    out_dates = prices.index[out_start:out_end]
    out_set = set(out_dates)
    out_orders = [trade for trade in result.trade_log if trade["date"] in out_set]

    report = performance_report(
        equity=result.equity_curve["total_value"].loc[out_dates],
        benchmark=result.benchmark_curve.loc[out_dates],
        trade_log=out_orders,
        positions=result.equity_curve["shares"].loc[out_dates],
    )
    return report, len(out_orders)


def walk_forward(
    prices: pd.DataFrame,
    initial_cash: float = INITIAL_CASH,
    in_sample: int = IN_SAMPLE_BARS,
    out_of_sample: int = OUT_OF_SAMPLE_BARS,
    step: int = STEP_BARS,
    commission: float = 0.0,
    progress: bool = True,
) -> List[Roll]:
    """Run the full walk-forward analysis.

    Args:
        prices: OHLCV history to walk through.
        initial_cash: Starting capital for every roll.
        in_sample: Length of each optimisation window, in bars.
        out_of_sample: Length of each evaluation window, in bars.
        step: Bars to advance between rolls.
        commission: Proportional commission, applied both when selecting and
            when evaluating, so the choice is made under the costs it will face.
        progress: Whether to print a line as each roll completes, since a full
            grid search per roll takes a while.

    Returns:
        One Roll per complete window pair, in chronological order.
    """
    rolls: List[Roll] = []
    boundaries = roll_boundaries(len(prices), in_sample, out_of_sample, step)

    for number, (start, out_start, out_end) in enumerate(boundaries, start=1):
        fast, slow, in_sample_sharpe = select(
            prices.iloc[start:out_start],
            initial_cash,
            commission=commission,
        )
        report, orders = evaluate(
            prices,
            initial_cash,
            fast,
            slow,
            start,
            out_start,
            out_end,
            commission=commission,
        )

        rolls.append(
            Roll(
                in_sample_start=prices.index[start],
                in_sample_end=prices.index[out_start - 1],
                out_start=prices.index[out_start],
                out_end=prices.index[out_end - 1],
                fast=fast,
                slow=slow,
                in_sample_sharpe=in_sample_sharpe,
                out_sharpe=float(report.loc[SHARPE, STRATEGY_COLUMN]),
                out_benchmark_sharpe=float(report.loc[SHARPE, BENCHMARK_COLUMN]),
                out_return=float(report.loc[TOTAL_RETURN, STRATEGY_COLUMN]),
                out_benchmark_return=float(
                    report.loc[TOTAL_RETURN, BENCHMARK_COLUMN]
                ),
                out_orders=orders,
                out_exposure=float(report.loc[EXPOSURE, STRATEGY_COLUMN]),
            )
        )

        if progress:
            print(f"  roll {number}/{len(boundaries)} done: chose "
                  f"{fast}/{slow}", flush=True)

    return rolls


def rolls_to_frame(rolls: List[Roll]) -> pd.DataFrame:
    """Turn the rolls into a numeric table for printing or export.

    Args:
        rolls: The rolls to tabulate.

    Returns:
        A DataFrame with one row per roll, keeping values numeric so the table
        can be exported or plotted without being parsed back out of strings.
    """
    return pd.DataFrame([roll._asdict() for roll in rolls])


def print_rolls(rolls: List[Roll]) -> None:
    """Print the per-roll table.

    Args:
        rolls: The rolls to display.
    """
    print(f"\n{'in-sample window':<25}{'chosen':>9}{'IS Sh':>8}"
          f"{'  out-of-sample window':<25}{'OOS Sh':>8}{'B&H Sh':>8}"
          f"{'OOS ret':>9}{'B&H ret':>9}{'ord':>5}{'expo':>8}")
    print("-" * 114)

    for roll in rolls:
        in_sample = (f"{roll.in_sample_start.date()} to "
                     f"{roll.in_sample_end.date()}")
        out_sample = (f"  {roll.out_start.date()} to {roll.out_end.date()}")
        print(f"{in_sample:<25}{f'{roll.fast}/{roll.slow}':>9}"
              f"{roll.in_sample_sharpe:>8.2f}{out_sample:<25}"
              f"{roll.out_sharpe:>8.2f}{roll.out_benchmark_sharpe:>8.2f}"
              f"{roll.out_return:>9.2%}{roll.out_benchmark_return:>9.2%}"
              f"{roll.out_orders:>5d}{roll.out_exposure:>8.0%}")

    print("\n  ord = orders filled inside the window, exposure = share of its "
          "bars invested.\n  A window with no orders and full exposure "
          "reproduces buy-and-hold exactly: the\n  strategy simply held what it "
          "already owned.")


def print_summary(rolls: List[Roll]) -> None:
    """Print the aggregate statistics and what they imply.

    Args:
        rolls: The rolls to summarise.
    """
    table = rolls_to_frame(rolls)

    in_sample_mean = table["in_sample_sharpe"].mean()
    out_mean = table["out_sharpe"].mean()
    penalty = in_sample_mean - out_mean

    # A window the strategy sat out entirely has no out-of-sample Sharpe to
    # report, and pandas resolves a comparison against NaN as False. Left alone,
    # such a roll would count as a failure in the numerator of every rate below
    # while the means above quietly dropped it, putting figures printed in the
    # same block on two different denominators. The rates are therefore taken
    # over the rolls where the comparison exists, and any exclusion is printed
    # rather than absorbed. On the default history nothing is excluded, so this
    # changes no published number; it stops a future run from misreporting one.
    scored = table[table["out_sharpe"].notna()]
    rated = scored[scored["out_benchmark_sharpe"].notna()]
    returns_rated = table[table["out_return"].notna()
                          & table["out_benchmark_return"].notna()]

    positive = (scored["out_sharpe"] > 0).mean()
    beat_sharpe = (rated["out_sharpe"] > rated["out_benchmark_sharpe"]).mean()
    beat_return = (returns_rated["out_return"]
                   > returns_rated["out_benchmark_return"]).mean()
    unscored = len(table) - len(scored)

    combinations = list(zip(table["fast"], table["slow"]))
    unchanged = sum(
        1 for earlier, later in zip(combinations, combinations[1:])
        if earlier == later
    )

    print(f"\n\nSUMMARY OVER {len(rolls)} ROLLS")
    print(f"  Mean in-sample Sharpe          {in_sample_mean:>7.3f}")
    print(f"  Mean out-of-sample Sharpe      {out_mean:>7.3f}")
    print(f"  Overfitting penalty            {penalty:>7.3f}   "
          f"(what optimisation promised and did not deliver)")
    print(f"  Mean buy-and-hold Sharpe (OOS) "
          f"{table['out_benchmark_sharpe'].mean():>7.3f}")
    print()
    print(f"  Out-of-sample Sharpe positive  {positive:>7.0%} of rolls")
    print(f"  Beat buy-and-hold on Sharpe    {beat_sharpe:>7.0%} of rolls")
    print(f"  Beat buy-and-hold on return    {beat_return:>7.0%} of rolls")
    print(f"  Median out-of-sample Sharpe    "
          f"{table['out_sharpe'].median():>7.3f}")
    if unscored:
        print(f"  ... the three rates above are over the {len(scored)} of "
              f"{len(rolls)} rolls whose\n      out-of-sample Sharpe is defined. "
              f"{unscored} window(s) were spent entirely in\n      cash, which "
              f"leaves no return series to score and no verdict to give.")
    print()
    print(f"  Distinct window pairs chosen   {len(set(combinations)):>7} "
          f"of {len(rolls)} rolls")
    print(f"  Kept the previous choice       {unchanged:>7} of "
          f"{max(len(rolls) - 1, 0)} transitions")

    silent = table[table["out_orders"] == 0]
    passive = silent[silent["out_exposure"] > 0.999]
    print()
    print(f"  Windows with no orders at all  {len(silent):>7} of {len(rolls)}")
    print(f"  ... of which fully invested    {len(passive):>7}   "
          f"(indistinguishable from buy-and-hold)")

    print("\nINTERPRETATION")

    # The penalty is the headline. It is the difference between what the
    # optimisation reported on data it had already seen and what the same
    # parameters delivered on data it had not, which is precisely the quantity
    # an in-sample-only study cannot report about itself.
    if penalty > 0.5:
        print(f"  The mean Sharpe falls by {penalty:.3f} from the in-sample "
              f"window to the very next\n  out-of-sample window, on identical "
              f"parameters and the same asset. Nothing about\n  the strategy "
              f"changed between the two, so the drop is not bad luck: it is the\n"
              f"  portion of the in-sample figure that was fitted to the "
              f"in-sample noise.")
    elif penalty > 0.1:
        print(f"  The mean Sharpe falls by {penalty:.3f} out-of-sample. Some of "
              f"the in-sample result\n  survives, but a meaningful part of it "
              f"was fitting rather than signal.")
    else:
        print(f"  The out-of-sample Sharpe holds up, falling only {penalty:.3f}. "
              f"On this history the\n  optimisation was not merely fitting "
              f"noise, which is the unusual outcome.")

    if len(set(combinations)) > len(rolls) / 2:
        print(f"\n  The chosen windows changed in "
              f"{len(rolls) - 1 - unchanged} of {len(rolls) - 1} transitions "
              f"and took {len(set(combinations))} distinct\n  values across "
              f"{len(rolls)} rolls. Parameters that genuinely described the "
              f"market would\n  be stable; these move because each window's "
              f"best fit is largely noise, which is\n  the same conclusion the "
              f"heatmap reached from a different direction.")

    if len(passive):
        print(f"\n  A caveat on the averages: {len(passive)} of {len(rolls)} "
              f"windows saw no order and stayed fully\n  invested, so in those "
              f"the strategy was buy-and-hold and scored identically by\n"
              f"  construction. The gap against the benchmark is therefore "
              f"decided by the {len(rolls) - len(passive)} windows\n  where the "
              f"strategy actually did something different, which is a smaller "
              f"sample than\n  the roll count suggests and a reason to hold the "
              f"conclusion loosely.")

    if beat_sharpe < 0.5:
        print(f"\n  Honestly optimised parameters beat buy-and-hold on "
              f"risk-adjusted terms in only\n  {beat_sharpe:.0%} of "
              f"out-of-sample windows, and the benchmark pays no transaction "
              f"costs\n  here while the strategy does. This is the answer to "
              f"the project's research\n  question for this rule on this asset, "
              f"and a negative answer honestly obtained is\n  a result, not a "
              f"failure.")
    else:
        print(f"\n  The parameters beat buy-and-hold out-of-sample in "
              f"{beat_sharpe:.0%} of windows, which on a\n  single asset over "
              f"one history is suggestive rather than conclusive.")


def main(
    ticker: str = TICKER,
    start: str = START,
    end: str = END,
    initial_cash: float = INITIAL_CASH,
    commission: float = 0.0,
) -> None:
    """Run the walk-forward analysis and report it.

    Args:
        ticker: Yahoo Finance symbol, served from cache after the first fetch.
        start: First date of the history, as YYYY-MM-DD.
        end: Last date of the history, as YYYY-MM-DD.
        initial_cash: Starting capital for every roll.
        commission: Proportional commission per trade, applied to both the
            selection and the evaluation.
    """
    prices = get_price_data(ticker, start, end)

    title = (f"WALK-FORWARD VALIDATION  {ticker}  "
             f"{period_label(prices, start, end)}  ({len(prices)} bars)")
    print(title)
    print("=" * len(title))
    print(f"In-sample {IN_SAMPLE_BARS} bars, out-of-sample "
          f"{OUT_OF_SAMPLE_BARS} bars, rolled by {STEP_BARS} so the "
          f"out-of-sample\nwindows do not overlap. Grid: "
          f"{len(FAST_GRID)} fast x {len(SLOW_GRID)} slow windows, "
          f"commission {commission:.2%}.")
    print("Parameters are chosen using in-sample bars only; the out-of-sample "
          "bars are not\npassed to the selection at all.\n")

    rolls = walk_forward(prices, initial_cash, commission=commission)
    print_rolls(rolls)
    print_summary(rolls)


if __name__ == "__main__":
    main()
