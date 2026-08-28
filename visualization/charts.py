"""Plotly figures for the backtest results.

Every function here takes plain pandas objects and returns a Plotly figure.
Nothing imports streamlit, so the same figures can be written to HTML from a
script, embedded in a notebook or dropped into a report without a dashboard
running. The __main__ block at the bottom does exactly that, which is both a
self-test and proof that the module stands on its own.

EACH CHART ANSWERS ONE QUESTION
The guideline asks for figures that are attractive without being empty, and
makes the test concrete: a chart earns its place by answering a question a
reader actually has.

    equity_curve_chart          Did the strategy beat buying and holding?
    drawdown_chart              How deep were the losses, and how long?
    trade_marker_chart          Where did the strategy actually act?
    returns_distribution_chart  Are the returns normal, or do they have fat
                                tails?
    parameter_heatmap_chart     Is this parameter choice robust, or an isolated
                                peak?

That framing is what keeps them uncluttered. Anything not needed to answer the
question is left out, which is why the equity chart carries two lines rather
than every metric that could be plotted against a date axis.

THE COLOUR CONVENTION IS FIXED HERE AND NOWHERE ELSE
Blue always means the strategy and grey always means the benchmark, in every
figure. Red always means loss: the drawdown fill and the sell markers share it.
Green appears only on buys. Because the meanings are constant, a reader who has
understood one chart has already understood the palette of the others, and no
figure needs to explain its own colours.

NOTHING DERIVED IS RECOMPUTED HERE
Every function that needs a derived quantity takes the equity curve and asks
analytics.risk for it: drawdown_chart calls drawdown_series, and
returns_distribution_chart calls periodic_returns, skewness and kurtosis. None
of them accepts the derived series from the caller. That costs a little
flexibility and buys something worth more: the shape on screen is necessarily
the quantity the performance report puts a number on, because both come from one
implementation. A chart that quietly disagreed with the table beside it would be
worse than no chart.

It also keeps the signatures uniform among the figures that describe a single
backtest: a caller passes an equity curve and, where a comparison makes sense, a
benchmark curve, without having to remember which figure wants raw values and
which wants a transformation.

parameter_heatmap_chart is the one exception, and unavoidably so. It does not
describe one backtest but a grid of several hundred, so it takes the swept grid
that analytics.validation already produces. Recomputing a sweep inside a plotting
function would be the wrong trade entirely: it would take minutes, and it would
put the choice of what to sweep in the hands of the module least equipped to
make it.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from analytics.risk import (
    drawdown_series,
    kurtosis,
    periodic_returns,
    skewness,
)
from constants import BUY, SELL

STRATEGY_COLOUR = "#1f77b4"
BENCHMARK_COLOUR = "#9aa0a6"
LOSS_COLOUR = "#d62728"
GAIN_COLOUR = "#2ca02c"
PRICE_COLOUR = "#5f6368"

# Deliberately outside the semantic palette above. The fitted normal is not a
# data series and must not borrow the grey that means benchmark; a dark dashed
# line is the conventional way to draw a theoretical reference.
MODEL_COLOUR = "#202124"

# Plotly renders a fill at 100% opacity by default, which buries the gridlines
# under it. The drawdown area reads better as a tint with its outline intact.
LOSS_FILL = "rgba(214, 39, 40, 0.28)"

STRATEGY_NAME = "Strategy"
BENCHMARK_NAME = "Buy and hold"

TEMPLATE = "plotly_white"
HEIGHT = 420

# The heatmap needs more vertical room than the time-series figures: its cells
# should stay close to square so that neither axis looks like the important one.
HEATMAP_HEIGHT = 560

# Bin count for the returns histogram. Enough to expose a fat tail without
# turning the body of the distribution into noise at the few hundred to few
# thousand returns a backtest of a handful of years produces.
BINS = 80

# Resolution of the fitted normal curve. It only has to look smooth.
CURVE_POINTS = 200


def equity_curve_chart(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio value against buy and hold",
) -> go.Figure:
    """Plot the portfolio's value over time against the benchmark.

    The two curves are left on their natural scale rather than being rebased to
    100, because the Backtester starts both from the same capital. They are
    therefore already directly comparable, and the axis keeps the useful
    property of reading in the same units as the initial cash input.

    Hover is unified on the date, so a single pointer position reports both
    curves at once. That is the whole question the chart exists to answer: at
    any moment, which one was ahead, and by how much.

    Args:
        equity: Portfolio value over time, indexed by date.
        benchmark: Buy-and-hold value over the same dates. Omitted cleanly if
            None, leaving a single-curve chart.
        title: Figure title.

    Returns:
        A Plotly figure.
    """
    figure = go.Figure()

    if benchmark is not None and not benchmark.empty:
        figure.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.to_numpy(dtype=float),
            name=BENCHMARK_NAME,
            mode="lines",
            line=dict(color=BENCHMARK_COLOUR, width=1.8, dash="dot"),
            hovertemplate=f"{BENCHMARK_NAME}: %{{y:,.2f}}<extra></extra>",
        ))

    figure.add_trace(go.Scatter(
        x=equity.index,
        y=equity.to_numpy(dtype=float),
        name=STRATEGY_NAME,
        mode="lines",
        line=dict(color=STRATEGY_COLOUR, width=2.2),
        hovertemplate=f"{STRATEGY_NAME}: %{{y:,.2f}}<extra></extra>",
    ))

    _apply_layout(figure, title, "Portfolio value")
    figure.update_yaxes(tickformat=",.0f")
    return figure


def drawdown_chart(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Drawdown from the running peak",
) -> go.Figure:
    """Plot how far below its own high-water mark the portfolio sat.

    The strategy's drawdown is a red area hanging below zero, since the quantity
    is never positive: zero marks every new peak, and the area's depth is the
    loss still outstanding at that date. Width matters as much as depth, because
    a shallow decline that takes two years to recover is its own kind of
    expensive, and only the horizontal axis shows that.

    The benchmark's drawdown is drawn as a plain grey line when available. A
    depth of 9% means little in isolation; against the 31% the same period
    handed a buy-and-hold holder, it means a great deal.

    Args:
        equity: Portfolio value over time, indexed by date.
        benchmark: Buy-and-hold value over the same dates, for comparison.
        title: Figure title.

    Returns:
        A Plotly figure.
    """
    figure = go.Figure()
    drawdown = drawdown_series(equity)

    figure.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.to_numpy(dtype=float),
        name=STRATEGY_NAME,
        mode="lines",
        line=dict(color=LOSS_COLOUR, width=1.6),
        fill="tozeroy",
        fillcolor=LOSS_FILL,
        hovertemplate=f"{STRATEGY_NAME}: %{{y:.2%}}<extra></extra>",
    ))

    if benchmark is not None and not benchmark.empty:
        reference = drawdown_series(benchmark)
        figure.add_trace(go.Scatter(
            x=reference.index,
            y=reference.to_numpy(dtype=float),
            name=BENCHMARK_NAME,
            mode="lines",
            line=dict(color=BENCHMARK_COLOUR, width=1.6, dash="dot"),
            hovertemplate=f"{BENCHMARK_NAME}: %{{y:.2%}}<extra></extra>",
        ))

    _apply_layout(figure, title, "Drawdown")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    return figure


def trade_marker_chart(
    prices: pd.Series,
    trade_log: Sequence[Dict[str, Any]],
    title: str = "Where the strategy traded",
) -> go.Figure:
    """Plot the close price with the strategy's entries and exits marked.

    Markers sit at the quoted close rather than at the fill price, so they land
    exactly on the line the reader is following. The fill actually paid, which
    differs from the quote whenever transaction costs are switched on, is
    reported on hover along with the size of the order. Putting the marker at
    the fill would lift it off the price line by the width of the spread and
    make the chart look misaligned.

    An empty trade log yields the price line alone, with no legend entries for
    trades that never happened. That is a real outcome for a slow strategy on a
    short history, not an error.

    Args:
        prices: Close price over time, indexed by date.
        trade_log: The Backtester's trade log. Each entry needs date, action,
            quoted_price and shares; price and cost enrich the hover when
            present.
        title: Figure title.

    Returns:
        A Plotly figure.
    """
    figure = go.Figure()

    figure.add_trace(go.Scatter(
        x=prices.index,
        y=prices.to_numpy(dtype=float),
        name="Close price",
        mode="lines",
        line=dict(color=PRICE_COLOUR, width=1.5),
        hovertemplate="Close: %{y:,.2f}<extra></extra>",
    ))

    marker_styles = (
        (BUY, "Buy", GAIN_COLOUR, "triangle-up"),
        (SELL, "Sell", LOSS_COLOUR, "triangle-down"),
    )
    for action, label, colour, symbol in marker_styles:
        orders = [order for order in trade_log if order["action"] == action]
        if not orders:
            continue

        figure.add_trace(go.Scatter(
            x=[order["date"] for order in orders],
            y=[float(order["quoted_price"]) for order in orders],
            name=label,
            mode="markers",
            marker=dict(color=colour, symbol=symbol, size=12,
                        line=dict(color="white", width=1)),
            customdata=[_order_hover(order) for order in orders],
            hovertemplate=(f"<b>{label}</b> at %{{y:,.2f}}"
                           f"%{{customdata}}<extra></extra>"),
        ))

    _apply_layout(figure, title, "Price")
    figure.update_yaxes(tickformat=",.2f")
    return figure


def returns_distribution_chart(
    equity: pd.Series,
    title: str = "Distribution of periodic returns",
) -> go.Figure:
    """Plot the histogram of periodic returns against a fitted normal curve.

    The normal curve is fitted to the same mean and standard deviation as the
    returns themselves, which is what makes the comparison fair and the
    departures readable. Two departures matter and both are visible without any
    statistics: a peak taller than the curve with tails poking out beyond it is
    excess kurtosis, the fat tails that parametric VaR underestimates, and a
    histogram leaning to one side of the curve is skew.

    Densities are plotted rather than counts, because a probability density is
    the only scale on which a histogram and a fitted curve are directly
    comparable without an arbitrary rescaling factor.

    TYING THE PICTURE TO THE TABLE
    The subtitle carries the sample size, the share of returns that are exactly
    zero, and the skewness and excess kurtosis. Those last two come from
    analytics.risk called with the same argument the performance report passes,
    so the numbers under the title are necessarily the Distribution rows of the
    table, not a second opinion on them.

    The share of exact zeros is always shown rather than only when it is large.
    It is the one number that explains the shape of a cash-heavy strategy's
    histogram, where a spike at zero dwarfs everything and the curve fitted
    through it describes the flat bars rather than the trading. Stating it
    unconditionally means the figure carries its own qualification wherever it
    is displayed, including outside the dashboard.

    Args:
        equity: Portfolio value over time. Returns are derived with
            periodic_returns rather than taken from the caller, so the sample
            drawn here is the sample the report describes.
        title: Figure title.

    Returns:
        A Plotly figure. An equity curve too short to yield returns gives an
        empty figure carrying an explanatory annotation rather than raising.
    """
    figure = go.Figure()
    returns = periodic_returns(equity)

    if returns.empty:
        _apply_layout(figure, title, "Density", x_label="Periodic return",
                      x_hoverformat=".2%",
                      subtitle="Not enough history to compute a single return.")
        return figure

    values = returns.to_numpy(dtype=float)
    figure.add_trace(go.Histogram(
        x=values,
        name="Actual returns",
        histnorm="probability density",
        nbinsx=BINS,
        marker=dict(color=STRATEGY_COLOUR, line=dict(color="white", width=0.4)),
        opacity=0.75,
        hovertemplate="Actual density: %{y:.1f}<extra></extra>",
    ))

    average = float(values.mean())
    deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    # A zero standard deviation means every return was identical, so there is no
    # bell curve to fit. The histogram alone still tells the story.
    if deviation > 0.0:
        grid = np.linspace(values.min(), values.max(), CURVE_POINTS)
        figure.add_trace(go.Scatter(
            x=grid,
            y=norm.pdf(grid, loc=average, scale=deviation),
            name="Fitted normal",
            mode="lines",
            line=dict(color=MODEL_COLOUR, width=2, dash="dash"),
            hovertemplate="Normal density: %{y:.1f}<extra></extra>",
        ))

    _apply_layout(figure, title, "Density", x_label="Periodic return",
                  x_hoverformat=".2%",
                  subtitle=_distribution_subtitle(equity, returns))
    figure.update_xaxes(tickformat=".1%")
    figure.update_layout(bargap=0.02)
    return figure


def parameter_heatmap_chart(
    grid: pd.DataFrame,
    center: float,
    selected: Optional[Tuple[int, int]] = None,
    metric_label: str = "Sharpe ratio",
    title: str = "Sharpe across the fast and slow window grid",
    subtitle: Optional[str] = None,
    annotate: bool = True,
) -> go.Figure:
    """Plot a parameter sweep as an interactive heatmap.

    The colour scale is diverging and centred on a reference score rather than
    on the grid's own midpoint, which is what turns a decorative gradient into a
    verdict: warm cells beat the reference, cool cells lose to it, and the
    boundary between them is a real threshold instead of an artefact of the
    range that happened to be swept. The reference is normally the benchmark's
    score, since beating buy-and-hold is the only bar that matters.

    WHY THE AXES ARE CATEGORICAL
    Window values are not evenly spaced once a user's own choice is inserted
    into a regular grid, and numeric axes would then draw cells of unequal
    width, implying that some combinations cover more ground than others. They
    do not: each cell is one backtest. Treating the labels as categories gives
    every cell equal weight on screen, which is the honest rendering.

    Invalid combinations, where the fast window is not shorter than the slow
    one, arrive as NaN and are left as gaps. A gap reads correctly as "not
    applicable" where a zero would read as "tested and worthless".

    Args:
        grid: Scores with fast windows as the index and slow windows as the
            columns, as returned by metric_grid.
        center: Score the colour scale pivots on, normally the benchmark's.
        selected: A (fast, slow) pair to outline, or None. It must be present in
            the grid, which is the caller's job to arrange.
        metric_label: Name of the plotted quantity, used on the colour bar.
        title: Figure title.
        subtitle: Optional second line under the title.
        annotate: Whether to print each cell's value inside it.

    Returns:
        A Plotly figure.
    """
    figure = go.Figure()

    fast_labels = [str(value) for value in grid.index]
    slow_labels = [str(value) for value in grid.columns]

    figure.add_trace(go.Heatmap(
        x=slow_labels,
        y=fast_labels,
        z=grid.to_numpy(dtype=float),
        colorscale="RdBu",
        reversescale=True,
        zmid=center if pd.notna(center) else None,
        colorbar=dict(title=metric_label),
        hovertemplate=("Fast %{y} · Slow %{x}<br>"
                       f"{metric_label} %{{z:.3f}}<extra></extra>"),
        texttemplate="%{z:.2f}" if annotate else None,
        textfont=dict(size=8),
        hoverongaps=False,
    ))

    if selected is not None:
        fast, slow = selected
        figure.add_trace(go.Scatter(
            x=[str(slow)],
            y=[str(fast)],
            name="Your selection",
            mode="markers",
            marker=dict(symbol="square-open", size=22, color=MODEL_COLOUR,
                        line=dict(width=3)),
            hovertemplate=(f"<b>Your selection</b><br>Fast {fast} · Slow {slow}"
                           "<extra></extra>"),
        ))

    _apply_layout(figure, title, "Fast window (bars)",
                  x_label="Slow window (bars)", x_hoverformat="",
                  subtitle=subtitle)

    # A heatmap wants the cell under the pointer, not every cell sharing an x
    # value, so the module's unified date hover is the wrong mode here.
    figure.update_layout(hovermode="closest", height=HEATMAP_HEIGHT)
    figure.update_xaxes(type="category", showspikes=False)
    figure.update_yaxes(type="category", showspikes=False)
    return figure


def _distribution_subtitle(equity: pd.Series, returns: pd.Series) -> str:
    """Summarise the plotted sample in one line under the title.

    Args:
        equity: The curve the returns came from, as the risk functions expect.
        returns: The periodic returns being plotted.

    Returns:
        A single line of plain text.
    """
    zero_share = float((returns == 0.0).mean())
    return (f"{len(returns):,} returns · {zero_share:.1%} exactly zero · "
            f"skewness {skewness(equity):.2f} · "
            f"excess kurtosis {kurtosis(equity, excess=True):.2f}")


def _order_hover(order: Dict[str, Any]) -> str:
    """Build the extra hover lines describing one filled order.

    Cost fields are only mentioned when they are non-zero, so a frictionless run
    does not pad every tooltip with three zeros.

    Args:
        order: One entry of the Backtester's trade log.

    Returns:
        A string of pre-formatted HTML lines, empty if there is nothing to add.
    """
    lines: List[str] = [f"{float(order['shares']):,.4f} shares"]

    cost = float(order.get("cost") or 0.0)
    if cost:
        lines.append(f"filled at {float(order['price']):,.4f}")
        lines.append(f"cost {cost:,.2f}")

    return "<br>" + "<br>".join(lines)


def _apply_layout(
    figure: go.Figure,
    title: str,
    y_label: str,
    x_label: str = "Date",
    x_hoverformat: str = "%d %b %Y",
    subtitle: Optional[str] = None,
) -> None:
    """Apply the styling every figure in this module shares.

    Centralising it is what makes the charts look like a set rather than four
    unrelated plots, and it is the only place a decision like the hover mode has
    to be made.

    Three of the four figures share a date axis, which is why the x-axis
    arguments default to dates rather than being spelled out at each call.

    Args:
        figure: The figure to style, modified in place.
        title: Figure title.
        y_label: Label for the vertical axis.
        x_label: Label for the horizontal axis.
        x_hoverformat: Format applied to the x value in the hover box.
        subtitle: Optional smaller second line under the title. Rendered with
            HTML rather than the layout's own subtitle field, which keeps the
            function working across Plotly versions.
    """
    heading = title if subtitle is None else f"{title}<br><sup>{subtitle}</sup>"

    figure.update_layout(
        title=heading,
        template=TEMPLATE,
        height=HEIGHT,
        # Unified hover is the point of these charts: one pointer position
        # reports every series on that date, which is what makes a comparison
        # possible without reading two tooltips side by side.
        hovermode="x unified",
        margin=dict(l=70, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1.0),
    )
    figure.update_xaxes(title_text=x_label, hoverformat=x_hoverformat,
                        showspikes=True, spikemode="across", spikethickness=1)
    figure.update_yaxes(title_text=y_label)


if __name__ == "__main__":
    from pathlib import Path

    from data.market_data import get_price_data
    from engine.backtester import Backtester
    from strategies.momentum import Momentum

    OUTPUT = Path(__file__).resolve().parent.parent / "output"
    OUTPUT.mkdir(exist_ok=True)

    print("Running a backtest to draw. Momentum on AAPL, with costs charged so "
          "the trade hover has something to show.")
    prices = get_price_data("AAPL", "2020-01-01", "2023-01-01")
    result = Backtester(prices, initial_cash=10_000.0,
                        strategy=Momentum(lookback=126, rebalance_freq=21),
                        commission=0.001).run()
    equity = result.equity_curve["total_value"]

    figures = {
        "equity_curve": equity_curve_chart(equity, result.benchmark_curve),
        "drawdown": drawdown_chart(equity, result.benchmark_curve),
        "trade_markers": trade_marker_chart(prices["Close"], result.trade_log),
        "returns_distribution": returns_distribution_chart(equity),
    }

    print()
    print("Each figure is built from plain pandas objects, with no dashboard "
          "involved, and written straight to HTML:")
    for name, figure in figures.items():
        path = OUTPUT / f"{name}.html"
        figure.write_html(path, include_plotlyjs="cdn")
        traces = ", ".join(trace.name for trace in figure.data)
        print(f"  {path.name:<22} {len(figure.data)} traces ({traces})")

    # The drawdown drawn here has to be the same number the report prints, which
    # is the reason drawdown_chart derives it rather than accepting it.
    print()
    drawn = min(figures["drawdown"].data[0].y)
    from analytics.risk import max_drawdown
    print(f"Deepest point of the plotted area: {drawn:.4%}")
    print(f"max_drawdown from analytics.risk:  {max_drawdown(equity):.4%}")
    assert abs(drawn - max_drawdown(equity)) < 1e-12
    print("The chart and the report cannot disagree: both call drawdown_series.")

    # An empty trade log is a real outcome, not an error, so it must not raise.
    print()
    empty = trade_marker_chart(prices["Close"], [])
    print(f"With no trades to mark: {len(empty.data)} trace "
          f"({empty.data[0].name}), no phantom legend entries.")

    single = equity_curve_chart(equity, None)
    print(f"With no benchmark supplied: {len(single.data)} trace "
          f"({single.data[0].name}).")

    # The subtitle exists to tie the picture to the Distribution rows of the
    # report, so the numbers it quotes have to be those rows and not a recount.
    print()
    subtitle = figures["returns_distribution"].layout.title.text.split("<sup>")[1]
    print(f"Histogram subtitle: {subtitle.replace('</sup>', '')}")
    print(f"skewness from analytics.risk:        {skewness(equity):.2f}")
    print(f"excess kurtosis from analytics.risk: "
          f"{kurtosis(equity, excess=True):.2f}")
    print("Same functions, same argument, so the subtitle cannot drift from "
          "the table.")

    # A curve too short to yield a single return must not raise: parameter
    # sweeps and short date ranges both reach this path.
    print()
    for label, curve in (("one bar", equity.iloc[:1]),
                         ("empty", equity.iloc[:0])):
        degenerate = returns_distribution_chart(curve)
        note = degenerate.layout.title.text.split("<sup>")[1]
        print(f"  {label:<8} -> {len(degenerate.data)} traces, "
              f"subtitle: {note.replace('</sup>', '')}")

    # A perfectly flat curve has a zero standard deviation, so there is no bell
    # curve to fit. The histogram must still be drawn.
    flat = pd.Series([10_000.0] * 50, index=equity.index[:50])
    figure = returns_distribution_chart(flat)
    print(f"  {'flat':<8} -> {len(figure.data)} trace "
          f"({figure.data[0].name}), no normal fitted to a zero deviation")
