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

    equity_curve_chart   Did the strategy beat buying and holding?
    drawdown_chart       How deep did the losses get, and how long did they last?
    trade_marker_chart   Where did the strategy actually act?

That framing is what keeps them uncluttered. Anything not needed to answer the
question is left out, which is why the equity chart carries two lines rather
than every metric that could be plotted against a date axis.

THE COLOUR CONVENTION IS FIXED HERE AND NOWHERE ELSE
Blue always means the strategy and grey always means the benchmark, in every
figure. Red always means loss: the drawdown fill and the sell markers share it.
Green appears only on buys. Because the meanings are constant, a reader who has
understood one chart has already understood the palette of the others, and no
figure needs to explain its own colours.

DRAWDOWNS ARE NOT RECOMPUTED HERE
drawdown_chart takes an equity curve and calls drawdown_series from
analytics.risk, rather than accepting a drawdown series from the caller. That
costs a little flexibility and buys something worth more: the shape on screen is
necessarily the same quantity the performance report puts a number on, since
both come from one implementation. A chart that quietly disagreed with the table
beside it would be worse than no chart.
"""

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from analytics.risk import drawdown_series
from constants import BUY, SELL

STRATEGY_COLOUR = "#1f77b4"
BENCHMARK_COLOUR = "#9aa0a6"
LOSS_COLOUR = "#d62728"
GAIN_COLOUR = "#2ca02c"
PRICE_COLOUR = "#5f6368"

# Plotly renders a fill at 100% opacity by default, which buries the gridlines
# under it. The drawdown area reads better as a tint with its outline intact.
LOSS_FILL = "rgba(214, 39, 40, 0.28)"

STRATEGY_NAME = "Strategy"
BENCHMARK_NAME = "Buy and hold"

TEMPLATE = "plotly_white"
HEIGHT = 420
DATE_HOVER = "%{x|%d %b %Y}"


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


def _apply_layout(figure: go.Figure, title: str, y_label: str) -> None:
    """Apply the styling every figure in this module shares.

    Centralising it is what makes the charts look like a set rather than three
    unrelated plots, and it is the only place a decision like the hover mode has
    to be made.

    Args:
        figure: The figure to style, modified in place.
        title: Figure title.
        y_label: Label for the vertical axis.
    """
    figure.update_layout(
        title=title,
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
    figure.update_xaxes(title_text="Date", hoverformat="%d %b %Y",
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
