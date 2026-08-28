"""Streamlit dashboard: the only file in this project that knows about a UI.

THE ARCHITECTURAL BOUNDARY THIS FILE DEFENDS
Everything below the UI layer, the engine, the strategies and the analytics, has
been written without a single import of streamlit, and that is the property this
file exists to preserve. Its whole job is to turn widget values into a plain
configuration object, hand that to functions that already exist, and render what
comes back.

The practical test of the boundary is that every number shown here can be
reproduced from the command line by the scripts at the project root, because both
call the same functions with the same arguments. If a calculation ever needed to
live in this file, that would be the signal it belongs in the analytics package
instead.

WHY THE PARAMETER WIDGETS ARE BUILT PER STRATEGY
The guideline is explicit that the parameters on screen must change with the
selected strategy rather than showing every strategy's knobs at once. That is
achieved structurally here: each strategy has its own builder function that draws
only its own widgets and returns a ready instance, and only the selected builder
is ever called. Widgets that do not apply are not hidden, they are never created,
so there is no way for a stale value from an unselected strategy to leak into a
run.

WHERE THE TRANSACTION COSTS ARE APPLIED
Nowhere in this file. The sidebar collects three percentages, divides them by a
hundred to reach the fractions the engine speaks, and hands them to the
Backtester, which passes them to the Broker that has priced fills since Phase 4.
The one cost figure this file displays, the drag of a round trip, is obtained by
asking a Broker what it would charge rather than by restating its formulas.

WHY INVALID INPUT IS PREVENTED RATHER THAN REPORTED
A moving average crossover needs fast < slow, and an RSI needs oversold <
overbought. Both could be checked after the fact and reported as an error, but it
is better for the slider bounds to make the invalid state unreachable: the slow
window's minimum is derived from the fast window's current value. The strategies
still validate their own arguments, since they are library code and cannot assume
a well-behaved caller, but a user should never see that exception.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from analytics.report import (
    BENCHMARK_COLUMN,
    EXPOSURE,
    MAX_DRAWDOWN,
    SHARPE,
    STRATEGY_COLUMN,
    TOTAL_RETURN,
    format_report,
    performance_report,
    report_caveats,
)
from data.market_data import get_price_data
from engine.backtester import BacktestResult, Backtester
from engine.broker import Broker
from engine.portfolio import Portfolio
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import RSIMeanReversion
from strategies.momentum import Momentum
from strategies.moving_average import MovingAverageCrossover
from visualization.charts import (
    drawdown_chart,
    equity_curve_chart,
    trade_marker_chart,
)

DEFAULT_TICKER = "AAPL"
DEFAULT_START = date(2020, 1, 1)
DEFAULT_END = date(2023, 1, 1)
DEFAULT_CASH = 10_000.0

MA_LABEL = "MA Crossover"
RSI_LABEL = "RSI Mean Reversion"
MOMENTUM_LABEL = "Momentum"

# Enough bars for the longest indicator warm-up plus something to trade on.
# Shorter ranges run without error but produce an all-HOLD signal series and an
# empty report, which looks like a bug rather than a short history.
MINIMUM_BARS = 60


@dataclass
class BacktestConfig:
    """Everything the engine needs, collected from the sidebar.

    This is the object the guideline's architecture principle is about. The UI
    produces one of these and nothing else; the engine consumes it and never
    learns where it came from. Keeping it a plain dataclass means the same
    configuration could just as well be built by a script, a test or a REST
    handler.

    The three cost fields are stored as fractions, not percentages, because that
    is the unit the Broker speaks. The sidebar asks for percentages, since that
    is how costs are quoted in practice, and converts once at the boundary. The
    conversion is the only arithmetic this file performs on a cost.

    Attributes:
        ticker: Yahoo Finance symbol to fetch.
        start: First date of the history.
        end: Last date of the history, exclusive.
        initial_cash: Starting capital.
        strategy: A ready-to-run strategy instance.
        commission: Proportional commission per trade as a fraction, so 0.001
            is 0.1% of the trade value.
        spread: Bid-ask spread as a fraction of price. Half is paid per side.
        slippage: Adverse price move as a fraction of price, per side.
    """

    ticker: str
    start: date
    end: date
    initial_cash: float
    strategy: BaseStrategy
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0

    @property
    def charges_costs(self) -> bool:
        """Whether any friction is switched on."""
        return bool(self.commission or self.spread or self.slippage)

    def problems(self) -> List[str]:
        """Reasons this configuration cannot be run, if any.

        Returns:
            A list of human-readable messages, empty when the config is valid.
        """
        issues: List[str] = []
        if not self.ticker.strip():
            issues.append("Enter a ticker symbol.")
        if self.start >= self.end:
            issues.append("The start date must fall before the end date.")
        if self.initial_cash <= 0:
            issues.append("Initial cash must be greater than zero.")
        return issues


def build_ma_strategy() -> BaseStrategy:
    """Draw the moving average inputs and return the configured strategy.

    The slow window's lower bound is tied to the fast window's current value, so
    the two sliders cannot express a combination the strategy would reject.
    """
    fast = st.sidebar.slider(
        "Fast window (bars)",
        min_value=5,
        max_value=150,
        value=50,
        step=5,
        help="Lookback of the short moving average.",
    )
    slow = st.sidebar.slider(
        "Slow window (bars)",
        min_value=fast + 5,
        max_value=400,
        value=max(200, fast + 5),
        step=5,
        help="Lookback of the long moving average. Always longer than the fast "
             "window, since two averages of equal length never cross.",
    )
    existing_trend = st.sidebar.checkbox(
        "Enter a trend already in progress",
        value=False,
        help="When the averages first become computable the fast one may "
             "already be above the slow one, with the crossing that caused it "
             "outside the data. Off, the conservative default, waits for a "
             "crossing it can actually observe. On measures the rule as a state "
             "rather than as an event, and is the fairer setting when comparing "
             "against momentum.",
    )

    return MovingAverageCrossover(
        fast_window=fast,
        slow_window=slow,
        enter_on_existing_trend=existing_trend,
    )


def build_rsi_strategy() -> BaseStrategy:
    """Draw the RSI inputs and return the configured strategy.

    The overbought threshold's lower bound follows the oversold value, for the
    same reason the slow window follows the fast one.
    """
    period = st.sidebar.slider(
        "RSI period (bars)",
        min_value=2,
        max_value=50,
        value=14,
        help="Bars the RSI averages gains and losses over. 14 is Wilder's own.",
    )
    oversold = st.sidebar.slider(
        "Oversold threshold",
        min_value=5.0,
        max_value=45.0,
        value=30.0,
        step=1.0,
        help="Buy when the RSI crosses down through this level.",
    )
    overbought = st.sidebar.slider(
        "Overbought threshold",
        min_value=oversold + 5.0,
        max_value=95.0,
        value=max(70.0, oversold + 5.0),
        step=1.0,
        help="Sell when the RSI crosses up through this level.",
    )

    return RSIMeanReversion(
        rsi_period=period,
        oversold=oversold,
        overbought=overbought,
    )


def build_momentum_strategy() -> BaseStrategy:
    """Draw the momentum inputs and return the configured strategy."""
    lookback = st.sidebar.slider(
        "Lookback (bars)",
        min_value=10,
        max_value=400,
        value=126,
        step=2,
        help="Window of the trailing return. 126 bars is roughly six months.",
    )
    frequency = st.sidebar.slider(
        "Review every (bars)",
        min_value=1,
        max_value=126,
        value=21,
        help="Bars between decisions. 1 reviews on every bar, 21 monthly. "
             "Reviewing more often trades more without necessarily scoring "
             "better.",
    )

    return Momentum(lookback=lookback, rebalance_freq=frequency)


# Only the selected builder runs, which is what makes the parameter section
# change with the strategy instead of showing every strategy's inputs at once.
STRATEGY_BUILDERS = {
    MA_LABEL: build_ma_strategy,
    RSI_LABEL: build_rsi_strategy,
    MOMENTUM_LABEL: build_momentum_strategy,
}


def cost_input(label: str, help_text: str) -> float:
    """Ask for one cost in percent and return it as a fraction.

    The upper bound is deliberately far tighter than the Broker's own limit,
    which only rejects a commission at or above 100%. Capping the field at the
    5% that run_cost_scenarios.py uses as its break-even search ceiling keeps
    the field inside the range the rest of the project treats as meaningful, and
    makes a Broker rejection unreachable from the UI.

    Args:
        label: Widget label, which must name the unit.
        help_text: Tooltip explaining what the number does.

    Returns:
        The value as a fraction of price, so 0.1% is returned as 0.001.
    """
    percent = st.sidebar.number_input(
        label,
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.01,
        format="%.3f",
        help=help_text,
    )
    return float(percent) / 100.0


def round_trip_drag(commission: float, spread: float, slippage: float) -> float:
    """Fraction of a position consumed by buying it and selling it again.

    The figure is obtained by asking a Broker what it would charge, rather than
    by restating its formulas here. Quoting a notional price of 1.0 makes the
    two fill prices read directly as fractions, and no Portfolio is ever
    debited: only the pricing methods are called.

    Args:
        commission: Proportional commission per trade, as a fraction.
        spread: Bid-ask spread as a fraction of price.
        slippage: Adverse price move as a fraction of price, per side.

    Returns:
        The fraction of the position lost to a single round trip.
    """
    broker = Broker(Portfolio(1.0), commission=commission, spread=spread,
                    slippage=slippage)
    return 1.0 - broker.sell_fill_price(1.0) / broker.buy_fill_price(1.0)


def sidebar() -> Tuple[BacktestConfig, bool]:
    """Draw the whole sidebar and collect it into a configuration.

    Returns:
        The BacktestConfig described by the current widget values, and whether
        the run button was pressed on this rerun.
    """
    st.sidebar.title("Backtest setup")

    st.sidebar.subheader("Data")
    ticker = st.sidebar.text_input("Ticker", value=DEFAULT_TICKER,
                                   help="Yahoo Finance notation, e.g. AAPL, "
                                        "MSFT, BRK-B, ^GSPC.")
    start = st.sidebar.date_input("Start date", value=DEFAULT_START)
    end = st.sidebar.date_input("End date", value=DEFAULT_END)
    cash = st.sidebar.number_input(
        "Initial cash",
        min_value=100.0,
        value=DEFAULT_CASH,
        step=1_000.0,
        format="%.2f",
    )

    st.sidebar.subheader("Strategy")
    choice = st.sidebar.selectbox("Strategy", list(STRATEGY_BUILDERS))

    st.sidebar.subheader(f"{choice} parameters")
    strategy = STRATEGY_BUILDERS[choice]()

    st.sidebar.subheader("Transaction Costs")
    st.sidebar.caption("All three are percentages of the traded price. Zero "
                       "reproduces a frictionless run exactly.")
    commission = cost_input(
        "Commission (%) per trade",
        "Charged on the value of every fill. Enter 0.1 for 0.1%, which is "
        "0.001 as a fraction.",
    )
    spread = cost_input(
        "Bid-ask spread (%)",
        "The gap between the buying and selling price. Half of it is paid on "
        "each side of a round trip, so 0.1 costs 0.05 per fill.",
    )
    slippage = cost_input(
        "Slippage (%) per fill",
        "Adverse price movement between the decision and the fill. Charged in "
        "full on each side.",
    )

    drag = round_trip_drag(commission, spread, slippage)
    if drag:
        st.sidebar.caption(f"One round trip now costs {drag:.3%} of the "
                           f"position before the price moves at all.")

    config = BacktestConfig(
        ticker=ticker,
        start=start,
        end=end,
        initial_cash=float(cash),
        strategy=strategy,
        commission=commission,
        spread=spread,
        slippage=slippage,
    )

    issues = config.problems()
    for issue in issues:
        st.sidebar.error(issue)

    run = st.sidebar.button(
        "Run Backtest",
        type="primary",
        disabled=bool(issues),
        width="stretch",
    )
    return config, run


def execute(config: BacktestConfig) -> Dict[str, Any]:
    """Run one backtest and gather everything the page needs to show it.

    No calculation happens here. The prices come from the cached data layer, the
    simulation from the Backtester and every figure from performance_report, so
    the dashboard cannot drift away from what the command-line scripts report.

    Args:
        config: The configuration to run.

    Returns:
        A dict holding the prices, the BacktestResult and the report.

    Raises:
        ValueError: Propagated from the data layer for an unknown ticker or an
            empty date range, and from the engine for too short a history.
    """
    prices = get_price_data(
        config.ticker.strip().upper(),
        config.start.isoformat(),
        config.end.isoformat(),
    )
    if len(prices) < MINIMUM_BARS:
        raise ValueError(
            f"Only {len(prices)} trading days in this range. At least "
            f"{MINIMUM_BARS} are needed for an indicator to warm up and still "
            f"have room to trade."
        )

    result = Backtester(
        prices,
        initial_cash=config.initial_cash,
        strategy=config.strategy,
        commission=config.commission,
        spread=config.spread,
        slippage=config.slippage,
    ).run()

    report = performance_report(
        equity=result.equity_curve["total_value"],
        benchmark=result.benchmark_curve,
        trade_log=result.trade_log,
        positions=result.equity_curve["shares"],
    )

    return {"prices": prices, "result": result, "report": report,
            "config": config}


def render_headline(report: pd.DataFrame) -> None:
    """Show the four figures a reader looks at first.

    Each is presented against the benchmark's own value, because none of them
    means anything in isolation: a 13% return is good or bad only relative to
    what holding the asset would have produced over the same days.

    Args:
        report: A report as returned by performance_report.
    """
    cards = (
        ("Total return", TOTAL_RETURN, "{:.2%}", True),
        ("Sharpe ratio", SHARPE, "{:.3f}", True),
        ("Max drawdown", MAX_DRAWDOWN, "{:.2%}", True),
        ("Exposure", EXPOSURE, "{:.2%}", False),
    )

    for column, (title, metric, template, compare) in zip(st.columns(4), cards):
        value = report.loc[metric, STRATEGY_COLUMN]
        reference = report.loc[metric, BENCHMARK_COLUMN]
        comparable = compare and pd.notna(value) and pd.notna(reference)

        # The delta carries the sign of the difference, which Streamlit colours
        # green when positive. That reads correctly for all three compared
        # metrics, including the drawdown: a strategy that fell 9% against the
        # benchmark's 31% gives a positive difference, and a shallower drawdown
        # is indeed the better outcome. The benchmark's own figure goes in the
        # tooltip, since a difference alone cannot say what it was measured from.
        column.metric(
            title,
            "n/a" if pd.isna(value) else template.format(value),
            delta=(f"{template.format(value - reference)} vs buy & hold"
                   if comparable else None),
            help=(f"Buy and hold: {template.format(reference)}"
                  if pd.notna(reference) else
                  "Not defined for the benchmark, which is always fully "
                  "invested."),
        )


def render_charts(state: Dict[str, Any]) -> None:
    """Render the three figures, each under the question it answers.

    The captions are not decoration. A chart that a reader cannot immediately
    connect to a question is the thing the guideline warns against, and stating
    the question above the figure is the cheapest way to prevent it.

    Args:
        state: The dict returned by execute.
    """
    result: BacktestResult = state["result"]
    equity = result.equity_curve["total_value"]

    st.subheader("Did the strategy beat buying and holding?")
    st.caption("Both curves start from the same capital, so they can be read "
               "against each other directly. Hover anywhere to see both values "
               "on that date; drag to zoom, double-click to reset.")
    st.plotly_chart(equity_curve_chart(equity, result.benchmark_curve),
                    width="stretch")

    st.subheader("How bad did the losses get, and how long did they last?")
    st.caption("Distance below zero is how far the portfolio sat under its own "
               "previous best. Width matters as much as depth: a shallow "
               "decline that takes years to recover is its own kind of "
               "expensive.")
    st.plotly_chart(drawdown_chart(equity, result.benchmark_curve),
                    width="stretch")

    st.subheader("Where did the strategy actually act?")
    st.caption("Green triangles mark entries, red triangles exits, placed at "
               "the closing price of the bar they were filled on. Hover for "
               "the size of each order.")
    st.plotly_chart(trade_marker_chart(state["prices"]["Close"],
                                       result.trade_log),
                    width="stretch")


def render_results(state: Dict[str, Any]) -> None:
    """Render a completed backtest.

    Args:
        state: The dict returned by execute.
    """
    config: BacktestConfig = state["config"]
    result: BacktestResult = state["result"]
    report: pd.DataFrame = state["report"]

    st.subheader(f"{config.ticker.strip().upper()} · {config.start} to "
                 f"{config.end} · {len(state['prices'])} bars")
    charged = (f"commission {config.commission:.3%}, spread "
               f"{config.spread:.3%}, slippage {config.slippage:.3%}"
               if config.charges_costs else "no transaction costs")
    st.caption(f"{config.strategy!r} · initial cash "
               f"{config.initial_cash:,.2f} · {len(result.trade_log)} orders "
               f"filled · {charged}")

    render_headline(report)

    if config.charges_costs:
        paid = sum(order["cost"] for order in result.trade_log)
        st.info(
            f"Costs are charged to the strategy only. The buy-and-hold "
            f"benchmark is shown uncharged, the same convention "
            f"run_cost_scenarios.py uses, which makes every comparison above "
            f"conservative for the strategy: it pays to trade while the "
            f"benchmark's single notional purchase does not. Frictions "
            f"consumed {paid:,.2f} across {len(result.trade_log)} orders, "
            f"{paid / config.initial_cash:.2%} of the starting capital."
        )

    # report_caveats reads warnings off the report's own contents. It was
    # written to return plain sentences precisely so a UI could render them as
    # banners instead of scraping them out of preformatted text.
    for note in report_caveats(report):
        st.warning(note)

    # The charts come before the table because the guideline calls the equity
    # curve the single most important figure, and because a shape is read faster
    # than thirty numbers. The table stays underneath for anyone who wants the
    # figure behind the shape.
    render_charts(state)

    st.subheader("Performance report")
    st.caption("Strategy against an uncharged buy-and-hold benchmark over the "
               "same dates.")

    # format_report is reused rather than reimplemented, so this table is
    # character-for-character what the command-line scripts print. Formatting a
    # cell correctly depends on which metric it belongs to, and that knowledge
    # already lives in the analytics package.
    st.code(format_report(report), language=None)

    with st.expander(f"Trade log ({len(result.trade_log)} orders)"):
        if not result.trade_log:
            st.write("The strategy never traded over this period.")
        else:
            log = pd.DataFrame(result.trade_log)
            st.dataframe(
                log[["date", "action", "quoted_price", "price", "shares",
                     "cost"]],
                hide_index=True,
                width="stretch",
            )
            st.caption("quoted_price is the market close, price the all-in "
                       "fill the account actually paid or received, and cost "
                       "the gap between them. The two prices coincide only "
                       "when every friction is zero.")


def main() -> None:
    """Draw the page."""
    st.set_page_config(page_title="Backtesting dashboard", layout="wide")

    st.title("Strategy backtesting dashboard")
    st.caption("Classical technical strategies measured against buy-and-hold. "
               "Configure a run in the sidebar, then press Run Backtest.")

    config, run = sidebar()

    # Streamlit reruns this script on every widget interaction, so a result held
    # only in a local variable would vanish the moment anything is touched.
    # Keeping it in session state lets the report stay on screen while the user
    # reads it and adjusts the next run's inputs.
    if run:
        try:
            with st.spinner("Running backtest..."):
                st.session_state["last_run"] = execute(config)
            st.session_state.pop("error", None)
        except (ValueError, TypeError) as error:
            st.session_state.pop("last_run", None)
            st.session_state["error"] = str(error)

    if "error" in st.session_state:
        st.error(st.session_state["error"])

    if "last_run" in st.session_state:
        render_results(st.session_state["last_run"])
    elif "error" not in st.session_state:
        st.info("Press **Run Backtest** in the sidebar to begin. Every input "
                "already has a sensible default, so it will run as it stands.")


if __name__ == "__main__":
    main()
