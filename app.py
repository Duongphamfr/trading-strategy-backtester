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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from analytics.report import (
    BENCHMARK_COLUMN,
    BETA,
    CONDITIONAL_VAR,
    EXCESS_KURTOSIS,
    EXPOSURE,
    HISTORICAL_VAR,
    LOW_EXPOSURE_THRESHOLD,
    MAX_DRAWDOWN,
    PARAMETRIC_VAR,
    PERCENT_METRICS,
    R_SQUARED,
    SHARPE,
    SKEWNESS,
    STRATEGY_COLUMN,
    TOTAL_RETURN,
    VOLATILITY,
    format_report,
    performance_report,
    report_caveats,
)
from analytics.validation import (
    FAST_WINDOWS,
    PLATEAU_TOLERANCE,
    SLOW_WINDOWS,
    assess,
    cell_neighbours,
    metric_grid,
    reference_value,
    sweep,
)
from data.market_data import (DataSourceUnavailable, get_price_data,
                              period_label)
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
    parameter_heatmap_chart,
    returns_distribution_chart,
    trade_marker_chart,
)

DEFAULT_TICKER = "AAPL"
DEFAULT_START = date(2020, 1, 1)
DEFAULT_END = date(2023, 1, 1)
DEFAULT_CASH = 10_000.0

# A short, deliberately opinionated menu, so that trying the dashboard does not
# start with recalling that Apple is AAPL. Names are carried beside the symbols
# for the same reason: a bare symbol list still assumes the knowledge it is meant
# to spare the reader.
#
# Chosen for liquidity and for spread across sectors, since a backtest of one
# mega-cap tech name after another teaches very little. The two index ETFs matter
# most of all: a single stock flatters trend-following, and comparing a strategy on
# SPY against the same strategy on TSLA is the quickest way to see how much of a
# result belongs to the asset rather than to the rules.
#
# The list is not exhaustive and is not meant to be. Anything Yahoo serves is
# reachable through the custom entry below.
PRESET_TICKERS: Dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth",
    "KO": "Coca-Cola",
    "PG": "Procter & Gamble",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "CAT": "Caterpillar",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
}

# The sentinel that reveals the free-text field. It is a member of the options
# list rather than a separate checkbox so that picking a preset and entering
# something exotic are visibly the same decision, made in one place.
#
# WHY NOT accept_new_options, WHICH IS THE NATIVE WAY TO DO THIS
# st.selectbox grew an accept_new_options flag that turns it into an editable
# combobox, and on the page it is plainly nicer than a sentinel: one widget, no
# reveal step. It is rejected here for a reason outside the browser. Streamlit's
# own AppTest harness cannot drive the off-list path in this version: both
# set_value and a direct session_state assignment raise "not in list", because the
# proxy resolves a value by looking up its index among the declared options. That
# would leave the custom-symbol path with no automated coverage at all, including
# the invalid-ticker and lost-connection banners that path has to reach.
#
# A sentinel keeps every state reachable from a test, since the sentinel itself is
# a declared option and the revealed field is an ordinary text input. Little is
# given up: selectbox already filters its options as you type, so a preset is
# still found by typing three letters of it. Worth revisiting if a later
# Streamlit teaches AppTest to type into a combobox.
CUSTOM_CHOICE = "Custom…"

TICKER_CHOICES: Tuple[str, ...] = (*PRESET_TICKERS, CUSTOM_CHOICE)

MA_LABEL = "MA Crossover"
RSI_LABEL = "RSI Mean Reversion"
MOMENTUM_LABEL = "Momentum"

# Enough bars for the longest indicator warm-up plus something to trade on.
# Shorter ranges run without error but produce an all-HOLD signal series and an
# empty report, which looks like a bug rather than a short history.
MINIMUM_BARS = 60

# WHY SOME DELTA BADGES GO GREY WHEN THE STRATEGY SAT IN CASH
#
# A metric card shows the gap to buy-and-hold and colours it green when the gap
# favours the strategy. That colouring is a claim, and for some metrics the claim
# stops being true once exposure collapses, at which point a green badge would
# contradict the caveat banner printed directly beneath it.
#
# Two different mechanisms are at work, which is why there are two sets rather
# than one. Both suppress the colour; they do so for opposite reasons, and the
# tooltip says which applies.
#
# CASH_DILUTED holds metrics computed as an average, a quantile or a regression
# over every bar. Flat bars enter that sample as exact zeros and drag the result
# toward zero, so the number is describing the cash the strategy was holding
# rather than the strategy itself. These are the rows report_caveats names.
#
# CASH_FLATTERED holds metrics whose value is honest but whose comparison is not.
# A drawdown depth is a path extremum, not an average, so flat bars cannot dilute
# it: the portfolio really did fall only that far. What does not survive scrutiny
# is the credit for it. Falling less than a fully invested benchmark, while
# holding cash on nine bars in ten, measures absence from the market and not
# better risk control, so the achievement badge is withdrawn even though the
# figure stands.
#
# Total Return, Sharpe and Sortino keep their colours. A return is a path
# endpoint that flat bars leave untouched, and in the ratios the dilution hits
# numerator and denominator together rather than one side only.
CASH_DILUTED = frozenset({
    VOLATILITY, HISTORICAL_VAR, PARAMETRIC_VAR, CONDITIONAL_VAR,
    SKEWNESS, EXCESS_KURTOSIS, BETA, R_SQUARED,
})
CASH_FLATTERED = frozenset({MAX_DRAWDOWN})

DILUTED_REASON = ("Badge greyed out: with exposure below "
                  "{threshold:.0%} this figure is mostly describing the cash "
                  "the strategy held, since flat bars enter it as zeros.")
FLATTERED_REASON = ("Badge greyed out: the depth itself is real, but beating a "
                    "fully invested benchmark on it while in cash below "
                    "{threshold:.0%} exposure reflects absence from the market "
                    "rather than better risk control.")


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


def ticker_label(choice: str) -> str:
    """Render one dropdown entry: the symbol, and whose it is.

    Args:
        choice: A preset symbol, or the custom sentinel.

    Returns:
        The text shown in the dropdown. The sentinel is passed through unchanged,
        having no company behind it.
    """
    company = PRESET_TICKERS.get(choice)
    return f"{choice} · {company}" if company else choice


def resolve_ticker(choice: str, typed: str) -> str:
    """Reduce the two ticker widgets to the one string the engine receives.

    The dashboard offers two ways to name an asset and the engine knows about
    none of them: BacktestConfig carries a symbol, exactly as it did when this was
    a single text field. Keeping that reduction in a function of its own, rather
    than inline in the sidebar, is what allows every combination to be tested
    without rendering a page.

    Typed input is upper-cased here so that the normalised symbol is what the
    config carries, and therefore what the results heading shows. The data layer
    normalises again on its own behalf, which is left alone: it protects the
    command-line scripts, which never pass through this function.

    An empty custom field deliberately resolves to the empty string rather than
    falling back to a preset. That is what keeps the Run button disabled with
    "Enter a ticker symbol." instead of quietly backtesting an asset the reader
    did not choose.

    Args:
        choice: The dropdown selection: a preset symbol or CUSTOM_CHOICE.
        typed: Contents of the custom field. Ignored unless the sentinel is
            selected, so switching back to a preset cannot resurrect stale text.

    Returns:
        The symbol to backtest, upper-cased and stripped, possibly empty.
    """
    if choice == CUSTOM_CHOICE:
        return typed.strip().upper()
    return choice


def ticker_input() -> str:
    """Draw the ticker chooser and return the symbol it names.

    Returns:
        The symbol to backtest, empty when the custom field is still blank.
    """
    choice = st.sidebar.selectbox(
        "Ticker",
        options=TICKER_CHOICES,
        index=TICKER_CHOICES.index(DEFAULT_TICKER),
        format_func=ticker_label,
        key="ticker_choice",
        help="Pick a liquid name, or choose Custom… to enter any other symbol.",
    )

    typed = ""
    if choice == CUSTOM_CHOICE:
        typed = st.sidebar.text_input(
            "Custom ticker",
            key="custom_ticker",
            placeholder="e.g. BRK-B, ^GSPC, ASML.AS",
            help="Yahoo Finance notation. Case does not matter, and an unknown "
                 "symbol is reported rather than guessed at.",
        )

    return resolve_ticker(choice, typed)


def sidebar() -> Tuple[BacktestConfig, bool]:
    """Draw the whole sidebar and collect it into a configuration.

    Returns:
        The BacktestConfig described by the current widget values, and whether
        the run button was pressed on this rerun.
    """
    st.sidebar.title("Backtest setup")

    st.sidebar.subheader("Data")
    ticker = ticker_input()
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
        DataSourceUnavailable: Propagated from the data layer when the download
            could not be made, as distinct from being made and coming back empty.
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


def delta_label(metric: str, difference: float) -> str:
    """Render a strategy-minus-benchmark difference in the unit it is really in.

    THE UNIT CHANGES WHEN YOU SUBTRACT, AND THE SYMBOL HAS TO FOLLOW
    A return of 41.71% less a return of 329.44% is not a return of -287.73%. It
    is a difference of 287.73 percentage points, and rendering it with the same
    "%" the two returns carry states something impossible: a long-only position
    cannot lose more than all of its capital, so no reader can take -287.73% at
    face value and still be reading about a return.

    On a benchmark that returned 74% the arithmetic never left the plausible
    range, which is why the mislabelling survived so long. A high-return asset
    exposes it immediately.

    Percentage points fix the symbol without touching the number. The subtraction
    was always the right quantity and is unchanged; only its rendering moves from
    a unit it never had to the one it always did.

    Which metrics need it is read from PERCENT_METRICS rather than listed here, so
    a card added later inherits the right unit from the same table that decides
    how its value is printed.

    Args:
        metric: The report row label, used to decide the unit.
        difference: Strategy value minus benchmark value, in the metric's own
            units, so a ratio for a percentage metric.

    Returns:
        A label ready for st.metric's delta, keeping the leading sign that
        Streamlit reads to choose the arrow's direction.
    """
    if metric in PERCENT_METRICS:
        return f"{difference * 100:+.2f} pp vs buy & hold"
    return f"{difference:+.3f} vs buy & hold"


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

    cash_heavy = is_cash_heavy(report)

    for column, (title, metric, template, compare) in zip(st.columns(4), cards):
        value = report.loc[metric, STRATEGY_COLUMN]
        reference = report.loc[metric, BENCHMARK_COLUMN]
        comparable = compare and pd.notna(value) and pd.notna(reference)

        reason = delta_disclaimer(metric) if cash_heavy else None

        # The delta carries the sign of the difference, which Streamlit colours
        # green when positive and red when negative. That colouring is an
        # editorial claim, so it is switched off for any metric whose claim the
        # caveat banner below would immediately dispute. delta_color="off"
        # keeps the number and drops the colour, leaving the reader the figure
        # without the verdict.
        if pd.notna(reference):
            tooltip = f"Buy and hold: {template.format(reference)}"
            if reason and comparable:
                tooltip = f"{tooltip}. {reason}"
        else:
            tooltip = ("Not defined for the benchmark, which is always fully "
                       "invested.")

        column.metric(
            title,
            "n/a" if pd.isna(value) else template.format(value),
            delta=(delta_label(metric, value - reference)
                   if comparable else None),
            delta_color="off" if reason else "normal",
            help=tooltip,
        )


def is_cash_heavy(report: pd.DataFrame,
                  threshold: float = LOW_EXPOSURE_THRESHOLD) -> bool:
    """Whether exposure is low enough to qualify how the report reads.

    The threshold is imported from analytics.report rather than restated here,
    so the greyed-out badges and the caveat banner can never disagree about when
    a report has become a description of cash. Moving the threshold in one place
    moves both.

    Args:
        report: A report as returned by performance_report.
        threshold: Exposure below which the qualification applies.

    Returns:
        True when exposure is present, known and below the threshold.
    """
    if EXPOSURE not in report.index or STRATEGY_COLUMN not in report.columns:
        return False

    exposure = report.loc[EXPOSURE, STRATEGY_COLUMN]
    return bool(pd.notna(exposure) and exposure < threshold)


def delta_disclaimer(metric: str,
                     threshold: float = LOW_EXPOSURE_THRESHOLD) -> Optional[str]:
    """The reason this metric's delta badge should not be coloured, if any.

    Args:
        metric: A row label from the report.
        threshold: Exposure threshold, quoted back in the message.

    Returns:
        A sentence for the tooltip, or None when the metric's comparison
        survives low exposure and keeps its colour.
    """
    if metric in CASH_DILUTED:
        return DILUTED_REASON.format(threshold=threshold)
    if metric in CASH_FLATTERED:
        return FLATTERED_REASON.format(threshold=threshold)
    return None


def render_charts(state: Dict[str, Any]) -> None:
    """Render the three figures, each under the question it answers.

    The captions are not decoration. A chart that a reader cannot immediately
    connect to a question is the thing the guideline warns against, and stating
    the question above the figure is the cheapest way to prevent it.

    Args:
        state: The dict returned by execute.
    """
    result: BacktestResult = state["result"]
    report: pd.DataFrame = state["report"]
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

    st.subheader("Are the returns normal, or do they have fat tails?")
    st.caption("Bars are the strategy's own returns, the dashed line a normal "
               "distribution fitted to the same mean and standard deviation. A "
               "taller peak with returns poking out past the ends of the curve "
               "is excess kurtosis, the fat tails that make Parametric VaR too "
               "optimistic. A histogram leaning to one side of the curve is "
               "skew. The subtitle repeats the two Distribution rows of the "
               "table below.")

    # The exposure threshold governs this note as it governs the caveat banner
    # and the greyed-out badges, so all three appear and disappear together
    # rather than each drawing its own line.
    if is_cash_heavy(report):
        st.info(
            "Exposure is low, so most bars are flat and their returns are "
            "exactly zero. That produces the single tall spike at the centre, "
            "and it is an artifact of sitting in cash rather than a property "
            "of the strategy's trading. The fitted curve is dragged toward "
            "that spike too, so read this chart as a description of a mostly "
            "idle account. The share of returns that are exactly zero is "
            "printed under the title. A strategy that stays invested produces "
            "a far more informative shape here."
        )

    st.plotly_chart(returns_distribution_chart(equity), width="stretch")

    render_sweep(state)


# The sweep runs one backtest per grid cell, so it is the only expensive thing
# the dashboard does. Caching on the inputs means a second look costs nothing,
# and the button means it is never paid for by someone who only wanted the
# equity curve. Both are needed: the cache alone would still charge the first
# view of every new ticker or cost setting.
@st.cache_data(show_spinner=False)
def cached_sweep(prices: pd.DataFrame, initial_cash: float, fast_windows: Tuple,
                 slow_windows: Tuple, commission: float, spread: float,
                 slippage: float, enter_on_existing_trend: bool) -> pd.DataFrame:
    """Run the parameter sweep, memoised on its inputs.

    The prices are passed in rather than re-fetched so the grid is computed on
    exactly the bars the displayed backtest used. Streamlit hashes the frame to
    build the cache key, which also means a forced data refresh invalidates the
    sweep automatically.

    Every argument is one the grid genuinely depends on, which is what makes the
    memoisation safe: an input left out of this signature would be an input the
    cache ignores, and the heatmap would go on showing a grid computed under the
    previous setting.

    Args:
        prices: OHLCV frame to sweep over.
        initial_cash: Starting capital.
        fast_windows: Fast windows to try. A tuple, since the cache key must be
            hashable.
        slow_windows: Slow windows to try.
        commission: Proportional commission per trade, as a fraction.
        spread: Bid-ask spread as a fraction of price.
        slippage: Adverse price move as a fraction of price, per side.
        enter_on_existing_trend: The warm-up boundary setting of the run being
            described, so every cell measures the same strategy the report does.

    Returns:
        The sweep result, as returned by analytics.validation.sweep.
    """
    return sweep(prices, initial_cash, fast_windows=fast_windows,
                 slow_windows=slow_windows, commission=commission,
                 spread=spread, slippage=slippage,
                 enter_on_existing_trend=enter_on_existing_trend)


def sweep_windows(selected: int, grid: Sequence[int]) -> Tuple[int, ...]:
    """The standard grid with the user's own value folded in.

    The sidebar sliders step in fives while the sweep grid steps in tens, so the
    combination actually being run is usually not a cell of the standard grid.
    Rather than silently snapping the marker to the nearest cell, which would
    show the user a score that is not theirs, the exact value is added to the
    axis. It costs one extra row or column of backtests and makes the marker
    truthful.

    Args:
        selected: The value the user chose.
        grid: The standard axis for this dimension.

    Returns:
        The sorted union of the two, without duplicates.
    """
    return tuple(sorted(set(grid) | {int(selected)}))


def render_sweep(state: Dict[str, Any]) -> None:
    """Render the parameter sweep section, for the MA strategy only.

    The heatmap answers a question none of the other figures can: whether the
    result just shown was a property of the rule or an accident of two numbers.
    It is specific to the moving average crossover, which is the strategy whose
    parameters Phase 4 found an isolated peak for, so it is not offered at all
    for the other two.

    Args:
        state: The dict returned by execute.
    """
    config: BacktestConfig = state["config"]
    strategy = config.strategy
    if not isinstance(strategy, MovingAverageCrossover):
        return

    fast = int(strategy.fast_window)
    slow = int(strategy.slow_window)

    st.subheader("Is this parameter choice robust, or an isolated peak?")
    st.caption(
        f"Every cell is a full backtest of one fast and slow pair over the same "
        f"dates, costs and warm-up setting, coloured against the benchmark's "
        f"own Sharpe: warm "
        f"beats buy-and-hold, cool loses to it, blank means the fast window was "
        f"not shorter than the slow one. Your {fast}/{slow} is outlined. A good "
        f"score surrounded by other good scores is a plateau worth trusting; one "
        f"surrounded by poor scores was luck. This runs a few hundred backtests "
        f"and takes a moment the first time, then it is cached."
    )

    # The opt-in is remembered so the heatmap survives the reruns that any
    # widget interaction triggers, exactly as the report does. It is deliberately
    # forgotten when a new backtest is run, because that may be a different
    # ticker or period where the sweep is a fresh multi-second cost, and
    # incurring that silently would defeat the point of the button.
    if st.button("Run parameter sweep", key="sweep_button"):
        st.session_state["sweep_requested"] = True

    if not st.session_state.get("sweep_requested"):
        st.caption("Not run yet.")
        return

    # The warm-up setting is read off the strategy object rather than off the
    # sidebar, so it is by construction the one the displayed report was produced
    # under. Sweeping under any other value would put a different strategy's
    # score in the cell the caption labels as the user's own.
    with st.spinner("Sweeping the fast and slow window grid..."):
        results = cached_sweep(
            state["prices"],
            config.initial_cash,
            sweep_windows(fast, FAST_WINDOWS),
            sweep_windows(slow, SLOW_WINDOWS),
            config.commission,
            config.spread,
            config.slippage,
            bool(strategy.enter_on_existing_trend),
        )

    grid = metric_grid(results, SHARPE)
    benchmark_sharpe = reference_value(results, SHARPE)

    st.plotly_chart(
        parameter_heatmap_chart(
            grid,
            center=benchmark_sharpe,
            selected=(fast, slow),
            subtitle=(f"{int(grid.notna().to_numpy().sum())} valid combinations · "
                      f"colour centred on the benchmark's Sharpe of "
                      f"{benchmark_sharpe:.3f}"),
        ),
        width="stretch",
    )

    render_robustness(grid, fast, slow, benchmark_sharpe)


def render_robustness(grid: pd.DataFrame, fast: int, slow: int,
                      benchmark_sharpe: float) -> None:
    """Print the robustness read for the selected cell.

    assess is reused for the grid-wide picture, and cell_neighbours for the
    user's own cell, which assess cannot describe because it only ever looks at
    the best one.

    Args:
        grid: The Sharpe grid.
        fast: Selected fast window.
        slow: Selected slow window.
        benchmark_sharpe: The benchmark's Sharpe over the same dates.
    """
    summary = assess(grid, benchmark_sharpe)
    own = float(grid.loc[fast, slow])
    neighbours = cell_neighbours(grid, fast, slow)
    neighbour_mean = (sum(neighbours) / len(neighbours) if neighbours
                      else float("nan"))

    left, right = st.columns(2)
    left.metric(f"Your {fast}/{slow}", f"{own:.3f}",
                delta=f"{own - benchmark_sharpe:+.3f} vs buy & hold")
    right.metric(f"Grid best {summary.best_fast}/{summary.best_slow}",
                 f"{summary.best_score:.3f}",
                 delta=f"{summary.best_score - own:+.3f} vs your choice",
                 delta_color="off")

    st.markdown(verdict(own, neighbour_mean, len(neighbours), summary))


def verdict(own: float, neighbour_mean: float, neighbour_count: int,
            summary: Any) -> str:
    """Compose the sentences interpreting the selected cell.

    The plateau test compares the neighbourhood's mean against the cell's own
    score using PLATEAU_TOLERANCE, the same band run_parameter_sweep uses to
    decide whether the grid's best score sits on a plateau. Sharing it keeps the
    dashboard's verdict and the command-line script's verdict on one definition.

    Args:
        own: Score of the selected cell.
        neighbour_mean: Mean score of its valid neighbours.
        neighbour_count: How many neighbours were valid.
        summary: The Robustness summary for the whole grid.

    Returns:
        Markdown text.
    """
    lines = [
        f"The {neighbour_count} combinations adjacent to yours average "
        f"**{neighbour_mean:.3f}**, against your own **{own:.3f}**. "
        f"Across the whole grid the median is {summary.median_score:.3f} and "
        f"{summary.beats_reference:.0%} of combinations beat the benchmark."
    ]

    if not neighbour_count or pd.isna(neighbour_mean):
        lines.append("Too few valid neighbours to judge the neighbourhood.")
    elif own <= 0:
        lines.append(
            "Your combination does not clear zero, so robustness is beside the "
            "point: there is no edge here to be robust."
        )
    elif neighbour_mean >= own * (1.0 - PLATEAU_TOLERANCE):
        lines.append(
            "The neighbours hold up, so this sits on a plateau rather than a "
            "spike. That is the outcome to want: it suggests the score came "
            "from the rule and would survive windows chosen a little "
            "differently."
        )
    elif neighbour_mean >= own * 0.5:
        lines.append(
            "The neighbours fall off noticeably. The score is not a single "
            "lucky cell, but it does depend on the windows more than a robust "
            "setting should."
        )
    else:
        lines.append(
            "The neighbours collapse, so this is an isolated peak. A score that "
            "vanishes when either window moves by one step is a property of "
            "this particular history, not of the strategy, and it is exactly "
            "what the walk-forward analysis found does not survive out of "
            "sample."
        )

    if summary.best_score > own:
        lines.append(
            f"The grid's best cell scores {summary.best_score - own:+.3f} more "
            f"than yours. Chasing it is the trap the whole exercise is meant to "
            f"expose: that cell was chosen by looking at this history, and no "
            f"trader had it in advance."
        )

    return "\n\n".join(lines)


def render_results(state: Dict[str, Any]) -> None:
    """Render a completed backtest.

    Args:
        state: The dict returned by execute.
    """
    config: BacktestConfig = state["config"]
    result: BacktestResult = state["result"]
    report: pd.DataFrame = state["report"]

    # Dates read off the bars rather than off the date pickers, so a ticker whose
    # listing postdates the chosen start reports the period it actually covers.
    period = period_label(state["prices"], config.start.isoformat(),
                          config.end.isoformat())
    st.subheader(f"{config.ticker.strip().upper()} · {period} · "
                 f"{len(state['prices'])} bars")
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
    # The three types caught here are the three ways a run can fail for a reason
    # the user can act on, and each already carries a sentence written for them.
    # DataSourceUnavailable is the one this handler used to miss: the transport's
    # exception escaped, and Streamlit answered a lost connection with a traceback
    # pointing into yfinance. It is listed by name rather than reached through its
    # OSError base so that a genuine filesystem failure, which the user cannot fix
    # by retrying, still surfaces.
    #
    # Nothing broader is caught on purpose. An AttributeError or a KeyError here
    # would be a bug in this project, and turning it into a tidy banner would only
    # make it harder to find.
    if run:
        try:
            with st.spinner("Running backtest..."):
                st.session_state["last_run"] = execute(config)
            st.session_state.pop("error", None)
            st.session_state.pop("sweep_requested", None)
        except (DataSourceUnavailable, ValueError, TypeError) as error:
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
