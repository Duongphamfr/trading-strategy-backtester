"""Automated tests for the engine: Portfolio, Broker and Backtester.

These are the checks that used to live in the __main__ blocks of the engine
modules, turned into assertions. The difference matters: a print statement in a
__main__ block only catches a regression if somebody happens to run that file and
read the numbers, whereas a failing assertion stops a commit.

Wherever a number was verified by hand earlier in the project, that same number
is the expected value here rather than a fresh one read off a run. Several of the
expectations are exact identities, derived on paper and asserted to floating
point precision, which is stronger than a tolerance chosen to make a test pass:

    a buy-and-hold strategy through the full engine must land on the analytic
    buy-and-hold value, because the benchmark is computed from the price series
    and never touches the Portfolio;

    a round trip at a flat quote must retain exactly
    (1 - h)(1 - c) / ((1 + h)(1 + c)) of the capital, where h is half the spread
    plus slippage and c the commission, since the market contributed nothing;

    the naive share sizing the Broker was fixed to avoid must overdraw by
    exactly commission times cash.
"""

import pandas as pd
import pytest

from constants import BUY, HOLD, SELL
from engine.backtester import Backtester
from engine.broker import MIN_ORDER_VALUE, Broker
from engine.portfolio import Portfolio


class BuyFirstBarAndHold:
    """Buys everything on the first bar and never sells.

    The engine-level equivalent of buy-and-hold, and the strategy the Phase 1
    correctness check rests on.
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(HOLD, index=data.index, dtype=object)
        signals.iloc[0] = BUY
        return signals


class FlipEvery:
    """Alternates fully invested and fully in cash on a fixed bar schedule.

    Not a strategy anyone would trade. It turns the portfolio over a known number
    of times whatever prices do, which is what makes cost drag predictable.
    """

    def __init__(self, period: int = 2) -> None:
        self.period = period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(HOLD, index=data.index, dtype=object)
        for position in range(0, len(data), self.period):
            signals.iloc[position] = BUY if (position // self.period) % 2 == 0 else SELL
        return signals


class BuyAboveThreshold:
    """Holds while the close is above a level and sits in cash below it.

    Strictly causal: the signal on bar T reads bar T's close and nothing else.
    That is what makes it usable for testing the engine's own causality, since
    any dependence on future bars would have to come from the engine rather than
    from the strategy.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        above = data["Close"] > self.threshold
        return pd.Series(
            [BUY if flag else SELL for flag in above],
            index=data.index,
            dtype=object,
        )


class AlwaysReturnsGarbage:
    """Returns a signal outside the BUY/SELL/HOLD vocabulary."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series("MAYBE", index=data.index, dtype=object)


def retained_fraction(commission: float, spread: float, slippage: float) -> float:
    """Fraction of capital surviving one round trip at an unchanged quote.

    Derived rather than measured. Buying pays price * (1 + h) * (1 + c) per share
    and selling receives price * (1 - h) * (1 - c), where h is half the spread
    plus the slippage, so the price cancels and the ratio is all that is left.

    Args:
        commission: Proportional commission per trade.
        spread: Bid-ask spread as a fraction of price.
        slippage: Adverse price move as a fraction of price, per side.

    Returns:
        The fraction of the starting cash left after buying and selling once.
    """
    half = spread / 2.0 + slippage
    return ((1.0 - half) * (1.0 - commission)) / ((1.0 + half) * (1.0 + commission))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def test_portfolio_starts_as_all_cash_and_no_position(initial_cash):
    portfolio = Portfolio(initial_cash)

    assert portfolio.cash == initial_cash
    assert portfolio.shares == 0.0
    assert portfolio.total_value(price=100.0) == initial_cash


def test_buy_deducts_cost_from_cash_and_adds_shares(initial_cash):
    portfolio = Portfolio(initial_cash)

    portfolio.buy(shares=50.0, price=100.0)

    assert portfolio.cash == 5_000.0
    assert portfolio.shares == 50.0


def test_sell_adds_proceeds_to_cash_and_reduces_shares(initial_cash):
    portfolio = Portfolio(initial_cash)
    portfolio.buy(shares=50.0, price=100.0)

    portfolio.sell(shares=20.0, price=120.0)

    # 5,000 left after the buy, plus 20 shares at 120.
    assert portfolio.cash == 7_400.0
    assert portfolio.shares == 30.0


def test_total_value_is_cash_plus_marked_position(initial_cash):
    portfolio = Portfolio(initial_cash)
    portfolio.buy(shares=50.0, price=100.0)
    portfolio.sell(shares=20.0, price=120.0)

    # 7,400 of cash plus 30 shares marked at 120.
    assert portfolio.position_value(price=120.0) == 3_600.0
    assert portfolio.total_value(price=120.0) == 11_000.0


def test_round_trip_profit_is_the_price_gain_times_the_shares(initial_cash):
    portfolio = Portfolio(initial_cash)

    portfolio.buy(shares=100.0, price=100.0)
    portfolio.sell(shares=100.0, price=125.0)

    assert portfolio.shares == 0.0
    assert portfolio.cash == initial_cash + 100.0 * (125.0 - 100.0)


def test_buying_more_than_the_cash_allows_raises(initial_cash):
    portfolio = Portfolio(initial_cash)

    with pytest.raises(ValueError, match="Insufficient cash"):
        portfolio.buy(shares=101.0, price=100.0)

    # The rejected order must leave the books untouched.
    assert portfolio.cash == initial_cash
    assert portfolio.shares == 0.0


def test_spending_exactly_all_the_cash_is_allowed(initial_cash):
    portfolio = Portfolio(initial_cash)

    portfolio.buy(shares=100.0, price=100.0)

    assert portfolio.cash == pytest.approx(0.0, abs=1e-12)


def test_selling_more_shares_than_held_raises_no_shorting(initial_cash):
    portfolio = Portfolio(initial_cash)
    portfolio.buy(shares=10.0, price=100.0)

    with pytest.raises(ValueError, match="Short selling is not supported"):
        portfolio.sell(shares=11.0, price=100.0)

    assert portfolio.shares == 10.0


def test_selling_with_no_position_raises(initial_cash):
    portfolio = Portfolio(initial_cash)

    with pytest.raises(ValueError, match="Short selling is not supported"):
        portfolio.sell(shares=1.0, price=100.0)


@pytest.mark.parametrize("shares, price", [(0.0, 100.0), (-1.0, 100.0),
                                           (1.0, 0.0), (1.0, -100.0)])
def test_non_positive_quantities_and_prices_are_rejected(initial_cash, shares, price):
    portfolio = Portfolio(initial_cash)

    with pytest.raises(ValueError):
        portfolio.buy(shares=shares, price=price)


@pytest.mark.parametrize("capital", [0.0, -1.0])
def test_initial_capital_must_be_strictly_positive(capital):
    with pytest.raises(ValueError, match="Initial capital"):
        Portfolio(capital)


def test_history_records_one_row_per_recorded_bar(initial_cash):
    portfolio = Portfolio(initial_cash)
    portfolio.buy(shares=50.0, price=100.0)
    portfolio.record(pd.Timestamp("2022-01-03"), price=100.0)
    portfolio.sell(shares=20.0, price=120.0)
    portfolio.record(pd.Timestamp("2022-01-04"), price=120.0)

    history = portfolio.to_dataframe()

    assert list(history.columns) == ["price", "cash", "shares",
                                     "position_value", "total_value"]
    assert len(history) == 2
    assert history["total_value"].iloc[0] == 10_000.0
    assert history["total_value"].iloc[1] == 11_000.0


def test_empty_history_still_has_the_declared_schema(initial_cash):
    history = Portfolio(initial_cash).to_dataframe()

    assert history.empty
    assert list(history.columns) == ["price", "cash", "shares",
                                     "position_value", "total_value"]


# ---------------------------------------------------------------------------
# Broker, frictionless
# ---------------------------------------------------------------------------

def test_buy_all_invests_the_whole_balance(initial_cash):
    portfolio = Portfolio(initial_cash)
    broker = Broker(portfolio)

    shares = broker.buy_all(price=100.0)

    assert shares == 100.0
    assert portfolio.cash == pytest.approx(0.0, abs=1e-12)
    assert portfolio.shares == 100.0


def test_sell_all_liquidates_the_whole_position(initial_cash):
    portfolio = Portfolio(initial_cash)
    broker = Broker(portfolio)
    broker.buy_all(price=100.0)

    shares = broker.sell_all(price=125.0)

    assert shares == 100.0
    assert portfolio.shares == 0.0
    assert portfolio.cash == pytest.approx(12_500.0, rel=1e-12)


def test_sell_all_with_no_position_does_nothing(initial_cash):
    portfolio = Portfolio(initial_cash)
    broker = Broker(portfolio)

    assert broker.sell_all(price=125.0) == 0.0
    assert portfolio.cash == initial_cash


def test_buy_all_declines_an_order_below_the_minimum_value():
    portfolio = Portfolio(MIN_ORDER_VALUE / 100.0)
    broker = Broker(portfolio)

    assert broker.buy_all(price=100.0) == 0.0
    assert portfolio.shares == 0.0


@pytest.mark.parametrize("price", [0.0, -100.0])
def test_broker_rejects_non_positive_prices(initial_cash, price):
    broker = Broker(Portfolio(initial_cash))

    with pytest.raises(ValueError, match="strictly positive"):
        broker.buy_all(price=price)


# ---------------------------------------------------------------------------
# Broker, with transaction costs
# ---------------------------------------------------------------------------

COST_CASES = [
    (0.0, 0.0, 0.0),
    (0.001, 0.0, 0.0),
    (0.0, 0.002, 0.0),
    (0.0, 0.0, 0.001),
    (0.0025, 0.001, 0.0005),
    (0.05, 0.02, 0.01),
]


def test_zero_costs_are_the_default(initial_cash):
    explicit = Broker(Portfolio(initial_cash), commission=0.0, spread=0.0,
                      slippage=0.0)
    default = Broker(Portfolio(initial_cash))

    assert explicit.buy_fill_price(100.0) == default.buy_fill_price(100.0) == 100.0
    assert explicit.sell_fill_price(100.0) == default.sell_fill_price(100.0) == 100.0


@pytest.mark.parametrize("commission, spread, slippage", COST_CASES[1:])
def test_buy_fills_above_the_quote_and_sell_fills_below(commission, spread,
                                                        slippage, initial_cash):
    broker = Broker(Portfolio(initial_cash), commission=commission,
                    spread=spread, slippage=slippage)

    assert broker.buy_fill_price(100.0) > 100.0
    assert broker.sell_fill_price(100.0) < 100.0


@pytest.mark.parametrize("commission, spread, slippage", COST_CASES)
def test_buy_all_never_overdraws_the_portfolio(commission, spread, slippage,
                                               initial_cash):
    """The trap the Broker's sizing was written to avoid.

    Sizing from the fill price and charging commission afterwards leaves the
    account short by commission times cash. The share count must come from the
    all-in price so the outlay equals the cash exactly.
    """
    portfolio = Portfolio(initial_cash)
    broker = Broker(portfolio, commission=commission, spread=spread,
                    slippage=slippage)

    broker.buy_all(price=100.0)

    assert portfolio.cash >= -1e-9
    assert portfolio.cash == pytest.approx(0.0, abs=1e-9)


def test_naive_sizing_would_overdraw_by_exactly_commission_times_cash(initial_cash):
    """Documents the size of the bug, not just its absence.

    An exact identity: the wrong formula overspends by commission * cash, no
    more and no less, which is why the correct one divides by the all-in price.
    """
    commission = 0.001
    broker = Broker(Portfolio(initial_cash), commission=commission)

    # The fill price before the commission is folded in.
    fill_without_commission = broker.buy_fill_price(100.0) / (1.0 + commission)

    naive_shares = initial_cash / fill_without_commission
    naive_outlay = naive_shares * fill_without_commission * (1.0 + commission)

    correct_shares = initial_cash / broker.buy_fill_price(100.0)
    correct_outlay = correct_shares * broker.buy_fill_price(100.0)

    assert naive_outlay - initial_cash == pytest.approx(commission * initial_cash,
                                                        rel=1e-12)
    assert correct_outlay == pytest.approx(initial_cash, rel=1e-12)


@pytest.mark.parametrize("commission, spread, slippage", COST_CASES)
def test_round_trip_at_a_flat_quote_loses_only_the_costs(commission, spread,
                                                         slippage, initial_cash):
    """With the price unchanged, every penny of the change is a cost."""
    portfolio = Portfolio(initial_cash)
    broker = Broker(portfolio, commission=commission, spread=spread,
                    slippage=slippage)

    broker.buy_all(price=100.0)
    broker.sell_all(price=100.0)

    expected = initial_cash * retained_fraction(commission, spread, slippage)

    assert portfolio.shares == 0.0
    assert portfolio.cash == pytest.approx(expected, rel=1e-12)


def test_higher_commission_retains_strictly_less(initial_cash):
    balances = []
    for commission in (0.0, 0.0005, 0.001, 0.0025, 0.005):
        portfolio = Portfolio(initial_cash)
        broker = Broker(portfolio, commission=commission)
        broker.buy_all(price=100.0)
        broker.sell_all(price=100.0)
        balances.append(portfolio.cash)

    assert balances == sorted(balances, reverse=True)
    assert balances[0] == pytest.approx(initial_cash, rel=1e-12)


@pytest.mark.parametrize("commission, spread, slippage", [
    (-0.001, 0.0, 0.0),
    (0.0, -0.001, 0.0),
    (0.0, 0.0, -0.001),
    (1.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.5),
])
def test_impossible_cost_parameters_are_rejected(initial_cash, commission,
                                                 spread, slippage):
    with pytest.raises(ValueError):
        Broker(Portfolio(initial_cash), commission=commission, spread=spread,
               slippage=slippage)


# ---------------------------------------------------------------------------
# Backtester: the Phase 1 correctness check
# ---------------------------------------------------------------------------

def test_buy_and_hold_strategy_matches_the_analytic_benchmark(wave_prices,
                                                              initial_cash):
    """The Phase 1 check, which the whole engine rests on.

    The benchmark is computed straight from the price series and never passes
    through the Portfolio or the Broker, so agreement between the two is a real
    test of the accounting. An engine bug could not hide in both.
    """
    result = Backtester(wave_prices, initial_cash=initial_cash,
                        strategy=BuyFirstBarAndHold()).run()

    assert len(result.trade_log) == 1
    assert abs(result.final_value - result.benchmark_final_value) < 1e-6


def test_buy_and_hold_matches_the_benchmark_on_a_flat_series(flat_prices,
                                                             initial_cash):
    result = Backtester(flat_prices, initial_cash=initial_cash,
                        strategy=BuyFirstBarAndHold()).run()

    assert result.final_value == pytest.approx(initial_cash, rel=1e-12)
    assert abs(result.final_value - result.benchmark_final_value) < 1e-6


def test_benchmark_curve_is_the_price_ratio_times_the_capital(wave_prices,
                                                              initial_cash):
    result = Backtester(wave_prices, initial_cash=initial_cash,
                        strategy=BuyFirstBarAndHold()).run()

    closes = wave_prices["Close"]
    expected_final = initial_cash * closes.iloc[-1] / closes.iloc[0]

    assert result.benchmark_curve.iloc[0] == pytest.approx(initial_cash, rel=1e-12)
    assert result.benchmark_curve.iloc[-1] == pytest.approx(expected_final,
                                                            rel=1e-12)
    # 100 -> 150 over the fixture, so exactly 1.5x the capital.
    assert expected_final == pytest.approx(15_000.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Backtester: costs
# ---------------------------------------------------------------------------

def test_omitting_costs_equals_passing_zero(wave_prices, initial_cash):
    default = Backtester(wave_prices, initial_cash=initial_cash,
                         strategy=FlipEvery(2)).run()
    explicit = Backtester(wave_prices, initial_cash=initial_cash,
                          strategy=FlipEvery(2), commission=0.0, spread=0.0,
                          slippage=0.0).run()

    assert default.final_value == explicit.final_value
    pd.testing.assert_frame_equal(default.equity_curve, explicit.equity_curve)


def test_rising_costs_strictly_reduce_the_final_value(wave_prices, initial_cash):
    values = [
        Backtester(wave_prices, initial_cash=initial_cash,
                   strategy=FlipEvery(2), commission=commission).run().final_value
        for commission in (0.0, 0.0005, 0.001, 0.0025, 0.005)
    ]

    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_costs_leave_the_benchmark_untouched(wave_prices, initial_cash):
    free = Backtester(wave_prices, initial_cash=initial_cash,
                      strategy=FlipEvery(2)).run()
    charged = Backtester(wave_prices, initial_cash=initial_cash,
                         strategy=FlipEvery(2), commission=0.01).run()

    pd.testing.assert_series_equal(free.benchmark_curve, charged.benchmark_curve)


def test_flat_market_flip_loses_exactly_the_modelled_costs(flat_prices,
                                                           initial_cash):
    """A closed-form check that the engine charges what the Broker models.

    Flipping on a flat quote can only lose costs, and the number of round trips
    is known from the schedule, so the final value is predictable on paper.
    """
    commission = 0.001
    result = Backtester(flat_prices, initial_cash=initial_cash,
                        strategy=FlipEvery(2), commission=commission).run()

    buys = sum(1 for order in result.trade_log if order["action"] == BUY)
    sells = sum(1 for order in result.trade_log if order["action"] == SELL)
    round_trips = min(buys, sells)

    expected = initial_cash * retained_fraction(commission, 0.0, 0.0) ** round_trips
    if buys > sells:
        # A position still open at the end has paid the buy leg only.
        expected *= 1.0 / (1.0 + commission)

    assert result.final_value == pytest.approx(expected, rel=1e-9)


def test_trade_log_separates_the_quote_from_the_fill(wave_prices, initial_cash):
    commission = 0.002
    result = Backtester(wave_prices, initial_cash=initial_cash,
                        strategy=FlipEvery(3), commission=commission).run()

    assert result.trade_log
    for order in result.trade_log:
        assert set(order) >= {"date", "action", "price", "shares",
                              "quoted_price", "cost"}
        assert order["cost"] > 0.0
        assert order["cost"] == pytest.approx(
            abs(order["price"] - order["quoted_price"]) * order["shares"],
            rel=1e-12,
        )
        if order["action"] == BUY:
            assert order["price"] > order["quoted_price"]
        else:
            assert order["price"] < order["quoted_price"]


def test_frictionless_fills_equal_the_quote(wave_prices, initial_cash):
    result = Backtester(wave_prices, initial_cash=initial_cash,
                        strategy=FlipEvery(3)).run()

    assert result.trade_log
    for order in result.trade_log:
        assert order["price"] == order["quoted_price"]
        assert order["cost"] == 0.0


# ---------------------------------------------------------------------------
# Backtester: no look-ahead
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_factory", [
    pytest.param(lambda: BuyAboveThreshold(threshold=105.0), id="threshold"),
    pytest.param(lambda: FlipEvery(1), id="flip-every-bar"),
    pytest.param(lambda: FlipEvery(2), id="flip-every-2-bars"),
    pytest.param(lambda: BuyFirstBarAndHold(), id="buy-and-hold"),
])
def test_truncating_future_bars_does_not_change_past_decisions(wave_prices,
                                                               initial_cash,
                                                               strategy_factory):
    """The engine-level causality check, mirroring the strategy-level one.

    Every strategy here is strictly causal, reading only the current bar or a
    fixed schedule. So if a shortened run ever disagreed with the prefix of the
    full run, the leak would have to be in the engine's loop rather than in the
    signals.

    WHY EVERY TRUNCATION LENGTH IS TESTED, NOT A SAMPLE
    A one-bar forward peek only shows up where consecutive signals differ, and
    at the final bar of a truncated run the peek has nothing left to read and
    falls back on the current bar. Those two facts together mean a handful of
    hand-picked cut points can miss a real leak by coincidence: an earlier
    version of this test chose four lengths and every one of them happened to
    land where the shifted signal matched the honest one. Sweeping all lengths
    removes the luck, and FlipEvery(1) is included because its signal changes on
    every bar, so no cut point can hide a shift.
    """
    full = Backtester(wave_prices, initial_cash=initial_cash,
                      strategy=strategy_factory()).run()

    for kept in range(1, len(wave_prices) + 1):
        truncated = Backtester(wave_prices.iloc[:kept], initial_cash=initial_cash,
                               strategy=strategy_factory()).run()

        pd.testing.assert_frame_equal(
            full.equity_curve.iloc[:kept],
            truncated.equity_curve,
            obj=f"equity curve truncated to {kept} bars",
        )

        cutoff = wave_prices.index[kept - 1]
        past_orders = [order for order in full.trade_log
                       if order["date"] <= cutoff]
        assert past_orders == truncated.trade_log, (
            f"trade log changed when the series was cut to {kept} bars"
        )


def test_a_bar_is_marked_after_its_own_trade_not_before(wave_prices, initial_cash):
    """The buy on bar 0 must already be reflected in bar 0's recorded value."""
    result = Backtester(wave_prices, initial_cash=initial_cash,
                        strategy=BuyFirstBarAndHold()).run()

    first = result.equity_curve.iloc[0]

    assert first["shares"] > 0.0
    assert first["cash"] == pytest.approx(0.0, abs=1e-9)
    assert first["total_value"] == pytest.approx(initial_cash, rel=1e-12)


def test_rerunning_the_same_backtester_gives_identical_results(wave_prices,
                                                               initial_cash):
    """run() rebuilds the books, so a second call must not compound the first."""
    backtester = Backtester(wave_prices, initial_cash=initial_cash,
                            strategy=FlipEvery(2), commission=0.001)

    first = backtester.run()
    second = backtester.run()

    assert first.final_value == second.final_value
    assert first.trade_log == second.trade_log
    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)


# ---------------------------------------------------------------------------
# Backtester: input validation
# ---------------------------------------------------------------------------

def test_empty_price_data_is_rejected(initial_cash):
    with pytest.raises(ValueError, match="empty price DataFrame"):
        Backtester(pd.DataFrame(), initial_cash=initial_cash,
                   strategy=BuyFirstBarAndHold())


def test_missing_close_column_is_rejected(wave_prices, initial_cash):
    with pytest.raises(ValueError, match="'Close' column"):
        Backtester(wave_prices.drop(columns=["Close"]),
                   initial_cash=initial_cash, strategy=BuyFirstBarAndHold())


def test_unsorted_price_data_is_rejected(wave_prices, initial_cash):
    with pytest.raises(ValueError, match="sorted chronologically"):
        Backtester(wave_prices.iloc[::-1], initial_cash=initial_cash,
                   strategy=BuyFirstBarAndHold())


def test_an_object_without_generate_signals_is_rejected(wave_prices, initial_cash):
    with pytest.raises(TypeError, match="generate_signals"):
        Backtester(wave_prices, initial_cash=initial_cash, strategy=object())


def test_a_signal_outside_the_vocabulary_is_rejected(wave_prices, initial_cash):
    backtester = Backtester(wave_prices, initial_cash=initial_cash,
                            strategy=AlwaysReturnsGarbage())

    with pytest.raises(ValueError):
        backtester.run()
