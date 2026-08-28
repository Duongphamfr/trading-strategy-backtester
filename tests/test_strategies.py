"""Automated tests for the strategy layer: the contract, the rules, causality.

These convert the self-checks from the __main__ blocks of the three strategy
modules into assertions, and add the parts those blocks could not express.

WHAT MAKES THE EXPECTED VALUES TRUSTWORTHY
Three different sources, none of which is "whatever the code printed last time":

    Hand-derived. The moving average and momentum cases use windows small
    enough to work the arithmetic out on paper. The comments show the derivation
    so a reader can check it without running anything.

    An independent reimplementation. The RSI is checked against a plain Python
    loop written straight from Wilder's recursion. That matters more than it
    might look: the production code reaches the same result by a vectorised
    trick, planting the seed at bar `period` and letting pandas' ewm skip the
    leading NaNs, and the loop is the only thing here capable of catching a
    mistake in that trick.

    A published reference. The RSI price series is the classic worked example,
    whose first reading is 70.46, so the numbers can be traced outside this
    project entirely.

WHY THE CAUSALITY TESTS SWEEP EVERY TRUNCATION LENGTH
The engine tests taught this the hard way. A first version of the no-look-ahead
check there picked four cut points by hand, and a deliberately broken engine
still passed it: a one-bar forward peek only shows up where consecutive signals
differ, and all four chosen points happened to land elsewhere. Sweeping every
length removes the luck. The tests below also assert that signals actually fire,
because a strategy that returns nothing but HOLD is trivially causal and would
otherwise report a meaningless pass.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from constants import BUY, HOLD, SELL
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import RSIMeanReversion
from strategies.momentum import Momentum
from strategies.moving_average import MovingAverageCrossover

VOCABULARY = {BUY, SELL, HOLD}

# The classic RSI worked example, rounded to the two decimals the published
# table prints. Its first reading is the widely quoted 70.46.
REFERENCE_CLOSES: List[float] = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
]

REFERENCE_PERIOD = 14

# The published readings, to the two decimals of the source table. Bar 14 is the
# seed, computed from the simple mean of the first fourteen changes; every later
# bar comes from Wilder's recursion.
PUBLISHED_RSI: Dict[int, float] = {
    14: 70.46, 15: 66.25, 16: 66.48, 17: 69.35, 18: 66.29, 19: 57.92,
    20: 62.88, 21: 63.21, 22: 56.01, 23: 62.34, 24: 54.67, 25: 50.39,
    26: 40.02, 27: 41.49, 28: 41.90, 29: 45.50,
}


def frame(closes: List[float]) -> pd.DataFrame:
    """Wrap a list of closes in the minimal frame the strategies read."""
    return pd.DataFrame({"Close": [float(price) for price in closes]})


def textbook_wilder_rsi(closes: List[float], period: int) -> Dict[int, float]:
    """Wilder's RSI by explicit recursion, as an independent reference.

    Written from the 1978 definition rather than from the project's code, and
    kept deliberately naive: a Python loop, no pandas, no exponential-average
    helper. Agreeing with the vectorised implementation to floating point
    precision is then real evidence, because the two share no machinery.

    The rearrangement 100 * gain / (gain + loss) is used rather than
    100 - 100 / (1 + gain / loss), which is algebraically the same but divides
    by zero when a window holds no losses.

    Args:
        closes: Closing prices, oldest first.
        period: Number of changes the averages span.

    Returns:
        A mapping from bar position to RSI, starting at bar `period`. Bars in
        the warm-up are absent, and so is any bar where the price never moved
        and the reading is genuinely undefined.
    """
    gains, losses = [], []
    for earlier, later in zip(closes, closes[1:]):
        change = later - earlier
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    readings: Dict[int, float] = {}
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    def reading(gain: float, loss: float) -> float:
        return 100.0 * gain / (gain + loss) if gain + loss > 0.0 else float("nan")

    readings[period] = reading(average_gain, average_loss)
    for position in range(period, len(gains)):
        average_gain += (gains[position] - average_gain) / period
        average_loss += (losses[position] - average_loss) / period
        readings[position + 1] = reading(average_gain, average_loss)

    return readings


def marks(signals: pd.Series) -> str:
    """Render a signal series as a compact string, B for buy, S for sell.

    Turns an assertion failure into something readable: comparing "..B..S..." to
    "..B...S.." shows immediately where a rule went wrong, which a diff of two
    long Series does not.
    """
    return "".join({BUY: "B", SELL: "S", HOLD: "."}[signal] for signal in signals)


def signal_positions(signals: pd.Series, action: str) -> List[int]:
    """Integer positions of the bars carrying a given signal."""
    return [position for position, signal in enumerate(signals) if signal == action]


# Every strategy is built through a factory rather than shared as an instance,
# so no test can be affected by another having mutated one.
STRATEGY_FACTORIES = [
    pytest.param(lambda: MovingAverageCrossover(fast_window=3, slow_window=8),
                 id="ma-3-8"),
    pytest.param(lambda: MovingAverageCrossover(fast_window=3, slow_window=8,
                                                enter_on_existing_trend=True),
                 id="ma-3-8-enter-on-trend"),
    pytest.param(lambda: MovingAverageCrossover(fast_window=5, slow_window=20),
                 id="ma-5-20"),
    pytest.param(lambda: RSIMeanReversion(rsi_period=5, oversold=40.0,
                                          overbought=60.0),
                 id="rsi-5-40-60"),
    pytest.param(lambda: RSIMeanReversion(rsi_period=14),
                 id="rsi-14-default-bands"),
    pytest.param(lambda: Momentum(lookback=5, rebalance_freq=1),
                 id="momentum-5-every-bar"),
    pytest.param(lambda: Momentum(lookback=6, rebalance_freq=4),
                 id="momentum-6-every-4-bars"),
]


# ---------------------------------------------------------------------------
# BaseStrategy contract
# ---------------------------------------------------------------------------

def test_base_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseStrategy()


def test_a_subclass_that_skips_generate_signals_cannot_be_instantiated():
    class Incomplete(BaseStrategy):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_a_subclass_implementing_the_method_is_usable(wave_prices):
    class AlwaysHold(BaseStrategy):
        def generate_signals(self, data):
            return self.hold_signals(data)

    strategy = AlwaysHold()
    signals = strategy.generate_signals(wave_prices)

    assert (signals == HOLD).all()
    assert signals.index.equals(wave_prices.index)


def test_hold_signals_is_neutral_and_aligned(wave_prices):
    signals = BaseStrategy.hold_signals(wave_prices)

    assert (signals == HOLD).all()
    assert signals.index.equals(wave_prices.index)
    assert len(signals) == len(wave_prices)


@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_signals_are_aligned_to_the_data_index(strategy_factory,
                                               oscillating_prices):
    signals = strategy_factory().generate_signals(oscillating_prices)

    assert isinstance(signals, pd.Series)
    assert signals.index.equals(oscillating_prices.index)
    assert len(signals) == len(oscillating_prices)


@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_signals_use_only_the_declared_vocabulary(strategy_factory,
                                                  oscillating_prices):
    signals = strategy_factory().generate_signals(oscillating_prices)

    assert set(signals.unique()) <= VOCABULARY
    assert not signals.isna().any()


@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_generating_signals_does_not_modify_the_price_data(strategy_factory,
                                                           oscillating_prices):
    """Contract point five: a strategy must leave the frame it is handed alone.

    Otherwise a parameter sweep, which reuses one frame across hundreds of runs,
    would have every run after the first read data the previous one altered.
    """
    original = oscillating_prices.copy(deep=True)

    strategy_factory().generate_signals(oscillating_prices)

    pd.testing.assert_frame_equal(oscillating_prices, original)


@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_the_warm_up_is_spent_holding(strategy_factory, oscillating_prices):
    """No strategy may act before its indicator is defined."""
    strategy = strategy_factory()
    signals = strategy.generate_signals(oscillating_prices)

    if isinstance(strategy, MovingAverageCrossover):
        warm_up = strategy.slow_window - 1
    elif isinstance(strategy, RSIMeanReversion):
        warm_up = strategy.rsi_period
    else:
        warm_up = strategy.lookback - 1

    assert (signals.iloc[:warm_up] == HOLD).all()


@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_params_and_name_describe_the_configuration(strategy_factory):
    strategy = strategy_factory()

    assert strategy.name == type(strategy).__name__
    assert strategy.params
    for key, value in strategy.params.items():
        assert getattr(strategy, key) == value
        assert key in repr(strategy)


@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_a_history_shorter_than_the_warm_up_never_trades(strategy_factory):
    """Too little data must produce silence rather than an error."""
    signals = strategy_factory().generate_signals(frame([100.0, 101.0, 102.0]))

    assert (signals == HOLD).all()


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fast, slow", [
    (50, 50),
    (200, 50),
    (0, 200),
    (-10, 200),
    (50, 0),
    (50, -200),
])
def test_moving_average_rejects_impossible_windows(fast, slow):
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=fast, slow_window=slow)


@pytest.mark.parametrize("fast, slow", [(10.5, 200), (10, 200.5)])
def test_moving_average_rejects_fractional_windows(fast, slow):
    with pytest.raises(ValueError, match="whole numbers"):
        MovingAverageCrossover(fast_window=fast, slow_window=slow)


def test_moving_average_accepts_a_valid_pair():
    strategy = MovingAverageCrossover(fast_window=50, slow_window=200)

    assert strategy.fast_window == 50
    assert strategy.slow_window == 200
    assert strategy.enter_on_existing_trend is False


@pytest.mark.parametrize("period", [0, -1, -14])
def test_rsi_rejects_a_non_positive_period(period):
    with pytest.raises(ValueError, match="rsi_period"):
        RSIMeanReversion(rsi_period=period)


def test_rsi_rejects_a_fractional_period():
    with pytest.raises(ValueError, match="whole number"):
        RSIMeanReversion(rsi_period=14.5)


@pytest.mark.parametrize("oversold, overbought", [
    (70.0, 30.0),
    (50.0, 50.0),
    (0.0, 70.0),
    (-10.0, 70.0),
    (30.0, 100.0),
    (30.0, 120.0),
    (-5.0, 105.0),
])
def test_rsi_rejects_thresholds_outside_zero_to_one_hundred_or_inverted(
        oversold, overbought):
    """RSI is bounded to [0, 100], so these can never fire or always would."""
    with pytest.raises(ValueError, match="Thresholds"):
        RSIMeanReversion(oversold=oversold, overbought=overbought)


def test_rsi_accepts_wilders_conventional_settings():
    strategy = RSIMeanReversion()

    assert strategy.rsi_period == 14
    assert strategy.oversold == 30.0
    assert strategy.overbought == 70.0


@pytest.mark.parametrize("lookback", [0, -1, -126])
def test_momentum_rejects_a_non_positive_lookback(lookback):
    with pytest.raises(ValueError, match="lookback"):
        Momentum(lookback=lookback)


@pytest.mark.parametrize("frequency", [0, -1, -21])
def test_momentum_rejects_a_rebalance_frequency_below_one(frequency):
    with pytest.raises(ValueError, match="rebalance_freq"):
        Momentum(rebalance_freq=frequency)


@pytest.mark.parametrize("lookback, frequency", [(12.5, 21), (126, 21.5)])
def test_momentum_rejects_fractional_bar_counts(lookback, frequency):
    with pytest.raises(ValueError, match="whole numbers"):
        Momentum(lookback=lookback, rebalance_freq=frequency)


def test_momentum_accepts_every_bar_reviewing():
    strategy = Momentum(lookback=126, rebalance_freq=1)

    assert strategy.lookback == 126
    assert strategy.rebalance_freq == 1


# ---------------------------------------------------------------------------
# RSI correctness
# ---------------------------------------------------------------------------

def test_rsi_matches_the_published_worked_example():
    """The external check: numbers traceable outside this project."""
    computed = RSIMeanReversion(rsi_period=REFERENCE_PERIOD).rsi(
        frame(REFERENCE_CLOSES))

    for position, expected in PUBLISHED_RSI.items():
        # The published table prints two decimals, so agreement is asserted to
        # half a unit in the last of them.
        assert computed.iloc[position] == pytest.approx(expected, abs=0.005), (
            f"bar {position}: expected the published {expected}, "
            f"got {computed.iloc[position]}"
        )


def test_rsi_matches_an_independent_textbook_implementation():
    """The structural check: the vectorised seeding trick against a plain loop."""
    computed = RSIMeanReversion(rsi_period=REFERENCE_PERIOD).rsi(
        frame(REFERENCE_CLOSES))
    reference = textbook_wilder_rsi(REFERENCE_CLOSES, REFERENCE_PERIOD)

    assert len(reference) == 16
    for position, expected in reference.items():
        assert computed.iloc[position] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("period", [2, 5, 9, 14])
def test_rsi_matches_the_textbook_implementation_at_other_periods(period,
                                                                  oscillating_prices):
    """The agreement must not be a coincidence of the period 14 seed."""
    closes = list(oscillating_prices["Close"])
    computed = RSIMeanReversion(rsi_period=period,
                               oversold=30.0, overbought=70.0).rsi(frame(closes))
    reference = textbook_wilder_rsi(closes, period)

    for position, expected in reference.items():
        assert computed.iloc[position] == pytest.approx(expected, abs=1e-10)


def test_rsi_seeds_at_the_period_bar_and_holds_nan_before_it():
    computed = RSIMeanReversion(rsi_period=REFERENCE_PERIOD).rsi(
        frame(REFERENCE_CLOSES))

    assert computed.iloc[:REFERENCE_PERIOD].isna().all()
    assert computed.notna().idxmax() == REFERENCE_PERIOD
    assert computed.iloc[REFERENCE_PERIOD] == pytest.approx(70.4641, abs=1e-4)


def test_rsi_stays_within_zero_and_one_hundred(oscillating_prices):
    computed = RSIMeanReversion().rsi(oscillating_prices).dropna()

    assert not computed.empty
    assert computed.min() >= 0.0
    assert computed.max() <= 100.0


def test_twenty_straight_gains_give_an_rsi_of_exactly_one_hundred():
    """No losses at all, so the ratio of gains to total is one."""
    computed = RSIMeanReversion().rsi(frame([100.0 + bar for bar in range(20)]))

    assert computed.iloc[-1] == pytest.approx(100.0, abs=1e-12)


def test_twenty_straight_losses_give_an_rsi_of_exactly_zero():
    computed = RSIMeanReversion().rsi(frame([100.0 - bar for bar in range(20)]))

    assert computed.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_a_price_that_never_moves_leaves_the_rsi_undefined():
    """Both averages are zero, so the reading is genuinely undefined, not 50."""
    flat = frame([100.0] * 20)
    strategy = RSIMeanReversion()

    assert strategy.rsi(flat).isna().all()
    assert (strategy.generate_signals(flat) == HOLD).all()


def test_a_history_no_longer_than_the_period_yields_no_rsi_at_all():
    """diff() consumes a bar, so the seed needs period + 1 bars to exist."""
    strategy = RSIMeanReversion(rsi_period=REFERENCE_PERIOD)

    assert strategy.rsi(frame(REFERENCE_CLOSES[:REFERENCE_PERIOD])).isna().all()
    assert strategy.rsi(frame(REFERENCE_CLOSES[:REFERENCE_PERIOD + 1])
                        ).notna().any()


def test_rsi_signals_fire_on_entering_an_extreme_not_on_every_bar_inside_it():
    """The transition rule, which keeps the trade log to one entry per visit.

    The series rises before it falls on purpose. A slide starting at bar zero
    would put the first oversold reading on the very first defined bar, where the
    rule cannot fire by design: there is no earlier reading to have crossed down
    from, so no crossing exists to detect. That is the documented behaviour, and
    a series arranged that way would test the warm-up rule instead of this one.
    """
    strategy = RSIMeanReversion(rsi_period=5, oversold=40.0, overbought=60.0)
    prices = frame([100.0 + bar for bar in range(8)]
                   + [104.0 - 3.0 * bar for bar in range(9)])

    rsi = strategy.rsi(prices)
    signals = strategy.generate_signals(prices)

    below = rsi < strategy.oversold
    first_below = int(below.idxmax())
    first_defined = int(rsi.notna().idxmax())

    # The crossing has to be observable for this test to mean anything.
    assert first_below > first_defined
    assert int(below.sum()) > 1

    assert signal_positions(signals, BUY) == [first_below]


# ---------------------------------------------------------------------------
# Moving average crossover, hand-derived
# ---------------------------------------------------------------------------

# Closes: 10 x4, 20 x4, 10 x4, with windows 2 and 4.
#   fast(2) from bar 1: 10, 10, 10, 15, 20, 20, 20, 15, 10, 10, 10
#   slow(4) from bar 3: 10, 12.5, 15, 17.5, 20, 17.5, 15, 12.5, 10
#   fast above slow:  bar 3 no (10 = 10), bars 4-6 yes, bar 7 no (20 = 20),
#                     bars 8-11 no
# So the only crossings visible are up at bar 4 and down at bar 7.
STEP_CLOSES = [10.0] * 4 + [20.0] * 4 + [10.0] * 4


def test_crossover_fires_once_up_and_once_down_on_a_hand_worked_series():
    signals = MovingAverageCrossover(fast_window=2,
                                     slow_window=4).generate_signals(
        frame(STEP_CLOSES))

    assert marks(signals) == "....B..S...."
    assert signal_positions(signals, BUY) == [4]
    assert signal_positions(signals, SELL) == [7]


def test_crossover_holds_when_the_slow_average_is_never_defined():
    signals = MovingAverageCrossover(fast_window=2,
                                     slow_window=20).generate_signals(
        frame(STEP_CLOSES))

    assert (signals == HOLD).all()


# A series rising from the first bar, so the fast average is already above the
# slow one the moment both exist, at bar 3. The crossing that put it there
# happened before the data starts and is therefore not observable.
#   fast(2) from bar 1: 15, 25, 35, 45, 55, 65, 75
#   slow(4) from bar 3: 25, 35, 45, 55, 65
#   at bar 3: 35 > 25, so the trend pre-exists.
PRE_EXISTING_TREND_CLOSES = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]


def test_by_default_the_crossover_refuses_a_trend_it_did_not_see_form():
    signals = MovingAverageCrossover(fast_window=2,
                                     slow_window=4).generate_signals(
        frame(PRE_EXISTING_TREND_CLOSES))

    assert (signals == HOLD).all()


def test_enter_on_existing_trend_buys_at_the_end_of_the_warm_up():
    signals = MovingAverageCrossover(
        fast_window=2, slow_window=4,
        enter_on_existing_trend=True).generate_signals(
        frame(PRE_EXISTING_TREND_CLOSES))

    assert signal_positions(signals, BUY) == [3]
    assert signal_positions(signals, SELL) == []


def test_enter_on_existing_trend_stays_out_when_the_warm_up_ends_in_a_downtrend():
    """The flag buys into an existing uptrend, not into any existing state."""
    falling = [80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]

    signals = MovingAverageCrossover(
        fast_window=2, slow_window=4,
        enter_on_existing_trend=True).generate_signals(frame(falling))

    assert (signals == HOLD).all()


def test_the_flag_changes_nothing_when_a_crossing_is_actually_visible():
    conservative = MovingAverageCrossover(fast_window=2, slow_window=4)
    eager = MovingAverageCrossover(fast_window=2, slow_window=4,
                                   enter_on_existing_trend=True)

    pd.testing.assert_series_equal(
        conservative.generate_signals(frame(STEP_CLOSES)),
        eager.generate_signals(frame(STEP_CLOSES)),
    )


# ---------------------------------------------------------------------------
# Momentum statefulness, hand-derived
# ---------------------------------------------------------------------------

# lookback 2, reviewed every 3 bars. The first decision falls on bar 2, so the
# review bars are 2, 5, 8 and 11. Trailing return is Close[T] / Close[T-2] - 1.
#
#   bar   close   trailing   review?   decision
#     2     110    +10.0%      yes      invest
#     3      90    -10.0%       no      (persists: invested)
#     4      90    -18.2%       no      (persists: invested)
#     5      80    -11.1%      yes      go to cash
#     6     120    +33.3%       no      (persists: cash)
#     7     120    +50.0%       no      (persists: cash)
#     8     130     +8.3%      yes      invest
#     9     100    -16.7%       no      (persists: invested)
#    10     100    -23.1%       no      (persists: invested)
#    11      90    -10.0%      yes      go to cash
#
# The point of the series is bars 3-4, 6-7 and 9-10, where the trailing return
# contradicts the position being held. A rule reading the indicator on every bar
# would trade there; this one is not due to look until the next review.
PERSISTENCE_CLOSES = [100.0, 100.0, 110.0, 90.0, 90.0, 80.0,
                      120.0, 120.0, 130.0, 100.0, 100.0, 90.0]

PERSISTENCE_STRATEGY = dict(lookback=2, rebalance_freq=3)
EXPECTED_REVIEW_BARS = [2, 5, 8, 11]
EXPECTED_INVESTED = [False, False, True, True, True, False,
                     False, False, True, True, True, False]


def test_momentum_holds_its_position_between_review_bars():
    holding = Momentum(**PERSISTENCE_STRATEGY).target_position(
        frame(PERSISTENCE_CLOSES))

    assert list(holding) == EXPECTED_INVESTED


def test_momentum_ignores_the_indicator_on_non_review_bars():
    """Bars 3, 4, 6 and 7 disagree with the position and must be overruled."""
    prices = frame(PERSISTENCE_CLOSES)
    strategy = Momentum(**PERSISTENCE_STRATEGY)

    trailing = strategy.momentum(prices)
    holding = strategy.target_position(prices)

    contradicting = [bar for bar in range(len(prices))
                     if bar not in EXPECTED_REVIEW_BARS
                     and not np.isnan(trailing.iloc[bar])
                     and (trailing.iloc[bar] > 0) != holding.iloc[bar]]

    assert contradicting == [3, 4, 6, 7, 9, 10]


def test_momentum_signals_only_the_bars_where_it_changes_its_mind():
    signals = Momentum(**PERSISTENCE_STRATEGY).generate_signals(
        frame(PERSISTENCE_CLOSES))

    assert marks(signals) == "..B..S..B..S"


def test_the_first_momentum_decision_falls_on_the_lookback_bar():
    strategy = Momentum(lookback=5, rebalance_freq=1)
    holding = strategy.target_position(frame([100.0 + bar for bar in range(12)]))

    assert not holding.iloc[:5].any()
    assert holding.iloc[5]


def test_a_rising_series_is_bought_once_and_held():
    """From the module's own hand-checkable cases: one entry, no exit."""
    strategy = Momentum(lookback=5, rebalance_freq=1)
    prices = frame([100.0 + bar for bar in range(12)])

    signals = strategy.generate_signals(prices)

    assert marks(signals) == ".....B......"
    # Invested from bar 5 to bar 11, so seven bars of twelve.
    assert strategy.target_position(prices).sum() == 7


def test_a_falling_series_is_never_bought():
    strategy = Momentum(lookback=5, rebalance_freq=1)
    prices = frame([100.0 - bar for bar in range(12)])

    assert (strategy.generate_signals(prices) == HOLD).all()
    assert not strategy.target_position(prices).any()


def test_a_flat_series_is_never_bought_because_zero_is_not_positive():
    strategy = Momentum(lookback=5, rebalance_freq=1)
    prices = frame([100.0] * 12)

    assert (strategy.generate_signals(prices) == HOLD).all()
    assert not strategy.target_position(prices).any()


def test_a_rise_then_a_fall_is_entered_and_exited_once():
    # Closes 100..107 then 107, 104, 101, 98. The trailing return over five bars
    # turns non-positive at bar 9, where 104 / 104 - 1 is exactly zero.
    strategy = Momentum(lookback=5, rebalance_freq=1)
    closes = [100.0 + bar for bar in range(8)] + [107.0 - 3.0 * bar
                                                 for bar in range(4)]

    signals = strategy.generate_signals(frame(closes))

    assert marks(signals) == ".....B...S.."
    # Invested on bars 5 to 8 only, so four bars of twelve.
    assert strategy.target_position(frame(closes)).sum() == 4


def test_reviewing_less_often_trades_less(oscillating_prices):
    """Turnover must fall as the review interval grows."""
    trades = [
        int((Momentum(lookback=6,
                      rebalance_freq=frequency).generate_signals(
            oscillating_prices) != HOLD).sum())
        for frequency in (1, 3, 10, 30)
    ]

    assert trades == sorted(trades, reverse=True)
    assert trades[0] > trades[-1]


# ---------------------------------------------------------------------------
# No look-ahead, for all three strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_truncating_the_history_never_changes_an_earlier_signal(strategy_factory,
                                                                oscillating_prices):
    """Recomputing on a shortened history must reproduce the overlap exactly.

    A direct test of the no-look-ahead contract rather than a proxy for it. If
    the signal on bar T drew on anything after T, deleting the tail would change
    it, and the two runs would disagree somewhere in the part they share.

    Every cut point is tried, for the reason set out in this module's docstring:
    hand-picked ones let a real leak through in the engine suite. The RSI is the
    case that most needs it, since Wilder's smoothing is recursive and a badly
    seeded version would drift differently depending on where the data ends.
    """
    strategy = strategy_factory()
    full = strategy.generate_signals(oscillating_prices)

    assert (full != HOLD).sum() > 0, (
        "this strategy never trades on the fixture, so the test would pass "
        "without testing anything"
    )

    for kept in range(1, len(oscillating_prices) + 1):
        truncated = strategy_factory().generate_signals(
            oscillating_prices.iloc[:kept])

        pd.testing.assert_series_equal(
            truncated,
            full.iloc[:kept],
            obj=f"signals recomputed on the first {kept} bars",
        )


@pytest.mark.parametrize("period", [5, 14])
def test_the_rsi_series_itself_does_not_drift_when_the_tail_is_removed(
        period, oscillating_prices):
    """The indicator, not just the signals, must be independent of the future."""
    strategy = RSIMeanReversion(rsi_period=period)
    full = strategy.rsi(oscillating_prices)

    for kept in range(period + 1, len(oscillating_prices) + 1):
        truncated = strategy.rsi(oscillating_prices.iloc[:kept])

        pd.testing.assert_series_equal(truncated, full.iloc[:kept])


def test_the_momentum_review_schedule_is_anchored_to_the_warm_up_not_the_end(
        oscillating_prices):
    """Anchoring backwards from the last bar would be a look-ahead violation.

    Worth isolating because it is the subtle way this strategy could cheat: the
    rule itself is causal, but a schedule counted back from the end of the
    sample would silently make every decision date depend on when the download
    happened to stop.
    """
    strategy = Momentum(lookback=6, rebalance_freq=7)
    full = strategy.target_position(oscillating_prices)

    for kept in range(1, len(oscillating_prices) + 1):
        truncated = strategy.target_position(oscillating_prices.iloc[:kept])

        pd.testing.assert_series_equal(truncated, full.iloc[:kept])


@pytest.mark.parametrize("strategy_factory", STRATEGY_FACTORIES)
def test_appending_future_bars_never_rewrites_the_past(strategy_factory,
                                                       oscillating_prices,
                                                       wave_prices):
    """The same property seen from the other end: growing the data is additive.

    Truncation and extension are logically equivalent tests, but this phrasing
    is closer to what actually happens in a walk-forward run, where each roll
    sees the previous window plus new bars.
    """
    half = len(oscillating_prices) // 2
    past = oscillating_prices.iloc[:half]

    before = strategy_factory().generate_signals(past)
    after = strategy_factory().generate_signals(oscillating_prices)

    pd.testing.assert_series_equal(before, after.iloc[:half])
