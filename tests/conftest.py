"""Shared fixtures for the test suite.

WHY THE PRICE DATA HERE IS SYNTHETIC
None of these fixtures touch the network or the on-disk cache. Every series is
written out literally, short enough to check by hand, and chosen so the expected
results can be derived on paper rather than copied from a previous run. A test
whose expected value came from running the code cannot detect that the code was
wrong to begin with.

That also keeps the suite fast and, more importantly, deterministic: yfinance
returns slightly different adjusted prices between calls, so a test pinned to
real prices would eventually fail for a reason that has nothing to do with the
code under test.
"""

import math
from typing import List

import pandas as pd
import pytest

# Ten bars at a constant price. The point of a flat quote is that any change in
# portfolio value must be a cost, since the market contributed nothing. It turns
# the cost arithmetic into an exact identity rather than an approximation.
FLAT_PRICE = 100.0
FLAT_BARS = 10

# A deliberately uneven path: up, back down below the start, then up to finish
# above it. Written out so every expected number below can be traced to a bar.
# The shape matters: it rises and falls more than once, so a strategy that
# trades has something to get right or wrong, and drawdowns actually exist.
WAVE_CLOSES: List[float] = [
    100.0, 110.0, 121.0, 108.9, 98.0, 88.2, 96.0, 105.6, 120.0, 132.0,
    125.0, 118.0, 130.0, 143.0, 150.0,
]


def _frame(closes: List[float]) -> pd.DataFrame:
    """Build a minimal OHLCV frame from a list of closes.

    Only Close is used by the engine, but the other columns are filled in so the
    fixture matches the shape market_data returns and cannot accidentally pass a
    test that would fail on real data.

    Args:
        closes: Closing prices, one per bar.

    Returns:
        A DataFrame indexed by consecutive business days.
    """
    index = pd.bdate_range("2022-01-03", periods=len(closes), name="Date")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [price * 1.01 for price in closes],
            "Low": [price * 0.99 for price in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=index,
    )


@pytest.fixture
def flat_prices() -> pd.DataFrame:
    """Ten bars at a constant price of 100."""
    return _frame([FLAT_PRICE] * FLAT_BARS)


@pytest.fixture
def wave_prices() -> pd.DataFrame:
    """Fifteen bars rising and falling, finishing at 150 from a start of 100."""
    return _frame(WAVE_CLOSES)


@pytest.fixture
def initial_cash() -> float:
    """Starting capital used across the engine tests."""
    return 10_000.0


# A drifting oscillation, long enough for the strategies to warm up and then
# change their minds several times. Generated from a closed-form expression
# rather than written out because the causality tests care about one property
# only, that the series is the same on every call, and not about any individual
# price. There is deliberately no random number generator involved, seeded or
# otherwise: a fixed seed still leaves the values at the mercy of the library's
# internals, whereas sine and a linear drift are pinned by arithmetic.
#
# The parameters are chosen so signals actually fire. A series too smooth for
# the moving averages to cross, or too tame for the RSI to reach its bands,
# would make a causality test pass by having nothing to disagree about, which is
# the failure mode the strategy tests below guard against explicitly.
OSCILLATING_BARS = 90
OSCILLATION_PERIOD = 25.0
OSCILLATION_AMPLITUDE = 0.18
OSCILLATION_DRIFT = 0.25


@pytest.fixture
def oscillating_prices() -> pd.DataFrame:
    """Ninety bars swinging up and down around a gently rising trend."""
    closes = [
        round(
            100.0
            * (1.0 + OSCILLATION_AMPLITUDE
               * math.sin(2.0 * math.pi * bar / OSCILLATION_PERIOD))
            + OSCILLATION_DRIFT * bar,
            4,
        )
        for bar in range(OSCILLATING_BARS)
    ]
    return _frame(closes)
