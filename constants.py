"""Project-wide constants.

Values that several packages must agree on, kept in one place so they cannot
drift apart. The signal vocabulary is the important case: strategies emit those
strings and the backtester interprets them, so a mismatch would break the
contract silently.

This module holds no logic and imports nothing from the project. It lives at the
project root rather than inside a package so that no package appears to depend on
another: engine, strategies and analytics all simply reach down to it.
"""

# Trading signals.
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

VALID_SIGNALS = frozenset({BUY, SELL, HOLD})

# Number of trading days in a year, the standard figure used to annualize daily
# statistics. Roughly 365 days less weekends and market holidays.
TRADING_DAYS_PER_YEAR = 252
