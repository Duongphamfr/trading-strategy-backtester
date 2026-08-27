"""Shared signal vocabulary.

The single place where the three trading signals are defined. Strategies emit
them and the backtester interprets them, so both sides must agree on the exact
strings; keeping one definition removes any chance of the two drifting apart.

This module holds no logic and imports nothing from the project. It lives at the
project root rather than inside a package so that neither engine nor strategies
appears to depend on the other: both simply reach down to a neutral vocabulary.
"""

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

VALID_SIGNALS = frozenset({BUY, SELL, HOLD})
