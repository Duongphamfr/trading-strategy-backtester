"""Shared interface that every trading strategy must implement."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from constants import HOLD


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    A strategy is a pure function of price history: it looks at the data and
    says what it would do on each bar. It never touches cash, positions, orders
    or the backtest loop. That separation is what lets strategies be swapped
    freely without the engine changing a single line, and what lets the UI pass
    parameters in without either side knowing about the other.

    Concrete strategies subclass this, pass their parameters to super().__init__
    so they are recorded in a uniform way, and implement generate_signals.

    Attributes:
        params: The parameters the strategy was constructed with, kept as a dict
            so results tables and the dashboard can report exactly which
            configuration produced a given backtest.
    """

    def __init__(self, **params: Any) -> None:
        """Store the strategy's parameters.

        Concrete strategies declare their own explicit keyword arguments, then
        forward them here, for example:

            class MovingAverageCrossover(BaseStrategy):
                def __init__(self, fast_window=50, slow_window=200):
                    super().__init__(fast_window=fast_window, slow_window=slow_window)

        Validating those parameters is each strategy's own responsibility, since
        only the strategy knows what a sensible value looks like.

        Args:
            **params: Arbitrary strategy parameters, stored on self.params.
        """
        self.params: Dict[str, Any] = dict(params)

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce one trading signal per bar of the price history.

        THE CONTRACT
        Implementations must satisfy every point below. The backtester relies on
        all of them and cannot verify most of them on its own.

        1. Alignment. The returned Series must be indexed by data.index, one
           value per bar, in the same order.

        2. Vocabulary. Every value must be "BUY", "SELL" or "HOLD". Bars the
           strategy has no opinion on, typically the warm-up period an indicator
           needs before it is defined, should be HOLD. The backtester also
           treats missing values as HOLD, but being explicit is clearer.

        3. NO LOOK-AHEAD BIAS. This is the important one. The signal for bar T
           may depend only on data up to and including bar T. Nothing the
           strategy could not have known on that date may influence it. The
           backtester computes all signals in one vectorised call before the
           simulation loop starts, purely for speed, so it has no way of
           catching a violation: a strategy that peeks at the future will simply
           produce impressive and completely fake results.

           Backward-looking operations are safe, for instance
           data["Close"].rolling(window).mean(), because a rolling window at bar
           T only ever covers bars at or before T.

           These leak future information and must be avoided, or shifted back
           by the strategy itself before the signals are returned:
             - a negative shift, such as .shift(-1), which pulls tomorrow back
               onto today;
             - centred windows, .rolling(window, center=True), whose window
               extends past bar T;
             - any statistic computed over the whole sample, such as
               data["Close"].mean() or a z-score scaled by the full-period
               standard deviation, since those embed knowledge of the entire
               history into every single bar;
             - resampling or reindexing that back-fills values from later dates.

        4. Execution convention. A signal emitted on bar T is filled at bar T's
           closing price. A strategy deriving its signal from that same close is
           therefore already at the optimistic edge of what is realistic; do not
           push further by reaching into bar T + 1.

        5. No side effects. Do not modify the DataFrame you are given. Work on a
           copy or on derived Series, so that the same data can be reused across
           strategies and parameter sweeps.

        Args:
            data: OHLCV price history indexed by date, sorted chronologically,
                as returned by market_data.get_price_data.

        Returns:
            A pandas Series of signals aligned to data.index.
        """
        raise NotImplementedError

    @staticmethod
    def hold_signals(data: pd.DataFrame) -> pd.Series:
        """Return an all-HOLD Series aligned to the data, ready to be filled in.

        Convenience for implementations, which typically start from a neutral
        stance and mark only the bars where something actually happens.

        Args:
            data: Price history whose index the signals should follow.

        Returns:
            A Series of HOLD values indexed by data.index.
        """
        return pd.Series(HOLD, index=data.index, dtype=object)

    @property
    def name(self) -> str:
        """Human-readable strategy name, used in results tables and the UI."""
        return type(self).__name__

    def __repr__(self) -> str:
        arguments = ", ".join(f"{key}={value!r}" for key, value in self.params.items())
        return f"{type(self).__name__}({arguments})"
