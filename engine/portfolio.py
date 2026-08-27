"""Portfolio accounting: cash, share position, and value over time."""

from typing import Any, Dict, List

import pandas as pd

# Cash comparisons are made with a small tolerance so that floating point noise
# does not reject an order that spends exactly the available cash.
CASH_TOLERANCE = 1e-9


class Portfolio:
    """Tracks cash, the share position, and total value of a single-asset portfolio.

    The Portfolio is deliberately passive: it knows nothing about strategies,
    prices feeds or order logic. It only applies the fills it is given and keeps
    a snapshot of its own state over time. The Broker decides *whether* and *at
    what price* a trade happens; the Portfolio just records the consequences.

    Share quantities are stored as floats, so fractional shares are allowed.
    Rounding to whole shares, if desired, is the Broker's responsibility.

    Attributes:
        initial_capital: Starting cash, kept for reference and return calculations.
        cash: Uninvested cash currently available.
        shares: Number of shares currently held. Never negative, since Phase 1
            does not support short selling.
    """

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        """Create a portfolio holding only cash.

        Args:
            initial_capital: Starting amount of cash. Must be strictly positive.

        Raises:
            ValueError: If initial_capital is not strictly positive.
        """
        if initial_capital <= 0:
            raise ValueError(
                f"Initial capital must be strictly positive, got {initial_capital}."
            )

        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.shares = 0.0
        self._history: List[Dict[str, Any]] = []

    def buy(self, shares: float, price: float) -> None:
        """Add shares to the position and deduct the cost from cash.

        Args:
            shares: Number of shares bought. Must be strictly positive.
            price: Execution price per share. Must be strictly positive.

        Raises:
            ValueError: If shares or price is not strictly positive, or if the
                order would cost more than the available cash.
        """
        self._validate_order(shares, price)

        cost = shares * price
        if cost > self.cash + CASH_TOLERANCE:
            raise ValueError(
                f"Insufficient cash: buying {shares} shares at {price} costs "
                f"{cost}, but only {self.cash} is available."
            )

        self.cash -= cost
        self.shares += shares

    def sell(self, shares: float, price: float) -> None:
        """Remove shares from the position and add the proceeds to cash.

        Args:
            shares: Number of shares sold. Must be strictly positive.
            price: Execution price per share. Must be strictly positive.

        Raises:
            ValueError: If shares or price is not strictly positive, or if the
                order would sell more shares than are currently held.
        """
        self._validate_order(shares, price)

        if shares > self.shares + CASH_TOLERANCE:
            raise ValueError(
                f"Cannot sell {shares} shares: only {self.shares} are held. "
                f"Short selling is not supported."
            )

        self.cash += shares * price
        self.shares -= shares

    def position_value(self, price: float) -> float:
        """Return the market value of the share position at the given price."""
        return self.shares * price

    def total_value(self, price: float) -> float:
        """Return total portfolio value: cash plus the value of the position."""
        return self.cash + self.position_value(price)

    def record(self, date: pd.Timestamp, price: float) -> None:
        """Append a snapshot of the portfolio state on a given date.

        Called once per bar by the backtester, this builds the equity curve used
        later by the analytics layer.

        Args:
            date: Date of the snapshot, used as the index of the history frame.
            price: Price used to mark the position to market on that date.
        """
        self._history.append(
            {
                "date": date,
                "price": price,
                "cash": self.cash,
                "shares": self.shares,
                "position_value": self.position_value(price),
                "total_value": self.total_value(price),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the recorded history as a DataFrame indexed by date.

        Returns:
            A pandas.DataFrame indexed by date with the columns price, cash,
            shares, position_value and total_value. If nothing has been recorded
            yet, an empty DataFrame with those same columns is returned, so that
            downstream code can rely on the schema either way.
        """
        columns = ["price", "cash", "shares", "position_value", "total_value"]

        if not self._history:
            empty_index = pd.DatetimeIndex([], name="date")
            return pd.DataFrame(columns=columns, index=empty_index, dtype=float)

        df = pd.DataFrame(self._history)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[columns]

    @staticmethod
    def _validate_order(shares: float, price: float) -> None:
        """Reject non-positive order quantities and prices."""
        if shares <= 0:
            raise ValueError(f"Share quantity must be strictly positive, got {shares}.")
        if price <= 0:
            raise ValueError(f"Price must be strictly positive, got {price}.")

    def __repr__(self) -> str:
        return (
            f"Portfolio(cash={self.cash:.2f}, shares={self.shares:.4f}, "
            f"initial_capital={self.initial_capital:.2f})"
        )


if __name__ == "__main__":
    portfolio = Portfolio(initial_capital=10_000.0)
    print(portfolio)

    portfolio.buy(shares=50.0, price=100.0)
    portfolio.record(pd.Timestamp("2022-01-03"), price=100.0)

    portfolio.sell(shares=20.0, price=120.0)
    portfolio.record(pd.Timestamp("2022-01-04"), price=120.0)

    print(portfolio)
    print(portfolio.to_dataframe())
