"""Order execution: turns trading decisions into portfolio fills."""

from engine.portfolio import Portfolio

# Orders below this notional value are treated as no-ops. Without it, leftover
# floating point cash would produce meaningless microscopic trades.
MIN_ORDER_VALUE = 1e-8


class Broker:
    """Executes orders against a Portfolio.

    The Broker is the only component that decides *whether* a trade happens and
    *at what price* it is filled. The Portfolio merely applies the resulting
    fill. Keeping that boundary clean is what allows transaction costs to be
    introduced in Phase 4 without touching the Portfolio or the strategies.

    In Phase 1 the Broker supports a single, deliberately simple position model:
    the portfolio is either fully invested in the asset or fully in cash. There
    is no partial sizing, no leverage and no short selling.

    Attributes:
        portfolio: The Portfolio whose cash and position this Broker updates.
    """

    def __init__(self, portfolio: Portfolio) -> None:
        """Bind the Broker to a portfolio.

        Args:
            portfolio: The Portfolio instance to execute orders against.
        """
        self.portfolio = portfolio

    def buy_all(self, price: float) -> float:
        """Invest all available cash in the asset at the given price.

        The share quantity is derived from the cash on hand, so the order is by
        construction affordable. Fractional shares are allowed.

        Args:
            price: Execution price per share. Must be strictly positive.

        Returns:
            The number of shares bought. Returns 0.0 when there is not enough
            cash to place a meaningful order, in which case nothing is executed.

        Raises:
            ValueError: If price is not strictly positive.
        """
        self._validate_price(price)

        cash = self.portfolio.cash
        if cash < MIN_ORDER_VALUE:
            return 0.0

        # PHASE 4 PLACEHOLDER — transaction costs.
        # A buy will fill above the quoted price once market frictions are
        # modelled. The effective price will become something like:
        #     effective_price = price * (1 + spread / 2 + slippage)
        # and a commission will be charged on the notional value. The share
        # quantity must then be computed from the effective price and from the
        # cash left after commission, so that the order stays affordable.
        # For now execution is frictionless: effective price == quoted price.
        effective_price = price

        shares = cash / effective_price
        self.portfolio.buy(shares, effective_price)
        return shares

    def sell_all(self, price: float) -> float:
        """Liquidate the entire position at the given price.

        Args:
            price: Execution price per share. Must be strictly positive.

        Returns:
            The number of shares sold. Returns 0.0 when no position is held, in
            which case nothing is executed.

        Raises:
            ValueError: If price is not strictly positive.
        """
        self._validate_price(price)

        shares = self.portfolio.shares
        if shares * price < MIN_ORDER_VALUE:
            return 0.0

        # PHASE 4 PLACEHOLDER — transaction costs.
        # A sell will fill below the quoted price, mirroring the buy side:
        #     effective_price = price * (1 - spread / 2 - slippage)
        # with a commission deducted from the proceeds.
        # For now execution is frictionless: effective price == quoted price.
        effective_price = price

        self.portfolio.sell(shares, effective_price)
        return shares

    @staticmethod
    def _validate_price(price: float) -> None:
        """Reject non-positive execution prices."""
        if price <= 0:
            raise ValueError(f"Execution price must be strictly positive, got {price}.")

    def __repr__(self) -> str:
        return f"Broker(portfolio={self.portfolio!r})"


if __name__ == "__main__":
    portfolio = Portfolio(initial_capital=10_000.0)
    broker = Broker(portfolio)

    bought = broker.buy_all(price=100.0)
    print(f"Bought {bought} shares -> {portfolio}")

    sold = broker.sell_all(price=125.0)
    print(f"Sold {sold} shares -> {portfolio}")

    print(f"Selling with no position returns {broker.sell_all(price=125.0)}")
