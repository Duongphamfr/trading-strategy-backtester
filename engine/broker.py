"""Order execution: turns trading decisions into portfolio fills."""

from engine.portfolio import Portfolio

# Orders below this notional value are treated as no-ops. Without it, leftover
# floating point cash would produce meaningless microscopic trades.
MIN_ORDER_VALUE = 1e-8


class Broker:
    """Executes orders against a Portfolio, charging transaction costs.

    The Broker is the only component that decides *whether* a trade happens and
    *at what price* it is filled. The Portfolio merely applies the resulting
    fill. Keeping that boundary clean is what allowed transaction costs to be
    introduced in Phase 4 without touching the Portfolio or the strategies.

    The Broker supports a single, deliberately simple position model: the
    portfolio is either fully invested in the asset or fully in cash. There is
    no partial sizing, no leverage and no short selling.

    THE THREE COSTS, AND WHY THEY ARE MODELLED SEPARATELY
    They behave differently even though all three reduce returns, and separating
    them lets a scenario say which friction the conclusion depends on.

    The bid-ask spread is the gap between what a buyer pays and a seller
    receives. A market order crosses half of it in each direction, hence the
    spread / 2 in each formula. It is a property of the instrument.

    Slippage is the difference between the price a trade was decided at and the
    price it actually filled at, from the market moving in the meantime. It is a
    property of the order and of how fast the market is going.

    Commission is what the broker charges, here proportional to the trade value.
    It is a property of the account.

    Spread and slippage move the fill price. Commission is levied on top of it.
    That distinction is not cosmetic, and getting it wrong is what causes the
    overdraft described in buy_all.

    ALL COSTS DEFAULT TO ZERO
    Frictionless execution stays the default, so every result produced before
    Phase 4 reproduces exactly. That matters more than convenience: a cost
    scenario is only interpretable against a zero-cost baseline computed by the
    identical code path.

    Attributes:
        portfolio: The Portfolio whose cash and position this Broker updates.
        commission: Proportional commission per trade, e.g. 0.001 for 0.1%.
        spread: Bid-ask spread as a fraction of price. Half is paid per side.
        slippage: Adverse price move as a fraction of price, per side.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        commission: float = 0.0,
        spread: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Bind the Broker to a portfolio and set its cost model.

        Args:
            portfolio: The Portfolio instance to execute orders against.
            commission: Proportional commission charged on the value of every
                trade. 0.001 means 0.1% per trade, charged on both the buy and
                the sell, so a round trip costs roughly 0.2%.
            spread: Bid-ask spread as a fraction of the quoted price. Half of it
                is crossed on each side, so 0.001 costs 0.05% per trade.
            slippage: Adverse price move as a fraction of the quoted price,
                applied against the trade on each side.

        Raises:
            ValueError: If any cost is negative, if commission reaches 1, or if
                spread / 2 + slippage reaches 1. The last two would drive a sale
                to zero or negative proceeds, which is not a cost model but an
                arithmetic breakdown, and is far better caught here than as a
                confusing failure deep inside the Portfolio.
        """
        commission = float(commission)
        spread = float(spread)
        slippage = float(slippage)

        if commission < 0.0 or spread < 0.0 or slippage < 0.0:
            raise ValueError(
                f"Transaction costs cannot be negative, got commission="
                f"{commission}, spread={spread}, slippage={slippage}. A negative "
                f"cost would pay the strategy to trade."
            )
        if commission >= 1.0:
            raise ValueError(
                f"commission must be below 1.0 (100% of trade value), got "
                f"{commission}."
            )
        if spread / 2.0 + slippage >= 1.0:
            raise ValueError(
                f"spread / 2 + slippage must be below 1.0, got spread={spread} "
                f"and slippage={slippage}, which totals "
                f"{spread / 2.0 + slippage}. At or beyond 1.0 a sale would "
                f"return nothing or less than nothing."
            )

        self.portfolio = portfolio
        self.commission = commission
        self.spread = spread
        self.slippage = slippage

    def buy_fill_price(self, price: float) -> float:
        """All-in cost per share of buying at the given quoted price.

        Spread and slippage push the fill above the quote; commission is then
        charged on that fill value. Folding the commission into a per-share
        figure is what keeps the rest of the accounting simple: this single
        number, multiplied by the share count, is the total cash that leaves the
        portfolio, so the Portfolio needs no notion of costs at all.

        Args:
            price: Quoted price per share.

        Returns:
            The effective per-share cash outflow, including commission.
        """
        return price * (1.0 + self.spread / 2.0 + self.slippage) * (1.0 + self.commission)

    def sell_fill_price(self, price: float) -> float:
        """All-in proceeds per share of selling at the given quoted price.

        The mirror image of buy_fill_price: spread and slippage push the fill
        below the quote, and commission is deducted from the proceeds.

        Args:
            price: Quoted price per share.

        Returns:
            The effective per-share cash inflow, net of commission.
        """
        return price * (1.0 - self.spread / 2.0 - self.slippage) * (1.0 - self.commission)

    def buy_all(self, price: float) -> float:
        """Invest all available cash in the asset at the given price.

        The share quantity is derived from the cash on hand, so the order is by
        construction affordable. Fractional shares are allowed.

        WHY THE SHARE COUNT DIVIDES BY THE ALL-IN PRICE
        This is the correctness trap the Phase 1 placeholder warned about, and it
        is worth spelling out because the wrong version looks right.

        The tempting formula is to size the order from the fill price and then
        charge the commission:

            shares = cash / (price * (1 + spread / 2 + slippage))
            commission_paid = commission * shares * fill_price

        That spends all the cash on shares first and only then discovers it owes
        the commission, so the total outlay exceeds the cash by exactly
        commission * cash. With a 0.1% commission on 10,000 of cash the account
        ends 10 in the red. The Portfolio would reject the order outright, and
        the backtest would crash rather than quietly mis-report, but only
        because the Portfolio happens to guard its cash; the sizing would still
        be wrong.

        The fix is to treat the commission as part of the unit price rather than
        as an afterthought. Solving

            shares * fill_price * (1 + commission) = cash

        for the share count gives the division below. Every component of the cost
        is inside buy_fill_price, so the arithmetic reduces to cash divided by
        the all-in price, the outlay is exactly the cash available, and the
        position can never overdraw.

        Args:
            price: Quoted price per share. Must be strictly positive.

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

        fill_price = self.buy_fill_price(price)
        shares = cash / fill_price

        # Booking the trade at the all-in price is what keeps the Portfolio free
        # of any cost logic: the cash it deducts is already the true total
        # outlay. The position is still marked to market at the quoted price
        # afterwards, so costs are recognised immediately as a loss of value,
        # which is exactly what they are.
        self.portfolio.buy(shares, fill_price)
        return shares

    def sell_all(self, price: float) -> float:
        """Liquidate the entire position at the given price.

        The sell side needs no equivalent of the buy-side sizing care: the share
        count is whatever is held, and the costs only reduce the proceeds. There
        is no budget to overrun.

        Args:
            price: Quoted price per share. Must be strictly positive.

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

        self.portfolio.sell(shares, self.sell_fill_price(price))
        return shares

    @staticmethod
    def _validate_price(price: float) -> None:
        """Reject non-positive execution prices."""
        if price <= 0:
            raise ValueError(f"Execution price must be strictly positive, got {price}.")

    def __repr__(self) -> str:
        return (
            f"Broker(commission={self.commission}, spread={self.spread}, "
            f"slippage={self.slippage})"
        )


if __name__ == "__main__":
    CAPITAL = 10_000.0
    QUOTE = 100.0

    print("Frictionless execution is unchanged")
    portfolio = Portfolio(initial_capital=CAPITAL)
    broker = Broker(portfolio)
    bought = broker.buy_all(price=QUOTE)
    print(f"  bought {bought:,.6f} shares at {QUOTE}, cash {portfolio.cash:.10f}")
    sold = broker.sell_all(price=125.0)
    print(f"  sold   {sold:,.6f} shares at 125.00, cash {portfolio.cash:,.10f}")
    print(f"  selling with no position returns {broker.sell_all(price=125.0)}")

    # THE OVERDRAFT THE NAIVE SIZING WOULD CAUSE
    # Sizing from the fill price and charging commission afterwards spends the
    # whole balance on shares and then owes the commission on top. The gap is
    # commission * cash exactly, which is a number worth seeing rather than
    # taking on trust.
    print("\nWhy the share count divides by the all-in price")
    commission = 0.001
    priced = Broker(Portfolio(CAPITAL), commission=commission)
    fill = priced.buy_fill_price(QUOTE) / (1.0 + commission)

    naive_shares = CAPITAL / fill
    naive_outlay = naive_shares * fill * (1.0 + commission)
    correct_shares = CAPITAL / priced.buy_fill_price(QUOTE)
    correct_outlay = correct_shares * priced.buy_fill_price(QUOTE)

    print(f"  naive:   {naive_shares:,.6f} shares -> outlay "
          f"{naive_outlay:,.4f} on {CAPITAL:,.2f} of cash "
          f"(over by {naive_outlay - CAPITAL:,.4f})")
    print(f"  correct: {correct_shares:,.6f} shares -> outlay "
          f"{correct_outlay:,.4f} (over by {correct_outlay - CAPITAL:.2e})")
    print(f"  the gap is exactly commission x cash: "
          f"{abs(naive_outlay - CAPITAL - commission * CAPITAL):.2e}")

    print("\nA round trip at a flat quote loses only the costs")
    print(f"  {'commission':>10} {'spread':>8} {'slippage':>9} "
          f"{'end cash':>12} {'round-trip':>11} {'expected':>11}")
    for commission, spread, slippage in [
        (0.0, 0.0, 0.0),
        (0.001, 0.0, 0.0),
        (0.0, 0.002, 0.0),
        (0.0, 0.0, 0.001),
        (0.001, 0.002, 0.001),
    ]:
        book = Portfolio(initial_capital=CAPITAL)
        desk = Broker(book, commission=commission, spread=spread, slippage=slippage)
        desk.buy_all(price=QUOTE)
        desk.sell_all(price=QUOTE)

        realised = book.cash / CAPITAL - 1.0
        # Buying then selling at the same quote multiplies capital by
        # sell_fill_price / buy_fill_price, so the loss is predictable in
        # closed form. Matching it proves the two sides are consistent.
        expected = desk.sell_fill_price(QUOTE) / desk.buy_fill_price(QUOTE) - 1.0
        print(f"  {commission:>10.4f} {spread:>8.4f} {slippage:>9.4f} "
              f"{book.cash:>12,.4f} {realised:>10.4%} {expected:>10.4%}")

    print("\nCash is never overdrawn, at any cost level")
    for commission in (0.0, 0.001, 0.01, 0.1, 0.5):
        book = Portfolio(initial_capital=CAPITAL)
        Broker(book, commission=commission, spread=0.01, slippage=0.005).buy_all(QUOTE)
        print(f"  commission {commission:>5.3f} -> cash after buy "
              f"{book.cash:>12.2e}  negative: {book.cash < -1e-9}")

    print("\nRejected cost parameters")
    for arguments in ({"commission": -0.001}, {"commission": 1.0},
                      {"spread": 1.5, "slippage": 0.3}, {"slippage": -1.0}):
        try:
            Broker(Portfolio(CAPITAL), **arguments)
        except ValueError as error:
            print(f"  {arguments} -> ValueError: {str(error)[:54]}...")
