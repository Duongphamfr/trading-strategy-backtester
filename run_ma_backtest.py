"""Exploration script: run the MA crossover strategy against buy-and-hold.

Temporary scaffolding for Phase 2. It only wires existing modules together, so
it doubles as a demonstration of the architecture: the configuration lives here,
at the edge, while the engine and the strategy stay unaware of who called them.
Proper performance measurement arrives in Phase 3.
"""

from analytics.report import performance_report, print_report
from data.market_data import get_price_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageCrossover

TICKER = "AAPL"
START = "2020-01-01"
END = "2023-01-01"
INITIAL_CASH = 10_000.0
FAST_WINDOW = 50
SLOW_WINDOW = 200


def main() -> None:
    """Run one backtest and print what happened."""
    prices = get_price_data(TICKER, START, END)
    strategy = MovingAverageCrossover(
        fast_window=FAST_WINDOW,
        slow_window=SLOW_WINDOW,
    )
    result = Backtester(prices, initial_cash=INITIAL_CASH, strategy=strategy).run()

    strategy_value = result.final_value
    benchmark_value = result.benchmark_final_value
    difference = strategy_value - benchmark_value

    print(f"{TICKER}  {START} to {END}  ({len(prices)} bars)")
    print(f"Strategy:        {strategy!r}")
    print(f"Initial cash:    {INITIAL_CASH:>12,.2f}")
    print()
    print(f"Strategy value:  {strategy_value:>12,.2f}  "
          f"({strategy_value / INITIAL_CASH - 1:+.2%})")
    print(f"Benchmark value: {benchmark_value:>12,.2f}  "
          f"({benchmark_value / INITIAL_CASH - 1:+.2%})")
    print(f"Difference:      {difference:>12,.2f}")
    print(f"Trades executed: {len(result.trade_log):>12}")

    print("\nTrade log")
    if not result.trade_log:
        print("  (no trades)")
    else:
        print(f"  {'Date':<12} {'Action':<6} {'Price':>10} {'Shares':>12}")
        for trade in result.trade_log:
            print(
                f"  {trade['date'].date()!s:<12} {trade['action']:<6} "
                f"{trade['price']:>10,.2f} {trade['shares']:>12,.4f}"
            )

    print_report(
        performance_report(
            equity=result.equity_curve["total_value"],
            benchmark=result.benchmark_curve,
        ),
        title="Performance report",
    )


if __name__ == "__main__":
    main()
