"""Module for downloading and cleaning historical price data from Yahoo Finance."""

import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV price data for a single ticker from Yahoo Finance.

    Args:
        ticker: Ticker symbol using Yahoo Finance notation, e.g. "AAPL", "MSFT".
        start: Start date in "YYYY-MM-DD" format (inclusive).
        end: End date in "YYYY-MM-DD" format (exclusive).

    Returns:
        A pandas.DataFrame indexed by trading date (a DatetimeIndex sorted in
        ascending order) with exactly five columns: Open, High, Low, Close and
        Volume. Prices are already adjusted for stock splits and dividends
        (auto_adjust=True), so the Close column can be fed straight into the
        backtest without any further adjustment.

    Raises:
        ValueError: If Yahoo Finance returns no data at all, which usually means
            an invalid ticker symbol, an invalid date range, or a network error.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"Failed to download data for ticker '{ticker}' between "
            f"{start} and {end}. Please check that the ticker symbol follows "
            f"Yahoo Finance notation and that the date range contains valid "
            f"trading days."
        )

    # yfinance returns MultiIndex columns (Price, Ticker) when downloading
    # several tickers, and in some versions even for a single ticker. Keep the
    # level that holds the OHLCV names.
    if isinstance(df.columns, pd.MultiIndex):
        price_level = 0
        for level in range(df.columns.nlevels):
            if set(OHLCV_COLUMNS) & set(df.columns.get_level_values(level)):
                price_level = level
                break
        df.columns = df.columns.get_level_values(price_level)

    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"The data returned for ticker '{ticker}' is missing required "
            f"columns: {', '.join(missing)}."
        )

    df = df[OHLCV_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    # Drop days with no closing price: the backtester values the portfolio on
    # Close, so a row without a Close is unusable.
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(
            f"After cleaning, no valid rows remain for ticker '{ticker}' "
            f"between {start} and {end}. Please try a different date range."
        )

    return df


if __name__ == "__main__":
    data = get_price_data("AAPL", "2022-01-01", "2023-01-01")
    print(data.head())
    print(f"\nTotal rows: {len(data)}")
