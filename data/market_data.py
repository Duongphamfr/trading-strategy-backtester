"""Download, clean and cache historical price data from Yahoo Finance.

WHY THERE IS A CACHE, AND WHY IT IS NOT AN OPTIMISATION
Yahoo Finance recomputes split and dividend adjustments continuously, so two
downloads of the same ticker over the same dates minutes apart can return prices
that differ in the last decimals. For a single backtest that is irrelevant. For
everything Phase 4 is about it is fatal: a parameter sweep compares hundreds of
runs against each other, and walk-forward validation compares in-sample against
out-of-sample windows. If the prices shift underneath those comparisons, the
differences being measured are partly noise from the data provider, and there is
no way to tell how much. Worse, a result becomes impossible to reproduce, which
is the one property a research project cannot do without.

Caching fixes the input. The first call downloads and writes the cleaned frame to
disk; every later call with the same arguments reads that file and therefore sees
byte-identical prices, today and next month. Speed is a side benefit.

WHY CSV, NOT PARQUET
Parquet would be the obvious choice: it stores dtypes natively, so a round trip
is exact by construction, and it is smaller and faster. It needs pyarrow or
fastparquet, neither of which is currently installed, and pulling in a large
binary dependency to cache a few hundred kilobytes of daily bars is a poor trade.
CSV needs nothing beyond pandas, and it stays readable, diffable and repairable
by hand, which matters for a file a researcher may want to inspect.

CSV does not round-trip exactly by default, and getting that right took measuring
rather than assuming. Writing is fine: pandas emits the shortest decimal string
that identifies each double uniquely. Reading is not. The default C parser uses a
fast float conversion that is not correctly rounded, and it returns values one
unit in the last place away from what was written, on roughly one row in twenty
of real AAPL data. Passing float_precision="round_trip" selects a correctly
rounded parser and makes the round trip bit-exact, verified on both real and
adversarial data. Without that single argument the cache would quietly reintroduce
the very drift it exists to remove, only smaller.

TWO WAYS A FETCH CAN FAIL, AND WHY THEY ARE DIFFERENT EXCEPTIONS
"There is no such ticker" and "I could not reach the source" call for different
answers from a caller, so this module raises different types. The first is a
ValueError: the argument was wrong and repeating the call cannot help. The second
is DataSourceUnavailable: the argument may be perfectly good and the same call may
well succeed in a minute. A user offline, behind a proxy or being rate limited is
in the second case, and telling them to check their ticker symbol would send them
looking in the wrong place.

Without the distinction the second case did not merely produce a poor message: the
transport's own exception escaped this module entirely, and the dashboard showed a
raw traceback because its handler had no reason to expect an exception type from a
library it never imports. Translating here is what lets every caller handle both
cases without knowing which HTTP stack yfinance currently uses.
"""

import hashlib
import os
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Anchored to this file rather than the working directory, so the cache is found
# whether a script is launched from the project root or anywhere else.
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Part of the cache key. Bump it whenever the cleaning below changes the shape or
# the dtypes of the result, so that files written by an older version are ignored
# rather than silently served in a format the rest of the project no longer
# expects. Old files become unreachable and can simply be deleted.
CACHE_VERSION = 1

# Both paths force the index to one resolution, which is what lets a cached frame
# be indistinguishable from a freshly downloaded one. yfinance currently yields
# datetime64[s] while read_csv yields datetime64[us]: same instants, different
# dtype, and enough to make an equality assertion fail. Seconds are chosen
# because they match what the download already returns and are far finer than
# daily bars require.
INDEX_DTYPE = "datetime64[s]"


class DataSourceUnavailable(OSError):
    """The price source could not be reached, so there is no answer yet.

    Distinct from the ValueError raised for an unknown ticker: this one says
    nothing about whether the request was reasonable, only that it could not be
    delivered. A caller may sensibly retry it, which is never true of a ValueError
    from this module.

    It derives from OSError rather than from Exception because that is the
    category its own causes come from, and it keeps a caller that already handles
    OSError working without having to learn this class exists.
    """


# How far the data may fall short of the request before that is worth explaining.
# A request landing on a weekend or a holiday needs no comment: the longest modern
# closure of the US equity market ran four days, which with the weekends on either
# side makes a week the natural cut. Beyond it the history genuinely does not
# reach, and saying so is the difference between a label and a guess.
RANGE_TOLERANCE_DAYS = 7


def covered_range(prices: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """First and last bar actually present in a price frame.

    Args:
        prices: A frame indexed by date, as returned by get_price_data.

    Returns:
        The first and last index entries.

    Raises:
        ValueError: If the frame is empty, which has no range to report.
    """
    if prices.empty:
        raise ValueError("An empty price frame covers no range.")
    return prices.index[0], prices.index[-1]


def period_label(
    prices: pd.DataFrame,
    requested_start: Optional[str] = None,
    requested_end: Optional[str] = None,
) -> str:
    """Describe the period a price frame actually covers.

    WHY THE REQUEST IS THE WRONG THING TO PRINT
    Yahoo returns what it has, not what was asked for, so a ticker listed in 2024
    answers a request starting in 2020 with an eight-times-shorter history and no
    complaint. Labelling the request made every report of such a ticker overstate
    its own study period: a header reading "2020-01-01 to 2026-08-01 (592 bars)"
    contradicted itself, since 6.6 years of daily bars is closer to 1,660 than to
    592, and a Sharpe ratio earned over 2.4 years was presented as the product of
    6.6. No computed figure was ever affected, all of them being driven by the
    bars themselves, but the sentence describing them was wrong.

    The request is still worth showing when it differs materially, otherwise
    someone who typed 2020 and reads 2024 would reasonably wonder whether their
    input was ignored. Within RANGE_TOLERANCE_DAYS the difference is just the
    calendar and is left unsaid.

    This lives beside the loader rather than in a display module because it states
    a fact about the data, and because five call sites print this label; one
    implementation is what stops them from disagreeing.

    Args:
        prices: A frame indexed by date, as returned by get_price_data.
        requested_start: The start that was asked for, as YYYY-MM-DD. Optional;
            without it no comparison is made and only the actual range is shown.
        requested_end: The end that was asked for, as YYYY-MM-DD.

    Returns:
        A string such as "2020-01-02 to 2022-12-30", with the request appended in
        parentheses when the data falls materially short of it.

    Raises:
        ValueError: If the frame is empty.
    """
    first, last = covered_range(prices)
    actual = f"{first.date()} to {last.date()}"

    if requested_start is None and requested_end is None:
        return actual

    tolerance = pd.Timedelta(days=RANGE_TOLERANCE_DAYS)
    clipped = False
    if requested_start is not None:
        clipped = clipped or (first - pd.Timestamp(requested_start)) > tolerance
    if requested_end is not None:
        clipped = clipped or (pd.Timestamp(requested_end) - last) > tolerance

    if not clipped:
        return actual

    asked = " to ".join(part for part in (requested_start, requested_end) if part)
    return f"{actual} (requested {asked})"


def get_price_data(
    ticker: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return historical OHLCV price data for a single ticker, cached on disk.

    The first call for a given ticker and date range downloads from Yahoo
    Finance, cleans the result and stores it under data/cache/. Later calls with
    the same arguments read that file and never touch the network, so repeated
    runs, parameter sweeps and walk-forward windows all see exactly the same
    prices. See the module docstring for why that matters more than it sounds.

    A cache file that is missing, unreadable, truncated or written by an older
    cache version is treated as absent: the function warns and downloads. There
    is therefore no state in which a corrupted cache can break a caller, and no
    need to clear the directory by hand after an interrupted run.

    Args:
        ticker: Ticker symbol using Yahoo Finance notation, e.g. "AAPL", "MSFT".
        start: Start date in "YYYY-MM-DD" format (inclusive).
        end: End date in "YYYY-MM-DD" format (exclusive).
        force_refresh: Download even if a cache file exists, and overwrite it.
            Use this to pick up genuinely new data, for instance when extending
            the end date, or to refresh prices that have since been re-adjusted.

    Returns:
        A pandas.DataFrame indexed by trading date (a DatetimeIndex sorted in
        ascending order) with exactly five columns: Open, High, Low, Close and
        Volume. Prices are already adjusted for stock splits and dividends
        (auto_adjust=True), so the Close column can be fed straight into the
        backtest without any further adjustment. A cached result is identical to
        a downloaded one, dtypes included.

    Raises:
        DataSourceUnavailable: If the network or the provider prevented the
            download. A cached range never reaches the network and so never
            raises this, which is worth knowing when working offline.
        ValueError: If Yahoo Finance was reached and returned no data at all,
            which usually means an invalid ticker symbol or a date range holding
            no trading days.
    """
    path = _cache_path(ticker, start, end)

    if not force_refresh:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    frame = _canonicalise(_download(ticker, start, end), ticker, start, end)
    _write_cache(path, frame)
    return frame


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch raw daily bars from Yahoo Finance.

    WHERE THE LINE IS DRAWN BETWEEN A FAILED FETCH AND A BUG
    Two exception trees are translated into DataSourceUnavailable, and everything
    else is left to propagate.

    OSError is Python's own category for an external resource failing, and every
    HTTP stack yfinance has shipped puts its exceptions inside it: requests and
    curl_cffi both derive RequestException from OSError, as do urllib's URLError
    and the socket layer's own errors, along with the builtin ConnectionError and
    TimeoutError. Catching the base rather than a list of leaves is what keeps
    this working when the library changes transport, which it has already done
    once.

    YFException is yfinance's own base, for the failures it detects itself rather
    than at the transport layer. YFRateLimitError is the one that matters: being
    throttled is a temporary condition with a perfectly valid ticker, and it does
    not arrive as an OSError.

    Anything outside those two trees propagates untouched, and that is the point
    of naming them rather than catching Exception. A TypeError means these
    arguments do not match the signature; an AttributeError or a KeyError means
    the library moved and the call needs revisiting. Those are our bugs, they are
    not fixed by retrying, and a banner reading "check your connection" would hide
    exactly the kind of breakage that must be loud during development.

    The residual risk is stated rather than papered over: an exception from
    neither tree would once again reach the caller unhandled. The test suite pins
    the current behaviour so that such a change is a failing test rather than a
    surprise in the browser.

    Args:
        ticker: Ticker symbol using Yahoo Finance notation.
        start: Start date in "YYYY-MM-DD" format (inclusive).
        end: End date in "YYYY-MM-DD" format (exclusive).

    Returns:
        Whatever yfinance returned, uncleaned.

    Raises:
        DataSourceUnavailable: If the source could not be reached or refused the
            request.
        ValueError: If the source was reached and returned nothing.
    """
    try:
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except (OSError, YFException) as error:
        raise DataSourceUnavailable(
            f"Could not reach Yahoo Finance to download '{ticker}' "
            f"({type(error).__name__}: {error}). This is a problem with the "
            f"connection to the data source, not with the ticker symbol. Check "
            f"that you are online, and if you are behind a proxy or have been "
            f"rate limited, waiting a moment and retrying usually works. Any "
            f"range already in the local cache still loads normally."
        ) from error

    if frame is None or frame.empty:
        raise ValueError(
            f"Failed to download data for ticker '{ticker}' between "
            f"{start} and {end}. Please check that the ticker symbol follows "
            f"Yahoo Finance notation and that the date range contains valid "
            f"trading days. A source that is refusing requests can also return "
            f"nothing, so if the symbol is definitely right, try again shortly."
        )

    return frame


def _canonicalise(
    frame: pd.DataFrame,
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Reduce a raw frame to the project's canonical shape.

    Both the download path and the cache-read path go through this one function,
    which is what makes a cached frame indistinguishable from a fresh one. Had
    each path done its own tidying, the two would drift apart on some detail such
    as column order or index resolution, and the reproducibility guarantee would
    hold only approximately. Every step is idempotent, so running it on data that
    is already canonical is harmless.

    Args:
        frame: Raw OHLCV data, from yfinance or from a cache file.
        ticker: Ticker symbol, used only in error messages.
        start: Start date, used only in error messages.
        end: End date, used only in error messages.

    Returns:
        A frame with exactly the OHLCV columns, a sorted DatetimeIndex named
        Date at INDEX_DTYPE resolution, and no rows lacking a closing price.

    Raises:
        ValueError: If required columns are absent, or if nothing survives
            cleaning.
    """
    # yfinance returns MultiIndex columns (Price, Ticker) when downloading
    # several tickers, and in some versions even for a single ticker. Keep the
    # level that holds the OHLCV names.
    if isinstance(frame.columns, pd.MultiIndex):
        price_level = 0
        for level in range(frame.columns.nlevels):
            if set(OHLCV_COLUMNS) & set(frame.columns.get_level_values(level)):
                price_level = level
                break
        frame.columns = frame.columns.get_level_values(price_level)

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"The data returned for ticker '{ticker}' is missing required "
            f"columns: {', '.join(missing)}."
        )

    frame = frame[OHLCV_COLUMNS].copy()

    # yfinance labels the column index "Price"; read_csv leaves it unnamed. Pure
    # metadata, invisible in every calculation, and still enough to make the two
    # paths compare as different frames. Clearing it is what makes the equality
    # assertion in __main__ a real check rather than a near-miss.
    frame.columns.name = None

    frame.index = pd.to_datetime(frame.index).astype(INDEX_DTYPE)
    frame.index.name = "Date"
    frame = frame.sort_index()

    # Drop days with no closing price: the backtester values the portfolio on
    # Close, so a row without a Close is unusable.
    frame = frame.dropna(subset=["Close"])

    if frame.empty:
        raise ValueError(
            f"After cleaning, no valid rows remain for ticker '{ticker}' "
            f"between {start} and {end}. Please try a different date range."
        )

    return frame


def _cache_path(ticker: str, start: str, end: str) -> Path:
    """Build the cache file path for one ticker and date range.

    The name carries a readable part so the directory can be understood at a
    glance, and a hash of the exact key so it cannot be misread. The hash is not
    decoration: sanitising a ticker for the filesystem is lossy, and "BRK-B",
    "BRK.B" and "BRK/B" would otherwise collapse onto one filename and serve each
    other's prices. Silently returning the wrong asset's data is the worst
    failure this module could have, and ten hex characters rule it out.

    The cache version is part of the hashed key, so bumping it retires every
    existing file at once.

    Args:
        ticker: Ticker symbol using Yahoo Finance notation.
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.

    Returns:
        The path the cache file for these arguments would occupy.
    """
    key = f"v{CACHE_VERSION}|{ticker}|{start}|{end}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]

    readable = re.sub(r"[^A-Za-z0-9._-]", "-", f"{ticker}_{start}_{end}")[:60]
    return CACHE_DIR / f"{readable}_{digest}.csv"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Load a cached frame, or return None if it cannot be trusted.

    float_precision="round_trip" is the argument that makes the cache honest.
    Without it the default parser returns values one unit in the last place away
    from what was written, which would put a small non-reproducible wobble back
    into the prices. See the module docstring.

    Every failure mode collapses to the same answer: return None and let the
    caller download. That is why the except clause is deliberately broad. A
    truncated file raises a parser error, a file from an interrupted write may
    parse but lack columns, a directory that is unreadable raises an OSError, and
    none of those should be a caller's problem when the data can simply be
    fetched again.

    Args:
        path: Cache file to read.

    Returns:
        The canonicalised frame, or None if the file is absent or unusable.
    """
    if not path.exists():
        return None

    try:
        frame = pd.read_csv(
            path,
            index_col="Date",
            parse_dates=["Date"],
            float_precision="round_trip",
        )
        return _canonicalise(frame, path.stem, "cache", "cache")
    except Exception as error:
        warnings.warn(
            f"Ignoring unusable cache file {path.name} ({type(error).__name__}: "
            f"{error}). The data will be downloaded again and the file "
            f"overwritten.",
            stacklevel=3,
        )
        return None


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    """Store a canonical frame, replacing any existing file atomically.

    The write goes to a temporary file and is then moved into place, because
    os.replace is atomic on every platform that matters. A process killed midway
    through leaves a stray .tmp file rather than a half-written cache entry.
    Since a truncated cache entry is exactly the corruption _read_cache has to
    cope with, it is worth not manufacturing any.

    A failure here is reported and swallowed. The caller already holds the data
    it asked for, and a full disk or a read-only checkout is no reason to fail a
    backtest; the only cost is that the next call downloads again.

    Args:
        path: Destination cache file.
        frame: Canonical frame to store.
    """
    temporary = path.with_name(path.name + ".tmp")

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(temporary)
        os.replace(temporary, path)
    except OSError as error:
        warnings.warn(
            f"Could not write cache file {path.name} ({type(error).__name__}: "
            f"{error}). The data is still returned, but the next call will "
            f"download again.",
            stacklevel=3,
        )
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    import time

    TICKER, START, END = "AAPL", "2022-01-01", "2023-01-01"
    path = _cache_path(TICKER, START, END)

    def timed(**kwargs) -> tuple:
        """Fetch the demo range and report how long it took."""
        started = time.perf_counter()
        frame = get_price_data(TICKER, START, END, **kwargs)
        return frame, time.perf_counter() - started

    print(f"Cache file: {path.relative_to(Path(__file__).resolve().parent.parent)}")

    # force_refresh guarantees the first leg really downloads, so the demo shows
    # the same thing whether or not a cache already existed from an earlier run.
    downloaded, download_seconds = timed(force_refresh=True)
    print(f"\n1. force_refresh=True  -> downloaded {len(downloaded)} rows "
          f"in {download_seconds:6.3f}s")
    print(f"   cache written: {path.exists()}  ({path.stat().st_size:,} bytes)")

    cached, cache_seconds = timed()
    print(f"2. second call         -> loaded     {len(cached)} rows "
          f"in {cache_seconds:6.3f}s"
          f"   ({download_seconds / max(cache_seconds, 1e-9):.0f}x faster)")

    # assert_frame_equal compares values, dtypes, column order and the index, so
    # this is the real claim: a cached frame is not merely close to a downloaded
    # one, it is indistinguishable from it.
    pd.testing.assert_frame_equal(downloaded, cached)
    print("3. assert_frame_equal(downloaded, cached) passed: identical values, "
          "dtypes and index")

    identical_bits = all(
        (downloaded[column].values == cached[column].values).all()
        for column in OHLCV_COLUMNS
    )
    print(f"   bit-for-bit identical on every column: {identical_bits}")

    print("\n4. Corrupted cache falls back to downloading")
    path.write_text("this is not a csv\x00\x01truncated")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recovered = get_price_data(TICKER, START, END)
    print(f"   warning raised: {caught[0].category.__name__}")
    print(f"   recovered {len(recovered)} rows, cache rewritten: "
          f"{path.stat().st_size:,} bytes")
    print(f"   index and columns match the original: "
          f"{recovered.index.equals(downloaded.index)}")

    # Whether the re-downloaded prices match the earlier ones is out of our
    # hands, and that is the whole argument for the cache. A mismatch here is
    # Yahoo having re-adjusted the series between two calls minutes apart.
    same_prices = downloaded["Close"].equals(recovered["Close"])
    print(f"   re-downloaded closes identical to the first download: "
          f"{same_prices}")
    if not same_prices:
        drift = (downloaded["Close"] - recovered["Close"]).abs().max()
        print(f"   -> Yahoo returned different adjusted prices, max drift "
              f"{drift:.2e}. This is exactly what the cache exists to freeze.")

    print("\n5. Reproducibility across repeated cached reads")
    first, second = get_price_data(TICKER, START, END), get_price_data(TICKER, START, END)
    pd.testing.assert_frame_equal(first, second)
    print("   two cached reads are identical: True")

    print(f"\n{cached.head()}")
    print(f"\nTotal rows: {len(cached)}")
