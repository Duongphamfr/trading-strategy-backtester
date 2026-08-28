"""Automated tests for the data layer's two failure modes.

WHY THESE TWO CASES DESERVE TESTS OF THEIR OWN
"There is no such ticker" and "I could not reach the source" are the only two ways
a fetch fails, and a caller has to tell them apart: the first is answered by fixing
the argument, the second by waiting. The module promises a different exception type
for each, and the dashboard's error banner is built entirely on that promise.

The promise had a hole. Only the empty-result case was translated, so a lost
connection let the transport's own exception escape the module, and the dashboard
answered it with a traceback pointing into yfinance. What is locked here is the
translation and, just as importantly, its limit: an exception that means this
project has a bug must still travel unchanged.

NOTHING HERE TOUCHES THE NETWORK
yfinance's download function is replaced in every test. That is what makes the
failure modes reachable at all, since a real rate limit cannot be arranged on
demand, and it keeps the suite runnable on a train.
"""

import socket
import urllib.error
from typing import List, Tuple

import pandas as pd
import pytest
import requests
import yfinance
from yfinance.exceptions import YFException, YFRateLimitError

from data import market_data
from data.market_data import DataSourceUnavailable, get_price_data

TICKER = "TESTTICKER"
START, END = "2020-01-01", "2020-02-01"

# Representatives of every route a fetch failure can arrive by: the builtins the
# socket layer raises, the DNS and urllib wrappers, and the exception classes of
# both HTTP stacks yfinance has shipped. requests and curl_cffi are covered by the
# OSError base; the rate limit error is the one that is not, which is exactly why
# it is named separately in the module under test.
TRANSPORT_FAILURES: List[Tuple[str, BaseException]] = [
    ("builtin-connection", ConnectionError("connection refused")),
    ("builtin-timeout", TimeoutError("timed out after 30s")),
    ("dns", socket.gaierror(8, "nodename nor servname provided")),
    ("urllib", urllib.error.URLError("dns failure")),
    ("requests", requests.exceptions.ConnectionError("max retries exceeded")),
    ("proxy", requests.exceptions.ProxyError("cannot connect to proxy")),
    ("rate-limit", YFRateLimitError()),
]

# Every one of these means the fault is here, not out there. Retrying cannot help
# and a reassuring banner would only delay the diagnosis.
PROGRAMMING_ERRORS: List[BaseException] = [
    TypeError("download() got an unexpected keyword argument 'auto_adjust'"),
    AttributeError("module 'yfinance' has no attribute 'download'"),
    KeyError("Close"),
    NameError("name 'yf' is not defined"),
    ImportError("cannot import name 'download'"),
]


def frame(bars: int = 20) -> pd.DataFrame:
    """A minimal frame shaped the way yfinance returns one."""
    index = pd.date_range("2020-01-01", periods=bars, freq="B", name="Date")
    closes = [100.0 + step for step in range(bars)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * bars,
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Point the cache at a fresh directory for every test.

    Without this a test would either read a file left by the developer's own runs,
    which makes the download path unreachable, or write one into the project's
    cache, which makes the next test depend on this one having run.
    """
    monkeypatch.setattr(market_data, "CACHE_DIR", tmp_path / "cache")


def raising(error: BaseException):
    """A stand-in for yfinance.download that always fails this way."""
    def download(*args, **kwargs):
        raise error
    return download


def returning(result):
    """A stand-in for yfinance.download that always returns this."""
    def download(*args, **kwargs):
        return result
    return download


# --------------------------------------------------------------------------
# A fetch that could not be made
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error",
    [pytest.param(error, id=name) for name, error in TRANSPORT_FAILURES],
)
def test_a_transport_failure_is_reported_as_the_source_being_unavailable(
    monkeypatch, error: BaseException,
):
    """Whatever the transport raises, callers see one type they can handle.

    The alternative is what used to happen: every caller would need to know which
    HTTP library yfinance currently uses, and would break when that changed.
    """
    monkeypatch.setattr(yfinance, "download", raising(error))

    with pytest.raises(DataSourceUnavailable):
        get_price_data(TICKER, START, END)


def test_the_unavailable_message_points_at_the_connection_not_the_ticker(
    monkeypatch,
):
    """The message has to send the reader somewhere useful.

    A valid symbol reported as possibly misspelled is worse than no message: it
    sends someone checking Yahoo's notation when the real fix is to reconnect.
    """
    monkeypatch.setattr(yfinance, "download",
                        raising(ConnectionError("connection refused")))

    with pytest.raises(DataSourceUnavailable) as caught:
        get_price_data(TICKER, START, END)

    message = str(caught.value)
    assert "not with the ticker symbol" in message
    assert "online" in message
    assert "rate limited" in message
    # The underlying cause is quoted, since it is the one detail a developer
    # needs and the user can ignore.
    assert "ConnectionError" in message


def test_the_original_error_is_kept_as_the_cause(monkeypatch):
    """Translating must not destroy the traceback underneath.

    The chained cause is what makes a real incident diagnosable from a log.
    """
    original = requests.exceptions.ProxyError("cannot connect to proxy")
    monkeypatch.setattr(yfinance, "download", raising(original))

    with pytest.raises(DataSourceUnavailable) as caught:
        get_price_data(TICKER, START, END)

    assert caught.value.__cause__ is original


def test_an_unavailable_source_is_an_oserror(monkeypatch):
    """The class sits in the category its causes come from.

    A script already written to handle OSError around a fetch keeps working
    without importing anything new.
    """
    monkeypatch.setattr(yfinance, "download",
                        raising(ConnectionError("connection refused")))

    with pytest.raises(OSError):
        get_price_data(TICKER, START, END)


# --------------------------------------------------------------------------
# A fetch that was made and came back empty
# --------------------------------------------------------------------------

@pytest.mark.parametrize("result", [pd.DataFrame(), None],
                         ids=["empty-frame", "none"])
def test_an_empty_result_stays_a_value_error(monkeypatch, result):
    """The unknown-ticker path must not be swept into the new type.

    It is a genuinely different situation: the source answered, and the answer was
    that it has nothing. Repeating the call cannot change that.
    """
    monkeypatch.setattr(yfinance, "download", returning(result))

    with pytest.raises(ValueError) as caught:
        get_price_data(TICKER, START, END)

    assert not isinstance(caught.value, DataSourceUnavailable)
    assert "ticker symbol follows" in str(caught.value)


def test_the_two_failures_read_differently(monkeypatch):
    """A reader must be able to tell which of the two happened.

    Identical wording would make the distinction in the type useless to anyone
    reading a banner rather than catching an exception.
    """
    monkeypatch.setattr(yfinance, "download", returning(pd.DataFrame()))
    with pytest.raises(ValueError) as empty:
        get_price_data(TICKER, START, END)

    monkeypatch.setattr(yfinance, "download",
                        raising(ConnectionError("connection refused")))
    with pytest.raises(DataSourceUnavailable) as unreachable:
        get_price_data(TICKER, START, END)

    assert str(empty.value) != str(unreachable.value)
    assert "Could not reach" in str(unreachable.value)
    assert "Could not reach" not in str(empty.value)


# --------------------------------------------------------------------------
# The limit of the translation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error",
    [pytest.param(error, id=type(error).__name__) for error in PROGRAMMING_ERRORS],
)
def test_a_programming_error_is_not_dressed_up_as_a_network_problem(
    monkeypatch, error: BaseException,
):
    """The reason the catch names two trees instead of catching Exception.

    Each of these says the call itself is wrong. Reporting them as a connection
    problem would send a developer to check their wifi while a broken signature
    or a renamed column sat in the code.
    """
    monkeypatch.setattr(yfinance, "download", raising(error))

    with pytest.raises(type(error)) as caught:
        get_price_data(TICKER, START, END)

    assert not isinstance(caught.value, DataSourceUnavailable)


def test_both_http_stacks_are_covered_by_the_oserror_base():
    """Why one base is enough for two libraries, asserted rather than assumed.

    yfinance has already moved from requests to curl_cffi once. Listing leaf
    classes would have broken silently at that point; both stacks root their
    exceptions in OSError, so the base survives the change. If a future stack
    does not, this fails and says so.
    """
    assert issubclass(requests.exceptions.RequestException, OSError)

    curl = pytest.importorskip("curl_cffi.requests.exceptions")
    assert issubclass(curl.RequestException, OSError)

    # The counterpart: yfinance's own tree is not an OSError, which is precisely
    # why the module names it separately.
    assert not issubclass(YFException, OSError)


# --------------------------------------------------------------------------
# The cache is the offline escape hatch
# --------------------------------------------------------------------------

def test_a_cached_range_loads_with_the_network_completely_broken(monkeypatch):
    """The claim the docstring makes to anyone working offline.

    Populate the cache while the source works, then break it entirely. A cached
    range must still load, because it never reaches the network at all.
    """
    monkeypatch.setattr(yfinance, "download", returning(frame()))
    downloaded = get_price_data(TICKER, START, END)

    monkeypatch.setattr(yfinance, "download",
                        raising(ConnectionError("connection refused")))
    cached = get_price_data(TICKER, START, END)

    pd.testing.assert_frame_equal(downloaded, cached)


def test_force_refresh_offline_fails_rather_than_serving_the_cache(monkeypatch):
    """force_refresh means what it says, even when it cannot be honoured.

    Quietly returning the cached copy would be friendlier and wrong: the caller
    asked for fresh prices, and pretending to have supplied them is how a stale
    figure ends up in a report.
    """
    monkeypatch.setattr(yfinance, "download", returning(frame()))
    get_price_data(TICKER, START, END)

    monkeypatch.setattr(yfinance, "download",
                        raising(ConnectionError("connection refused")))

    with pytest.raises(DataSourceUnavailable):
        get_price_data(TICKER, START, END, force_refresh=True)
