"""Automated tests for what the dashboard shows when a run fails.

WHY THE UI IS TESTED AT ALL
Everything else in the suite checks a value. This file checks a rendering, because
the difference between a handled failure and an unhandled one is invisible in any
value: both leave the same session state and neither returns anything. It is only
in what reaches the page that a clean banner and a stack trace part company, and
the whole point of the handler is which of the two a user meets.

Streamlit's AppTest runs app.py headlessly and exposes the elements it produced,
which is what makes that observable. An uncaught exception lands in at.exception,
and that is exactly the red traceback a browser would render; a handled one lands
in at.error as a banner.

WHAT IS LOCKED
The three outcomes, one per row of the handler:

    A lost connection is a banner. This is the regression: the handler caught only
    ValueError and TypeError, so the data layer's transport exception went
    straight through and the page answered a dropped wifi connection with a
    traceback into yfinance.

    An unknown ticker is a banner, and a different one. It always was, and it must
    stay that way now that a second failure type shares the handler.

    A bug in this project is still a traceback. That is the deliberate limit, and
    it is worth a test precisely because it is the thing a careless widening of
    the except clause would destroy without breaking anything else.

No test here touches the network.
"""

from pathlib import Path

import pandas as pd
import pytest
import yfinance
from streamlit.testing.v1 import AppTest

from data import market_data

APP = Path(__file__).resolve().parent.parent / "app.py"

# Not a real symbol and not one any cache holds, so the run is forced down the
# download path where the stub is waiting. A cached ticker would never reach it.
UNCACHED_TICKER = "ZZTESTNET"

# AppTest reruns the whole script per interaction, and the sidebar alone builds a
# few dozen widgets, so the default four seconds is tight on a loaded machine.
TIMEOUT = 60


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Keep the dashboard's reads and writes out of the project's cache."""
    monkeypatch.setattr(market_data, "CACHE_DIR", tmp_path / "cache")


def run_dashboard(ticker: str = UNCACHED_TICKER) -> AppTest:
    """Load the app, type a ticker, press Run Backtest, return the page."""
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.text_input[0].set_value(ticker)
    at = at.run()

    for button in at.button:
        if "run backtest" in button.label.lower():
            return button.click().run()

    raise AssertionError("The Run Backtest button was not found on the page.")


def banners(at: AppTest) -> str:
    """Every error banner the page is showing, joined for easy searching."""
    return "\n".join(element.value for element in at.error)


def tracebacks(at: AppTest) -> str:
    """Every uncaught exception, which is what a browser renders in red."""
    return "\n".join(element.value for element in at.exception)


def break_the_network(monkeypatch, error: BaseException) -> None:
    """Make every download attempt fail this way."""
    def download(*args, **kwargs):
        raise error
    monkeypatch.setattr(yfinance, "download", download)


def synthetic_history(bars: int = 500, first: str = "2024-03-21") -> pd.DataFrame:
    """An oscillating history, long enough for a 200-bar window to warm up.

    Deliberately starting years after the sidebar's default 2020 so the rendered
    period label has something to be wrong about.
    """
    import numpy as np

    index = pd.date_range(first, periods=bars, freq="B", name="Date")
    steps = np.arange(bars)
    closes = 100.0 * (1 + 0.25 * np.sin(steps / 30.0) + steps / bars * 0.4)
    return pd.DataFrame(
        {"Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
         "Close": closes, "Volume": [1_000_000] * bars},
        index=index,
    )


def supply_history(monkeypatch, prices: pd.DataFrame) -> None:
    """Make every download return this history instead of reaching Yahoo."""
    monkeypatch.setattr(yfinance, "download",
                        lambda *args, **kwargs: prices.copy())


def empty_the_source(monkeypatch) -> None:
    """Make the source answer, with nothing, as it does for an unknown symbol."""
    monkeypatch.setattr(yfinance, "download",
                        lambda *args, **kwargs: pd.DataFrame())


# --------------------------------------------------------------------------
# A lost connection
# --------------------------------------------------------------------------

def test_a_network_failure_shows_a_banner_and_not_a_traceback(monkeypatch):
    """The regression this file exists for."""
    break_the_network(monkeypatch, ConnectionError(
        "Max retries exceeded: failed to resolve 'query2.finance.yahoo.com'"))

    at = run_dashboard()

    assert tracebacks(at) == ""
    assert "Could not reach Yahoo Finance" in banners(at)


def test_the_network_banner_tells_the_user_what_to_do(monkeypatch):
    """A banner that only says something went wrong is barely better than a trace."""
    break_the_network(monkeypatch, ConnectionError("connection refused"))

    at = run_dashboard()
    message = banners(at)

    assert "not with the ticker symbol" in message
    assert "retrying" in message


def test_a_rate_limit_is_handled_like_any_other_unreachable_source(monkeypatch):
    """Being throttled does not arrive as an OSError, so it is worth its own test.

    yfinance signals it through its own exception tree. If only the transport tree
    were translated this case would still reach the browser as a traceback, on a
    perfectly valid ticker, which is the most confusing version of the bug.
    """
    from yfinance.exceptions import YFRateLimitError

    break_the_network(monkeypatch, YFRateLimitError())

    at = run_dashboard()

    assert tracebacks(at) == ""
    assert "Could not reach Yahoo Finance" in banners(at)


def test_no_results_are_left_on_screen_after_a_failed_run(monkeypatch):
    """A banner above a previous run's numbers would be worse than the traceback.

    The results live in session state so they survive widget interaction, which
    means clearing them on failure has to be deliberate.
    """
    break_the_network(monkeypatch, ConnectionError("connection refused"))

    at = run_dashboard()

    assert tracebacks(at) == ""
    assert len(at.metric) == 0
    assert "Performance report" not in [s.value for s in at.subheader]


# --------------------------------------------------------------------------
# An unknown ticker, which must stay distinguishable
# --------------------------------------------------------------------------

def test_an_unknown_ticker_still_shows_its_own_banner(monkeypatch):
    """The pre-existing behaviour, pinned now that it shares a handler."""
    empty_the_source(monkeypatch)

    at = run_dashboard()
    message = banners(at)

    assert tracebacks(at) == ""
    assert "ticker symbol follows" in message
    assert "Could not reach Yahoo Finance" not in message


def test_an_empty_ticker_cannot_even_be_run():
    """Validation catches this before any download is attempted.

    Worth pinning because it is the cheapest of the three defences and the only
    one that stops the user rather than reporting to them.
    """
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.text_input[0].set_value("   ")
    at = at.run()

    button = next(element for element in at.button
                  if "run backtest" in element.label.lower())

    assert button.proto.disabled
    assert "Enter a ticker symbol." in banners(at)


# --------------------------------------------------------------------------
# A run that succeeds, which is where the rendering code actually lives
# --------------------------------------------------------------------------
#
# Every test above ends in a banner, so between them they never render a result.
# That left the page's own presentation logic untested, which is awkward when it
# is the presentation that keeps changing: the period label and the delta labels
# are both computed here and nowhere else.

def test_a_successful_run_renders_its_results(monkeypatch):
    """The happy path, so a rendering change cannot break the page unnoticed."""
    supply_history(monkeypatch, synthetic_history())

    at = run_dashboard()

    assert tracebacks(at) == ""
    assert banners(at) == ""
    assert len(at.metric) >= 4, "the headline cards should be showing"


def test_the_rendered_period_is_the_one_the_data_covers(monkeypatch):
    """The dashboard's own version of the header-dates fix.

    The sidebar defaults ask for 2020 onward; this history begins in 2024. The
    subheader has to describe the bars, and say that the request was wider.
    """
    prices = synthetic_history()
    supply_history(monkeypatch, prices)

    at = run_dashboard()
    headings = " ".join(element.value for element in at.subheader)

    assert f"{prices.index[0].date()} to {prices.index[-1].date()}" in headings
    assert "requested 2020-01-01" in headings


def test_the_headline_deltas_are_labelled_in_percentage_points(monkeypatch):
    """The dashboard's version of the return-gap fix, seen as a reader sees it."""
    supply_history(monkeypatch, synthetic_history())

    at = run_dashboard()
    deltas = [element.delta for element in at.metric if element.delta]

    assert deltas, "at least one card should carry a delta"
    assert any("pp" in delta for delta in deltas)
    # A percentage delta would have carried a "%", which is the mislabelling.
    assert not any("% vs buy & hold" in delta for delta in deltas)


# --------------------------------------------------------------------------
# The limit: our own bugs stay loud
# --------------------------------------------------------------------------

@pytest.mark.parametrize("error", [
    pytest.param(AttributeError("'NoneType' object has no attribute 'columns'"),
                 id="AttributeError"),
    pytest.param(KeyError("Close"), id="KeyError"),
    pytest.param(RuntimeError("something nobody anticipated"), id="RuntimeError"),
])
def test_a_bug_in_this_project_still_reaches_the_developer(monkeypatch, error):
    """The test that would fail if the except clause were widened carelessly.

    Catching Exception here would make every one of these a soothing banner
    telling the user to check their connection, and the actual fault would have to
    be found by reading the code instead of the screen.
    """
    break_the_network(monkeypatch, error)

    at = run_dashboard()

    assert tracebacks(at) != ""
    assert "Could not reach Yahoo Finance" not in banners(at)
