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

from app import CUSTOM_CHOICE, PRESET_TICKERS
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


def open_dashboard() -> AppTest:
    """Load the app with every widget at its default."""
    return AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()


def name_ticker(at: AppTest, ticker: str) -> AppTest:
    """Put a ticker into the sidebar, by whichever of the two routes it needs.

    A preset is one selection. Anything else takes two steps, because the free
    text field does not exist until the sentinel is chosen; the extra run in
    between is what creates it. Both widgets are addressed by key rather than by
    position so that reordering the sidebar cannot silently retarget a test.
    """
    if ticker in PRESET_TICKERS:
        at.selectbox(key="ticker_choice").set_value(ticker)
        return at.run()

    at.selectbox(key="ticker_choice").set_value(CUSTOM_CHOICE)
    at = at.run()
    at.text_input(key="custom_ticker").set_value(ticker)
    return at.run()


def click_run(at: AppTest) -> AppTest:
    """Press Run backtest and return the resulting page."""
    for button in at.button:
        if "run backtest" in button.label.lower():
            return button.click().run()

    raise AssertionError("The Run Backtest button was not found on the page.")


def run_dashboard(ticker: str = UNCACHED_TICKER) -> AppTest:
    """Load the app, name a ticker, press Run backtest, return the page."""
    return click_run(name_ticker(open_dashboard(), ticker))


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
# Naming a ticker: a dropdown of presets, or anything at all
# --------------------------------------------------------------------------
#
# The engine's side of this is unchanged: BacktestConfig still carries one symbol
# string, exactly as when the sidebar held a single text field. What is new is that
# two widgets can supply it, so what is worth locking is the reduction of those two
# to that one, and the fact that neither route bypasses validation.
#
# Note that every error-handling test in this file already runs through the custom
# route, UNCACHED_TICKER being deliberately absent from the presets. The clean
# banners for an unknown symbol, a lost connection and a rate limit are therefore
# all exercised on typed input rather than on a preset.

def test_a_preset_is_passed_through_untouched():
    """Picking from the list needs no normalising: the list is already canonical."""
    from app import resolve_ticker

    assert resolve_ticker("MSFT", "") == "MSFT"


def test_a_typed_symbol_is_upper_cased_and_trimmed():
    """So that "aapl" and " aapl " name the same asset as "AAPL"."""
    from app import resolve_ticker

    assert resolve_ticker(CUSTOM_CHOICE, "  brk-b  ") == "BRK-B"


def test_text_left_behind_cannot_resurrect_itself():
    """Switching back to a preset must ignore whatever the field still holds.

    Streamlit keeps a keyed widget's value across reruns, so the custom text
    survives the field being hidden. Reading it regardless of the dropdown would
    backtest a symbol the reader had visibly moved away from.
    """
    from app import resolve_ticker

    assert resolve_ticker("KO", "TSLA") == "KO"


@pytest.mark.parametrize("typed", ["", "   ", "\t\n"])
def test_an_unfilled_custom_field_resolves_to_nothing(typed: str):
    """Not to a fallback. Empty is what keeps the Run button disabled."""
    from app import resolve_ticker

    assert resolve_ticker(CUSTOM_CHOICE, typed) == ""


def test_the_preset_list_is_a_clean_set_of_symbols():
    """Guards the table itself: no duplicates, no stray case, no empty entries."""
    symbols = list(PRESET_TICKERS)

    assert len(symbols) == len(set(symbols)), "a duplicated preset"
    assert all(symbol == symbol.strip().upper() for symbol in symbols)
    assert all(PRESET_TICKERS[symbol] for symbol in symbols), "a preset with no name"
    assert 15 <= len(symbols) <= 20, "the list is meant to stay short"


def test_the_list_spans_more_than_one_kind_of_asset():
    """A menu of mega-cap tech would teach the reader that any strategy works.

    The index ETFs are the ones that matter: comparing a strategy on SPY with the
    same strategy on a single volatile name is how the reader sees how much of a
    result belongs to the asset.
    """
    assert {"SPY", "QQQ"} <= set(PRESET_TICKERS)
    assert {"KO", "JNJ", "XOM", "JPM"} <= set(PRESET_TICKERS)


def test_the_sentinel_is_not_mistakable_for_a_symbol():
    """It shares the dropdown with real symbols, so it must not look like one."""
    from app import TICKER_CHOICES

    assert CUSTOM_CHOICE not in PRESET_TICKERS
    assert TICKER_CHOICES[-1] == CUSTOM_CHOICE, "the sentinel belongs last"
    assert not CUSTOM_CHOICE.isupper()


def test_the_dropdown_opens_on_the_previous_default():
    """The field used to default to AAPL, and existing screenshots assume it."""
    at = open_dashboard()

    assert at.selectbox(key="ticker_choice").value == "AAPL"


def test_each_entry_shows_whose_symbol_it_is():
    """The point of the list is to spare the reader recalling that Apple is AAPL."""
    from app import ticker_label

    assert ticker_label("AAPL") == "AAPL · Apple"
    # The sentinel has no company behind it and is passed through.
    assert ticker_label(CUSTOM_CHOICE) == CUSTOM_CHOICE


def test_the_free_text_field_appears_only_when_it_is_asked_for():
    """A permanently visible box beside the dropdown would be two ways to answer."""
    at = open_dashboard()
    assert len(at.text_input) == 0

    at.selectbox(key="ticker_choice").set_value(CUSTOM_CHOICE)
    at = at.run()
    assert len(at.text_input) == 1

    at.selectbox(key="ticker_choice").set_value("AAPL")
    at = at.run()
    assert len(at.text_input) == 0


def test_a_preset_runs_end_to_end(monkeypatch):
    """The quick path, all the way to a rendered result."""
    supply_history(monkeypatch, synthetic_history())

    at = run_dashboard("KO")

    assert tracebacks(at) == ""
    assert banners(at) == ""
    assert any("KO" in element.value for element in at.subheader)


def test_a_lower_case_custom_symbol_reaches_the_engine_upper_cased(monkeypatch):
    """The normalisation, observed where the reader sees it rather than in a unit."""
    supply_history(monkeypatch, synthetic_history())

    at = run_dashboard("brk-b")

    assert tracebacks(at) == ""
    headings = " ".join(element.value for element in at.subheader)
    assert "BRK-B" in headings
    assert "brk-b" not in headings


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
    one that stops the user rather than reporting to them. Reaching an empty
    ticker now takes choosing Custom… and typing whitespace, the dropdown having
    no blank entry to select.
    """
    at = name_ticker(open_dashboard(), "   ")

    button = next(element for element in at.button
                  if "run backtest" in element.label.lower())

    assert button.proto.disabled
    assert "Enter a ticker symbol." in banners(at)


def test_choosing_custom_without_typing_yet_does_not_offer_to_run():
    """The state between revealing the field and filling it.

    A blank custom field must not fall back to the preset that was showing a
    moment ago: backtesting an asset the reader did not choose is worse than
    asking them to finish typing.
    """
    at = open_dashboard()
    at.selectbox(key="ticker_choice").set_value(CUSTOM_CHOICE)
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
