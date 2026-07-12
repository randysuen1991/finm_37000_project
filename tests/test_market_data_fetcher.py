"""Unit tests for MarketDataFetcher (skeleton, Issues #6/#7).

The private fetch methods are not implemented yet, so:
- pure helpers (symbol parsing, spread symbol building, schema, spec math)
  are tested directly;
- the public methods are tested as *composition* with mocked privates,
  pinning down the call contract before implementation;
- the intended behavior of ``next_contract_month`` is written as xfail tests
  that will start passing once it is implemented (remove the marks then).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from market_data_fetcher import (
    BOOK_INDEX_NAME,
    MONTH_CODES,
    TRADE_COLUMNS,
    CalendarSpreadContractSpec,
    CalendarSpreadData,
    MarketDataFetcher,
    book_columns,
    next_contract_month,
    split_symbol,
)

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

LEVELS = 2  # small books are enough for unit tests


def make_book(timestamps, bid_px, ask_px, bid_sz=10.0, ask_sz=10.0, levels=LEVELS):
    """Build a schema-conforming book frame with identical prices offset per level."""
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True), name=BOOK_INDEX_NAME)
    data = {}
    for lvl in range(levels):
        data[f"bid_px_{lvl:02d}"] = [px - 0.25 * lvl for px in bid_px]
        data[f"ask_px_{lvl:02d}"] = [px + 0.25 * lvl for px in ask_px]
        data[f"bid_sz_{lvl:02d}"] = bid_sz
        data[f"ask_sz_{lvl:02d}"] = ask_sz
    return pd.DataFrame(data, index=idx)[book_columns(levels)]


def make_trades(timestamps, prices, sizes=None, sides=None):
    """Build a schema-conforming trade frame."""
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True), name=BOOK_INDEX_NAME)
    n = len(prices)
    return pd.DataFrame(
        {
            "price": prices,
            "size": sizes if sizes is not None else [1.0] * n,
            "side": sides if sides is not None else ["B"] * n,
        },
        index=idx,
    )


@pytest.fixture
def es_books():
    """README ES snapshot as one-row book frames (front, back, spread)."""
    ts = ["2026-06-10T16:00:00Z"]
    return {
        "front": make_book(ts, bid_px=[7340.00], ask_px=[7340.25]),
        "back": make_book(ts, bid_px=[7401.00], ask_px=[7401.25]),
        "spread": make_book(ts, bid_px=[60.70], ask_px=[60.75]),
    }


@pytest.fixture
def es_trades():
    """Sample trade frames: a buyer lifts the spread ask, a seller hits the front bid."""
    return {
        "ESM6": make_trades(["2026-06-10T16:00:00.5Z"], [7340.00], [5.0], ["A"]),
        "ESU6": make_trades(["2026-06-10T16:00:00.7Z"], [7401.25], [2.0], ["B"]),
        "ESM6-ESU6": make_trades(["2026-06-10T16:00:00.9Z"], [60.75], [10.0], ["B"]),
    }


@pytest.fixture
def es_spec():
    return CalendarSpreadContractSpec(
        product_code="ES",
        front_symbol="ESM6",
        back_symbol="ESU6",
        spread_symbol="ESM6-ESU6",
        outright_tick_size=0.25,
        spread_tick_size=0.05,
        contract_multiplier=50.0,
    )


@pytest.fixture
def fetcher():
    return MarketDataFetcher(client=None, levels=LEVELS)


# --------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------- #

def test_book_columns_names_and_count():
    cols = book_columns(2)
    assert cols == [
        "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
        "bid_px_01", "ask_px_01", "bid_sz_01", "ask_sz_01",
    ]
    assert len(book_columns(10)) == 40


def test_make_book_conforms_to_schema(es_books):
    front = es_books["front"]
    assert front.index.name == BOOK_INDEX_NAME
    assert str(front.index.tz) == "UTC"
    assert list(front.columns) == book_columns(LEVELS)
    assert front["bid_px_00"].iloc[0] == 7340.00
    assert front["ask_px_00"].iloc[0] == 7340.25


def test_spec_tick_values(es_spec):
    # ES: one outright tick (0.25 pt) = $12.50, one spread tick (0.05 pt) = $2.50
    assert es_spec.outright_tick_value == pytest.approx(12.50)
    assert es_spec.spread_tick_value == pytest.approx(2.50)


def test_make_trades_conforms_to_schema(es_trades):
    trades = es_trades["ESM6-ESU6"]
    assert trades.index.name == BOOK_INDEX_NAME
    assert str(trades.index.tz) == "UTC"
    assert list(trades.columns) == TRADE_COLUMNS
    assert trades["price"].iloc[0] == 60.75
    assert trades["side"].iloc[0] == "B"


def test_calendar_spread_data_holds_raw_frames(es_books, es_trades):
    data = CalendarSpreadData(
        front_symbol="ESM6", back_symbol="ESU6", spread_symbol="ESM6-ESU6",
        front=es_books["front"], back=es_books["back"], spread=es_books["spread"],
        front_trades=es_trades["ESM6"],
        back_trades=es_trades["ESU6"],
        spread_trades=es_trades["ESM6-ESU6"],
    )
    # frames keep their own raw timestamps — no alignment is imposed
    assert data.front["bid_px_00"].iloc[0] == 7340.00
    assert data.spread_trades.index[0] > data.front.index[0]


# --------------------------------------------------------------------- #
# Symbol helpers
# --------------------------------------------------------------------- #

def test_split_symbol():
    assert split_symbol("ESM6") == ("ES", "M", "6")
    assert split_symbol("ZNH7") == ("ZN", "H", "7")
    assert split_symbol("6EU6") == ("6E", "U", "6")


def test_split_symbol_rejects_garbage():
    with pytest.raises(ValueError):
        split_symbol("NOT_A_SYMBOL")


# Intended behavior of next_contract_month, TDD-style — these become the spec;
# remove the xfail marks once it is implemented (Issue #7).

@pytest.mark.xfail(raises=NotImplementedError, reason="Issue #7: next_contract_month not implemented")
def test_next_contract_quarterly():
    assert next_contract_month("H6") == "M6"
    assert next_contract_month("M6") == "U6"
    assert next_contract_month("U6") == "Z6"
    assert next_contract_month("Z6") == "H7"  # year rollover


@pytest.mark.xfail(raises=NotImplementedError, reason="Issue #7: next_contract_month not implemented")
def test_next_contract_monthly_cycle():
    assert next_contract_month("F6", cycle=MONTH_CODES) == "G6"
    assert next_contract_month("Z6", cycle=MONTH_CODES) == "F7"


@pytest.mark.xfail(raises=NotImplementedError, reason="Issue #7: next_contract_month not implemented")
def test_next_contract_rejects_bad_input():
    with pytest.raises(ValueError):
        next_contract_month("A6")  # 'A' not a month code in the quarterly cycle
    with pytest.raises(ValueError):
        next_contract_month("M")  # missing year digit


@pytest.mark.xfail(raises=NotImplementedError, reason="Issue #7: next_contract_month not implemented")
def test_resolve_symbols(fetcher):
    assert fetcher._resolve_symbols("M6", "ES") == ("ESM6", "ESU6", "ESM6-ESU6")
    assert fetcher._resolve_symbols("Z6", "NQ") == ("NQZ6", "NQH7", "NQZ6-NQH7")
    assert fetcher._resolve_symbols("F6", "CL", cycle=MONTH_CODES) == (
        "CLF6", "CLG6", "CLF6-CLG6"
    )


def test_build_spread_symbol(fetcher):
    assert fetcher._build_spread_symbol("ESM6", "ESU6") == "ESM6-ESU6"


def test_build_spread_symbol_rejects_mixed_roots(fetcher):
    with pytest.raises(ValueError, match="root"):
        fetcher._build_spread_symbol("ESM6", "NQU6")


def test_build_spread_symbol_rejects_wrong_leg_order(fetcher):
    with pytest.raises(ValueError, match="before"):
        fetcher._build_spread_symbol("ESU6", "ESM6")  # back leg first
    with pytest.raises(ValueError, match="before"):
        fetcher._build_spread_symbol("ESM6", "ESM6")  # same contract


def test_build_spread_symbol_handles_year_rollover(fetcher):
    # Z6 (Dec 2026) expires before H7 (Mar 2027)
    assert fetcher._build_spread_symbol("ESZ6", "ESH7") == "ESZ6-ESH7"


# --------------------------------------------------------------------- #
# fetch_calendar_spread_data — composition contract (privates mocked)
# --------------------------------------------------------------------- #

def test_fetch_calendar_spread_data_composes_privates(fetcher, es_books, es_trades):
    books = {"ESM6": es_books["front"], "ESU6": es_books["back"],
             "ESM6-ESU6": es_books["spread"]}

    with patch.object(
        fetcher, "_resolve_symbols",
        return_value=("ESM6", "ESU6", "ESM6-ESU6"),
    ), patch.object(
        fetcher, "_fetch_single_instrument",
        side_effect=lambda sym, start, end: books[sym],
    ) as mock_fetch, patch.object(
        fetcher, "_fetch_single_instrument_trades",
        side_effect=lambda sym, start, end: es_trades[sym],
    ) as mock_trades:
        result = fetcher.fetch_calendar_spread_data(
            "M6", "ES", "2026-06-10T15:00:00Z", "2026-06-10T17:00:00Z"
        )

    # each instrument's book and trades fetched separately, with the requested window
    for mock in (mock_fetch, mock_trades):
        assert mock.call_count == 3
        fetched_symbols = [c.args[0] for c in mock.call_args_list]
        assert fetched_symbols == ["ESM6", "ESU6", "ESM6-ESU6"]
        for c in mock.call_args_list:
            assert c.args[1] == "2026-06-10T15:00:00Z"
            assert c.args[2] == "2026-06-10T17:00:00Z"

    # result contract
    assert isinstance(result, CalendarSpreadData)
    assert result.front_symbol == "ESM6"
    assert result.back_symbol == "ESU6"
    assert result.spread_symbol == "ESM6-ESU6"
    # books and trades pass through untouched, on raw event timestamps
    assert result.front["bid_px_00"].iloc[0] == 7340.00
    assert result.spread["ask_px_00"].iloc[0] == 60.75
    assert result.spread_trades["price"].iloc[0] == 60.75
    assert result.front_trades["side"].iloc[0] == "A"


@pytest.mark.xfail(raises=NotImplementedError, reason="Issue #7: next_contract_month not implemented")
def test_fetch_calendar_spread_data_validates_before_fetching(fetcher):
    with patch.object(fetcher, "_fetch_single_instrument") as mock_fetch:
        with pytest.raises(ValueError):
            fetcher.fetch_calendar_spread_data("A6", "ES", "t0", "t1")  # bad month code
    mock_fetch.assert_not_called()


# --------------------------------------------------------------------- #
# iter_calendar_spread_data — lazy per-day composition (privates mocked)
# --------------------------------------------------------------------- #

def test_iter_calendar_spread_data_is_lazy_and_yields_per_day(fetcher):
    windows = [
        (pd.Timestamp("2026-06-10T00:00:00Z"), pd.Timestamp("2026-06-11T00:00:00Z")),
        (pd.Timestamp("2026-06-11T00:00:00Z"), pd.Timestamp("2026-06-12T00:00:00Z")),
    ]
    day1, day2 = object(), object()  # sentinels standing in for CalendarSpreadData

    with patch.object(
        fetcher, "_trading_days", return_value=windows
    ), patch.object(
        fetcher, "fetch_calendar_spread_data", side_effect=[day1, day2]
    ) as mock_fetch:
        gen = fetcher.iter_calendar_spread_data(
            "M6", "ES", "2026-06-10T00:00:00Z", "2026-06-12T00:00:00Z"
        )

        # generator: nothing fetched until consumed
        mock_fetch.assert_not_called()

        assert next(gen) is day1
        assert mock_fetch.call_count == 1  # only one day in memory so far

        assert next(gen) is day2
        with pytest.raises(StopIteration):
            next(gen)

    # each day fetched with its own sub-window
    assert mock_fetch.call_args_list[0].args == ("M6", "ES", *windows[0], "HMUZ")
    assert mock_fetch.call_args_list[1].args == ("M6", "ES", *windows[1], "HMUZ")


# --------------------------------------------------------------------- #
# fetch_calendar_spread_contract_specification — composition contract
# --------------------------------------------------------------------- #

def test_fetch_contract_specification_composes_privates(fetcher, es_spec):
    fake_defs = pd.DataFrame({"raw_symbol": ["ESM6", "ESU6", "ESM6-ESU6"]})

    with patch.object(
        fetcher, "_resolve_symbols",
        return_value=("ESM6", "ESU6", "ESM6-ESU6"),
    ), patch.object(
        fetcher, "_fetch_definitions", return_value=fake_defs
    ) as mock_defs, patch.object(
        fetcher, "_build_contract_spec", return_value=es_spec
    ) as mock_build:
        spec = fetcher.fetch_calendar_spread_contract_specification("M6", "ES")

    mock_defs.assert_called_once_with("ESM6", "ESU6", "ESM6-ESU6")
    mock_build.assert_called_once_with("ESM6", "ESU6", "ESM6-ESU6", fake_defs)
    assert spec == es_spec


# --------------------------------------------------------------------- #
# Not-yet-implemented surface (Issues #6/#7)
# --------------------------------------------------------------------- #

def test_unimplemented_privates_raise(fetcher, es_books):
    with pytest.raises(NotImplementedError):
        next_contract_month("M6")
    with pytest.raises(NotImplementedError):
        fetcher._fetch_single_instrument("ESM6", "t0", "t1")
    with pytest.raises(NotImplementedError):
        fetcher._fetch_single_instrument_trades("ESM6", "t0", "t1")
    with pytest.raises(NotImplementedError):
        fetcher._trading_days("t0", "t1")
    with pytest.raises(NotImplementedError):
        fetcher._fetch_definitions("ESM6", "ESU6", "ESM6-ESU6")
