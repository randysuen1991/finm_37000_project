"""MarketDataFetcher: interface + data contracts for calendar-spread market data (Issue #6/#7).

This module defines the *schema* that downstream modules (legging cost #8,
expected P&L engine #9, lead-lag analysis #11) can build against, plus a
skeleton ``MarketDataFetcher`` class. Fetching/syncing logic is intentionally
not implemented yet — the private methods raise ``NotImplementedError`` and
the public methods are pure composition, so they can be unit-tested today
with mocked privates.

Data contracts
--------------
1. Book frame (one per instrument), the unit of market data:
   - index: ``ts_event`` — tz-aware UTC ``DatetimeIndex``, ascending.
   - columns (for each level ``lvl`` in ``0..levels-1``, Databento MBP naming):
       ``bid_px_{lvl:02d}``, ``ask_px_{lvl:02d}`` : float (NaN if level empty)
       ``bid_sz_{lvl:02d}``, ``ask_sz_{lvl:02d}`` : float
   Level 00 is top of book.

2. Trade frame (one per instrument), the unit of executed-trade data used to
   simulate fills:
   - index: ``ts_event`` — tz-aware UTC ``DatetimeIndex``, ascending.
   - columns:
       ``price`` : float — trade price in points
       ``size``  : float — traded quantity (contracts)
       ``side``  : str   — aggressor side, ``"A"`` (ask/seller-initiated),
                   ``"B"`` (bid/buyer-initiated) or ``"N"`` (unknown)
   Trades are events, not state: they are *not* synced onto the book index.
   A simulation replays them in time order against the synced books.

3. ``CalendarSpreadData`` — the three book frames (front leg, back leg,
   spread instrument) *time-synced onto a common index*, the three raw
   trade frames, plus their symbols.

4. ``CalendarSpreadContractSpec`` — the static CME parameters needed to turn
   points into dollars (tick sizes, multiplier, quote scaling, price format).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

if TYPE_CHECKING:  # avoid a hard databento dependency at import time
    import databento as db

TimeLike = Union[str, pd.Timestamp]

CME_DATASET = "GLBX.MDP3"
BOOK_INDEX_NAME = "ts_event"

#: CME month codes in calendar order.
MONTH_CODES = "FGHJKMNQUVXZ"

_SYMBOL_RE = re.compile(rf"^(?P<root>[A-Z0-9]+?)(?P<month>[{MONTH_CODES}])(?P<year>\d)$")


#: Columns of a trade frame (index: ``ts_event``).
TRADE_COLUMNS = ["price", "size", "side"]


def book_columns(levels: int = 10) -> list[str]:
    """Column names of a book frame with ``levels`` price levels."""
    cols: list[str] = []
    for lvl in range(levels):
        cols += [
            f"bid_px_{lvl:02d}",
            f"ask_px_{lvl:02d}",
            f"bid_sz_{lvl:02d}",
            f"ask_sz_{lvl:02d}",
        ]
    return cols


#: Default contract cycle: quarterly (equity index, FX, rates).
#: Products listing every month (e.g. CL, NG) should pass cycle=MONTH_CODES.
QUARTERLY_CYCLE = "HMUZ"


def next_contract(front_month: str, cycle: str = QUARTERLY_CYCLE) -> str:
    """Return the contract month following ``front_month`` in the given cycle.

    ``front_month`` is a month-year code like ``"M6"``. Handles year rollover.

    >>> next_contract("M6")
    'U6'
    >>> next_contract("Z6")
    'H7'
    """
    if len(front_month) != 2 or front_month[0] not in cycle or not front_month[1].isdigit():
        raise ValueError(
            f"Invalid front month {front_month!r} for cycle {cycle!r} "
            "(expected month-year code like 'M6')"
        )
    month, year = front_month[0], int(front_month[1])
    pos = cycle.index(month)
    if pos + 1 < len(cycle):
        return f"{cycle[pos + 1]}{year}"
    return f"{cycle[0]}{(year + 1) % 10}"


def split_symbol(symbol: str) -> tuple[str, str, str]:
    """Split a CME outright symbol into (root, month_code, year_digit).

    >>> split_symbol("ESM6")
    ('ES', 'M', '6')
    """
    m = _SYMBOL_RE.match(symbol)
    if m is None:
        raise ValueError(f"Cannot parse CME outright symbol: {symbol!r}")
    return m.group("root"), m.group("month"), m.group("year")


@dataclass(frozen=True)
class CalendarSpreadData:
    """Order books and trades for the two legs and the spread instrument.

    The three book frames follow the book-frame schema and share the same
    index (the result of ``MarketDataFetcher._sync_books``). The three trade
    frames follow the trade-frame schema and keep their raw event timestamps.
    """

    front_symbol: str
    back_symbol: str
    spread_symbol: str
    front: pd.DataFrame
    back: pd.DataFrame
    spread: pd.DataFrame
    front_trades: pd.DataFrame
    back_trades: pd.DataFrame
    spread_trades: pd.DataFrame

    @property
    def is_aligned(self) -> bool:
        """True if all three frames share an identical index."""
        return self.front.index.equals(self.back.index) and self.front.index.equals(
            self.spread.index
        )


@dataclass(frozen=True)
class CalendarSpreadContractSpec:
    """Static CME contract parameters for a calendar spread (Issue #7 schema).

    ``contract_multiplier`` is dollars per index point (ES: 50), so
    tick value in dollars = tick size × multiplier.
    """

    product_code: str            # e.g. "ES"
    front_symbol: str            # e.g. "ESM6"
    back_symbol: str             # e.g. "ESU6"
    spread_symbol: str           # e.g. "ESM6-ESU6"
    outright_tick_size: float    # points, e.g. 0.25
    spread_tick_size: float      # points, e.g. 0.05
    contract_multiplier: float   # $ per point, e.g. 50.0
    quote_scaling: float = 1.0   # raw-quote -> point conversion (e.g. cents quotes)
    price_display_format: str = "decimal"  # "decimal" | "fractional"
    fractional_denominator: Optional[int] = None  # e.g. 32 for ZN, if fractional
    front_expiration: Optional[pd.Timestamp] = None
    back_expiration: Optional[pd.Timestamp] = None

    @property
    def outright_tick_value(self) -> float:
        """Dollar value of one outright tick."""
        return self.outright_tick_size * self.contract_multiplier

    @property
    def spread_tick_value(self) -> float:
        """Dollar value of one spread tick."""
        return self.spread_tick_size * self.contract_multiplier


class MarketDataFetcher:
    """Fetches time-synced calendar-spread market data from Databento.

    Public interface (stable — downstream modules code against this):
      - ``fetch_calendar_spread_data(symbol1, symbol2, time_start, time_end)``
      - ``fetch_calendar_spread_contract_specification(symbol1, symbol2)``

    The private fetch/sync/definition methods are the implementation surface
    for Issues #6/#7 and currently raise ``NotImplementedError``.
    """

    def __init__(
        self,
        client: Optional["db.Historical"] = None,
        dataset: str = CME_DATASET,
        levels: int = 10,
    ) -> None:
        self._client = client
        self._dataset = dataset
        self._levels = levels

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_calendar_spread_data(
        self,
        front_month: str,
        product: str,
        time_start: TimeLike,
        time_end: TimeLike,
        cycle: str = QUARTERLY_CYCLE,
    ) -> CalendarSpreadData:
        """Fetch synced books and raw trades for the front month, back month, and spread.

        Args:
            front_month: Month-year code of the front leg, e.g. ``"M6"``.
            product: CME product code, e.g. ``"ES"``.
            time_start / time_end: Requested time window.
            cycle: Listed contract cycle used to derive the back month
                (default quarterly ``"HMUZ"``; use ``MONTH_CODES`` for
                monthly products like CL).

        The back month is the next contract in ``cycle`` after
        ``front_month``. Each instrument's book is fetched separately, then
        the three book streams are synced onto a common time index. Trades
        are fetched per instrument and returned on their raw event
        timestamps (for fill simulation).
        """
        front_symbol, back_symbol, spread_symbol = self._resolve_symbols(
            front_month, product, cycle
        )

        front = self._fetch_single_instrument(front_symbol, time_start, time_end)
        back = self._fetch_single_instrument(back_symbol, time_start, time_end)
        spread = self._fetch_single_instrument(spread_symbol, time_start, time_end)

        front, back, spread = self._sync_books(front, back, spread)

        front_trades = self._fetch_single_instrument_trades(front_symbol, time_start, time_end)
        back_trades = self._fetch_single_instrument_trades(back_symbol, time_start, time_end)
        spread_trades = self._fetch_single_instrument_trades(spread_symbol, time_start, time_end)

        return CalendarSpreadData(
            front_symbol=front_symbol,
            back_symbol=back_symbol,
            spread_symbol=spread_symbol,
            front=front,
            back=back,
            spread=spread,
            front_trades=front_trades,
            back_trades=back_trades,
            spread_trades=spread_trades,
        )

    def fetch_calendar_spread_contract_specification(
        self,
        front_month: str,
        product: str,
        cycle: str = QUARTERLY_CYCLE,
    ) -> CalendarSpreadContractSpec:
        """Fetch the static contract parameters for the legs and their spread."""
        front_symbol, back_symbol, spread_symbol = self._resolve_symbols(
            front_month, product, cycle
        )
        definitions = self._fetch_definitions(front_symbol, back_symbol, spread_symbol)
        return self._build_contract_spec(
            front_symbol, back_symbol, spread_symbol, definitions
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _resolve_symbols(
        self,
        front_month: str,
        product: str,
        cycle: str = QUARTERLY_CYCLE,
    ) -> tuple[str, str, str]:
        """Resolve (front, back, spread) symbols from a front month and product.

        >>> MarketDataFetcher()._resolve_symbols("M6", "ES")
        ('ESM6', 'ESU6', 'ESM6-ESU6')
        """
        back_month = next_contract(front_month, cycle)
        front_symbol = f"{product}{front_month}"
        back_symbol = f"{product}{back_month}"
        spread_symbol = self._build_spread_symbol(front_symbol, back_symbol)
        return front_symbol, back_symbol, spread_symbol

    def _build_spread_symbol(self, symbol1: str, symbol2: str) -> str:
        """Build and validate the CME calendar spread symbol, e.g. ``ESM6-ESU6``.

        Validates that both legs share the same product root and that
        ``symbol1`` expires before ``symbol2``.
        """
        root1, month1, year1 = split_symbol(symbol1)
        root2, month2, year2 = split_symbol(symbol2)
        if root1 != root2:
            raise ValueError(
                f"Legs must share a product root: {symbol1!r} vs {symbol2!r}"
            )
        order1 = (int(year1), MONTH_CODES.index(month1))
        order2 = (int(year2), MONTH_CODES.index(month2))
        if order1 >= order2:
            raise ValueError(
                f"Front leg must expire before back leg: {symbol1!r} !< {symbol2!r}"
            )
        return f"{symbol1}-{symbol2}"

    def _fetch_single_instrument(
        self,
        symbol: str,
        time_start: TimeLike,
        time_end: TimeLike,
    ) -> pd.DataFrame:
        """Fetch the MBP book time series for one instrument.

        Returns a book frame (see module docstring): ``ts_event`` UTC index,
        ``bid/ask _px/_sz`` columns for ``self._levels`` levels.
        """
        raise NotImplementedError  # Issue #6

    def _fetch_single_instrument_trades(
        self,
        symbol: str,
        time_start: TimeLike,
        time_end: TimeLike,
    ) -> pd.DataFrame:
        """Fetch executed trades for one instrument (Databento ``trades`` schema).

        Returns a trade frame (see module docstring): ``ts_event`` UTC index,
        ``price``, ``size``, ``side`` columns, in ascending time order.
        """
        raise NotImplementedError  # Issue #6

    def _sync_books(
        self,
        front: pd.DataFrame,
        back: pd.DataFrame,
        spread: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Align the three book frames onto a common time index.

        Intended behavior: build the union of event timestamps, forward-fill
        each book to that index (as-of semantics: at time t, each book shows
        its most recent state at or before t), and drop leading rows where
        any book has no state yet. All returned frames share one index.
        """
        raise NotImplementedError  # Issue #6

    def _fetch_definitions(
        self,
        symbol1: str,
        symbol2: str,
        spread_symbol: str,
    ) -> pd.DataFrame:
        """Fetch Databento instrument definitions for the legs and the spread."""
        raise NotImplementedError  # Issue #7

    def _build_contract_spec(
        self,
        symbol1: str,
        symbol2: str,
        spread_symbol: str,
        definitions: pd.DataFrame,
    ) -> CalendarSpreadContractSpec:
        """Map raw definitions (+ static JSON mapping file) to a ``CalendarSpreadContractSpec``."""
        raise NotImplementedError  # Issue #7
