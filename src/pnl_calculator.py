"""PnLCalculator: expected P&L engine for calendar-spread market making (Issue #9).

Streaming design — data is consumed one spread transaction at a time, so it
works identically whether the driver iterates a list of daily
``CalendarSpreadData`` or the memory-bounded ``iter_calendar_spread_data``
generator:

    calc = PnLCalculator()
    cost_calc = LeggingCostCalculator()          # Issue #8
    for datum in fetcher.iter_calendar_spread_data(...):
        replay(datum, cost_calc, calc)
    pnl_of_the_period = calc.generate_pnl()

Components:
- ``OrderBook``      — as-of book state lookup for one instrument (for
                       ad-hoc queries and analysis).
- ``replay``         — driver: merges book updates and spread trades of one
                       ``CalendarSpreadData`` into a single time-ordered
                       event stream and walks it once. Book events refresh
                       the cached current state; each trade event gets the
                       legging cost (#8) and is fed to the accumulator with
                       the three cached snapshots.
- ``PnLCalculator``  — stateful accumulator: ``consume`` one transaction at
                       a time, ``generate_pnl`` aggregates at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from legging_cost_calculator import LeggingCostCalculator
from market_data_fetcher import BOOK_INDEX_NAME, CalendarSpreadData


class OrderBook:
    """Time-series order book for one instrument with as-of state lookup.

    Wraps a book frame (see ``market_data_fetcher`` schema): ``ts_event``
    UTC index, ``bid/ask_px/sz_{level}`` columns.
    """

    def __init__(self, symbol: str, book: pd.DataFrame) -> None:
        if not isinstance(book.index, pd.DatetimeIndex) or book.index.tz is None:
            raise ValueError(
                f"{symbol}: book index must be a tz-aware DatetimeIndex "
                f"named {BOOK_INDEX_NAME!r}"
            )
        if not book.index.is_monotonic_increasing:
            raise ValueError(f"{symbol}: book index must be ascending")
        if "bid_px_00" not in book.columns or "ask_px_00" not in book.columns:
            raise ValueError(f"{symbol}: book frame missing top-of-book columns")
        self.symbol = symbol
        self._book = book

    def state_at(self, ts: pd.Timestamp | str) -> Optional[pd.Series]:
        """Book state as of ``ts``: the most recent row at or before ``ts``.

        Returns ``None`` if the book has no state yet (``ts`` earlier than
        the first event).
        """
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        pos = self._book.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        return self._book.iloc[pos]

    def best_bid(self, ts: pd.Timestamp | str) -> Optional[float]:
        """Best bid price as of ``ts`` (None if no state yet)."""
        state = self.state_at(ts)
        return None if state is None else state["bid_px_00"]

    def best_ask(self, ts: pd.Timestamp | str) -> Optional[float]:
        """Best ask price as of ``ts`` (None if no state yet)."""
        state = self.state_at(ts)
        return None if state is None else state["ask_px_00"]

    def mid(self, ts: pd.Timestamp | str) -> Optional[float]:
        """Mid price as of ``ts`` (None if no state yet)."""
        state = self.state_at(ts)
        if state is None:
            return None
        return (state["bid_px_00"] + state["ask_px_00"]) / 2.0

    def __len__(self) -> int:
        return len(self._book)


@dataclass(frozen=True)
class SpreadTransaction:
    """One spread trade with the market context the P&L model needs."""

    ts: pd.Timestamp
    price: float
    size: float
    side: str  # aggressor: "B" buyer-initiated, "A" seller-initiated, "N" unknown
    front_snapshot: pd.Series
    back_snapshot: pd.Series
    spread_snapshot: pd.Series
    cost: float  # legging cost (points) from Issue #8's calculator


class PnLCalculator:
    """Stateful P&L accumulator for spread transactions (Issue #9).

    Feed it one transaction at a time with ``consume``; call
    ``generate_pnl`` once at the end of the period.
    """

    def __init__(self) -> None:
        self._transactions: list[SpreadTransaction] = []

    def consume(
        self,
        front_snapshot: pd.Series,
        back_snapshot: pd.Series,
        spread_snapshot: pd.Series,
        trade: pd.Series,
        cost: float,
    ) -> None:
        """Record one spread transaction.

        Args:
            front_snapshot / back_snapshot / spread_snapshot: book-frame rows
                as of the trade's timestamp.
            trade: trade-frame row (``price``, ``size``, ``side``; its
                ``name`` is the event timestamp).
            cost: legging cost in points from the #8 calculator.
        """
        self._transactions.append(
            SpreadTransaction(
                ts=trade.name,
                price=float(trade["price"]),
                size=float(trade["size"]),
                side=str(trade["side"]),
                front_snapshot=front_snapshot,
                back_snapshot=back_snapshot,
                spread_snapshot=spread_snapshot,
                cost=float(cost),
            )
        )

    @property
    def transaction_count(self) -> int:
        return len(self._transactions)

    def generate_pnl(self) -> pd.DataFrame:
        """Aggregate consumed transactions into expected P&L for the period.

        Intended output (to be built next): expected P&L per contract and
        for the period, decomposed into spread edge earned, legging cost
        paid, adverse selection, and inventory risk — the inputs Issue #10
        needs to solve for the break-even incentive.
        """
        raise NotImplementedError  # next step of Issue #9


def _merged_events(data: CalendarSpreadData):
    """Yield all events (book updates + spread trades) in time order.

    Each event is ``(ts, priority, kind, payload)``. At equal timestamps,
    book updates are processed before trades (priority 0 < 1): a trade
    executed against the book state stamped at the same instant.
    """
    streams = [
        ("front", 0, data.front),
        ("back", 0, data.back),
        ("spread", 0, data.spread),
        ("trade", 1, data.spread_trades),
    ]
    events = []
    for kind, priority, frame in streams:
        for ts, row in frame.iterrows():
            events.append((ts, priority, kind, row))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def replay(
    data: CalendarSpreadData,
    cost_calculator: LeggingCostCalculator,
    pnl_calculator: PnLCalculator,
) -> int:
    """Drive one ``CalendarSpreadData`` through the P&L pipeline.

    Event-driven replay: all book updates and spread trades are merged into
    one time-ordered stream and walked once (O(n + m)). Book updates refresh
    the cached "current state" per instrument; each spread trade triggers a
    legging-cost query (#8) and is fed to ``pnl_calculator.consume`` with
    the three cached snapshots. Trades that occur before all three books
    have a state are skipped.

    Returns the number of transactions consumed.
    """
    books: dict[str, Optional[pd.Series]] = {"front": None, "back": None, "spread": None}

    consumed = 0
    for ts, _priority, kind, payload in _merged_events(data):
        if kind != "trade":
            books[kind] = payload  # update cached book state
            continue

        if books["front"] is None or books["back"] is None or books["spread"] is None:
            continue  # a book has no state yet at this timepoint

        cost = cost_calculator.get_cost(books["front"], books["back"], books["spread"])
        pnl_calculator.consume(
            books["front"], books["back"], books["spread"], payload, cost
        )
        consumed += 1
    return consumed
