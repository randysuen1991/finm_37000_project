"""Run the P&L pipeline end-to-end (Issue #9 demo).

Wires the components together:

    MarketDataFetcher  ->  CalendarSpreadData  ->  replay  ->  PnLCalculator

Uses the TEMP ``HardcodedESFetcher`` (ES Jun/Sep 2026, real Databento data,
disk-cached) until Issues #6/#7 land, and the placeholder (random)
``LeggingCostCalculator`` until Issue #8 lands. ``generate_pnl`` is the next
step of #9, so for now the script reports what was consumed.

Run:
    uv run src/run_pnl_pipeline.py
"""

from __future__ import annotations

import databento as db
import pandas as pd

from demo_print_books import HardcodedESFetcher
from legging_cost_calculator import LeggingCostCalculator
from pnl_calculator import PnLCalculator, replay
from util import get_databento_api_key

# Two demo sessions: a 60-second window on each day, matching the windows
# demo_print_books.py already cached to disk — so this runs offline.
DAYS = ["2026-06-09", "2026-06-10"]
SESSION_START_UTC = "16:00:00"  # 10:00 AM CT, the README snapshot hour
SESSION_LENGTH = pd.Timedelta(seconds=60)


def main() -> None:
    client = db.Historical(get_databento_api_key())
    fetcher = HardcodedESFetcher(client=client, levels=1)
    cost_calculator = LeggingCostCalculator()  # placeholder costs (Issue #8)
    pnl_calculator = PnLCalculator()

    for day in DAYS:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        print(f"\n{day} {SESSION_START_UTC}–{t1:%H:%M:%S} UTC  fetching...", flush=True)

        data = fetcher.fetch_calendar_spread_data("M6", "ES", t0, t1)
        print(
            f"  book events: front={len(data.front)}, back={len(data.back)}, "
            f"spread={len(data.spread)}; spread trades={len(data.spread_trades)}"
        )

        consumed = replay(data, cost_calculator, pnl_calculator)
        print(f"  consumed {consumed} spread transaction(s)")

    print(f"\nTotal transactions consumed: {pnl_calculator.transaction_count}")

    try:
        pnl = pnl_calculator.generate_pnl()
        print(pnl)
    except NotImplementedError:
        print("generate_pnl() not implemented yet — next step of Issue #9.")


if __name__ == "__main__":
    main()
