"""LeggingCostCalculator: cost of hedging a spread fill in the outrights (Issue #8).

PLACEHOLDER — ``get_cost`` currently returns a random number so the P&L
pipeline (#9) can run end-to-end. The real calculation belongs to Issue #8.
The P&L engine calls ``get_cost`` for each spread transaction, passing the
three book snapshots as of the trade's timestamp.
"""

from __future__ import annotations

import random

import pandas as pd


class LeggingCostCalculator:
    """Computes the mechanical cost of legging out a spread fill (Issue #8)."""

    def get_cost(
        self,
        front_snapshot: pd.Series,
        back_snapshot: pd.Series,
        spread_snapshot: pd.Series,
    ) -> float:
        """Legging cost in points for a spread fill at this moment.

        Snapshots are book-frame rows (as-of the trade timestamp) with
        ``bid/ask_px/sz_{level}`` fields. Per the README worked example:
        the synthetic spread from crossing the outrights (back ask - front
        bid) versus the spread instrument's own quote — ES snapshot gives
        61.25 - 60.75 = 0.50 points (= $25/contract).

        TODO(#8): replace this placeholder — it ignores the snapshots and
        returns a random cost in [0, 1) points.
        """
        return random.random()  # placeholder for Issue #8
