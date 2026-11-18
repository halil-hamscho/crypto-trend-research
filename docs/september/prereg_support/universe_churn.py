"""Reproducer for the reconstitution-frequency table in the pre-registration.

Universe only: membership, entries, exits and weight turnover. No strategy
returns, no P&L. Applies the registered 10-in / 13-out membership buffer.

Run:  python docs/september/prereg_support/universe_churn.py  (from the repo root)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from data.universe.selector import PoolTooShallow, UniverseSelector

N_IN, N_OUT = 10, 13


def eligible(sel_out, sel_in, day):
    """Cap-ordered eligible contracts out to rank 13, with a rank-10 fallback."""
    for sel in (sel_out, sel_in):
        try:
            members, _ = sel.select(day)
            return list(members.contract), dict(zip(members.contract, members.market_cap))
        except PoolTooShallow:
            continue
    raise RuntimeError(f"no eligible set for {day}")


def run(dates, label, sel_out, sel_in):
    held, held_w = [], {}
    entries = exits = 0
    turnover = 0.0
    for stamp in dates:
        order, caps = eligible(sel_out, sel_in, stamp.date())
        keep = [c for c in held if c in order][:N_IN]          # incumbents inside rank 13
        for contract in order:                                  # fill from the top
            if len(keep) == N_IN:
                break
            if contract not in keep:
                keep.append(contract)
        total = sum(caps[c] for c in keep)
        weights = {c: caps[c] / total for c in keep}
        entries += len([c for c in keep if c not in held])
        exits += len([c for c in held if c not in keep])
        turnover += sum(abs(weights.get(c, 0) - held_w.get(c, 0))
                        for c in set(weights) | set(held_w)) / 2
        held, held_w = keep, weights
    years = (dates[-1] - dates[0]).days / 365.25
    print(f"{label:8s} rebalances={len(dates):4d}  entries={entries:3d} exits={exits:3d}  "
          f"{entries/years:4.1f} entries/yr  turnover={turnover/years*100:5.1f}%/yr")


def main() -> None:
    logging.disable(logging.CRITICAL)
    sel_out = UniverseSelector(min_history_days=0, n=N_OUT)
    sel_in = UniverseSelector(min_history_days=0, n=N_IN)
    run(list(pd.date_range("2020-02-01", "2026-08-01", freq="MS")), "monthly", sel_out, sel_in)
    run(list(pd.date_range("2020-02-03", "2026-08-31", freq="W-MON")), "weekly", sel_out, sel_in)


if __name__ == "__main__":
    main()
