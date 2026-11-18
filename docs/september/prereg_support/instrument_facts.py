"""Reproducer for the instrument facts quoted in the protocol and the rationale.

Universe and contract properties only — no strategy, no positions, no P&L.
Every number printed here is quoted somewhere in the documents; this script is
what makes them checkable, with the window stated rather than implied.

Run from the repo root:  python docs/september/prereg_support/instrument_facts.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.universe.coinmarketcap.new_fetcher import normalize, panel  # noqa: E402

WINDOW = ("2020-02-09", "2026-08-31")      # the registered basket window
FUNDING_CONTRACTS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]


def top_n_share(n: int = 10) -> None:
    """Share of top-100 market cap held by the top n, per day."""
    universe = normalize.load()
    universe = universe[(universe.date >= pd.Timestamp(WINDOW[0]).date())
                        & (universe.date <= pd.Timestamp(WINDOW[1]).date())]
    universe = universe.sort_values(["date", "market_cap"], ascending=[True, False])
    share = universe.groupby("date").apply(
        lambda d: d.market_cap.head(n).sum() / d.market_cap.sum(), include_groups=False)
    print(f"top-{n} share of top-100 market cap, {WINDOW[0]}..{WINDOW[1]} "
          f"({len(share)} days): mean {share.mean():.3%}, min {share.min():.3%}, "
          f"max {share.max():.3%}, last {share.iloc[-1]:.3%}")


def funding_cost() -> None:
    """Annualised funding, the holding cost a long pays.

    Each row is one settlement, so the annualisation uses that row's own
    interval: a 4-hour contract settles six times a day, an 8-hour one three.
    Reported as the mean of per-settlement annualised rates over the window, and
    also as the realised total, which is what a position actually pays.
    """
    print(f"\nfunding, {WINDOW[0]}..{WINDOW[1]} (positive = paid by longs):")
    for contract in FUNDING_CONTRACTS:
        rows = panel.load_funding(contract)
        stamps = pd.to_datetime(rows.timestamp, format="mixed", utc=True)
        rows = rows[(stamps >= pd.Timestamp(WINDOW[0], tz="UTC"))
                    & (stamps <= pd.Timestamp(WINDOW[1], tz="UTC") + pd.Timedelta(days=1))]
        per_year = 365 * 24 / rows.funding_interval_hours.astype(float)
        annualised = (rows.funding_rate.astype(float) * per_year).mean()

        years = (stamps.max() - stamps.min()).days / 365.25
        realised = rows.funding_rate.astype(float).sum() / years
        print(f"  {contract:9s} {len(rows):6d} settlements | "
              f"mean annualised {annualised:6.2%} | realised {realised:6.2%}/yr")


def contract_events() -> None:
    """Contracts that stopped trading inside the window."""
    ranges = panel.load_traded_ranges()
    dead = ranges[ranges.last_traded < WINDOW[1]]
    print(f"\ncontracts that stopped trading before {WINDOW[1]}: {len(dead)}")
    for row in dead.itertuples():
        print(f"  {row.contract:14s} {row.first_traded} -> {row.last_traded}  "
              f"traded {row.traded_days:5d}  halted {row.halted_days:4d}  "
              f"zombie bars {row.zombie_bars:5d}")


def funding_intervals() -> None:
    """Settlement intervals actually present — never assume 8 hours."""
    counts = {}
    for path in sorted(Path("data/derived/binance/funding").glob("*.csv")):
        rows = pd.read_csv(path, usecols=["funding_interval_hours"])
        for hours, count in rows.funding_interval_hours.value_counts().items():
            counts[hours] = counts.get(hours, 0) + int(count)
    print("\nfunding settlement intervals across all contracts (hours -> rows):",
          dict(sorted(counts.items())))


if __name__ == "__main__":
    top_n_share()
    funding_cost()
    funding_intervals()
    contract_events()
