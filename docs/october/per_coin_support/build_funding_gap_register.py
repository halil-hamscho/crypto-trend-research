"""Build the stage-B funding-gap register: every place a stage-B contract's funding chain breaks the
sealed spacing rule, each verified to be Binance's own schedule rather than a lost payment.

Run from the repo root: PYTHONPATH=. python docs/october/per_coin_support/build_funding_gap_register.py

The sealed check (`basket.funding_gaps`) accepts a spacing equal to either neighbouring row's recorded
interval. It flags three things in the top-25/50 contracts that are not missing data:
  interval_change  Binance moved a contract between cadences (8h to 2h for FTT on 2022-11-10; 1h back
                   to 4h for several 2025-26 contracts) off the old grid, so one spacing matches
                   neither interval
  exchange_skip    the 2026-06-24 04:00 settlement absent for every 4h contract: the sealed protocol's
                   Part 10 Exception 1, carried by the per-coin config (`funding.carried_exception`),
                   charged zero as in the sealed run
  no_trading       a gap where the contract did not trade at all (LIT between its old and new coin;
                   PUMP before its launch), so no position can lie inside it
  settlement_tail  the first archived row after an exchange settlement (FTT 2022-11-14), never charged
  mark_hole        a recorded settlement at an hour with no mark bar (AERGO, CTK, CVC, CVX, SLP in 2025):
                   it cannot be valued, so it is dropped, and the ledger refuses any HELD window that
                   touches it (none does: each is outside every eligible stretch)

Each gap is accepted only if Binance's `fapi/v1/fundingRate` returns exactly the archive's rows around
it (raw responses kept in data/raw/binance/api/fundingRate/<contract>/), or, for `no_trading`, if the
daily klines show no trade inside it. Anything else raises. Prices and rates only; no strategy or
control return is computed.
"""
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from data.universe.coinmarketcap.new_fetcher import panel
from portfolio.basket import SCORED_END, funding_gaps
from portfolio.per_coin_runs import (STAGE_B_UNIVERSES, contracts_of, load_event_register, load_funding_supplemented,
                                     read_settlements_supplemented, universe_members)
from portfolio.per_coin_universe import traded_days

OUT = Path(__file__).resolve().parent / "funding_gap_register.csv"
API = "https://fapi.binance.com/fapi/v1/fundingRate"
SKIP = pd.Timestamp("2026-06-24 04:00", tz="UTC")


def ms(t) -> int:
    return int(pd.Timestamp(t).value // 10**6)


def api_rows(contract, lo, hi) -> set:
    """The API's settlement minutes in [lo, hi], cached as raw JSON."""
    path = panel.RAW_DIR / "api" / "fundingRate" / contract / f"gap-{ms(lo)}-{ms(hi)}.json"
    if not path.exists():
        body = requests.get(API, params={"symbol": contract, "startTime": ms(lo), "endTime": ms(hi), "limit": 1000},
                            timeout=30)
        body.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body.content)
        time.sleep(0.25)
    rows = json.loads(path.read_text())
    if len(rows) >= 1000:
        raise ValueError(f"{contract}: API window {lo}..{hi} too wide")
    return set(pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True).floor("min")), path


def traded_inside(traded, g):
    return traded[(traded > g.previous.tz_localize(None).normalize()) & (traded < g.stamp.tz_localize(None).normalize())]


SKIP_WINDOW = (pd.Timestamp("2026-06-24 00:00", tz="UTC"), pd.Timestamp("2026-06-24 08:00", tz="UTC"))


def mark_holes(contract) -> list:
    """Funding rows inside the mark range whose hour has no mark bar, as bracketing register rows."""
    f = load_funding_supplemented(contract)
    marks = panel.load_mark_prices(contract)
    stamp = f.timestamp.dt.floor("h")
    inside = (stamp >= marks.index.min()) & (stamp <= marks.index.max())
    valued = stamp[inside & stamp.isin(marks.index)].sort_values()
    out = []
    for t in stamp[inside & ~stamp.isin(marks.index)]:
        before, after = valued[valued < t], valued[valued > t]
        if len(before) and len(after) and not any(a < t < b for a, b, *_ in out):
            out.append((before.iloc[-1], after.iloc[0], t))
    return [{"contract": contract, "start": a, "end": b, "kind": "mark_hole", "spacing_hours": (b - a) / pd.Timedelta(hours=1),
             "intervals": "", "evidence": f"settlement at {t} with no hourly mark bar in {contract}.csv; dropped, "
                                         "and no held window may touch it"} for a, b, t in out]


def classify(contract, g, settled_at, traded) -> str:
    if (SKIP_WINDOW[0] <= g.previous < SKIP < g.stamp <= SKIP_WINDOW[1]
            and g.interval_hours == g.previous_interval_hours):
        return "exchange_skip"
    if any(g.previous <= t < g.stamp for t in settled_at or ()):
        return "settlement_tail"
    if g.spacing_hours > 24 and not len(traded_inside(traded, g)):
        return "no_trading"
    if g.previous_interval_hours != g.interval_hours:
        return "interval_change"
    raise ValueError(f"{contract}: unexplained gap {g.previous} -> {g.stamp}")


def build() -> pd.DataFrame:
    contracts = sorted({c for u in STAGE_B_UNIVERSES for c in contracts_of(universe_members(u))})
    settled = {}
    for e in load_event_register().itertuples():
        settled.setdefault(e.contract, []).append(e.instant)
    rows = []
    for c in contracts:
        if not (panel.MARK_DIR / f"{c}.csv").exists():
            continue                                  # no marks archived: never tradable (U6), never held
        holes = mark_holes(c)
        rows += holes
        gaps = funding_gaps(read_settlements_supplemented(c, holes=pd.DataFrame(holes) if holes else None))
        gaps = gaps[[not any(h["start"] == g.previous and h["end"] == g.stamp for h in holes)
                     for g in gaps.itertuples()]]
        if not len(gaps):
            continue
        archive = load_funding_supplemented(c)
        traded = traded_days(panel.load_ohlcv(c))
        for g in gaps.itertuples(index=False):
            kind = classify(c, g, settled.get(c), traded)
            lo, hi = g.previous - pd.Timedelta(hours=1), g.stamp + pd.Timedelta(hours=1)
            if kind == "no_trading":
                evidence = f"no traded daily bar strictly inside the gap ({c}_1d.csv)"
            else:
                api, path = api_rows(c, lo, hi)
                mine = set(archive[(archive.timestamp >= lo) & (archive.timestamp <= hi)].timestamp.dt.floor("min"))
                if api != mine:
                    raise ValueError(f"{c}: archive and API differ around {g.previous} -> {g.stamp}: "
                                     f"API only {sorted(api - mine)}, archive only {sorted(mine - api)}")
                evidence = f"archive == fapi/v1/fundingRate on [{lo}, {hi}] ({path.name}, fetched 2026-10-26)"
            rows.append({"contract": c, "start": g.previous, "end": g.stamp, "kind": kind,
                         "spacing_hours": g.spacing_hours,
                         "intervals": f"{g.previous_interval_hours:g}h->{g.interval_hours:g}h", "evidence": evidence})
    rows += [t for t in settlement_tails(contracts)
             if not any(r["contract"] == t["contract"] and r["start"] <= t["start"] and r["end"] >= t["instant_end"]
                        for r in rows)]
    for r in rows:
        r.pop("instant_end", None)
    return pd.DataFrame(rows)


def settlement_tails(contracts) -> list:
    """A settlement due between the last recorded one and the settlement instant T, but absent (OM:
    hourly funding to 08:00, trading until 09:08, no 09:00 settlement). Accepted, and charged nothing,
    only if Binance's API also has no row there; registered from the last row to one interval past T,
    so a held window ending at T is complete."""
    out = []
    for e in load_event_register().itertuples():
        if e.contract not in contracts or not (panel.MARK_DIR / f"{e.contract}.csv").exists():
            continue
        f = load_funding_supplemented(e.contract)
        before = f[f.timestamp <= e.instant]
        if before.empty:
            continue
        last = before.iloc[-1]
        due = last.timestamp.floor("h") + pd.Timedelta(hours=float(last.funding_interval_hours))
        if due > e.instant:
            continue
        api, path = api_rows(e.contract, last.timestamp - pd.Timedelta(hours=1), e.instant)
        mine = set(before[before.timestamp >= last.timestamp - pd.Timedelta(hours=1)].timestamp.dt.floor("min"))
        if api != mine:
            raise ValueError(f"{e.contract}: the API has settlements before T={e.instant} the archive lacks: {sorted(api - mine)}")
        out.append({"contract": e.contract, "start": last.timestamp.floor("h"),
                    "end": e.instant + pd.Timedelta(hours=float(last.funding_interval_hours)), "kind": "settlement_tail",
                    "instant_end": e.instant,
                    "spacing_hours": None, "intervals": f"{last.funding_interval_hours:g}h",
                    "evidence": f"settlement due {due} before the settlement instant {e.instant} is absent from the "
                                f"archive and from fapi/v1/fundingRate ({path.name}); charged nothing"})
    return out


if __name__ == "__main__":
    register = build()
    register.to_csv(OUT, index=False)
    print(register.groupby("kind").size().to_string())
    print(f"{len(register)} gaps registered in {register.contract.nunique()} contracts -> {OUT}")
