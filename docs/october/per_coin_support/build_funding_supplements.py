"""Funding before the archive starts, from Binance's own API (stage-B specification §4.3).

Run from the repo root: PYTHONPATH=. python docs/october/per_coin_support/build_funding_supplements.py

Why: for three relisted or redenominated contracts the monthly fundingRate archive starts late, so a
position held earlier would have no funding record:
  ICPUSDT  archive from 2022-09-01; the old contract traded from 2021-05-11 to its 2022-06-10 settlement
  BNXUSDT  archive from 2023-02-01; traded from 2022-04-01 (token swap 2023-02-11 -> 02-22)
  TLMUSDT  archive from 2023-03-01; traded from 2021-07-16 to its 2022-06-09 settlement
The archive has no `..._SETTLED` funding folder for them; the history exists only behind
`fapi/v1/fundingRate`.

What: the raw API responses (data/raw/binance/api/fundingRate/<contract>/, fetched 2026-10-26) are
checked against the archive over the first twelve months both cover: every row must match exactly
in rate and millisecond timestamp, or this raises. The rows before the archive's first settlement are
then written, in the derived funding format, to data/derived/binance/funding_supplement/<contract>.csv,
joining the archive without a gap. The archive-built files are not touched, so the frozen inputs of
earlier runs are unchanged. The ledger's stage-B settlement reader joins the two
(portfolio/per_coin_runs.py). Rates only; no strategy or control return is computed.
"""
import hashlib
import json
import time

import pandas as pd
import requests

from data.universe.coinmarketcap.new_fetcher import panel

CONTRACTS = ("ICPUSDT", "BNXUSDT", "TLMUSDT")
API = "https://fapi.binance.com/fapi/v1/fundingRate"
CHECK_MONTHS = 12


def ms(t) -> int:
    return int(pd.Timestamp(t).value // 10**6)


def fetch(contract: str, start, end) -> list:
    """Raw API responses for [start, end), month by month (well under 1,000 rows each), cached."""
    folder = panel.RAW_DIR / "api" / "fundingRate" / contract
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for a in pd.date_range(start, end, freq="MS", tz="UTC", inclusive="left"):
        b = a + pd.offsets.MonthBegin(1)
        path = folder / f"{contract}-fundingRate-api-month-{a:%Y-%m}.json"
        if not path.exists():
            body = requests.get(API, params={"symbol": contract, "startTime": ms(a), "endTime": ms(b) - 1,
                                             "limit": 1000}, timeout=30)
            body.raise_for_status()
            path.write_bytes(body.content)
            time.sleep(0.25)
        if len(json.loads(path.read_text())) >= 1000:
            raise ValueError(f"{path.name}: a month at the API limit; paginate")
        files.append(path)
    return files


def load_api(files) -> pd.DataFrame:
    frame = pd.concat([pd.DataFrame(json.loads(p.read_text())) for p in files])
    frame = frame.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)
    return pd.DataFrame({"timestamp": pd.to_datetime(frame.fundingTime.astype("int64"), unit="ms", utc=True),
                         "funding_rate": frame.fundingRate.astype(float),
                         "funding_interval_hours": frame.get("fundingIntervalHours", pd.Series(dtype=float))})


def intervals_from_spacing(contract, stamps: pd.Series, archive_start) -> list:
    """Each row's interval as the hours to the next settlement (the last row's to the archive's first).
    Inferring intervals from spacing could hide a missing payment, so a spacing that differs from BOTH
    of its neighbours while they agree with each other (2, 2, 4, 2, 2: a hole in a steady cadence)
    raises; a different spacing between two regimes (8, 8, 4, 2, 2: Binance switching cadence off its
    grid, as FTT on 2022-11-10) is a cadence change and is accepted. Only 1, 2, 4 and 8 hours are
    Binance cadences; a switch may also leave one 3, 5, 6 or 7-hour step."""
    nxt = list(stamps.iloc[1:]) + [archive_start]
    hours = [round((b - a) / pd.Timedelta(hours=1), 2) for a, b in zip(stamps, nxt)]
    for i, h in enumerate(hours):
        before, after = hours[i - 1] if i else None, hours[i + 1] if i + 1 < len(hours) else None
        steady = before is not None and after is not None and before == after
        if h not in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) or (steady and h != before):
            raise ValueError(f"{contract}: settlement spacing {h}h at {stamps.iloc[i]} inside a steady "
                             f"{before}h cadence: a missing payment, not a cadence change")
    cadences = (1.0, 2.0, 4.0, 8.0)
    out = []
    for i, h in enumerate(hours):       # a switch step keeps the cadence it leaves, so the check sees the change
        out.append(h if h in cadences else next((x for x in hours[i - 1::-1] if x in cadences), None)
                   or next(x for x in hours[i + 1:] if x in cadences))
    return out


def build(contract: str) -> pd.DataFrame:
    archive = panel.load_funding(contract)
    archive_start = archive.timestamp.min()
    listed = pd.read_csv(panel.TRADED_RANGES).set_index("contract").at[contract, "first_traded"]
    month0 = pd.Timestamp(listed, tz="UTC").replace(day=1)
    check_end = (archive_start.floor("D").replace(day=1) + pd.DateOffset(months=CHECK_MONTHS)).tz_convert("UTC")
    files = fetch(contract, month0, check_end)
    api = load_api(files)
    a = api[(api.timestamp >= archive_start) & (api.timestamp < check_end)].reset_index(drop=True)
    b = archive[archive.timestamp < check_end].reset_index(drop=True)
    if len(a) != len(b) or not (a.timestamp.equals(b.timestamp) and (a.funding_rate == b.funding_rate).all()):
        raise ValueError(f"{contract}: the API does not reproduce the archive exactly on the overlap; not using it")
    early = api[api.timestamp < archive_start].reset_index(drop=True)
    spacing = early.timestamp.diff().dropna().dt.total_seconds().div(3600).round(2)
    if early.funding_interval_hours.isna().all():                   # older API rows carry no interval
        early["funding_interval_hours"] = [int(h) for h in intervals_from_spacing(contract, early.timestamp, archive_start)]
    gap = archive_start - early.timestamp.iloc[-1]
    if gap - pd.Timedelta(hours=float(early.funding_interval_hours.iloc[-1])) > pd.Timedelta(seconds=1):
        raise ValueError(f"{contract}: the supplement does not join the archive without a gap ({gap})")
    out = panel.DERIVED_DIR / "funding_supplement" / f"{contract}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    early.to_csv(out, index=False)
    print(f"{contract}: checked {len(a)} overlapping rows, identical; wrote {len(early)} rows "
          f"{early.timestamp.iloc[0]} .. {early.timestamp.iloc[-1]}")
    for p in files:
        print(f"  source {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
    return early


if __name__ == "__main__":
    for c in CONTRACTS:
        build(c)
