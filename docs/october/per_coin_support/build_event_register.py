"""Build the stage-B event register: every contract end a stage-B book can meet while a coin may be
held (docs/october/per_coin_stage_b_specification.md §4.2).

Run from the repo root: PYTHONPATH=. python docs/october/per_coin_support/build_event_register.py

Which contracts. For each stage-B universe (top 25, top 50, registry-wide $10m), a coin may be held on
holding day D if it was eligible at the decision for D, or was held the day before and could not
trade. Every contract with such a day on which it cannot trade is listed. In these books every one is
a contract end (a stop, a relisting more than 30 days later, or BNX's token swap, a registered span
break): no halt inside a span is met. Anything else raises.

The settlement instant T. Binance keeps archiving hourly MARK klines after a settlement (FTT settled
at 04:00 on 2022-11-14; its marks run to 23:00), so the last mark bar cannot date it. The TRADE
klines can: the last 1-minute bar with volume is the settlement print. T is that minute rounded to
the nearest half hour (Binance's scheduling grid) when it lies within 5 minutes of it; otherwise (an
unscheduled end) T is the close of that last traded minute, flagged in the evidence. T is checked
against the dated announcement where one is on file (docs/october/binance_delisting_rules_2026-10-13.md).
The register covers T's day and the two after it: under the one-day lag a coin stays eligible at the
decision for two more holding days, and those days need an entry so the ledger books them as
untradable rather than stopping. The 1-minute index files settlement_price reads are fetched if absent.

Prices only; no strategy or control return is computed.
"""
import hashlib
import io
import zipfile
from datetime import timedelta
from pathlib import Path

import pandas as pd

from data.universe.coinmarketcap.new_fetcher import panel
from portfolio.basket import SCORED_END, SCORED_START
from portfolio.per_coin_ledger import _tradable_frame
from portfolio.per_coin_runs import index_days, stage_b_universe_inputs, STAGE_B_UNIVERSES
from portfolio.per_coin_signal import load_per_coin_config
from portfolio.per_coin_universe import eligibility, forced_starts, traded_days

OUT = Path(__file__).resolve().parent / "event_register.csv"
ANNOUNCED = {"LUNAUSDT": ("2022-05-12 15:30", "ef3ce76d5c2d45ee9b6dc3f281cde744"),
             "FTTUSDT": ("2022-11-14 04:00", "3b4c7c08479b48d48189191e37a7adca"),
             "ICPUSDT": ("2022-06-10 09:00", "af469aeeab074738bb4a276070a9d11b")}
FOLLOWING_DAYS = 2
TOLERANCE = pd.Timedelta(minutes=5)


def possibly_held_ends() -> dict:
    """contract -> the last traded day before its first possibly-held untradable day, over every
    stage-B universe."""
    config = load_per_coin_config()
    days = [SCORED_START + timedelta(days=i) for i in range(-1, (SCORED_END - SCORED_START).days + 2)]
    ends = {}
    for universe in STAGE_B_UNIVERSES:
        members, bars, counts = stage_b_universe_inputs(universe)
        contracts = sorted(bars)
        ok, _, _ = eligibility(members, days[:-1], bars, counts, config)
        tradable = _tradable_frame(tuple(contracts), days, bars, counts, config.min_hourly_marks, config.span_break_days)
        for c in ok.columns:
            held = False
            for prev, day in zip(days[:-1], days[1:]):
                decided = bool(ok.at[prev, c])
                if (decided or held) and not tradable.at[day, c]:
                    tdays = traded_days(bars[c])
                    before, after = tdays[tdays < pd.Timestamp(day)], tdays[tdays >= pd.Timestamp(day)]
                    resumes_new_span = (not len(after) or (after[0] - before[-1]).days - 1 > config.span_break_days
                                        or after[0] in forced_starts(bars[c]))
                    if not resumes_new_span:
                        raise ValueError(f"{c}: untradable on {day} while possibly held, inside a span: a halt "
                                         f"(ledger step 7 is not built)")
                    last = before[-1].date()
                    ends.setdefault(c, set()).add(last)
                    held = False                          # the exchange closes the position at the end
                    continue
                held = decided
    return ends


def last_traded_minute(contract: str, day) -> tuple:
    key = f"data/futures/um/daily/klines/{contract}/1m/{contract}-1m-{day}.zip"
    path = panel.fetch(key)                       # verified against Binance's own checksum
    data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = z.read(z.namelist()[0]).decode()
    frame = pd.read_csv(io.StringIO(text), header=None if text[:1].isdigit() else 0)
    frame.columns = range(frame.shape[1])
    traded = frame[frame[5].astype(float) > 0]
    return pd.to_datetime(int(traded[0].iloc[-1]), unit="ms", utc=True), key, hashlib.sha256(data).hexdigest()


def build() -> pd.DataFrame:
    rows = []
    for contract, lasts in sorted(possibly_held_ends().items()):
        for day in sorted(lasts):
            minute, key, digest = last_traded_minute(contract, day)
            instant = minute.round("30min")
            off_grid = abs(minute - instant) > TOLERANCE
            if off_grid:                                 # an unscheduled end: the close of the last traded minute
                instant = minute + pd.Timedelta(minutes=1)
            if contract in ANNOUNCED and pd.Timestamp(ANNOUNCED[contract][0], tz="UTC") != instant:
                raise ValueError(f"{contract}: data say {instant}, the announcement {ANNOUNCED[contract][0]}")
            for d in index_days(instant):
                panel.fetch(f"data/futures/um/daily/indexPriceKlines/{contract}/1m/{contract}-1m-{d}.zip")
            evidence = (f"last traded 1m bar {minute:%Y-%m-%d %H:%M} UTC in {key} (sha256 {digest[:16]})"
                        + ("; off Binance's half-hour grid: T = the close of that minute" if off_grid else ""))
            if contract in ANNOUNCED:
                evidence += f"; Binance announcement {ANNOUNCED[contract][1]}"
            first = instant.date()
            rows.append({"contract": contract, "kind": "settlement", "first_day": first,
                         "last_day": first + timedelta(days=FOLLOWING_DAYS), "instant": instant.isoformat(),
                         "evidence": evidence})
    return pd.DataFrame(rows).sort_values(["first_day", "contract"]).reset_index(drop=True)


if __name__ == "__main__":
    register = build()
    register.to_csv(OUT, index=False)
    print(f"{len(register)} settlements -> {OUT}")
    print(register.drop(columns="evidence").to_string())
