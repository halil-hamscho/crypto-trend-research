"""Binance USDT-M daily klines + funding for every mapped contract, from Data Vision.

Raw: every zip exactly as served, next to Binance's own .CHECKSUM file,
verified on download. Derived: one OHLCV CSV and one funding CSV per contract,
plus traded_ranges.csv — the first and last day each contract ACTUALLY traded.

traded_ranges matters as much as the prices. The archive keeps writing files
for halted contracts — FTTUSDT last traded 2022-11-14 and is archived through
2026-08 at a frozen 1.59 with zero volume; EOSUSDT's funding files outlive its
klines by 15 months. "A file exists" is not "tradable". Eligibility must use
traded_ranges, never the registry's months.

Adjustments (1000x scaling, the LUNA -> 1000LUNC splice) are NOT applied here.
Derived files are faithful per-contract transcriptions; adjustments happen at
load time so a stored file is never mutated.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as Date
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR
from data.universe.coinmarketcap.new_fetcher import contract_map, registry

logger = logging.getLogger(__name__)

DATA_VISION = "https://data.binance.vision"
RAW_DIR = DATA_DIR / "raw" / "binance"
DERIVED_DIR = DATA_DIR / "derived" / "binance"
TRADED_RANGES = DERIVED_DIR / "traded_ranges.csv"


# --------------------------------------------------------------------------- raw

def archive_keys(contract: str) -> list[str]:
    """Every zip Binance holds for `contract`: monthly 1d klines, monthly
    funding, and this month's DAILY klines (monthly files only cover complete
    months). There is no daily funding archive (HTTP 404), so the current month
    ends up with prices but no funding — left missing downstream, never zero.

    Listed rather than generated, so no key is guessed and a 404 on a listed
    key is a genuine error rather than an expected gap.
    """
    this_month = Date.today().isoformat()[:7]
    prefixes = (
        f"data/futures/um/monthly/klines/{contract}/1d/",
        f"data/futures/um/monthly/fundingRate/{contract}/",
        # S3 prefixes match partial key names, so this lists exactly the daily
        # files published so far this month — and nothing for a dead contract.
        f"data/futures/um/daily/klines/{contract}/1d/{contract}-1d-{this_month}",
    )
    keys: list[str] = []
    for prefix in prefixes:
        _, found = registry._s3_list_all(prefix)
        keys += [k for k in found if k.endswith(".zip")]
    return keys


def raw_path(key: str) -> Path:
    return RAW_DIR / key.removeprefix("data/")


def fetch(key: str) -> Path:
    """Download one zip and its vendor .CHECKSUM, verify, write both. Idempotent.

    Unlike the CMC side this DOES verify a hash, because here the hash is
    Binance's own: a match proves the bytes are what Binance served.
    """
    path = raw_path(key)
    if path.exists():
        return path

    body = requests.get(f"{DATA_VISION}/{key}", timeout=30)
    body.raise_for_status()
    check = requests.get(f"{DATA_VISION}/{key}.CHECKSUM", timeout=30)
    check.raise_for_status()

    expected = check.text.split()[0]
    actual = hashlib.sha256(body.content).hexdigest()
    if actual != expected:
        raise ValueError(f"{key}: sha256 {actual[:12]}... != vendor {expected[:12]}...")

    path.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{path}.CHECKSUM").write_text(check.text)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(body.content)
    tmp.rename(path)                 # the zip lands last; exists() keys off it
    return path


def download(contracts: list[str] | None = None, workers: int = 12) -> None:
    """Fetch every archived file for every mapped contract.

    Threaded for the same reason as the registry: Data Vision is S3/CloudFront
    with no rate cap we hold ourselves to, and latency dominates. Tolerant per
    file, one retry pass, and anything still failing raises — a missing month
    is a hole in a price series, not a rounding error.
    """
    contracts = contracts or sorted(contract_map.load().contract.unique())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        listings = list(pool.map(archive_keys, contracts))
    keys = [k for batch in listings for k in batch]
    todo = [k for k in keys if not raw_path(k).exists()]
    logger.info("%d contracts, %d files, %d to fetch", len(contracts), len(keys), len(todo))

    failed = _fetch_all(todo, workers)
    if failed:
        logger.warning("retrying %d failed files", len(failed))
        failed = _fetch_all(sorted(failed), workers=2)
    if failed:
        raise RuntimeError(f"download incomplete: {len(failed)} files, e.g. {sorted(failed)[:5]}")


def fill_gaps(contracts: list[str] | None = None, workers: int = 12) -> int:
    """Fetch DAILY kline files for calendar days the monthly files dropped.

    Measured: 14 contracts are missing exactly 2022-02-26..28 and 2022-04-01..02
    from their monthly files, although they traded normally (XRPUSDT
    2022-02-27: $1.04bn volume). The daily archive has those days. Filling makes
    every derived OHLCV series calendar-complete, so that a missing row can only
    ever mean "no data", never "no trading" — halts (ICPUSDT 2022-06 -> 09)
    then show up honestly as zero-volume bars rather than as holes.

    Returns the number of daily files fetched; re-run build_all() afterwards.
    """
    contracts = contracts or sorted(contract_map.load().contract.unique())
    keys = []
    for contract in contracts:
        days = set(load_ohlcv(contract).timestamp.dt.date)
        full = pd.date_range(min(days), max(days), freq="D").date
        keys += [
            f"data/futures/um/daily/klines/{contract}/1d/{contract}-1d-{day}.zip"
            for day in full if day not in days
        ]
    todo = [k for k in keys if not raw_path(k).exists()]
    failed = _fetch_all(todo, workers)

    # A 404 here is a genuine absence, not a failure: the contract did not exist
    # that day. ICPUSDT has no file for 2022-09-22..26, between its delisting
    # and its relisting on 09-27. Anything else is a real error.
    absent = {k for k, err in failed.items() if "404" in err}
    if absent:
        logger.info("no daily file (contract not listed that day): %d, e.g. %s",
                    len(absent), sorted(absent)[:3])
    errors = {k: e for k, e in failed.items() if k not in absent}
    if errors:
        raise RuntimeError(f"gap fill incomplete: {len(errors)} files, e.g. {sorted(errors)[:5]}")
    return len(todo) - len(absent)


def _fetch_all(keys: list[str], workers: int) -> dict[str, str]:
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, k): k for k in keys}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                fut.result()
            except Exception as exc:
                failed[futures[fut]] = f"{type(exc).__name__}: {exc}"
            if i % 1000 == 0:
                logger.info("  %d/%d files (%d failed)", i, len(keys), len(failed))
    return failed


# ------------------------------------------------------------ mark prices (1h)
#
# Funding settles against the MARK price at the settlement instant, not the
# daily close, so the ledger needs an hourly mark series. These are separate
# functions rather than folded into archive_keys/download/build_all: marks are
# scoped to each contract's TRADED range (traded_ranges.csv, which build_all
# writes), and the daily-kline pipeline must behave exactly as before.

MARK_DIR = DERIVED_DIR / "mark_1h"


def _traded_span(contract: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[first, last] traded day, as UTC timestamps at 00:00, from traded_ranges."""
    row = load_traded_ranges().set_index("contract").loc[contract]
    return (pd.Timestamp(row.first_traded, tz="UTC"), pd.Timestamp(row.last_traded, tz="UTC"))


def mark_archive_keys(contract: str) -> list[str]:
    """Every markPriceKlines 1h zip covering the contract's traded range: monthly
    files for the months it touches, plus this month's DAILY files (monthly files
    cover complete months only). Listed, not guessed, like archive_keys. Nothing
    outside [first_traded, last_traded] is fetched, so a halted contract's
    archived-but-dead tail is skipped."""
    first, last = _traded_span(contract)
    first_month, last_month = first.strftime("%Y-%m"), last.strftime("%Y-%m")
    this_month = Date.today().isoformat()[:7]

    _, monthly = registry._s3_list_all(f"data/futures/um/monthly/markPriceKlines/{contract}/1h/")
    keys = [k for k in monthly if k.endswith(".zip")
            and first_month <= k[-11:-4] <= last_month]

    # only if the traded range reaches into the current month
    if last_month >= this_month:
        _, daily = registry._s3_list_all(
            f"data/futures/um/daily/markPriceKlines/{contract}/1h/{contract}-1h-{this_month}")
        keys += [k for k in daily if k.endswith(".zip")
                 and first.date().isoformat() <= k[-14:-4] <= last.date().isoformat()]
    return keys


def download_marks(contracts: list[str] | None = None, workers: int = 12) -> int:
    """Fetch every mark-price zip for every mapped contract, over its traded range.
    Same policy as download(): idempotent, checksum-verified, one retry pass, and
    anything still failing raises. Returns total bytes of zips on disk for these
    keys."""
    contracts = contracts or sorted(contract_map.load().contract.unique())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        listings = list(pool.map(mark_archive_keys, contracts))
    keys = [k for batch in listings for k in batch]
    todo = [k for k in keys if not raw_path(k).exists()]
    logger.info("marks: %d contracts, %d files, %d to fetch", len(contracts), len(keys), len(todo))

    failed = _fetch_all(todo, workers)
    if failed:
        logger.warning("retrying %d failed mark files", len(failed))
        failed = _fetch_all(sorted(failed), workers=2)
    if failed:
        raise RuntimeError(f"mark download incomplete: {len(failed)} files, e.g. {sorted(failed)[:5]}")
    return sum(raw_path(k).stat().st_size for k in keys)


def fill_mark_gaps(contracts: list[str] | None = None, workers: int = 12) -> int:
    """Fetch DAILY mark files for calendar days that the built series lacks a full
    24 hours for, inside the traded range (the mark analogue of fill_gaps).
    A 404 is a genuine absence and is left as a gap, never filled. Returns the
    number of daily files fetched; re-run build_mark_prices() afterwards."""
    contracts = contracts or sorted(contract_map.load().contract.unique())
    keys = []
    for contract in contracts:
        first, last = _traded_span(contract)
        hours = pd.read_csv(MARK_DIR / f"{contract}.csv", usecols=["timestamp"])
        counts = pd.to_datetime(hours.timestamp, utc=True).dt.floor("D").value_counts()
        for day in pd.date_range(first, last, freq="D"):
            if counts.get(day, 0) < 24:
                d = day.date().isoformat()
                keys.append(f"data/futures/um/daily/markPriceKlines/{contract}/1h/{contract}-1h-{d}.zip")
    todo = [k for k in keys if not raw_path(k).exists()]
    failed = _fetch_all(todo, workers)

    absent = {k for k, err in failed.items() if "404" in err}
    if absent:
        logger.info("no daily mark file: %d, e.g. %s", len(absent), sorted(absent)[:3])
    errors = {k: e for k, e in failed.items() if k not in absent}
    if errors:
        raise RuntimeError(f"mark gap fill incomplete: {len(errors)} files, e.g. {sorted(errors)[:5]}")
    return len(todo) - len(absent)


# ----------------------------------------------------------------------- derived

def _read_zip(path: Path) -> pd.DataFrame:
    """One archived CSV. Kline files are headerless through 2021-12 and carry a
    header from 2022-01; funding files always have one. Sniff, don't assume."""
    with zipfile.ZipFile(path) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    return pd.read_csv(io.StringIO(text), header=None if text[:1].isdigit() else 0)


def _to_utc(epoch: pd.Series) -> pd.Series:
    """Epoch -> UTC timestamp. Milliseconds today, but guard against the
    microsecond switch Binance made on spot data rather than silently decoding
    it as the year 50,000."""
    unit = "us" if epoch.iloc[0] > 1e14 else "ms"
    return pd.to_datetime(epoch, unit=unit, utc=True)


def build_ohlcv(contract: str) -> pd.DataFrame:
    kline_dirs = (
        RAW_DIR / "futures/um/monthly/klines" / contract / "1d",
        RAW_DIR / "futures/um/daily/klines" / contract / "1d",
    )
    frames = []
    for folder in kline_dirs:                      # monthly first, so it wins ties
        for path in sorted(folder.glob("*.zip")):
            # 0 open_time, 1-4 OHLC, 5 base volume, 7 quote volume. The quote
            # column is the dollar volume the investability screen reads, so it
            # is kept rather than trimmed away with the rest of the tail.
            raw = _read_zip(path).iloc[:, [0, 1, 2, 3, 4, 5, 7]]
            raw.columns = ["open_time", "open", "high", "low", "close",
                           "volume", "quote_volume"]
            frames.append(raw)

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "timestamp", _to_utc(df.pop("open_time")))
    df = df.drop_duplicates("timestamp", keep="first").sort_values("timestamp")

    out = DERIVED_DIR / "ohlcv" / f"{contract}_1d.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def build_mark_prices(contract: str) -> pd.DataFrame:
    """Hourly mark-price OHLC, one row per hour open, trimmed to the traded range.
    Faithful transcription: no fill, no interpolation. A missing hour stays
    missing so the caller sees it."""
    first, last = _traded_span(contract)
    frames = []
    for kind in ("monthly", "daily"):              # monthly first, so it wins ties
        folder = RAW_DIR / f"futures/um/{kind}/markPriceKlines" / contract / "1h"
        for path in sorted(folder.glob("*.zip")):
            raw = _read_zip(path).iloc[:, [0, 1, 2, 3, 4]]
            raw.columns = ["open_time", "open", "high", "low", "close"]
            frames.append(raw)

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "timestamp", _to_utc(df.pop("open_time")))
    df = df.drop_duplicates("timestamp", keep="first").sort_values("timestamp")
    df = df[(df.timestamp >= first) & (df.timestamp < last + pd.Timedelta(days=1))]

    MARK_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MARK_DIR / f"{contract}.csv", index=False)
    return df


def build_all_marks(contracts: list[str] | None = None) -> None:
    for contract in contracts or sorted(contract_map.load().contract.unique()):
        build_mark_prices(contract)


def build_funding(contract: str) -> pd.DataFrame:
    """Per-settlement rows, NOT daily sums: intervals vary (PUMP settles every
    4h), so the interval is carried through and summing is the loader's job."""
    frames = [
        _read_zip(p)
        for p in sorted((RAW_DIR / "futures/um/monthly/fundingRate" / contract).glob("*.zip"))
    ]
    out = DERIVED_DIR / "funding" / f"{contract}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        df = pd.DataFrame(columns=["timestamp", "funding_rate", "funding_interval_hours"])
    else:
        raw = pd.concat(frames, ignore_index=True)
        df = pd.DataFrame({
            "timestamp": _to_utc(raw.calc_time),
            "funding_rate": raw.last_funding_rate,
            "funding_interval_hours": raw.funding_interval_hours,
        }).drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(out, index=False)
    return df


def traded_range(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> dict:
    traded = ohlcv[ohlcv.volume > 0]
    first, last = traded.timestamp.min(), traded.timestamp.max()
    within = ohlcv[(ohlcv.timestamp >= first) & (ohlcv.timestamp <= last)]
    return {
        "first_traded": first.date(),
        "last_traded": last.date(),
        "traded_days": len(traded),
        # days INSIDE [first, last] with no trade: a halt and relist, e.g.
        # ICPUSDT 2022-06 -> 09. [first, last] is therefore NOT "tradable
        # throughout" — per-day eligibility must read the bars.
        "halted_days": int((within.volume == 0).sum())
        + ((last - first).days + 1 - len(within)),
        # archived-but-dead tails: bars and funding rows after the last trade
        "zombie_bars": int((ohlcv.timestamp > last).sum()),
        "zombie_funding_rows": int((funding.timestamp > last + pd.Timedelta(days=1)).sum())
        if len(funding) else 0,
    }


def build_all(contracts: list[str] | None = None) -> pd.DataFrame:
    contracts = contracts or sorted(contract_map.load().contract.unique())
    rows = []
    for contract in contracts:
        ohlcv = build_ohlcv(contract)
        funding = build_funding(contract)
        rows.append({"contract": contract, **traded_range(ohlcv, funding)})

    ranges = pd.DataFrame(rows)
    ranges.to_csv(TRADED_RANGES, index=False)
    return ranges


def _read_timestamped(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # format="ISO8601": funding calc_times carry a few stray milliseconds
    # ("...00:00.006+00:00") alongside whole-second rows, and without this pandas
    # silently leaves the whole column as strings.
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, format="ISO8601")
    return df


def load_ohlcv(contract: str) -> pd.DataFrame:
    return _read_timestamped(DERIVED_DIR / "ohlcv" / f"{contract}_1d.csv")


def load_funding(contract: str) -> pd.DataFrame:
    return _read_timestamped(DERIVED_DIR / "funding" / f"{contract}.csv")


def load_traded_ranges() -> pd.DataFrame:
    return pd.read_csv(TRADED_RANGES, parse_dates=["first_traded", "last_traded"])


def load_mark_prices(contract: str) -> pd.DataFrame:
    """Hourly mark klines indexed by UTC hour-open. `open` at hour H is the mark
    price at H:00, which is what a funding settlement floored to H joins to."""
    return _read_timestamped(MARK_DIR / f"{contract}.csv").set_index("timestamp")
