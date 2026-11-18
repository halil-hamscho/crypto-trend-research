"""Binance USDT-M perp contract registry, built from S3 listings alone.

REGISTRY FIRST, DOWNLOAD SECOND. The point of this module is to answer "was
this contract tradable on date D?" WITHOUT fetching a single price file. That
distinction is the whole reason the old pipeline was wrong: it tried to load a
coin and treated a failure as "not tradable", which conflates two opposite
things —

    BSVUSDT in 2020  -> no contract existed. Correct exclusion; promote someone.
    EOSUSDT in 2020  -> contract existed, we just never downloaded it. A BUG
                        that silently dropped a rank-8 name for six years.

A listing distinguishes them; a failed download cannot.

Note the live API is NOT usable here: fapi.binance.com returns HTTP 451
(geo-blocked) from this machine, so S3 is the only path to contract metadata.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as Date

import pandas as pd
import requests

from config import DATA_DIR

logger = logging.getLogger(__name__)

S3_BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINES_PREFIX = "data/futures/um/monthly/klines/"
REGISTRY_TABLE = DATA_DIR / "derived" / "contracts.csv"

# Contracts quoting a scaled unit of the asset: the price is per 1,000 (or
# 1,000,000) tokens. Daily % returns are unaffected, which is why this has been
# survivable so far, but the cmc_id join and any absolute-price logic are not.
#
# Hard-coded rather than regexed, deliberately. `1000XUSDT` is 1000x of the
# token "X" — a lstrip("1000") yields "XUSDT" by luck and `.removeprefix("1000")`
# on "1000000MOGUSDT" yields "000MOGUSDT" by bug. build() asserts this list still
# matches S3, so a new scaled contract fails the build instead of silently
# rescaling a coin by 10^3.
CONTRACT_SCALE = {
    "1000000BOBUSDT": 1_000_000,
    "1000000MOGUSDT": 1_000_000,
    "1000BONKUSDT": 1_000,
    "1000BTTCUSDT": 1_000,
    "1000CATUSDT": 1_000,
    "1000CHEEMSUSDT": 1_000,
    "1000FLOKIUSDT": 1_000,
    "1000LUNCUSDT": 1_000,
    "1000PEPEUSDT": 1_000,
    "1000RATSUSDT": 1_000,
    "1000SATSUSDT": 1_000,
    "1000SHIBUSDT": 1_000,
    "1000WHYUSDT": 1_000,
    "1000XECUSDT": 1_000,
    "1000XUSDT": 1_000,
}


def _s3_list(prefix: str, timeout: float = 15.0) -> str:
    """Raw XML for one listing page.

    raise_for_status() is load-bearing: an error page (busy, throttled, a blip)
    contains no <Key> elements, so without it a failed request reads as "this
    contract has no files" — i.e. "it never existed". That is the exact
    confusion this module exists to prevent, and it would fail open. Raising
    instead lets the caller's retry pass handle it.
    """
    response = requests.get(f"{S3_BUCKET}?delimiter=/&prefix={prefix}", timeout=timeout)
    response.raise_for_status()
    if "<ListBucketResult" not in response.text:
        raise ValueError(f"not an S3 listing for {prefix!r}: {response.text[:120]!r}")
    return response.text


def _s3_list_all(prefix: str) -> tuple[list[str], list[str]]:
    """(common prefixes, keys) across every page.

    S3 caps a listing at 1,000 entries and sets <IsTruncated>true; the klines
    prefix returns ~1,018, so the NextMarker loop is mandatory, not optional.
    """
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""

    while True:
        url = prefix + (f"&marker={requests.utils.quote(marker)}" if marker else "")
        xml = _s3_list(url)
        prefixes += re.findall(r"<Prefix>([^<]+)</Prefix>", xml)
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        if not re.search(r"<IsTruncated>true", xml):
            return prefixes, keys
        found = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        if not found:
            return prefixes, keys
        marker = found.group(1)


def list_contracts() -> list[str]:
    """Every USDT-M perp contract that has EVER existed in the um archive.

    Measured 2026-09-17: 1,018 contract directories, 864 USDT-quoted.
    """
    prefixes, _ = _s3_list_all(KLINES_PREFIX)
    depth = KLINES_PREFIX.count("/") + 1
    names = [p.rstrip("/").split("/")[-1] for p in prefixes if p.count("/") == depth]

    contracts = []
    for name in names:
        if not name.endswith("USDT"):
            continue          # USDC/BUSD-quoted variants: a different instrument
        if "SETTLED" in name:
            continue          # e.g. AERGOUSDTSETTLEDSETTLED — delisting artifacts
        contracts.append(name)
    return sorted(contracts)


def contract_lifetime(contract: str) -> tuple[str, str] | None:
    """(first_month, last_month) as 'YYYY-MM', or None if nothing is archived.

    Verified: LUNAUSDT -> ('2021-01', '2022-05'), 17 months — the contract died
    with the chain. BSVUSDT -> ('2023-10', ...), which is why it is correctly
    absent from the 2020 universe despite being a top-5 coin then.
    """
    _, keys = _s3_list_all(f"{KLINES_PREFIX}{contract}/1d/")
    months = sorted({k.split("-1d-")[1][:7] for k in keys if k.endswith(".zip")})
    return (months[0], months[-1]) if months else None


def build(contracts: list[str] | None = None, workers: int = 8) -> pd.DataFrame:
    """Build the registry. One listing per contract; NO price downloads.

    Threaded, unlike the CMC backfill, and the measurement is why: an S3 listing
    takes ~1.2s and the bucket imposes no rate cap we hold ourselves to, so 864
    serial listings is ~17 minutes of pure waiting. (The CMC fetcher is single
    -threaded because there a 0.145s fetch against a self-imposed 5 req/s cap
    means one thread already saturates it — threads would overlap latency the
    limiter throws away. Different numbers, different answer.)

    Tolerant per contract: a transient timeout records the contract and moves on,
    then a second pass retries the failures. Anything still failing raises — a
    partial registry silently understates what was tradable, which is exactly
    the class of error this module exists to prevent.
    """
    contracts = contracts if contracts is not None else list_contracts()

    # A scaled contract we do not know about would be silently mis-priced, so
    # make it a build failure rather than a runtime surprise.
    seen_scaled = {c for c in contracts if re.match(r"^1000\d*[A-Z]", c)}
    unknown = seen_scaled - set(CONTRACT_SCALE)
    if unknown:
        raise ValueError(
            f"unknown scaled contract(s) {sorted(unknown)} — add them to "
            f"CONTRACT_SCALE with the correct multiplier before continuing"
        )

    spans, failed = _lifetimes(contracts, workers)

    if failed:
        logger.warning("retrying %d failed listings", len(failed))
        retried, failed = _lifetimes(sorted(failed), workers=2)
        spans.update(retried)

    if failed:
        raise RuntimeError(
            f"registry incomplete: {len(failed)} of {len(contracts)} listings "
            f"still failing, e.g. {sorted(failed)[:5]}"
        )

    frame = pd.DataFrame([
        {
            "contract": c,
            "first_month": spans[c][0] if spans[c] else None,
            "last_month": spans[c][1] if spans[c] else None,
            "scale": CONTRACT_SCALE.get(c, 1),
        }
        for c in contracts
    ]).sort_values("contract").reset_index(drop=True)

    REGISTRY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REGISTRY_TABLE, index=False)
    return frame


def _lifetimes(
    contracts: list[str], workers: int
) -> tuple[dict[str, tuple[str, str] | None], dict[str, str]]:
    """(contract -> span-or-None, contract -> error) for one pass."""
    spans: dict[str, tuple[str, str] | None] = {}
    failed: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(contract_lifetime, c): c for c in contracts}
        for i, fut in enumerate(as_completed(futures), start=1):
            contract = futures[fut]
            try:
                spans[contract] = fut.result()
            except Exception as exc:
                failed[contract] = f"{type(exc).__name__}: {exc}"
            if i % 200 == 0:
                logger.info("  %d/%d listings (%d failed)", i, len(contracts), len(failed))

    return spans, failed


def load() -> pd.DataFrame:
    return pd.read_csv(REGISTRY_TABLE)


def is_live(registry: pd.DataFrame, contract: str, on: Date | str) -> bool:
    """Was `contract` archived in the month containing `on`?

    UPPER BOUND ONLY — "archived" is not "tradable". A halted contract keeps
    getting monthly files with its price frozen and zero volume: FTTUSDT last
    traded 2022-11-14 and is archived through 2026-08, pinned at 1.59. 9 of the
    62 mapped contracts have such tails (MATIC, TON, LUNA, FTT, SYS, XEM, KLAY,
    OM, BTT). A backtest trusting this alone would hold a dead coin at a frozen
    price — zero return, zero volatility, the "safest" asset in the book. The
    true first/last TRADED dates come from the bars (volume > 0) when the panel
    is downloaded; use those for eligibility. This is only the cheap prefilter.

    Monthly granularity is the archive's, not a choice: files are per-month, so
    a contract that listed mid-month reads as live from the 1st. That biases
    toward INCLUDING a few untradable early days, which the warmup filter then
    removes anyway (a contract with days of history has no usable signal).
    """
    month = (on if isinstance(on, str) else on.isoformat())[:7]
    row = registry.loc[registry.contract == contract]
    if row.empty or pd.isna(row.first_month.iloc[0]):
        return False
    return str(row.first_month.iloc[0]) <= month <= str(row.last_month.iloc[0])
