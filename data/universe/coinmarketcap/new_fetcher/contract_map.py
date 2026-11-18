"""cmc_id -> Binance USDT-M contract map, date-ranged.

The bridge between the universe (keyed by cmc_id) and the price panel (keyed by
contract name). Symbols are NOT a safe join key, because over a coin's life:

    - the symbol changes    MIOTA -> IOTA (1720), TON -> GRAM (11419)
    - the symbol is reused  "LUNA" is cmc_id 4172 until 2022-05 and a different
                            coin, 20314, afterwards
    - the contract changes  4172: LUNAUSDT, then 1000LUNCUSDT
    - the contract is scaled SHIB trades as 1000SHIBUSDT

THE MATCHING RULE. A contract belongs to a cmc_id iff the contract's name is that
coin's symbol (or its scaled form) AND the contract's lifetime overlaps the dates
on which that cmc_id actually carried that symbol. The overlap is what makes the
rule safe: LUNAUSDT (2021-01 -> 2022-05) overlaps 4172's "LUNA" years and not
20314's, so symbol reuse disambiguates itself.

The row's VALIDITY is the contract's own lifetime, not the symbol span — the
contract was tradable whatever CMC called the coin at the time. That is how
IOTAUSDT covers 1720's MIOTA years even though it matched via "IOTA".

Every match is then checked against prices (validate_prices): Binance's close
must agree with CMC's price x scale. A name match is a hypothesis; a price match
is the evidence.

Identity only — no exclusions here. USDC maps to USDCUSDT because that IS its
contract; dropping stablecoins is the selector's job, where it gets logged.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from config import DATA_DIR
from data.universe.coinmarketcap.new_fetcher import normalize, registry

# Committed, not derived: this is hand-reviewable data, and the one table in the
# pipeline where a human signing off on the rows is the point.
MAP_TABLE = DATA_DIR / "universe" / "cmc_contract_map.csv"

POOL_DEPTH = 20        # max rank needed to field 10 tradable names is 16 (§4.2)

# The deeper map for the per-coin top-25 and top-50 sensitivities (per-coin protocol
# Amendment 2). A separate file, so the sealed basket run's map above is never rewritten.
# CMC's daily table holds the top 100 by rank, so 100 is the deepest pool available.
DEEP_MAP_TABLE = DATA_DIR / "universe" / "cmc_contract_map_top100.csv"
DEEP_POOL_DEPTH = 100

# Hand-resolved rows the symbol rule cannot find. Empty on first build: every
# pool coin with a contract resolved automatically. Same columns as the output.
MANUAL: list[dict] = []

DATA_VISION = "https://data.binance.vision/data/futures/um/monthly/klines"


def pool_ids(universe: pd.DataFrame, depth: int = POOL_DEPTH) -> set[int]:
    """cmc_ids that were in the market-cap top-`depth` on any date.

    Ranked by market_cap, NOT cmcRank — the two disagree on 12.6% of dates
    inside the top 20.
    """
    ranked = universe.sort_values(["date", "market_cap"], ascending=[True, False])
    cap_rank = ranked.groupby("date").cumcount() + 1
    return set(ranked.loc[cap_rank <= depth, "cmc_id"])


def symbol_spans(universe: pd.DataFrame) -> pd.DataFrame:
    """One row per (cmc_id, symbol): the first and last date it carried it."""
    spans = (
        universe.groupby(["cmc_id", "symbol"])["date"]
        .agg(first_date="min", last_date="max")
        .reset_index()
    )
    spans["first_month"] = spans.first_date.map(lambda d: d.isoformat()[:7])
    spans["last_month"] = spans.last_date.map(lambda d: d.isoformat()[:7])
    return spans


def _candidates(symbol: str) -> list[tuple[str, int]]:
    """(contract, scale) names that could belong to `symbol`.

    Scaled bases are derived from registry.CONTRACT_SCALE, never by stripping
    digits — '1000XUSDT' is 1000x the token 'X'.
    """
    out = [(f"{symbol}USDT", 1)]
    for contract, scale in registry.CONTRACT_SCALE.items():
        base = contract[len(str(scale)):-len("USDT")]
        if base == symbol:
            out.append((contract, scale))
    return out


def build(
    universe: pd.DataFrame | None = None,
    contracts: pd.DataFrame | None = None,
    depth: int = POOL_DEPTH,
    table=MAP_TABLE,
    allow_shared: bool = False,
) -> pd.DataFrame:
    universe = universe if universe is not None else normalize.load()
    contracts = contracts if contracts is not None else registry.load()
    lifetimes = contracts.set_index("contract")

    pool = pool_ids(universe, depth)
    spans = symbol_spans(universe)

    # The monthly archive only holds COMPLETE months, so a contract trading today
    # reads as ending last month — which empties the basket for the whole current
    # month. A contract archived in the archive's latest month is still listed;
    # give it an open end. (Upper bound, like every archive date: a halted
    # contract is archived too. True last-traded dates come from the bars.)
    latest_archived = contracts.last_month.max()
    OPEN_END = "9999-12"
    spans = spans[spans.cmc_id.isin(pool)]

    rows = []
    for span in spans.itertuples():
        for contract, scale in _candidates(span.symbol):
            if contract not in lifetimes.index:
                continue
            life = lifetimes.loc[contract]
            overlaps = life.first_month <= span.last_month and span.first_month <= life.last_month
            if not overlaps:
                continue      # same name, different era: e.g. 20314 vs LUNAUSDT
            rows.append({
                "cmc_id": span.cmc_id,
                "symbol_at_match": span.symbol,
                "contract": contract,
                "valid_from": life.first_month,
                "valid_to": OPEN_END if life.last_month == latest_archived else life.last_month,
                "scale": scale,
                "source": "auto",
            })

    rows += [{**row, "source": "manual"} for row in MANUAL]
    mapping = (
        pd.DataFrame(rows)
        .drop_duplicates(["cmc_id", "contract"])
        .sort_values(["cmc_id", "valid_from"])
        .reset_index(drop=True)
    )

    # A contract claimed by two coins means the overlap rule failed somewhere. The deep map
    # (allow_shared=True) hands such contracts to an explicit, price-based resolution instead.
    claimed = mapping.groupby("contract").cmc_id.nunique()
    if (claimed > 1).any() and not allow_shared:
        raise ValueError(f"contracts claimed by >1 cmc_id: {claimed[claimed > 1].to_dict()}")

    table.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(table, index=False)
    return mapping


def unmapped(mapping: pd.DataFrame, universe: pd.DataFrame | None = None,
             depth: int = POOL_DEPTH) -> pd.DataFrame:
    """Pool coins with no contract at all, and the symbols they carried."""
    universe = universe if universe is not None else normalize.load()
    missing = pool_ids(universe, depth) - set(mapping.cmc_id)
    names = universe[universe.cmc_id.isin(missing)].groupby("cmc_id").symbol.agg(
        lambda s: "/".join(dict.fromkeys(s))
    )
    return names.rename("symbols").reset_index()


def load(table=MAP_TABLE) -> pd.DataFrame:
    return pd.read_csv(table)


def _monthly_closes(contract: str, month: str) -> pd.Series:
    """Daily closes for one archived month, indexed by date — TRADED bars only.

    Data Vision CSVs are headerless through 2021-12 and carry a header from
    2022-01, so sniff line 1 rather than assume either.

    Zero-volume bars are dropped. A halted contract keeps getting archived with
    its price frozen (FTTUSDT: last trade 2022-11-14, then ~4 years pinned at
    1.59), and a frozen price must never count as agreement with CMC.
    """
    url = f"{DATA_VISION}/{contract}/1d/{contract}-1d-{month}.zip"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    has_header = text.startswith("open_time")
    frame = pd.read_csv(io.StringIO(text), header=0 if has_header else None)
    frame = frame.iloc[:, [0, 4, 5]]
    frame.columns = ["open_time", "close", "volume"]
    frame = frame[frame.volume > 0]
    dates = pd.to_datetime(frame.open_time, unit="ms", utc=True).dt.date
    return pd.Series(frame.close.to_numpy(), index=dates)


def validate_prices(
    mapping: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    tolerance: float = 0.05,
    depth: int = POOL_DEPTH,
    closes_for=None,
) -> pd.DataFrame:
    """For each row, compare one month of Binance closes with CMC price x scale.

    Validates where the mapping MATTERS: the months the coin was in the pool
    (market-cap top-`depth`) and the contract was archived. Only if there are
    none — e.g. 1000LUNCUSDT, which lists after 4172 left the pool for good — does
    it fall back to any month the coin appears in the universe. Takes the middle
    month, away from listing and delisting artifacts. Passes when the median
    ratio of traded closes is within `tolerance` of 1.
    """
    universe = universe if universe is not None else normalize.load()
    ranked = universe.sort_values(["date", "market_cap"], ascending=[True, False])
    in_pool = ranked[ranked.groupby("date").cumcount() < depth]
    closes_for = closes_for or _monthly_closes
    results = []

    for row in mapping.itertuples():
        coin = universe[universe.cmc_id == row.cmc_id]
        pooled = in_pool[in_pool.cmc_id == row.cmc_id].date

        def months_within(dates):
            return sorted({
                d.isoformat()[:7] for d in dates
                if row.valid_from <= d.isoformat()[:7] <= row.valid_to
            })

        months = months_within(pooled) or months_within(coin.date)
        if not months:
            results.append({"cmc_id": row.cmc_id, "contract": row.contract,
                            "month": None, "ratio": None, "ok": False,
                            "note": "no overlap with universe data"})
            continue

        month = months[len(months) // 2]
        try:
            closes = closes_for(row.contract, month)
        except Exception as exc:
            results.append({"cmc_id": row.cmc_id, "contract": row.contract,
                            "month": month, "ratio": None, "ok": False,
                            "note": f"{type(exc).__name__}: {exc}"})
            continue

        cmc = coin.set_index("date").price
        both = pd.concat([closes.rename("binance"), cmc.rename("cmc")], axis=1).dropna()
        if both.empty:
            results.append({"cmc_id": row.cmc_id, "contract": row.contract,
                            "month": month, "ratio": None, "ok": False,
                            "note": "no traded bars overlapping CMC data"})
            continue
        ratio = float((both.binance / (both.cmc * row.scale)).median())
        results.append({"cmc_id": row.cmc_id, "contract": row.contract,
                        "month": month, "ratio": ratio,
                        "ok": abs(ratio - 1) <= tolerance, "note": f"{len(both)} days"})

    return pd.DataFrame(results)
