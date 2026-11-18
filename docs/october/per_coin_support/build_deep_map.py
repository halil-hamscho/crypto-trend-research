"""Build and validate the top-100 cmc_id -> contract map for the per-coin top-25 and top-50
sensitivities (per-coin protocol Amendment 2). Prices only; no strategy return is computed.

1. Rebuilds the depth-20 map to a scratch file and compares it with the committed sealed map,
   so the builder is shown not to have drifted.
2. Builds the depth-100 map into data/universe/cmc_contract_map_top100.csv, resolving contracts
   claimed by more than one cmc_id (token migrations, duplicate listings) by price: each claim is
   limited to the days the contract's close is within 5% of that coin's CMC price x scale; where
   two claims still overlap, the coin with more matching days keeps the day; a claim that never
   matches is dropped. Every row gets a day-level claim range (claim_from, claim_to).
3. Validates every row: one month of Binance closes (from the local panel, traded bars only)
   against CMC price x scale, median ratio within 5%.
4. Lists pool coins with no contract.

Run from the repo root:  python docs/october/per_coin_support/build_deep_map.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.universe.coinmarketcap.new_fetcher import contract_map, normalize, panel  # noqa: E402

OUT = Path(__file__).resolve().parent


def local_monthly_closes(contract: str, month: str) -> pd.Series:
    bars = panel.load_ohlcv(contract)
    bars = bars[(bars.volume > 0) & (bars.timestamp.dt.strftime("%Y-%m") == month)]
    return pd.Series(bars.close.astype(float).to_numpy(), index=bars.timestamp.dt.date.to_numpy())


def price_match_days(contract: str, cmc_id: int, scale: int, universe: pd.DataFrame) -> set:
    bars = panel.load_ohlcv(contract)
    bars = bars[bars.volume > 0]
    close = pd.Series(bars.close.astype(float).to_numpy(), index=bars.timestamp.dt.date.to_numpy())
    cmc = universe[universe.cmc_id == cmc_id].set_index("date").price
    both = pd.concat([close.rename("b"), cmc.rename("c")], axis=1).dropna()
    ratio = both.b / (both.c * scale)
    return set(both.index[(ratio - 1).abs() <= 0.05])


def resolve_claims(mapping: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Day-level claim ranges. Unshared rows claim the contract's whole traded life (open ends);
    shared contracts are split by the price rule in the module docstring."""
    out, log = [], []
    for contract, rows in mapping.groupby("contract"):
        if rows.cmc_id.nunique() == 1:
            for r in rows.itertuples(index=False):
                out.append({**r._asdict(), "claim_from": None, "claim_to": None})
            continue
        matches = {r.cmc_id: price_match_days(contract, r.cmc_id, r.scale, universe) for r in rows.itertuples()}
        by_id = {r.cmc_id: r for r in rows.itertuples(index=False)}
        taken = set()
        for cmc_id in sorted(matches, key=lambda k: -len(matches[k])):        # most matching days first
            if not matches[cmc_id]:
                log.append(f"{contract}: dropped cmc {cmc_id} (no price-matching day)")
                continue
            first, last = min(matches[cmc_id]), max(matches[cmc_id])
            free = [d.date() for d in pd.date_range(first, last, freq="D") if d.date() not in taken]
            if not free:
                log.append(f"{contract}: dropped cmc {cmc_id} (its matching range {first} -> {last} "
                           f"lies inside a coin with more matching days)")
                continue
            segments, start = [], free[0]
            for a, b in zip(free, free[1:] + [None]):
                if b is None or (b - a).days > 1:
                    segments.append((start, a)); start = b
            for a, b in segments:
                out.append({**by_id[cmc_id]._asdict(), "claim_from": a, "claim_to": b, "source": "resolved"})
                log.append(f"{contract}: cmc {cmc_id} claims {a} -> {b} "
                           f"(price matches on {len(matches[cmc_id])} days, {first} -> {last})")
            taken |= {d.date() for d in pd.date_range(first, last, freq="D")}
    resolved = pd.DataFrame(out).sort_values(["cmc_id", "valid_from"]).reset_index(drop=True)
    return resolved, log


def main() -> None:
    universe = normalize.load()

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = contract_map.build(universe, depth=contract_map.POOL_DEPTH, table=Path(tmp) / "map20.csv")
    sealed = contract_map.load()
    key = ["cmc_id", "contract", "valid_from", "scale"]
    merged = sealed[key].merge(rebuilt[key], how="outer", indicator=True)
    drift = merged[merged._merge != "both"]
    print(f"depth-20 rebuild vs committed sealed map: {len(sealed)} sealed rows, {len(rebuilt)} rebuilt, "
          f"{len(drift)} differing on {key}")
    if len(drift):
        print(drift.to_string(index=False))

    raw = contract_map.build(universe, depth=contract_map.DEEP_POOL_DEPTH, table=contract_map.DEEP_MAP_TABLE,
                             allow_shared=True)
    deep, log = resolve_claims(raw, universe)
    print("\nshared contracts resolved by price:")
    print("\n".join("  " + line for line in log))
    overlap = []
    for contract, rows in deep[deep.claim_from.notna()].groupby("contract"):
        spans = sorted(zip(rows.claim_from, rows.claim_to))
        overlap += [contract for a, b in zip(spans, spans[1:]) if b[0] <= a[1]]
    if overlap:
        raise ValueError(f"claim ranges still overlap: {overlap}")
    deep.to_csv(contract_map.DEEP_MAP_TABLE, index=False)
    print(f"\ndepth-100 map: {len(deep)} rows, {deep.cmc_id.nunique()} coins, {deep.contract.nunique()} contracts")
    missing_sealed = set(sealed.contract) - set(deep.contract)
    print("sealed contracts absent from the deep map:", sorted(missing_sealed) or "none")

    checked = contract_map.validate_prices(deep, universe, depth=contract_map.DEEP_POOL_DEPTH,
                                           closes_for=local_monthly_closes)
    checked.to_csv(OUT / "deep_map_validation.csv", index=False)
    bad = checked[~checked.ok]
    print(f"price validation: {int(checked.ok.sum())} of {len(checked)} rows pass (|median ratio - 1| <= 5%)")
    if len(bad):
        print(bad.to_string(index=False))

    gaps = contract_map.unmapped(deep, universe, depth=contract_map.DEEP_POOL_DEPTH)
    gaps.to_csv(OUT / "deep_map_unmapped.csv", index=False)
    print(f"\npool coins (ever top-100) with no Binance USDT-M contract: {len(gaps)}")


if __name__ == "__main__":
    main()
