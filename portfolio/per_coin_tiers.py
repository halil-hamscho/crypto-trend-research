"""Trial 62: a fixed risk budget by coin size, in place of equal risk per coin.

Design: docs/october/trial_62_size_tier_budget_design_2026-10-30.md (approved 2026-11-05, §2 and §5). On
decision day X every contract eligible at X is ranked by its coin's CoinMarketCap market cap on day X - 1
(Constraint 1: market_cap, never CMC's rank field; Constraint 4: CMC's day D is usable on D + 1), among the
contracts eligible at X. Ranks 1-10 are tier 1, 11-50 tier 2, the rest tier 3. A contract with no mapped
coin that day, or whose coin is not in that day's CMC table, is tier 3. Each non-empty tier gets an equal
share of the risk budget (an empty tier's share goes equally to the others) split equally among its coins:

    w_i = (1 / number of non-empty tiers) / (coins in i's tier)

With one non-empty tier, w_i = 1/N exactly and the book is stage C's. The coin of a contract comes from the
deep map (`cmc_contract_map_top100.csv`): a row with a claim range applies only inside it, judged on the
CMC day X - 1; a contract claimed by two coins on one day stops the run. Ties in market cap go to the lower
cmc_id, then the contract name. Pure functions over frames; the loader at the bottom reads disk.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

TIER_RANKS = (10, 50)          # last rank of tier 1 and of tier 2
TIERS = (1, 2, 3)
CMC_LAG_DAYS = 1               # CMC's day D is usable on D + 1 (Constraint 4)


@dataclass(frozen=True)
class Tiering:
    tier: pd.DataFrame          # [decision day x contract]: 1, 2 or 3 where eligible, 0 otherwise
    weight: pd.DataFrame        # [decision day x contract]: w_i where eligible, 0 otherwise
    cap: pd.DataFrame           # [decision day x contract]: the market cap ranked on (NaN: none, so tier 3)
    empty_tier_days: dict       # decision day -> the tiers that were empty (their budget redistributed)


def claimed_coins(mapping: pd.DataFrame) -> dict:
    """contract -> [(cmc_id, (claim_from, claim_to) or None)] from the deep map."""
    out = {}
    claims = "claim_from" in mapping.columns
    for row in mapping.itertuples():
        span = None
        if claims and pd.notna(row.claim_from):
            span = (date.fromisoformat(str(row.claim_from)), date.fromisoformat(str(row.claim_to)))
        entry = (int(row.cmc_id), span)
        if entry not in out.setdefault(row.contract, []):
            out[row.contract].append(entry)
    return out


def coin_on(entries, day: date) -> int | None:
    """The one coin a contract belongs to on `day`, None if no row applies; two coins stop the run."""
    hits = sorted({cmc for cmc, span in entries if span is None or span[0] <= day <= span[1]})
    if len(hits) > 1:
        raise ValueError(f"a contract is claimed by coins {hits} on {day}")
    return hits[0] if hits else None


def caps_by_day(universe: pd.DataFrame) -> dict:
    """CMC date -> Series(cmc_id -> market_cap)."""
    if universe.duplicated(["date", "cmc_id"]).any():
        raise ValueError("the CMC table has a coin twice on one date")
    return {d: g.set_index("cmc_id").market_cap.astype(float) for d, g in universe.groupby("date")}


def tier_table(eligible: pd.DataFrame, universe: pd.DataFrame, mapping: pd.DataFrame,
               ranks: tuple = TIER_RANKS) -> Tiering:
    """The tiers and weights on every decision day of `eligible` ([day x contract], decision-time)."""
    coins, caps = claimed_coins(mapping), caps_by_day(universe)
    days, contracts = list(eligible.index), list(eligible.columns)
    tier = pd.DataFrame(0, index=days, columns=contracts)
    weight = pd.DataFrame(0.0, index=days, columns=contracts)
    cap = pd.DataFrame(np.nan, index=days, columns=contracts)
    empty = {}
    for x in days:
        names = [c for c in contracts if bool(eligible.at[x, c])]
        if not names:
            continue
        seen = x - timedelta(days=CMC_LAG_DAYS)
        if seen not in caps:
            raise ValueError(f"no CMC snapshot for {seen}, the day before decision day {x}")
        today = caps[seen]
        ranked, unranked = [], []
        for c in names:
            coin = coin_on(coins.get(c, ()), seen)
            value = float(today.get(coin, np.nan)) if coin is not None else np.nan
            if np.isfinite(value) and value > 0.0:
                ranked.append((-value, coin, c))
                cap.at[x, c] = value
            else:
                unranked.append(c)
        level = {c: (1 if r <= ranks[0] else 2 if r <= ranks[1] else 3)
                 for r, (_, _, c) in enumerate(sorted(ranked), start=1)}
        level.update({c: 3 for c in unranked})
        counts = {k: sum(1 for v in level.values() if v == k) for k in TIERS}
        live = [k for k in TIERS if counts[k]]
        if len(live) < len(TIERS):
            empty[x] = [k for k in TIERS if not counts[k]]
        budget = 1.0 / len(live)
        for c, k in level.items():
            tier.at[x, c] = k
            weight.at[x, c] = budget / counts[k]
    return Tiering(tier=tier, weight=weight, cap=cap, empty_tier_days=empty)


def load_tier_inputs() -> tuple:
    """(the CMC table, the deep map) from disk, read through pandas so the checked loader sees them."""
    from data.universe.coinmarketcap.new_fetcher import contract_map, normalize
    return normalize.load(), contract_map.load(contract_map.DEEP_MAP_TABLE)


def tier_files() -> list:
    """The two files a tier book reads beyond stage B's, as relative paths."""
    from data.universe.coinmarketcap.new_fetcher import contract_map, normalize
    from portfolio.per_coin_inputs import relative
    return [relative(normalize.UNIVERSE_TABLE), relative(contract_map.DEEP_MAP_TABLE)]
