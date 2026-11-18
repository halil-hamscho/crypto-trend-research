"""Run from the repo root: python -m portfolio.test_per_coin_tiers

Trial 62's size tiers (docs/october/trial_62_size_tier_budget_design_2026-10-30.md §4): the tier table, the
weights in dynamic sizing, and the target_fn through the real ledger, on synthetic data only. No real-data
return is computed.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import portfolio.test_per_coin_model as M
from portfolio.per_coin_ledger import run_per_coin_ledger
from portfolio.per_coin_model import make_target_fn
from portfolio.per_coin_sizing import SIZING, dynamic_positions, risk_shares
from portfolio.per_coin_tiers import tier_table

X = M.DAYS[320]


def cmc(caps: dict, days=M.DAYS) -> pd.DataFrame:
    """The CMC table: {cmc_id: cap} every day, or {cmc_id: callable(day) -> cap}."""
    rows = [{"date": d, "cmc_id": i, "market_cap": float(v(d) if callable(v) else v)} for d in days
            for i, v in caps.items()]
    return pd.DataFrame(rows)


def deep_map(rows) -> pd.DataFrame:
    """(cmc_id, contract[, claim_from, claim_to])."""
    return pd.DataFrame([{"cmc_id": r[0], "contract": r[1], "claim_from": r[2] if len(r) > 2 else np.nan,
                          "claim_to": r[3] if len(r) > 2 else np.nan} for r in rows])


MAP = deep_map([(1, "AAA"), (2, "BBB"), (3, "CCC")])                 # BTCUSDT unmapped: tier 3
CAPS = {1: 400e9, 2: 300e9, 3: 200e9}


def eligible_all(days=M.DAYS[1:]) -> pd.DataFrame:
    """Every coin eligible from the second day (the first has no CMC day before it)."""
    return pd.DataFrame(True, index=list(days), columns=list(M.COINS))


def assert_raises(call, *fragments):
    try:
        call()
    except ValueError as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return
    raise AssertionError("expected ValueError")


# ------------------------------------------------------------------ the tier table
def test_ranks_tiers_and_equal_budgets():
    t = tier_table(eligible_all(), cmc(CAPS), MAP, ranks=(1, 2))
    assert t.tier.loc[X].to_dict() == {"AAA": 1, "BBB": 2, "CCC": 3, "BTCUSDT": 3}
    w = t.weight.loc[X]
    assert w.AAA == w.BBB == 1.0 / 3 and w.CCC == w.BTCUSDT == (1.0 / 3) / 2
    assert abs(w.sum() - 1.0) < 1e-15 and np.isnan(t.cap.at[X, "BTCUSDT"]) and t.cap.at[X, "AAA"] == 400e9
    assert X not in t.empty_tier_days


def test_an_empty_tier_is_shared_equally_and_logged():
    t = tier_table(eligible_all(), cmc(CAPS), MAP)                     # ranks 10 / 50: tier 2 is empty
    assert t.tier.loc[X].to_dict() == {"AAA": 1, "BBB": 1, "CCC": 1, "BTCUSDT": 3}
    assert t.weight.at[X, "AAA"] == 0.5 / 3 and t.weight.at[X, "BTCUSDT"] == 0.5
    assert t.empty_tier_days[X] == [2]


def test_one_tier_gives_exactly_one_over_n():
    full = deep_map([(1, "AAA"), (2, "BBB"), (3, "CCC"), (4, "BTCUSDT")])
    t = tier_table(eligible_all(), cmc({**CAPS, 4: 900e9}), full)
    assert (t.weight.loc[X] == 1.0 / 4).all() and (t.tier.loc[X] == 1).all()


def test_only_eligible_coins_are_ranked_and_weighted():
    elig = eligible_all()
    elig.loc[X, "AAA"] = False
    t = tier_table(elig, cmc(CAPS), MAP, ranks=(1, 2))
    assert t.tier.at[X, "AAA"] == 0 and t.weight.at[X, "AAA"] == 0.0
    assert t.tier.loc[X].to_dict() == {"AAA": 0, "BBB": 1, "CCC": 2, "BTCUSDT": 3}   # BBB moves up


def test_the_rank_reads_cmc_of_the_day_before_only():
    base = tier_table(eligible_all(), cmc(CAPS), MAP, ranks=(1, 2))
    flip_on_x = {1: lambda d: 100e9 if d >= X else 400e9, 2: 300e9, 3: 200e9}
    later = tier_table(eligible_all(), cmc(flip_on_x), MAP, ranks=(1, 2))
    assert later.tier.loc[:X].equals(base.tier.loc[:X])                   # CMC day X is not used at X
    assert later.tier.at[X + timedelta(days=1), "AAA"] == 3               # ...it is at X + 1
    flip_before = {1: lambda d: 100e9 if d >= X - timedelta(days=1) else 400e9, 2: 300e9, 3: 200e9}
    assert tier_table(eligible_all(), cmc(flip_before), MAP, ranks=(1, 2)).tier.at[X, "AAA"] == 3
    gap = cmc(CAPS)
    gap = gap[gap.date != X - timedelta(days=1)]
    assert_raises(lambda: tier_table(eligible_all(), gap, MAP), "no CMC snapshot")


def test_market_cap_not_cmc_rank_and_ties_by_cmc_id():
    table = cmc({1: 300e9, 2: 300e9, 3: 400e9})
    table["rank"] = table.cmc_id                                          # CMC's own field, wrong on purpose
    t = tier_table(eligible_all(), table, MAP, ranks=(1, 2))
    assert t.tier.loc[X, ["CCC", "AAA", "BBB"]].tolist() == [1, 2, 3]


def test_claim_ranges_decide_the_coin_and_a_double_claim_stops():
    switch = X - timedelta(days=1)
    m = deep_map([(1, "AAA", str(M.D0), str(switch - timedelta(days=1))), (9, "AAA", str(switch), "2099-12-31"),
                  (2, "BBB"), (3, "CCC")])
    t = tier_table(eligible_all(), cmc({**CAPS, 9: 1e9}), m, ranks=(1, 2))
    assert t.cap.at[X - timedelta(days=1), "AAA"] == 400e9 and t.cap.at[X, "AAA"] == 1e9   # judged on X - 1
    overlap = deep_map([(1, "AAA"), (9, "AAA"), (2, "BBB")])
    assert_raises(lambda: tier_table(eligible_all(), cmc({**CAPS, 9: 1e9}), overlap), "claimed by coins")


# ------------------------------------------------------------------ the sizing
def _book(n, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-20, 20, n), rng.uniform(0.4, 1.5, n), rng.normal(0, 0.02, n) * (rng.random(n) < 0.6))


def test_weights_of_one_over_n_reproduce_stage_c_bit_for_bit():
    for seed in range(20):
        f, sigma, held = _book(37, seed)
        for rho in (0.35, np.nan):
            a, b = dynamic_positions(f, sigma, rho, held), dynamic_positions(f, sigma, rho, held,
                                                                             weights=np.full(37, 1.0 / 37))
            assert all(np.array_equal(x, y) for x, y in zip(a, b)), seed


def test_at_full_forecasts_each_tier_takes_its_budget_of_standalone_risk():
    rng = np.random.default_rng(4)
    groups = np.repeat([1, 2, 3], [2, 5, 13])
    w = np.array([(1.0 / 3) / {1: 2, 2: 5, 3: 13}[k] for k in groups])
    f = 10.0 * rng.choice([-1.0, 1.0], len(groups))
    sigma = rng.uniform(0.4, 1.6, len(groups))
    _, ideal, m, _ = dynamic_positions(f, sigma, 0.4, np.zeros(len(groups)), weights=w)
    shares = risk_shares(ideal, sigma, 0.4, groups, (1, 2, 3))
    assert all(abs(shares[k]["standalone"] - 1.0 / 3) < 1e-12 for k in (1, 2, 3))
    assert abs(sum(s["variance"] for s in shares.values()) - 1.0) < 1e-12
    assert m > 0 and abs(float(np.sqrt(((1 - 0.4) * (ideal * sigma) @ (ideal * sigma)
                                        + 0.4 * ((ideal * sigma).sum()) ** 2)) - SIZING.target_vol)) < 1e-12


def test_weights_must_be_positive_and_sum_to_one():
    f, sigma, held = _book(4)
    for bad in ([0.5, 0.5, 0.0, 0.0], [0.3, 0.3, 0.3, 0.3], [0.25, 0.25, 0.25], [np.nan, 0.5, 0.25, 0.25]):
        assert_raises(lambda: dynamic_positions(f, sigma, 0.3, held, weights=bad), "instrument weights")


# ------------------------------------------------------------------ the target_fn through the ledger
def _run(tables, inputs, tiering=None, diagnostics=None):
    return run_per_coin_ledger(make_target_fn(tables, 1, "dynamic", diagnostics=diagnostics, tiering=tiering), inputs)


def test_one_tier_through_the_ledger_is_the_stage_c_book():
    bars, counts, elig, tables, inputs = M.setup()
    full = deep_map([(1, "AAA"), (2, "BBB"), (3, "CCC"), (4, "BTCUSDT")])
    one = tier_table(elig, cmc({**CAPS, 4: 900e9}), full)
    a, b = _run(tables, inputs), _run(tables, inputs, one)
    assert a.rows.equals(b.rows) and a.fingerprint == b.fingerprint
    three = tier_table(elig, cmc(CAPS), MAP, ranks=(1, 2))
    assert not _run(tables, inputs, three).rows.equals(a.rows)


def test_tier_diagnostics_and_the_twin_long_only_at_target():
    bars, counts, elig, tables, inputs = M.setup(variant="k1")
    three = tier_table(elig, cmc(CAPS), MAP, ranks=(1, 2))
    record = []
    ledger = _run(tables, inputs, three, record)
    assert (ledger.rows.quantity_held >= 0).all() and (ledger.rows.quantity_held > 0).any()
    assert all(abs(r["v_ideal"] - SIZING.target_vol) < 1e-12 for r in record) and len(record) == len(inputs.days) - 1
    for r in record:
        assert (r["n_tier1"], r["n_tier2"], r["n_tier3"]) == (1, 1, 2)
        assert abs(r["risk_alone_tier1"] + r["risk_alone_tier2"] + r["risk_alone_tier3"] - 1.0) < 1e-12


def test_size_tiers_need_dynamic_sizing():
    bars, counts, elig, tables, inputs = M.setup()
    three = tier_table(elig, cmc(CAPS), MAP)
    assert_raises(lambda: make_target_fn(tables, 1, "primary", tiering=three), "dynamic sizing only")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
