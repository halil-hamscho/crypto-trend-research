"""Run from the repo root: python -m portfolio.test_per_coin_attribution

The market / coin-specific split (protocol Part 7) and the two-sided validity check (Part 9).
Synthetic data only; no real-data return is computed.
"""
from datetime import timedelta

import numpy as np

from portfolio.per_coin_attribution import (TWO_SIDED, attribute, split, two_sided_days, validity)


def close(a, b, tol=1e-15):
    return abs(a - b) <= tol


# ------------------------------------------------------------------ the identity
def test_hand_computed_split():
    """w = (0.10, -0.05, 0), r = (2%, 1%, -3%): W = 0.05, mean r = 0, so all of it is coin-specific."""
    market, coin, ineligible = split(np.array([0.10, -0.05, 0.0]), np.array([0.02, 0.01, -0.03]),
                                     np.array([True, True, True]))
    assert close(market, 0.0) and close(coin, 0.0015) and ineligible == 0.0
    assert close(market + coin + ineligible, 0.10 * 0.02 - 0.05 * 0.01)


def test_equal_weights_on_one_side_are_pure_market():
    w = np.full(4, 0.05)
    market, coin, _ = split(w, np.array([0.03, -0.01, 0.02, 0.0]), np.ones(4, dtype=bool))
    assert close(coin, 0.0) and close(market, 0.05 * 0.04)                  # W x mean r


def test_a_dollar_neutral_book_has_no_market_term():
    market, coin, _ = split(np.array([0.1, -0.1]), np.array([0.05, -0.02]), np.ones(2, dtype=bool))
    assert close(market, 0.0) and close(coin, 0.1 * 0.05 + 0.1 * 0.02)


def test_held_but_ineligible_coins_form_the_third_term():
    w, r = np.array([0.1, 0.2, -0.05]), np.array([0.01, 0.02, 0.03])
    eligible = np.array([True, True, False])
    market, coin, ineligible = split(w, r, eligible)
    assert close(ineligible, -0.05 * 0.03)
    assert close(market + coin + ineligible, float(w @ r))


def test_nothing_eligible_puts_everything_in_the_third_term():
    market, coin, ineligible = split(np.array([0.1]), np.array([0.02]), np.array([False]))
    assert market == 0.0 and coin == 0.0 and close(ineligible, 0.002)


def test_the_identity_holds_on_random_books():
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = rng.integers(1, 12)
        w, r = rng.normal(0, 0.1, n), rng.normal(0, 0.05, n)
        eligible = rng.random(n) < 0.8
        assert abs(sum(split(w, r, eligible)) - float(w @ r)) < 1e-15


# ------------------------------------------------------------------ the two-sided check
def test_two_sided_days_need_both_sides_above_ten_percent_of_gross():
    longs = np.array([0.2, 0.1, 0.0])
    assert two_sided_days([longs, np.array([0.2, -0.05, 0.0]), np.array([0.2, -0.01, 0.0]), np.zeros(3)]) \
        == [False, True, False, False]                                     # 0.05/0.25 = 20% ok; 0.01/0.21 < 10%


def test_validity_flags_a_book_that_is_rarely_two_sided():
    ok = validity([True] * 3 + [False] * 7)
    assert close(ok["two_sided_share"], 0.3) and not ok["collapsed_to_market_timing"]
    bad = validity([True] * 1 + [False] * 9)
    assert bad["collapsed_to_market_timing"] and TWO_SIDED["min_share_of_days"] == 0.20


# ------------------------------------------------------------------ on a real ledger (synthetic data)
def ledger_run(variant="own"):
    """A synthetic world run end to end: the real model through the real ledger."""
    import portfolio.test_per_coin_model as m
    from portfolio.per_coin_ledger import run_per_coin_ledger
    from portfolio.per_coin_model import build_decision_tables, make_target_fn
    bars, counts = m.world()
    elig = m.calendar_eligibility(bars, counts)
    tables = build_decision_tables(bars, elig, m.CONFIG, variant=variant)
    inputs = m.build_ledger_inputs(m.COINS, m.START, m.END, elig, bars=bars, mark_counts=counts,
                                   settlements=m.settlements(bars))
    return run_per_coin_ledger(make_target_fn(tables, lag=1), inputs), inputs


def test_the_split_reproduces_every_days_price_and_funding_pnl():
    ledger, inputs = ledger_run()
    daily = attribute(ledger, inputs)                                      # raises if any day fails
    cap = inputs.capital
    assert np.allclose(daily.price_market + daily.price_coin + daily.price_ineligible,
                       ledger.daily.price_pnl / cap, rtol=0, atol=1e-15)
    assert np.allclose(daily.funding_market + daily.funding_coin + daily.funding_ineligible,
                       ledger.daily.funding_pnl / cap, rtol=0, atol=1e-15)
    assert np.allclose(daily.costs, (ledger.daily.fee + ledger.daily.slippage) / cap, rtol=0, atol=1e-18)
    assert (daily.price_coin != 0).any() and (daily.price_market != 0).any()
    assert list(daily.index) == list(ledger.daily.index)


def test_funding_per_unit_is_independent_of_the_position():
    """Flat eligible coins still get a funding return, so the funding split averages over them too."""
    from portfolio.per_coin_attribution import funding_per_unit
    ledger, inputs = ledger_run()
    rows = ledger.rows
    f = funding_per_unit(rows, inputs.settlements)
    held = rows.quantity_held != 0.0
    notional = rows.quantity_held * rows.fill_price
    assert np.allclose(f[held], (rows.funding_pnl / notional)[held], rtol=1e-12, atol=0)
    assert np.isfinite(f[(rows.event == "flat").to_numpy()]).all()


def test_the_no_signal_book_is_almost_all_market():
    """K1 holds every eligible coin long at equal risk: the coin-specific term comes only from unequal
    weights (different vols), so it should be small next to the market term."""
    ledger, inputs = ledger_run(variant="k1")
    daily = attribute(ledger, inputs)
    assert daily.price_coin.abs().sum() < 0.5 * daily.price_market.abs().sum()
    check = validity(daily.two_sided)                                      # long-only: never two-sided
    assert check["two_sided_share"] == 0.0 and check["collapsed_to_market_timing"]


def test_excluded_rows_are_counted_not_silently_dropped():
    import portfolio.test_per_coin_ledger as L
    from portfolio.per_coin_ledger import build_ledger_inputs, run_per_coin_ledger
    halted = L.START + timedelta(days=4)
    bars, marks, counts, eligible = L.synthetic(halted=(("CCC", halted),))
    inputs = build_ledger_inputs(L.CONTRACTS, L.START, L.END, eligible, bars=bars, marks=marks, mark_counts=counts,
                                 settlements=L.settlements_for(L.CONTRACTS), events=L.halt_event("CCC", halted, halted))
    ledger = run_per_coin_ledger(L.scripted({L.START: {"AAA": 1_000.0, "BBB": -500.0}}), inputs)
    daily = attribute(ledger, inputs)
    assert daily.n_excluded.sum() == 2 and (daily.loc[halted, "n_eligible"] == 2)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
