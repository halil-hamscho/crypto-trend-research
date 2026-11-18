"""Run from the repo root: python -m portfolio.test_strategy_runner  (the real-data test takes ~1 min)

Every test uses synthetic data except the last, which asserts structure only and prints no return.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.basket import build_inputs, reference_basket, run_ledger
from portfolio.strategy_runner import (buffered_target, daily_returns_over, full_target, make_exposure_fn,
                                       h2_diagnostics, run_strategy, write_run_record)
from portfolio.test_basket import LEDGER_DAYS, assert_raises, notional_fixture, utc
from portfolio.trend_signal import basket_forecast, load_signal_config

CONFIG = load_signal_config()
CAP, BAND = CONFIG.leverage_cap, CONFIG.band_half_width


# ------------------------------------------------------------------ the two execution rules
def test_buffered_band_one_case_per_row():
    std = 1.0                                                   # band = 0.10
    assert buffered_target(0.55, 0.50, std, BAND, CAP) == 0.55                      # inside: hold
    assert buffered_target(0.60, 0.50, std, BAND, CAP) == 0.60                      # on the edge: hold
    assert np.isclose(buffered_target(0.90, 0.50, std, BAND, CAP), 0.60)            # above: trade to the upper edge
    assert np.isclose(buffered_target(0.10, 0.50, std, BAND, CAP), 0.40)            # below: trade to the lower edge
    assert np.isclose(buffered_target(1.00, -0.50, std, BAND, CAP), -0.40)          # sign flip: through zero to the edge
    assert np.isclose(buffered_target(-1.00, 0.50, std, BAND, CAP), 0.40)
    assert buffered_target(0.0, 0.05, std, BAND, CAP) == 0.0                        # flat and ideal near zero: stay flat
    assert buffered_target(0.0, 1.0, std, BAND, CAP) == 0.90                        # opening trades to the near edge, not to x*


def test_buffered_caps_again_after_the_band_logic():
    assert buffered_target(0.5, 2.0, 1.0, BAND, CAP) == 1.9                         # cap does not bind: edge
    assert buffered_target(2.05, 2.0, 1.0, BAND, CAP) == 2.0                        # held inside the band but above the cap
    assert buffered_target(0.0, 2.0, 5.0, BAND, CAP) == 1.5
    assert buffered_target(-3.0, -2.0, 20.0, BAND, CAP) == -2.0                     # a wide band cannot lift it past the cap


def test_full_target_is_the_ideal_exposure_clipped():
    assert full_target(0.7, CAP) == 0.7 and full_target(3.0, CAP) == CAP and full_target(-3.0, CAP) == -CAP


# ------------------------------------------------------------------ the hook the runner builds
def handcrafted_signal(x_star, standard=1.0):
    n = len(LEDGER_DAYS)
    x = list(x_star) + [x_star[-1]] * (n - len(x_star))
    std = list(standard) + [standard[-1]] * (n - len(standard)) if isinstance(standard, (list, tuple)) else standard
    return pd.DataFrame({"x_star": x, "standard": std, "forecast": 0.0}, index=LEDGER_DAYS)


def test_the_hook_reads_the_signal_on_the_decision_day_and_lagged_by_one_row_on_request():
    signal = handcrafted_signal([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    log, lagged = [], []
    fn0 = make_exposure_fn(signal, CONFIG, "full", log)
    fn1 = make_exposure_fn(signal, CONFIG, "full", lagged, signal_lag_days=1)
    assert fn0(LEDGER_DAYS[3], 0.0, 1e5) == 0.4 and log[-1]["signal_day"] == LEDGER_DAYS[3]
    assert fn1(LEDGER_DAYS[3], 0.0, 1e5) == 0.3 and lagged[-1]["signal_day"] == LEDGER_DAYS[2]
    assert_raises(lambda: fn1(LEDGER_DAYS[0], 0.0, 1e5), "no signal")


def test_the_hook_raises_when_the_signal_is_missing_and_for_an_unknown_policy():
    signal = handcrafted_signal([np.nan, 0.2])
    assert_raises(lambda: make_exposure_fn(signal, CONFIG, "full", [])(LEDGER_DAYS[0], 0.0, 1e5), "not available")
    assert_raises(lambda: make_exposure_fn(signal, CONFIG, "sometimes", []), "policy")


# ------------------------------------------------------------------ behaviour through the real ledger
def run_hand(x_star, policy, standard=1.0, capital_policy="fixed"):
    log = []
    fn = make_exposure_fn(handcrafted_signal(x_star, standard), CONFIG, policy, log)
    led = run_ledger(**notional_fixture(), mode="notional", exposure_fn=fn, capital_policy=capital_policy)
    return led, pd.DataFrame(log).set_index("day")


def test_full_policy_books_the_ideal_exposure_every_day():
    led, log = run_hand([0.5, 0.8, 1.2, 0.3, 0.9], "full")
    assert np.allclose(log.target, [0.5, 0.8, 1.2, 0.3, 0.9])
    assert (led.groupby("holding_period_start").target_exposure.first().values == log.target.values).all()


def test_buffered_policy_stays_within_the_band_of_the_ideal_and_holds_inside_it():
    led, log = run_hand([1.0] * 5, "buffered", standard=1.0)        # constant ideal exposure, A drifts +10%/day
    assert (np.abs(log.target - log.x_star) <= BAND * log.standard + 1e-12).all()
    inside = np.abs(log.held - log.x_star) <= BAND * log.standard
    assert (log.target[inside] == log.held[inside]).all() and inside.any()
    assert (led[led.event == "hold"].trade.abs() < 1e-6)[led[led.event == "hold"].holding_period_start
                                                          .isin([utc(str(d)) for d in log.index[inside]])].all()


def test_zero_ideal_exposure_full_closes_with_costs_and_buffered_may_keep_inside_the_band():
    full, _ = run_hand([1.0, 0.0, 0.0, 0.0, 0.0], "full")
    day2 = full[full.holding_period_start == utc("2024-01-02")]
    assert (day2.quantity_after == 0).all() and (day2.fee + day2.slippage).sum() > 0     # closed, closing costs paid
    assert utc("2024-01-03") not in set(full.holding_period_start)                      # flat and staying flat: no rows
    buffered, log = run_hand([1.0, 0.0, 0.0, 0.0, 0.0], "buffered", standard=[1.0, 20.0])   # opens, then a band of 2.0
    assert (log.target.iloc[1:] == log.held.iloc[1:]).all()                                  # swallows a zero ideal
    assert (buffered.quantity_after != 0).all() and utc("2024-01-02") in set(buffered.holding_period_start)


def test_a_book_that_is_flat_and_stays_flat_costs_nothing_and_earns_nothing():
    led, _ = run_hand([0.0] * 5, "full")
    assert len(led) == 0                                                                  # never traded


def test_reconstitution_targets_apply_under_the_buffer():
    fx = notional_fixture(recon=True)
    log = []
    fn = make_exposure_fn(handcrafted_signal([1.0] * 5), CONFIG, "buffered", log)
    led = run_ledger(**fx, mode="notional", exposure_fn=fn)
    recon = led[led.event == "reconstitution"].set_index("contract")
    assert recon.loc["B", "quantity_after"] == 0.0 and recon.loc["C", "quantity_after"] > 0.0


def test_flat_days_stay_in_the_daily_returns_and_the_base_continues():
    led, _ = run_hand([1.0, 0.0, 0.0, 1.0, 1.0], "full")
    days = LEDGER_DAYS[:5]
    fixed = daily_returns_over(led, days, 100_000.0, "fixed")
    assert len(fixed) == 5 and fixed.loc[days[2], ["net", "price", "funding", "fee"]].abs().sum() == 0.0
    assert (fixed.base == 100_000.0).all() and fixed.loc[days[2], "net_return"] == 0.0
    led_e, _ = run_hand([1.0, 0.0, 0.0, 1.0, 1.0], "full", capital_policy="equity")
    eq = daily_returns_over(led_e, days, 100_000.0, "equity")
    assert np.isclose(eq.base.iloc[2], eq.base.iloc[1] + eq.net.iloc[1]) and eq.base.iloc[3] == eq.base.iloc[2]


def test_a_base_that_disagrees_with_the_continued_base_raises():
    led, _ = run_hand([1.0] * 5, "full", capital_policy="equity")
    assert_raises(lambda: daily_returns_over(led, LEDGER_DAYS[:5], 100_000.0, "fixed"), "differs")


# ------------------------------------------------------------------ the whole runner on a synthetic basket
def synthetic_inputs(n=780, seed=3):
    days = [d.date() for d in pd.date_range("2020-02-09", periods=n)]
    rng = np.random.default_rng(seed)
    marks = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(m, 0.03, n))) for c, m in
                          (("A", 0.0008), ("B", 0.0002), ("C", 0.0004))}, index=days)
    ab = pd.DataFrame({"contract": ["A", "B"], "weight": [0.6, 0.4]})
    ac = pd.DataFrame({"contract": ["A", "C"], "weight": [0.7, 0.3]})
    members = {days[0]: ab, **{d: (ac if i % 2 else ab) for i, d in enumerate(d for d in days if d.day == 1)}}
    index, weights = reference_basket(members, marks)
    settlements = {c: pd.DataFrame({"stamp": pd.to_datetime([pd.Timestamp(d, tz="UTC") for d in days], utc=True),
                                    "rate": 1e-4, "mark": marks[c].values, "interval_hours": 24.0,
                                    "exchange_skipped": False}) for c in marks}
    return members, marks, settlements, weights, index


INPUTS = synthetic_inputs()
START, END = INPUTS[1].index[530], INPUTS[1].index[-3]


def go(policy="buffered", capital_policy="fixed", **kwargs):
    return run_strategy(policy, capital_policy, inputs=INPUTS, start=START, end=END, **kwargs)


def test_the_runner_covers_every_scored_day_for_both_policies_and_both_capital_policies():
    for policy in ("full", "buffered"):
        for capital_policy in ("fixed", "equity"):
            r = go(policy, capital_policy)
            days = [d for d in INPUTS[1].index if START <= d <= END]
            assert list(r.daily.index) == days and list(r.paired.index) == days and set(r.decisions.index) >= set(days)
            assert not r.paired.isna().any().any() and r.stats["identity_max_residual"] < 1e-9
            assert r.decisions.target.abs().max() <= CAP + 1e-12


def test_paired_series_are_the_strategy_and_the_same_policy_benchmark():
    r = go("buffered", "equity")
    assert np.allclose(r.paired.strategy_net, r.daily.net_return)
    assert (r.paired.benchmark_net != r.paired.strategy_net).any()
    assert np.allclose(r.paired.strategy_gross, r.paired.strategy_price + r.daily.funding_return)


def test_full_trades_to_the_ideal_every_day_and_buffered_is_measured_not_assumed():
    full, buffered = go("full"), go("buffered")
    assert np.allclose(full.decisions.target, full.decisions.x_star.clip(-CAP, CAP))
    assert full.stats["turnover_per_year"] > 0 and buffered.stats["turnover_per_year"] > 0   # the ordering is empirical


def test_nothing_after_the_decision_day_enters_a_decision_or_an_earlier_row():
    base = go()
    k = 600
    marks = INPUTS[1].copy()
    marks.iloc[k + 1:] = marks.iloc[k + 1:] * 2.5                      # change every price after day k
    index, weights = reference_basket(INPUTS[0], marks)
    altered = (INPUTS[0], marks, {c: s.assign(mark=marks[c].values) for c, s in INPUTS[2].items()}, weights, index)
    other = run_strategy("buffered", "fixed", inputs=altered, start=START, end=END)
    day_k = INPUTS[1].index[k]
    before = [d for d in base.decisions.index if d <= day_k]
    pd.testing.assert_frame_equal(base.decisions.loc[before], other.decisions.loc[before])
    early = base.ledger[base.ledger.holding_period_start < utc(str(day_k))]
    pd.testing.assert_frame_equal(early.reset_index(drop=True),
                                  other.ledger[other.ledger.holding_period_start < utc(str(day_k))].reset_index(drop=True))


def test_the_one_day_lag_uses_yesterdays_signal():
    r0, r1 = go(), go(signal_lag_days=1)
    shifted = pd.Series(r1.decisions.signal_day.values, index=r1.decisions.index)
    assert (shifted != pd.Series(r1.decisions.index, index=r1.decisions.index)).all()
    assert (r0.decisions.signal_day == r0.decisions.index).all()
    assert r1.record["parameters"]["signal_lag_days"] == 1


def test_stats_report_the_registered_diagnostics():
    s = go().stats
    for key in ("turnover_per_year", "trade_count", "mean_abs_target_exposure", "mean_abs_forecast",
                "forecast_cap_hit_rate", "x_star_cap_hit_rate", "leverage_cap_bind_rate", "vol_floor_bind_rate",
                "funding_pnl_long_days", "funding_pnl_short_days", "flat_days"):
        assert np.isfinite(s[key]), key
    assert 0 <= s["leverage_cap_bind_rate"] <= 1 and s["trade_count"] > 0


def test_a_run_is_deterministic_and_its_record_reproduces():
    a, b = go(), go()
    assert a.record["result_fingerprint"] == b.record["result_fingerprint"]
    assert a.record["fingerprints"] == b.record["fingerprints"] and a.record["parameters"] == b.record["parameters"]
    with tempfile.TemporaryDirectory() as folder:
        saved = json.loads(write_run_record(a, Path(folder)).read_text())
    assert saved["parameters"]["policy"] == "buffered" and saved["parameters"]["config"]["fdm"] == 1.15
    assert saved["fingerprints"]["config_sha256"] == CONFIG.config_hash and "commit" in saved["git"]
    assert go("full").record["result_fingerprint"] != a.record["result_fingerprint"]


def test_unknown_policies_raise():
    assert_raises(lambda: run_strategy("sometimes", inputs=INPUTS), "policy")
    assert_raises(lambda: run_strategy("full", "compounded", inputs=INPUTS), "capital_policy")


# ------------------------------------------------------------------ the BTC-only comparator (H2)
def single_asset_inputs(asset="A"):
    """The comparator on the synthetic data: one member at weight 1 on the same dates."""
    marks = INPUTS[1][[asset]]
    members = {d: pd.DataFrame({"contract": [asset], "weight": [1.0]}) for d in INPUTS[0]}
    index, weights = reference_basket(members, marks)
    return members, marks, {asset: INPUTS[2][asset]}, weights, index


def test_a_single_member_index_is_the_assets_own_price_and_the_weight_is_one():
    members, marks, settlements, weights, index = single_asset_inputs()
    assert np.allclose(index.pct_change().dropna().values, marks["A"].pct_change().dropna().values)
    assert (weights["A"] == 1.0).all() and list(settlements) == ["A"]


def test_the_comparator_signal_is_the_signal_on_the_assets_own_price():
    members, marks, settlements, weights, index = single_asset_inputs()
    own = basket_forecast(pd.Series(marks["A"].values, index=marks.index), CONFIG)       # scale-invariant signal
    chained = basket_forecast(index, CONFIG)
    pd.testing.assert_series_equal(own.x_star, chained.x_star, check_names=False, rtol=1e-9, atol=1e-12)
    pd.testing.assert_series_equal(own.forecast, chained.forecast, check_names=False, rtol=1e-9, atol=1e-12)


def btc_like(policy="buffered", capital_policy="fixed"):
    return run_strategy(policy, capital_policy, inputs=single_asset_inputs(), start=START, end=END, instrument="btc")


def test_the_comparator_holds_only_its_one_contract_with_the_same_machinery():
    r = btc_like()
    assert set(r.ledger.contract) == {"A"} and r.stats["identity_max_residual"] < 1e-9
    assert list(r.daily.index) == [d for d in INPUTS[1].index if START <= d <= END]
    assert r.decisions.target.abs().max() <= CAP + 1e-12 and r.record["parameters"]["instrument"] == "btc"


def test_h2_diagnostics_pair_like_with_like_and_the_vol_matched_gap_is_by_hand():
    basket, comp = go("buffered", "fixed"), btc_like("buffered", "fixed")
    out = h2_diagnostics(basket, comp)
    assert np.allclose(out["gap_daily"]["net"], basket.daily.net_return - comp.daily.net_return)
    v_b, v_t = basket.daily.net_return.std() * np.sqrt(365), comp.daily.net_return.std() * np.sqrt(365)
    by_hand = (basket.daily.net_return - comp.daily.net_return * v_b / v_t).mean() * 365
    assert np.isclose(out["ex_post_vol_matched_gap_annual"], by_hand)
    assert np.isclose(out["gap_mean_annual"]["net"], out["gap_daily"]["net"].mean() * 365)
    for key in ("realised_vol", "mean_abs_exposure", "turnover_per_year"):
        assert set(out[key]) == {"basket", "btc"}
    assert_raises(lambda: h2_diagnostics(go("full"), comp), "like with like")


def test_unknown_instruments_raise():
    assert_raises(lambda: build_inputs(instrument="eth"), "instrument")
    assert_raises(lambda: run_strategy("full", instrument="eth", inputs=INPUTS), "instrument")


def test_real_btc_inputs_are_btc_alone_and_the_comparator_runs_end_to_end():
    members, marks, settlements, weights, index = build_inputs(instrument="btc")
    assert list(marks.columns) == ["BTCUSDT"] and (weights["BTCUSDT"] == 1.0).all()
    assert np.allclose(index.pct_change().dropna().values, marks["BTCUSDT"].pct_change().dropna().values)
    r = run_strategy("buffered", "fixed", inputs=(members, marks, settlements, weights, index))
    assert r.stats["days"] == 1883 and set(r.ledger.contract) == {"BTCUSDT"}
    assert r.stats["identity_max_residual"] < 1e-9 and not r.paired.isna().any().any()


# ------------------------------------------------------------------ the real basket: structure only, no return is printed
def test_real_basket_runs_end_to_end_with_the_registered_structure():
    r = run_strategy("buffered", "fixed")
    assert r.stats["days"] == 1883 and len(r.daily) == 1883 and len(r.paired) == 1883
    assert r.stats["identity_max_residual"] < 1e-9 and r.decisions.target.abs().max() <= CAP + 1e-12
    assert not r.paired.isna().any().any() and r.record["git"]["commit"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
