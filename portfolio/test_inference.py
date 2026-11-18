"""Run from the repo root: python -m portfolio.test_inference

Synthetic data with a known truth throughout. No real strategy return is used.
"""
import copy

import numpy as np
import pandas as pd
import yaml

from portfolio.inference import (CHUNK, check_layers, drag_gap, drag_table, evaluate_drags, REGISTERED_PRIMARY, check_experiment_identity, FAILS, FRAGILE, INCONCLUSIVE, SUPPORTED, alpha_attribution, alpha_beta,
                                 bootstrap_alpha, classify_full_vs_buffered, classify_h1, evaluate, gap_test,
                                 gross_tripwire, h2_conclusion, load_inference_config, newey_west_alpha, rng_for,
                                 sharpe, sharpe_intervals, stationary_bootstrap_indices, _batch_alpha_beta)
from portfolio.trend_signal import FROZEN_CONFIG

CFG = load_inference_config()
RAW = yaml.safe_load(FROZEN_CONFIG.read_text())


def assert_raises(call, *fragments):
    try:
        call()
    except ValueError as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return
    raise AssertionError("expected ValueError")


def pair(n=1200, alpha=0.0, beta=0.6, noise=0.01, rho=0.0, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0004, 0.03, n)
    e = np.zeros(n)
    shocks = rng.normal(0, noise, n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + shocks[t]
    days = pd.RangeIndex(n)
    return pd.Series(alpha + beta * x + e, index=days), pd.Series(x, index=days)


# ------------------------------------------------------------------ configuration
def test_config_carries_the_registered_settings():
    assert (CFG.alpha_level, CFG.annualisation, CFG.replications, CFG.mean_block_days) == (0.05, 365, 10_000, 20)
    assert CFG.sensitivity_block_days == (10, 40, 60) and CFG.newey_west_lags == 21
    assert (CFG.hurdle_annual, CFG.primary_bp, CFG.stress_bp, CFG.cash_rate) == (0.05, 10.0, 20.0, 0.0)


def test_config_raises_on_missing_keys_and_unimplemented_methods():
    for path in (("tests", "inference", "replications"), ("tests", "h2", "hurdle_delta_annual_return"),
                 ("costs", "stress_bp_per_side")):
        raw = copy.deepcopy(RAW)
        node = raw
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
        assert_raises(lambda: load_inference_config(raw=raw), ".".join(path))
    raw = copy.deepcopy(RAW)
    raw["tests"]["inference"]["method"] = "iid_bootstrap"
    assert_raises(lambda: load_inference_config(raw=raw), "tests.inference.method")
    raw = copy.deepcopy(RAW)
    raw["tests"]["inference"]["replications"] = 5
    assert_raises(lambda: load_inference_config(raw=raw), "replications")


# ------------------------------------------------------------------ the bootstrap itself
def test_stationary_indices_have_the_registered_block_length_and_wrap_around():
    rng = np.random.default_rng(0)
    n, mean = 500, 20
    idx = stationary_bootstrap_indices(n, mean, 4000, rng)
    assert idx.shape == (4000, n) and idx.min() >= 0 and idx.max() < n
    continues = idx[:, 1:] == (idx[:, :-1] + 1) % n                       # a step that does not start a block
    new_block_rate = 1.0 - continues.mean()                                # ~ 1/mean (a random restart can also continue by chance)
    assert abs(new_block_rate - (1 / mean) * (1 - 1 / n)) < 0.003
    assert (idx[:, 0] >= 0).all() and len(np.unique(idx[:, 0])) > n // 2   # the first block starts anywhere
    wrapped = (idx[:, :-1] == n - 1) & continues
    assert (idx[:, 1:][wrapped] == 0).all() and wrapped.any()              # continues from the last day to day 0


def test_a_mean_block_of_one_is_the_ordinary_iid_bootstrap():
    idx = stationary_bootstrap_indices(300, 1, 50, np.random.default_rng(1))
    assert (idx[:, 1:] != (idx[:, :-1] + 1) % 300).mean() > 0.98           # essentially every step is a fresh draw


def test_bad_arguments_raise():
    rng = np.random.default_rng(0)
    for args in ((1, 20, 10), (10, 0.5, 10), (10, 20, 0)):
        assert_raises(lambda: stationary_bootstrap_indices(*args, rng), "need")


def test_batched_ols_equals_a_loop_and_numpy():
    y, x = pair(200)
    idx = stationary_bootstrap_indices(200, 10, 5, np.random.default_rng(2))
    a, b = _batch_alpha_beta(y.to_numpy(), x.to_numpy(), idx)
    for r in range(5):
        slope, intercept = np.polyfit(x.to_numpy()[idx[r]], y.to_numpy()[idx[r]], 1)
        assert np.isclose(a[r], intercept) and np.isclose(b[r], slope)
    assert np.allclose(alpha_beta(y.to_numpy(), x.to_numpy()), np.polyfit(x, y, 1)[::-1])


def test_same_seed_same_draws_and_different_seed_different_draws():
    y, x = pair(400)
    a = bootstrap_alpha(y, x, CFG, seed=5, reps=600)
    b = bootstrap_alpha(y, x, CFG, seed=5, reps=600)
    c = bootstrap_alpha(y, x, CFG, seed=6, reps=600)
    assert np.array_equal(a["alphas"], b["alphas"]) and not np.array_equal(a["alphas"], c["alphas"])
    assert abs(a["alpha_p_low"] - c["alpha_p_low"]) < 0.5 * (a["alpha_p_high"] - a["alpha_p_low"])
    assert not np.array_equal(rng_for(1, "h1").random(3), rng_for(1, "h2").random(3))


def test_more_than_one_chunk_of_replications_is_handled():
    y, x = pair(300)
    out = bootstrap_alpha(y, x, CFG, seed=1, reps=CHUNK + 250)
    assert len(out["alphas"]) == CHUNK + 250 and out["reps"] == CHUNK + 250


# ------------------------------------------------------------------ does it find a known truth, and only that?
def test_a_real_alpha_is_recovered_and_passes_the_gate():
    y, x = pair(1500, alpha=0.0025, noise=0.012)                           # ~91%/yr alpha, small noise
    r = bootstrap_alpha(y, x, CFG, seed=3, reps=1000)
    assert abs(r["alpha_annual"] - 0.0025 * 365) < 0.35 and abs(r["beta"] - 0.6) < 0.05
    assert r["passes"] and r["alpha_p_low"] > 0 and r["p_value_one_sided"] < 0.05


def test_no_alpha_rarely_passes_the_gate_so_the_test_has_roughly_its_stated_size():
    wins = sum(bootstrap_alpha(*pair(900, alpha=0.0, noise=0.02, seed=s), CFG, seed=s, reps=300)["passes"] for s in range(200))
    assert wins / 200 <= 0.12                                               # nominal 5%: Monte Carlo and finite-sample slack


def test_a_negative_alpha_fails_and_the_interval_sits_below_zero():
    r = bootstrap_alpha(*pair(1500, alpha=-0.002, noise=0.012), CFG, seed=2, reps=800)
    assert not r["passes"] and r["alpha_p_high"] < 0 and r["p_value_one_sided"] > 0.95


def test_blocks_widen_the_interval_when_residuals_are_autocorrelated():
    y, x = pair(1500, alpha=0.0004, noise=0.01, rho=0.6, seed=4)
    wide = bootstrap_alpha(y, x, CFG, seed=1, mean_block=40, reps=1500)
    narrow = bootstrap_alpha(y, x, CFG, seed=1, mean_block=1, reps=1500)
    assert (wide["alpha_p_high"] - wide["alpha_p_low"]) > 1.15 * (narrow["alpha_p_high"] - narrow["alpha_p_low"])


def test_the_pair_is_resampled_together_not_separately():
    y, x = pair(800, alpha=0.0, beta=0.9, noise=0.002)                     # y almost exactly 0.9 x
    r = bootstrap_alpha(y, x, CFG, seed=1, reps=500)
    assert r["beta_ci95"][0] > 0.85 and r["beta_ci95"][1] < 0.95           # a separate shuffle would wreck beta


def test_misaligned_or_non_finite_returns_raise():
    y, x = pair(100)
    assert_raises(lambda: bootstrap_alpha(y.iloc[1:], x, CFG, seed=1, reps=100), "same days")
    bad = y.copy()
    bad.iloc[5] = np.nan
    assert_raises(lambda: bootstrap_alpha(bad, x, CFG, seed=1, reps=100), "finite")


# ------------------------------------------------------------------ Newey-West
def test_newey_west_with_no_lags_is_the_white_robust_standard_error_and_matches_iid_ols():
    y, x = pair(2000, alpha=0.0003, noise=0.01)
    nw0 = newey_west_alpha(y, x, 0, 365)
    X = np.column_stack([np.ones(len(y)), x.to_numpy()])
    inv = np.linalg.inv(X.T @ X)
    e = y.to_numpy() - X @ (inv @ X.T @ y.to_numpy())
    white = np.sqrt((inv @ (X.T * e ** 2) @ X @ inv)[0, 0]) * 365
    assert np.isclose(nw0["se_annual"], white)
    ols_se = np.sqrt(e.var(ddof=2) * inv[0, 0]) * 365
    assert abs(nw0["se_annual"] / ols_se - 1) < 0.1                        # no heteroskedasticity here, so they agree


def test_newey_west_widens_with_autocorrelation_and_matches_an_explicit_calculation():
    y, x = pair(1500, noise=0.01, rho=0.6, seed=7)
    lags = 21
    nw = newey_west_alpha(y, x, lags, 365)
    X = np.column_stack([np.ones(len(y)), x.to_numpy()])
    inv = np.linalg.inv(X.T @ X)
    e = y.to_numpy() - X @ (inv @ X.T @ y.to_numpy())
    u = X * e[:, None]
    S = sum((1 - abs(k) / (lags + 1)) * (u[max(k, 0):].T @ u[:len(u) - max(k, 0)] if k >= 0 else u[:k].T @ u[-k:])
            for k in range(-lags, lags + 1))
    assert np.isclose(nw["se_annual"], np.sqrt((inv @ S @ inv)[0, 0]) * 365)
    assert nw["se_annual"] > 1.3 * newey_west_alpha(y, x, 0, 365)["se_annual"]
    assert nw["lags"] == 21 and 0 <= nw["p_value_one_sided"] <= 1


def test_the_two_methods_agree_on_a_clean_case():
    y, x = pair(1500, alpha=0.0015, noise=0.012, seed=9)
    boot, nw = bootstrap_alpha(y, x, CFG, seed=1, reps=1500), newey_west_alpha(y, x, 21, 365)
    boot_se = (boot["alpha_p_high"] - boot["alpha_p_low"]) / (2 * 1.645)
    assert 0.6 < boot_se / nw["se_annual"] < 1.6 and abs(boot["alpha_annual"] - nw["alpha_annual"]) < 1e-9


# ------------------------------------------------------------------ Sharpe intervals
def test_sharpe_intervals_cover_the_point_estimate_are_paired_and_reproducible():
    y, x = pair(900, alpha=0.0008, noise=0.015)
    out = sharpe_intervals({"a": y, "b": x, "copy": y.copy()}, CFG, seed=1, reps=800)
    assert out["a"]["ci95"][0] < out["a"]["sharpe"] < out["a"]["ci95"][1]
    assert out["a"]["ci95"] == out["copy"]["ci95"]                          # same days drawn for every series
    assert np.isclose(out["a"]["sharpe"], y.mean() * 365 / (y.std() * np.sqrt(365)))
    assert out == sharpe_intervals({"a": y, "b": x, "copy": y.copy()}, CFG, seed=1, reps=800)
    assert out["a"]["ci90"][0] > out["a"]["ci95"][0] and out["a"]["ci90"][1] < out["a"]["ci95"][1]
    assert np.isclose(sharpe(np.array([[0.01, 0.02, 0.03]]))[0], 0.02 * 365 / (0.01 * np.sqrt(365)))
    assert_raises(lambda: sharpe_intervals({"a": y, "b": x.iloc[1:]}, CFG, seed=1, reps=100), "same days")


# ------------------------------------------------------------------ H2
def test_h2_gap_cases_against_zero_and_the_hurdle():
    rng = np.random.default_rng(5)
    days = pd.RangeIndex(1800)
    big = pd.Series(0.20 / 365 + rng.normal(0, 0.003, 1800), index=days)         # +20%/yr, tiny noise
    zero = pd.Series(rng.normal(0, 0.006, 1800), index=days)
    neg = pd.Series(-0.10 / 365 + rng.normal(0, 0.001, 1800), index=days)
    clears = gap_test(big, CFG, seed=1, reps=800)
    assert clears["rejects_h0"] and clears["clears_hurdle"] and not clears["rules_out_hurdle"]
    flat = gap_test(zero, CFG, seed=1, reps=800)
    assert not flat["rejects_h0"] and not flat["clears_hurdle"]
    ruled_out = gap_test(neg, CFG, seed=1, reps=800)
    assert ruled_out["rules_out_hurdle"] and not ruled_out["rejects_h0"]
    assert clears["hurdle"] == 0.05
    between = pd.Series(0.03 / 365 + rng.normal(0, 0.0005, 1800), index=days)    # beats zero, below the hurdle
    mid = gap_test(between, CFG, seed=1, reps=800)
    assert mid["rejects_h0"] and not mid["clears_hurdle"]


def test_h2_gating_table_and_non_finite_gap():
    assert h2_conclusion(True, True).startswith("basket trend adds value")
    assert "unproven" in h2_conclusion(True, False) and "passive beta" in h2_conclusion(False, True)
    assert h2_conclusion(False, False) == "headline claim unsupported"
    bad = pd.Series(np.linspace(-0.01, 0.01, 40))
    bad.iloc[7] = np.nan
    assert_raises(lambda: gap_test(bad, CFG, seed=1, reps=100), "finite")


# ------------------------------------------------------------------ alpha attribution
def layers(n=600, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.RangeIndex(n)
    def frame(scale):
        price = rng.normal(0.0006, 0.02, n) * scale
        funding = -0.0003 * np.ones(n) * scale + rng.normal(0, 0.0002, n)
        cost = np.abs(rng.normal(0.00005, 0.00003, n)) * scale
        return pd.DataFrame({"price_return": price, "funding_return": funding, "cost_return": cost,
                             "net_return": price + funding - cost}, index=days)
    return frame(0.5), frame(1.0)


def test_attribution_adds_up_exactly_at_one_beta_and_the_headline_is_the_ols_alpha():
    s, b = layers()
    out = alpha_attribution(s, b)
    assert abs(out["alpha_net"] - (out["alpha_price"] + out["alpha_funding"] - out["alpha_cost"])) < 1e-12
    a_hat, beta = alpha_beta(s.net_return.to_numpy(), b.net_return.to_numpy())
    assert np.isclose(out["beta"], beta) and np.isclose(out["alpha_net"], a_hat * 365)
    assert out["alpha_cost"] > 0 or out["alpha_cost"] < 0                    # a number, reported with its sign


def test_attribution_raises_when_layers_do_not_add_up_or_days_differ():
    s, b = layers()
    broken = s.copy()
    broken.loc[10, "net_return"] += 1e-3
    assert_raises(lambda: alpha_attribution(broken, b), "do not add up")
    assert_raises(lambda: alpha_attribution(s.iloc[1:], b), "same days")


# ------------------------------------------------------------------ the registered classification
def test_h1_classification_all_four_cases():
    assert classify_h1(0.30, 0.10, 0.08, 0.05, 0.02) == SUPPORTED
    assert classify_h1(0.30, 0.10, -0.01, 0.05, 0.02) == FRAGILE            # fails at 40-day blocks
    assert classify_h1(0.30, 0.10, 0.08, -0.02, 0.02) == FRAGILE            # fails at 60-day blocks
    assert classify_h1(0.30, 0.10, 0.08, 0.05, -0.03) == FRAGILE            # fails at the stress cost
    assert classify_h1(0.12, -0.04, 0.08, 0.05, 0.02) == INCONCLUSIVE       # positive, interval includes zero
    assert classify_h1(-0.05, -0.20, -0.2, -0.2, -0.2) == FAILS
    assert classify_h1(0.0, 0.1, 0.1, 0.1, 0.1) == FAILS                    # alpha must be strictly positive


def test_full_versus_buffered_table_and_the_gross_tripwire():
    assert classify_full_vs_buffered(0.1, 0.2) == "survives even naive trading"
    assert classify_full_vs_buffered(-0.1, 0.2) == "exists only with execution discipline"
    assert classify_full_vs_buffered(-0.1, -0.2) == "fails"
    assert classify_full_vs_buffered(0.1, -0.2).startswith("suspicious")
    assert gross_tripwire(0.0012, 0.0010) and not gross_tripwire(0.0010, 0.0012)


# ------------------------------------------------------------------ the orchestrator on synthetic runner outputs
def synthetic_results():
    from portfolio.test_strategy_runner import go, btc_like
    out = {"buffered": go("buffered", "fixed", cost_bp_per_side=10.0),
           "buffered_stress": go("buffered", "fixed", cost_bp_per_side=20.0),
           "full": go("full", "fixed"), "btc": btc_like("buffered", "fixed")}
    return out


def test_evaluate_runs_every_registered_analysis_and_labels_the_result():
    results = synthetic_results()
    out = evaluate(results, seed=11, reps=400)
    for key in ("h1", "h1_block_sensitivity", "h1_stress_cost", "newey_west", "attribution", "sharpe",
                "classification", "record", "full_vs_buffered", "h2"):
        assert key in out, key
    assert set(out["h1_block_sensitivity"]) == {10, 40, 60}
    assert out["classification"] in (SUPPORTED, INCONCLUSIVE, FAILS, FRAGILE)
    assert out["record"]["seed"] == 11 and out["record"]["registered_run"] is False and out["record"]["replications"] == 400
    assert out["h2"]["gated_on_h1"] is True and out["h2"]["h1_passes"] == out["h1"]["passes"]
    assert set(out["sharpe"]) == {"strategy", "benchmark", "btc_strategy"}
    assert abs(out["attribution"]["identity_residual"]) < 1e-9
    assert "alphas" not in out["h1"]
    assert out == evaluate(results, seed=11, reps=400)                       # reproducible


def test_the_stress_cost_run_costs_more_and_never_helps():
    r = synthetic_results()
    assert r["buffered_stress"].daily.cost.sum() > r["buffered"].daily.cost.sum()
    assert r["buffered_stress"].paired.strategy_net.sum() < r["buffered"].paired.strategy_net.sum()


def test_evaluate_without_the_optional_runs_still_works_and_requires_the_primary_pair():
    r = synthetic_results()
    only = {"buffered": r["buffered"], "buffered_stress": r["buffered_stress"]}
    out = evaluate(only, seed=1, reps=300)
    assert "full_vs_buffered" not in out and "h2" not in out and "btc_strategy" not in out["sharpe"]
    try:
        evaluate({"buffered": r["buffered"]}, seed=1, reps=300)
    except KeyError:
        return
    raise AssertionError("expected KeyError for the missing stress run")


# ------------------------------------------------------------------ step 1: the right experiment, defined statistics
def results_set():
    from portfolio.test_strategy_runner import go, btc_like
    return {"buffered": go("buffered", "fixed", cost_bp_per_side=10.0),
            "buffered_stress": go("buffered", "fixed", cost_bp_per_side=20.0),
            "full": go("full", "fixed"), "btc": btc_like("buffered", "fixed")}


def test_a_correctly_matched_set_passes_identity_but_is_not_called_registered_off_the_real_window():
    out = check_experiment_identity(results_set(), CFG)
    assert out["registered"] is False and any("start" in r for r in out["reasons"])


def test_a_mismatched_pair_raises_for_every_way_it_can_differ():
    from portfolio.test_strategy_runner import START, END, go
    from portfolio.strategy_runner import run_strategy
    good = results_set()
    wrong_cost = go("buffered", "fixed", cost_bp_per_side=15.0)
    wrong_capital = go("buffered", "equity", cost_bp_per_side=20.0)
    wrong_lag = go("buffered", "fixed", cost_bp_per_side=20.0, signal_lag_days=1)
    for bad in (wrong_cost, wrong_capital, wrong_lag):
        assert_raises(lambda: check_experiment_identity({**good, "buffered_stress": bad}, CFG), "not the same experiment")
    assert_raises(lambda: check_experiment_identity({**good, "full": good["buffered"]}, CFG), "differs in")
    assert_raises(lambda: check_experiment_identity({**good, "btc": good["buffered"]}, CFG), "differs in")
    assert_raises(lambda: check_experiment_identity({**good, "buffered": go("buffered", "fixed", cost_bp_per_side=15.0)}, CFG),
                  "registered 10 bp")
    from portfolio.test_strategy_runner import synthetic_inputs
    other_data = run_strategy("buffered", "fixed", inputs=synthetic_inputs(seed=4), start=START, end=END, cost_bp_per_side=20.0)
    assert_raises(lambda: check_experiment_identity({**good, "buffered_stress": other_data}, CFG), "different input data")
    try:
        check_experiment_identity({"buffered": good["buffered"]}, CFG)
    except KeyError:
        return
    raise AssertionError("expected KeyError for the missing stress run")


def test_registered_is_true_only_for_the_exact_registered_primary_on_clean_code():
    from dataclasses import replace
    from portfolio.trend_signal import load_signal_config
    good = results_set()

    def with_params(**changes):
        params = {**good["buffered"].record["parameters"], **REGISTERED_PRIMARY, **changes}
        params["config"] = {**params["config"], "config_hash": load_signal_config().config_hash}
        record = {**good["buffered"].record, "parameters": params, "git": {"commit": "abc", "dirty": False}}
        return replace(good["buffered"], record=record)

    def paired(primary, key, **change):
        r = good[key]
        params = {**primary.record["parameters"], **change}
        return replace(r, record={**r.record, "parameters": params, "git": {"commit": "abc", "dirty": False}})

    primary = with_params()
    results = {"buffered": primary, "buffered_stress": paired(primary, "buffered_stress", cost_bp_per_side=20.0)}
    results["buffered_stress"] = replace(results["buffered_stress"], record={**results["buffered_stress"].record,
                                         "fingerprints": primary.record["fingerprints"]})
    out = check_experiment_identity(results, CFG)
    assert out == {"registered": True, "reasons": []}, out
    dirty = replace(primary, record={**primary.record, "git": {"commit": "abc", "dirty": True}})
    assert not check_experiment_identity({**results, "buffered": dirty}, CFG)["registered"]
    def stale(run):                                             # both runs used a configuration that is not the current file
        params = {**run.record["parameters"], "config": {**run.record["parameters"]["config"], "config_hash": "0" * 64}}
        return replace(run, record={**run.record, "parameters": params})
    stale_set = {"buffered": stale(primary), "buffered_stress": stale(results["buffered_stress"])}
    assert any("configuration" in r for r in check_experiment_identity(stale_set, CFG)["reasons"])
    assert_raises(lambda: check_experiment_identity({**results, "buffered": stale(primary)}, CFG), "not the same experiment")


def test_duplicate_unsorted_short_and_degenerate_series_are_rejected():
    y, x = pair(200)
    dup = pd.Series(y.values, index=[0] * 100 + list(range(100, 200)))
    assert_raises(lambda: bootstrap_alpha(dup, x.set_axis(dup.index), CFG, seed=1, reps=100), "duplicate")
    assert_raises(lambda: bootstrap_alpha(y.iloc[::-1], x.iloc[::-1], CFG, seed=1, reps=100), "increasing")
    assert_raises(lambda: bootstrap_alpha(y.iloc[:20], x.iloc[:20], CFG, seed=1, reps=100), "at least")
    assert_raises(lambda: bootstrap_alpha(y, x * 0.0, CFG, seed=1, reps=100), "no variance")
    flat = pd.Series(0.001, index=y.index)
    assert_raises(lambda: sharpe_intervals({"a": y, "flat": flat}, CFG, seed=1, reps=100), "no variance")
    assert_raises(lambda: gap_test(dup, CFG, seed=1, reps=100), "duplicate")
    inf = y.copy()
    inf.iloc[3] = np.inf
    assert_raises(lambda: newey_west_alpha(inf, x, 21, 365), "finite")


def test_an_undefined_statistic_cannot_be_classified():
    nan = float("nan")
    for args in ((nan, 0.1, 0.1, 0.1, 0.1), (0.2, nan, 0.1, 0.1, 0.1), (0.2, 0.1, nan, 0.1, 0.1),
                 (0.2, 0.1, 0.1, nan, 0.1), (0.2, 0.1, 0.1, 0.1, nan), (float("inf"), 0.1, 0.1, 0.1, 0.1)):
        assert_raises(lambda: classify_h1(*args), "undefined")


# ------------------------------------------------------------------ funding and cost drag
def drag_layers(n=900, exposure_mean=0.6, rate=3e-4, cost=4e-5, seed=2, persistent=True):
    """Layers with a KNOWN funding drag (rate x exposure) and cost drag, plus the long/short side."""
    rng = np.random.default_rng(seed)
    phi, w = 0.97 if persistent else 0.0, np.zeros(n)
    for t in range(1, n):
        w[t] = phi * w[t - 1] + rng.normal(0, 0.15)
    exposure = np.clip(exposure_mean + w, -1.0, 2.0)
    price = rng.normal(0.0004, 0.02, n) * exposure
    funding = -rate * exposure + rng.normal(0, 3e-5, n)
    costs = np.abs(rng.normal(cost, cost / 3, n))
    days = pd.RangeIndex(n)
    layers = pd.DataFrame({"price_return": price, "funding_return": funding, "cost_return": costs,
                           "net_return": price + funding - costs}, index=days)
    return layers, pd.Series(np.sign(exposure), index=days), exposure


def test_drag_point_estimates_are_the_annualised_layer_means_and_the_layers_add_up():
    layers, side, _ = drag_layers()
    out = drag_table(layers, CFG, seed=1, name="t", side=side, reps=300)
    assert np.isclose(out["funding_drag"]["estimate"], -layers.funding_return.mean() * 365)
    assert np.isclose(out["cost_drag"]["estimate"], layers.cost_return.mean() * 365)
    assert np.isclose(out["total_drag"]["estimate"], out["funding_drag"]["estimate"] + out["cost_drag"]["estimate"])
    assert np.isclose(out["net"]["estimate"], out["price"]["estimate"] - out["total_drag"]["estimate"])
    assert np.isclose(out["gross"]["estimate"], out["price"]["estimate"] - out["funding_drag"]["estimate"])
    for key in ("funding_drag", "cost_drag", "total_drag"):
        lo, hi = out[key]["ci90"]
        assert lo < out[key]["estimate"] < hi and out[key]["ci95"][0] < lo and out[key]["ci95"][1] > hi


def test_the_side_split_adds_to_the_total_and_signs_follow_the_position():
    layers, side, exposure = drag_layers(exposure_mean=0.0)                    # the book is both long and short
    out = drag_table(layers, CFG, seed=1, name="t", side=side, reps=300)
    parts = sum(out[f"funding_drag_{k}_days"]["estimate"] for k in ("long", "short", "flat"))
    assert np.isclose(parts, out["funding_drag"]["estimate"])
    assert out["funding_drag_long_days"]["estimate"] > 0 > out["funding_drag_short_days"]["estimate"]   # longs pay, shorts receive


def test_registered_block_drag_intervals_are_too_narrow_for_persistent_funding_so_the_envelope_is_quoted():
    truth = 3e-4 * 0.6028 * 365                       # long-run mean exposure after clipping, by simulation
    n, hits_registered, hits_envelope = 80, 0, 0
    for k in range(n):
        layers, _, _ = drag_layers(n=1883, seed=500 + k)
        cell = drag_table(layers, CFG, seed=k, name="cover", reps=250)["funding_drag"]
        hits_registered += cell["ci90"][0] <= truth <= cell["ci90"][1]
        hits_envelope += cell["ci90_conservative"][0] <= truth <= cell["ci90_conservative"][1]
    assert hits_registered / n < 0.85                # documents the shortfall (nominal 90%): 20-day blocks under-cover here
    assert hits_envelope / n >= 0.78 and hits_envelope >= hits_registered     # the envelope covers more, though still not 90%


def test_the_envelope_contains_the_registered_interval_and_lists_every_block():
    layers, side, _ = drag_layers()
    cell = drag_table(layers, CFG, seed=1, name="t", side=side, reps=300)["funding_drag"]
    assert set(cell["by_block_ci90"]) == {20, 60, 120}
    assert cell["ci90_conservative"][0] <= cell["ci90"][0] and cell["ci90_conservative"][1] >= cell["ci90"][1]
    assert cell["by_block_ci90"][20] == cell["ci90"]


def test_the_gap_between_two_runs_is_the_difference_in_drag():
    heavy, _, _ = drag_layers(exposure_mean=1.0, seed=3, persistent=False)
    light, _, _ = drag_layers(exposure_mean=0.4, seed=3, persistent=False)
    out = drag_gap(heavy, light, CFG, seed=1, name="g", reps=400)
    assert np.isclose(out["funding_drag"]["estimate"], -(heavy.funding_return - light.funding_return).mean() * 365)
    assert out["funding_drag"]["ci90"][0] > 0                                  # the heavier book bears clearly more funding
    assert np.isclose(out["total_drag"]["estimate"], out["funding_drag"]["estimate"] + out["cost_drag"]["estimate"])


def test_drag_inputs_that_do_not_add_up_or_do_not_match_raise():
    layers, side, _ = drag_layers()
    broken = layers.copy()
    broken.loc[5, "net_return"] += 1e-3
    assert_raises(lambda: check_layers(broken), "do not add up")
    assert_raises(lambda: drag_table(broken, CFG, seed=1, name="b", reps=100), "do not add up")
    assert_raises(lambda: drag_table(layers.drop(columns="cost_return"), CFG, seed=1, name="b", reps=100), "lack")
    assert_raises(lambda: drag_table(layers, CFG, seed=1, name="b", side=side.iloc[1:], reps=100), "same days")
    assert_raises(lambda: drag_gap(layers, layers.iloc[1:], CFG, seed=1, name="b", reps=100), "same days")


def test_drag_results_are_reproducible_and_seeded():
    layers, side, _ = drag_layers()
    a = drag_table(layers, CFG, seed=5, name="r", side=side, reps=300)
    assert a == drag_table(layers, CFG, seed=5, name="r", side=side, reps=300)
    assert a != drag_table(layers, CFG, seed=6, name="r", side=side, reps=300)


def test_evaluate_drags_covers_every_run_and_the_stress_cost_is_exactly_double():
    out = evaluate_drags(results_set(), seed=1, reps=300)
    assert set(out) == {"strategy", "benchmark", "strategy_minus_benchmark", "cost_stress", "btc_strategy", "btc_benchmark"}
    assert np.isclose(out["cost_stress"]["cost_drag"]["estimate"], 2 * out["strategy"]["cost_drag"]["estimate"])
    assert out["cost_stress"]["bp"] == 20.0
    gap = out["strategy_minus_benchmark"]["funding_drag"]["estimate"]
    assert np.isclose(gap, out["strategy"]["funding_drag"]["estimate"] - out["benchmark"]["funding_drag"]["estimate"])
    assert "funding_drag_long_days" in out["strategy"] and "funding_drag_long_days" not in out["benchmark"]
    only = {k: v for k, v in results_set().items() if k in ("buffered", "buffered_stress")}
    assert set(evaluate_drags(only, seed=1, reps=200)) == {"strategy", "benchmark", "strategy_minus_benchmark", "cost_stress"}
    try:
        evaluate_drags({"buffered": only["buffered"]}, seed=1, reps=100)
    except KeyError:
        return
    raise AssertionError("expected KeyError for the missing stress run")


def test_evaluate_includes_the_drags():
    out = evaluate(results_set(), seed=2, reps=300)
    assert "drags" in out and set(out["drags"]) >= {"strategy", "benchmark", "cost_stress"}


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
