"""Run from the repo root: python -m portfolio.test_placebo"""
import numpy as np
import pandas as pd

from portfolio.inference import check_experiment_identity, load_inference_config
from portfolio.placebo import WARMUP, foresight, one_day_leak, world
from portfolio.strategy_runner import run_strategy
from portfolio.test_basket import assert_raises
from portfolio.test_strategy_runner import END, INPUTS, START

CFG = load_inference_config()


def test_a_world_is_deterministic_complete_and_has_the_real_index_length():
    (m1, k1, s1, w1, i1), start, end = world("trend", 0.0, 3)
    (m2, k2, s2, w2, i2), _, _ = world("trend", 0.0, 3)
    assert i1.equals(i2) and k1.equals(k2) and len(k1) == 2396
    assert start == k1.index[WARMUP] and end == k1.index[-2]
    assert set(k1.columns) == {"A", "B", "C"} and set(s1) == {"A", "B", "C"} and np.allclose(w1.sum(axis=1), 1.0)
    assert not world("trend", 0.0, 4)[0][4].equals(i1)


def test_a_planted_trend_adds_persistence_a_placebo_does_not():
    def lag_corr(index):
        r = index.pct_change().dropna().rolling(20).sum().dropna()
        return np.corrcoef(r.values[:-20], r.values[20:])[0, 1]
    placebo = np.mean([lag_corr(world("trend", 0.0, s)[0][4]) for s in range(6)])
    planted = np.mean([lag_corr(world("trend", 4.0, s)[0][4]) for s in range(6)])
    assert planted > placebo + 0.1 and abs(placebo) < 0.15
    meanrev = np.mean([lag_corr(world("meanrev", 0.0, s)[0][4]) for s in range(6)])
    assert meanrev < planted


def test_the_leaks_do_what_they_say():
    signal = pd.DataFrame({"x_star": np.arange(6.0), "standard": 1.0, "forecast": np.arange(6.0)},
                          index=pd.RangeIndex(6))
    leaked = one_day_leak(signal, pd.Series(np.arange(6.0)))
    assert list(leaked.x_star) == [1.0, 2.0, 3.0, 4.0, 5.0, 5.0] and leaked.index.equals(signal.index)
    index = pd.Series([100.0, 110.0, 99.0, 99.0, 120.0, 90.0])
    cheat = foresight(signal, index)
    assert list(cheat.x_star) == [1.5, -1.5, 0.0, 1.5, -1.5, 0.0] and (cheat.standard == 1.0).all()


def test_an_overridden_run_is_recorded_and_can_never_be_called_registered_or_paired_with_a_clean_one():
    clean = run_strategy("buffered", inputs=INPUTS, start=START, end=END)
    leaked = run_strategy("buffered", inputs=INPUTS, start=START, end=END, signal_override=one_day_leak)
    assert clean.record["parameters"]["signal_override"] is False and leaked.record["parameters"]["signal_override"] is True
    stress = run_strategy("buffered", inputs=INPUTS, start=START, end=END, cost_bp_per_side=20.0)
    assert_raises(lambda: check_experiment_identity({"buffered": leaked, "buffered_stress": stress}, CFG), "differs in")
    assert any("signal_override" in r for r in check_experiment_identity(
        {"buffered": leaked, "buffered_stress": run_strategy("buffered", inputs=INPUTS, start=START, end=END,
                                                            cost_bp_per_side=20.0, signal_override=one_day_leak)}, CFG)["reasons"])
    assert not np.allclose(clean.daily.net_return, leaked.daily.net_return)


def test_perfect_foresight_makes_money_on_the_synthetic_basket():
    cheat = run_strategy("full", inputs=INPUTS, start=START, end=END, signal_override=foresight)
    assert cheat.daily.price_return.sum() > 0 and cheat.stats["identity_max_residual"] < 1e-9
    honest = run_strategy("full", inputs=INPUTS, start=START, end=END)
    assert cheat.daily.net_return.mean() > 10 * abs(honest.daily.net_return.mean())


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
