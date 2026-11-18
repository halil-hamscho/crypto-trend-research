"""Run from the repo root: python -m portfolio.test_secondaries  (synthetic data)"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio.freeze as fz
import portfolio.secondaries as sec
from portfolio.basket import cap_btc, build_inputs
from portfolio.strategy_runner import run_strategy
from portfolio.test_basket import assert_raises
from portfolio.test_strategy_runner import END, INPUTS, START
from portfolio.trend_signal import FORECAST_UNIT, basket_forecast, load_signal_config, sign_only, single_speed_config

CONFIG = load_signal_config()


def test_btc_cap_moves_the_excess_pro_rata_and_leaves_an_uncapped_basket_alone():
    m = pd.DataFrame({"contract": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"], "weight": [0.66, 0.20, 0.10, 0.04]})
    out = cap_btc(m, 0.30).set_index("contract").weight
    assert np.isclose(out["BTCUSDT"], 0.30) and np.isclose(out.sum(), 1.0)
    rest = m.set_index("contract").weight.drop("BTCUSDT")
    assert np.allclose(out.drop("BTCUSDT") / out.drop("BTCUSDT").sum(), rest / rest.sum())       # pro rata
    assert cap_btc(m, 0.70).equals(m) and cap_btc(m.iloc[1:], 0.30).equals(m.iloc[1:])
    assert_raises(lambda: cap_btc(m, 1.2), "cap")
    assert_raises(lambda: build_inputs(instrument="btc", btc_cap=0.3), "basket only")


def test_a_single_speed_keeps_its_scalar_and_drops_the_blend_multiplier():
    one = single_speed_config(CONFIG, (16, 64))
    assert one.speeds == ((16, 64),) and one.fdm == 1.0 and one.scalars["16/64"] == 3.75
    assert_raises(lambda: single_speed_config(CONFIG, (5, 20)), "not a registered speed")
    index = INPUTS[4]
    alone, blend = basket_forecast(index, one), basket_forecast(index, CONFIG)
    scored = alone.forecast.dropna().index
    assert np.allclose(alone.loc[scored, "forecast"], blend.loc[scored, "f_16/64"].clip(-20, 20))


def test_sign_only_keeps_the_direction_at_the_average_size():
    signal = basket_forecast(INPUTS[4], CONFIG)
    out = sign_only(signal, CONFIG)
    scored = signal.forecast.dropna().index
    assert set(np.unique(out.loc[scored, "forecast"])) <= {-FORECAST_UNIT, 0.0, FORECAST_UNIT}
    assert (np.sign(out.loc[scored, "forecast"]) == np.sign(signal.loc[scored, "forecast"])).all()
    assert np.allclose(out.loc[scored, "x_star"], (np.sign(signal.loc[scored, "forecast"]) * signal.loc[scored, "standard"]).clip(-2, 2))
    assert out.forecast.isna().sum() == signal.forecast.isna().sum()                              # warm-up stays NaN


def test_the_variant_is_a_recorded_parameter_and_an_unknown_one_raises():
    r = run_strategy("buffered", inputs=INPUTS, start=START, end=END, signal_variant="sign_only")
    assert r.record["parameters"]["signal_variant"] == "sign_only"
    assert_raises(lambda: run_strategy("buffered", inputs=INPUTS, start=START, end=END, signal_variant="x"), "signal_variant")
    one = run_strategy("buffered", inputs=INPUTS, start=START, end=END, config=single_speed_config(CONFIG, (8, 32)))
    recorded = one.record["parameters"]["config"]
    assert [list(v) for v in recorded["speeds"]] == [[8, 32]] and recorded["fdm"] == 1.0


def test_the_manifest_lists_exactly_s2_to_s8_defers_the_rest_and_references_the_primary():
    m = sec.build_manifest("c3e68cb1f90a")
    assert list(m["analyses"]) == ["S2_btc_cap_30", "S3_btc_cap_35", "S4_speed_8_32", "S5_speed_16_64",
                                   "S6_speed_32_128", "S7_speed_64_256", "S8_sign_only"]
    assert m["references_primary_manifest"] == "c3e68cb1f90a" and set(m["deferred"]) >= {"S9 weekly reconstitution",
                                                                                          "S10 daily reconstitution", "S16 per-coin TSMOM"}
    assert m["seed"] == 20261004 and "portfolio/secondaries.py" in m["code_sha256"]
    assert m == sec.build_manifest("c3e68cb1f90a")


def test_a_synthetic_secondary_run_stores_seals_and_logs_everything():
    with tempfile.TemporaryDirectory() as store:
        folder = sec.run_secondaries(sec.build_manifest("synthetic"), Path(store), check_git=False,
                                     inputs_provider=lambda cap: INPUTS, reps=300, window=(START, END))
        assert fz.verify(folder)
        import json
        stats = json.loads((folder / "secondaries.json").read_text())
        assert set(stats) == set(sec.SECONDARIES)
        for name, cell in stats.items():
            assert (folder / name / "paired.csv.gz").exists()
            assert {"net_annual", "sharpe", "alpha", "newey_west", "turnover_per_year"} <= set(cell)
        attempts = fz.read_attempts(folder)
        assert [a["status"] for a in attempts] == ["started", "completed"] and attempts[-1]["registered"] is False
        assert_raises(lambda: sec.run_secondaries(sec.build_manifest("synthetic"), Path(store), check_git=False,
                                                  inputs_provider=lambda cap: INPUTS, reps=300, window=(START, END)),
                      "rerun needs a stated reason")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
