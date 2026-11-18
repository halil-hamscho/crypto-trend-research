"""Run from the repo root: python -m portfolio.test_trend_signal  (the real-data test takes ~30 s)"""
import copy
from datetime import date

import numpy as np
import pandas as pd
import yaml

from portfolio.trend_signal import (FORECAST_UNIT, FROZEN_CONFIG, basket_forecast, load_signal_config)
from strategies.combination.forecast_combiner import ForecastCombiner

CONFIG = load_signal_config()
RAW = yaml.safe_load(FROZEN_CONFIG.read_text())
DAYS = pd.date_range("2020-02-09", periods=900, freq="D")
rng = np.random.default_rng(7)
INDEX = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.035, len(DAYS)))), index=[d.date() for d in DAYS])
FORECAST = basket_forecast(INDEX, CONFIG)


def assert_raises(call, *fragments):
    try:
        call()
    except ValueError as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return
    raise AssertionError("expected ValueError")


# ------------------------------------------------------------------ the configuration
def test_config_carries_the_registered_values():
    assert CONFIG.speeds == ((8, 32), (16, 64), (32, 128), (64, 256))
    assert CONFIG.scalars == {"8/32": 5.3, "16/64": 3.75, "32/128": 2.65, "64/256": 1.87}
    assert (CONFIG.fdm, CONFIG.cap, CONFIG.vol_span, CONFIG.target_vol, CONFIG.leverage_cap) == (1.15, 20.0, 32, 0.30, 2.0)
    assert (CONFIG.floor_quantile, CONFIG.floor_window, CONFIG.floor_min_days) == (0.05, 500, 100)
    assert CONFIG.annualisation == 365 and CONFIG.warmup_observations == 512
    assert len(CONFIG.config_hash) == 64


def test_a_missing_key_raises_and_names_it():
    for path in (("signal", "fdm"), ("risk", "target_vol"), ("signal", "forecast_scalars"), ("sample", "warmup_observations")):
        raw = copy.deepcopy(RAW)
        del raw[path[0]][path[1]]
        assert_raises(lambda: load_signal_config(raw=raw), ".".join(path))


def test_a_structure_the_runner_does_not_implement_raises():
    for path, value in ((("signal", "ewma_adjust"), False), (("signal", "weights"), "correlation"),
                        (("risk", "vol_floor_rule"), "none"), (("signal", "pipeline_order"), ["raw"])):
        raw = copy.deepcopy(RAW)
        raw[path[0]][path[1]] = value
        assert_raises(lambda: load_signal_config(raw=raw), ".".join(path))


def test_a_speed_without_a_scalar_raises():
    raw = copy.deepcopy(RAW)
    del raw["signal"]["forecast_scalars"]["64/256"]
    assert_raises(lambda: load_signal_config(raw=raw), "64/256")


# ------------------------------------------------------------------ the maths, recomputed independently
def reference_forecast(index: pd.Series) -> pd.DataFrame:
    returns = index.pct_change()
    vol = returns.ewm(span=32, adjust=True).std()
    floor = vol.rolling(500, min_periods=100).quantile(0.05)
    vol = vol.where(vol >= floor, floor).fillna(vol)
    speeds = {}
    for (fast, slow), scalar in zip(CONFIG.speeds, (5.3, 3.75, 2.65, 1.87)):
        raw = index.ewm(span=fast, adjust=True).mean() - index.ewm(span=slow, adjust=True).mean()
        speeds[f"f_{fast}/{slow}"] = (raw / (index * vol) * scalar).clip(-20, 20)
    frame = pd.DataFrame(speeds)
    forecast = (frame.mean(axis=1, skipna=False) * 1.15).clip(-20, 20)
    x_star = (forecast / 10 * 0.30 / (vol * np.sqrt(365))).clip(-2, 2)
    return pd.DataFrame({"vol": vol, "forecast": forecast, "x_star": x_star}, index=index.index).join(frame)


def test_forecast_and_sizing_match_an_independent_calculation():
    ref = reference_forecast(INDEX)
    scored = FORECAST.index[511:]
    for column in ("f_8/32", "f_16/64", "f_32/128", "f_64/256", "forecast", "x_star"):
        assert np.allclose(FORECAST.loc[scored, column], ref.loc[scored, column], rtol=1e-10, atol=1e-12), column
    assert np.allclose(FORECAST.vol_daily.iloc[511:], ref.vol.iloc[511:])


def test_x_star_is_forecast_over_ten_times_the_standard_exposure():
    scored = FORECAST.iloc[511:]
    assert np.allclose(scored.standard, 0.30 / scored.vol_annual)
    raw = scored.forecast / FORECAST_UNIT * scored.standard
    assert np.allclose(scored.x_star, raw.clip(-2.0, 2.0))


def test_the_registered_pipeline_agrees_with_the_forecast_combiner_on_finite_rows():
    columns = [f"f_{f}/{s}" for f, s in CONFIG.speeds]
    scored = FORECAST.iloc[511:]
    combiner = ForecastCombiner(fdm=1.15)
    dummy = {c: pd.Series(0.0, index=scored.index) for c in columns}
    combined = combiner.combine({c: scored[c] for c in columns}, dummy, weight_method="equal")
    assert np.allclose(combined, scored.forecast)


# ------------------------------------------------------------------ no look-ahead, warm-up, caps
def test_nothing_after_the_decision_instant_enters_the_forecast():
    for k in (520, 600, 777, 899):
        truncated = basket_forecast(INDEX.iloc[:k], CONFIG)
        pd.testing.assert_frame_equal(truncated, FORECAST.iloc[:k], check_exact=False, rtol=1e-12, atol=1e-12)


def test_changing_the_future_does_not_change_the_past():
    altered = INDEX.copy()
    altered.iloc[700:] = altered.iloc[700:] * 3.0
    result = basket_forecast(altered, CONFIG)
    pd.testing.assert_frame_equal(result.iloc[:700], FORECAST.iloc[:700], check_exact=False, rtol=1e-12, atol=1e-12)


def test_position_is_nan_until_the_512th_observation_and_not_converted_to_zero():
    assert FORECAST.forecast.iloc[:511].isna().all() and FORECAST.x_star.iloc[:511].isna().all()
    assert FORECAST.forecast.iloc[511:].notna().all()
    assert FORECAST.index[511] == date(2021, 7, 4)               # observation 512 of an index that starts 2020-02-09
    assert not FORECAST.forecast_cap_hit.iloc[:511].any()


def test_forecast_and_exposure_respect_their_caps_and_the_flags_are_right():
    trend = pd.Series(100 * 1.02 ** np.arange(700), index=[d.date() for d in pd.date_range("2020-02-09", periods=700)])
    trend.iloc[::7] *= 1.0005                                    # a little noise so vol is nonzero
    out = basket_forecast(trend, CONFIG)
    assert out.forecast.abs().max() <= 20.0 and out.x_star.abs().max() <= 2.0
    assert out.forecast_cap_hit.any() and out.x_star_cap_hit.any()
    assert (out.forecast.abs()[out.forecast_cap_hit] == 20.0).all()


def test_a_nan_in_the_index_raises_and_a_scored_day_is_never_silently_zero():
    broken = INDEX.copy()
    broken.iloc[600] = np.nan
    assert_raises(lambda: basket_forecast(broken, CONFIG), "finite")
    unsorted = INDEX.iloc[::-1]
    assert_raises(lambda: basket_forecast(unsorted, CONFIG), "increasing")


def test_non_finite_zero_and_negative_index_values_raise_instead_of_becoming_a_forecast():
    for value in (np.inf, -np.inf, 0.0, -5.0):
        broken = INDEX.copy()
        broken.iloc[600] = value
        assert_raises(lambda: basket_forecast(broken, CONFIG), "finite and positive")


def test_a_config_with_invalid_domains_or_unimplemented_semantics_raises():
    cases = [(("signal", "cap"), 10.0, "signal.cap"), (("signal", "normalisation"), "raw / price", "signal.normalisation"),
             (("signal", "fdm"), float("inf"), "fdm"), (("signal", "fdm"), -1.0, "fdm"),
             (("risk", "target_vol"), -0.3, "target_vol"), (("risk", "leverage_cap"), 0.0, "leverage_cap"),
             (("risk", "vol_floor_quantile"), 1.5, "quantile"), (("risk", "vol_span_days"), 1, "vol span"),
             (("risk", "vol_floor_min_days"), 600, "floor_min_days"), (("sample", "warmup_observations"), 0, "warm-up"),
             (("signal", "speeds"), [[8, 32], [8, 32]], "unique"),
             (("signal", "forecast_scalars"), {"8/32": 5.3, "16/64": -1.0, "32/128": 2.65, "64/256": 1.87}, "scalars")]
    for path, value, fragment in cases:
        raw = copy.deepcopy(RAW)
        raw[path[0]][path[1]] = value
        assert_raises(lambda: load_signal_config(raw=raw), fragment)


def test_a_reversed_speed_raises():
    raw = copy.deepcopy(RAW)
    raw["signal"]["speeds"] = [[32, 8]]
    raw["signal"]["forecast_scalars"] = {"32/8": 1.0}
    assert_raises(lambda: load_signal_config(raw=raw), "fast < slow")


def test_the_vol_floor_binds_and_is_backward_looking():
    quiet = INDEX.copy()
    quiet.iloc[560:600] = quiet.iloc[559]                        # a flat stretch: raw vol collapses
    out = basket_forecast(quiet, CONFIG)
    raw = quiet.pct_change().ewm(span=32, adjust=True).std()
    assert (out.vol_daily.iloc[580:600] > raw.iloc[580:600]).all()      # the floor lifted it
    assert out.vol_daily.iloc[580:600].min() > 0
    early = basket_forecast(quiet.iloc[:559], CONFIG)
    pd.testing.assert_series_equal(early.vol_daily, out.vol_daily.iloc[:559], check_names=False)


def test_the_scalars_are_fixed_not_estimated_from_the_sample():
    half = basket_forecast(INDEX.iloc[:600], CONFIG)
    pd.testing.assert_series_equal(half["f_8/32"].iloc[511:], FORECAST["f_8/32"].iloc[511:600], check_names=False,
                                   rtol=1e-12)                    # a rolling median scalar would differ


# ------------------------------------------------------------------ real reference index
def test_real_reference_index_signal_is_finite_from_2021_07_04_to_the_end():
    from portfolio.basket import build_inputs
    _, _, _, _, index = build_inputs()
    out = basket_forecast(index, CONFIG)
    assert index.index[0] == date(2020, 2, 9) and out.forecast.first_valid_index() == date(2021, 7, 4)
    scored = out.loc[date(2021, 7, 4):]
    assert scored[["forecast", "x_star", "vol_annual"]].notna().all().all()
    assert scored.forecast.abs().max() <= 20.0 and scored.x_star.abs().max() <= 2.0
    print("   real index:", len(scored), "signal-available days; mean |forecast| %.1f; forecast cap-hit %.1f%%; "
          "x* cap-hit %.1f%%; vol_annual %.0f%%-%.0f%%; x* range %.2f to %.2f" % (
              scored.forecast.abs().mean(), 100 * scored.forecast_cap_hit.mean(), 100 * scored.x_star_cap_hit.mean(),
              100 * scored.vol_annual.min(), 100 * scored.vol_annual.max(), scored.x_star.min(), scored.x_star.max()))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
