"""Run from the repo root: python -m portfolio.test_reporting  (synthetic data)"""
import numpy as np
import pandas as pd

from portfolio.inference import load_inference_config
from portfolio.reporting import (_phi, _phi_inv, appraisal, calendar_year_intervals, convexity, deflated_sharpe,
                                 drawdowns, monthly, terminal_liquidation, two_factor_alpha)

CFG = load_inference_config()
DAYS = pd.date_range("2021-01-01", periods=1200, freq="D")


def series(values, start="2021-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_a_fixed_base_drawdown_can_exceed_the_starting_capital_while_the_compounded_one_cannot():
    r = series([0.5, -0.6, -0.6, 0.1])
    out = drawdowns(r)
    assert np.isclose(out["fixed_base_drawdown"], -1.2)                 # +50% then two -60% losses on a refilled base
    assert np.isclose(out["compounded_drawdown"], 1.5 * 0.4 * 0.4 / 1.5 - 1) and out["compounded_drawdown"] > -1
    assert out["fixed_base_peak"] == "2021-01-01" and out["fixed_base_trough"] == "2021-01-03"


def test_appraisal_ratio_is_alpha_over_residual_volatility():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0005, 0.03, 1200)
    y = 0.001 + 0.5 * x + rng.normal(0, 0.01, 1200)
    out = appraisal(pd.Series(y, index=DAYS), pd.Series(x, index=DAYS))
    assert abs(out["alpha_annual"] - 0.365) < 0.25 and abs(out["beta"] - 0.5) < 0.03
    assert np.isclose(out["appraisal_ratio"], out["alpha_annual"] / out["residual_vol_annual"])


def test_convexity_recovers_a_planted_square_term_and_the_centring_keeps_the_intercept():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.03, 1200)
    y = 0.0002 + 0.2 * x + 3.0 * (x ** 2 - np.mean(x ** 2)) + rng.normal(0, 0.002, 1200)
    out = convexity(pd.Series(y), pd.Series(x), lags=5, periods_per_year=365)
    assert abs(out["gamma"] - 3.0) < 0.3 and out["gamma_t"] > 10 and abs(out["beta"] - 0.2) < 0.02
    intercepts = []                                                     # the intercept is unbiased (one seed is noisy)
    for seed in range(40):
        g = np.random.default_rng(seed)
        xs = g.normal(0, 0.03, 1200)
        ys = 0.0002 + 0.2 * xs + 3.0 * (xs ** 2 - np.mean(xs ** 2)) + g.normal(0, 0.002, 1200)
        intercepts.append(convexity(pd.Series(ys), pd.Series(xs), lags=5, periods_per_year=365)["alpha_c_annual"])
    assert abs(np.mean(intercepts) - 0.0002 * 365) < 0.012              # standard error of the mean of 40 is ~0.003


def test_monthly_sums_are_non_overlapping_calendar_months():
    r = series(np.ones(62) * 0.01)                                        # Jan (31) + Feb (28) + 3 days of Mar
    m = monthly(r)
    assert list(m.round(2)) == [0.31, 0.28, 0.03]


def test_the_inverse_normal_is_accurate():
    for p in (1e-6, 0.001, 0.02, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert abs(_phi(_phi_inv(p)) - p) < 1e-8


def test_deflated_sharpe_falls_with_more_trials_and_reduces_to_psr_for_one_trial():
    rng = np.random.default_rng(3)
    r = series(rng.normal(0.001, 0.02, 1500))
    one = deflated_sharpe(r, 1)
    assert np.isclose(one["dsr"], one["psr_vs_zero"]) and one["sr0_annual"] == 0.0
    values = [deflated_sharpe(r, n)["dsr"] for n in (2, 10, 50, 500)]
    assert all(a > b for a, b in zip(values, values[1:]))
    strong = series(rng.normal(0.01, 0.02, 1500))                       # a very large edge survives many trials
    assert deflated_sharpe(strong, 500)["dsr"] > 0.99
    assert deflated_sharpe(series(rng.normal(0.0, 0.02, 1500)), 35)["dsr"] < 0.5


def test_two_factor_alpha_removes_both_exposures():
    rng = np.random.default_rng(4)
    b, t = rng.normal(0, 0.03, 1200), rng.normal(0, 0.02, 1200)
    s = 0.3 * b + 0.8 * t + rng.normal(0, 0.003, 1200)
    out = two_factor_alpha(pd.Series(s), pd.Series(b), pd.Series(t), lags=5)
    assert abs(out["beta_benchmark"] - 0.3) < 0.02 and abs(out["beta_btc_trend"] - 0.8) < 0.02 and abs(out["alpha_t"]) < 3


def test_calendar_years_sum_the_year_and_bracket_it():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"a": rng.normal(0.001, 0.02, 800)}, index=pd.date_range("2021-07-05", periods=800))
    out = calendar_year_intervals(frame, CFG, seed=1, reps=400)
    assert set(out) == {2021, 2022, 2023}
    for year, cell in out.items():
        block = frame[frame.index.year == year]
        assert cell["days"] == len(block) and np.isclose(cell["a"]["return"], block.a.sum())
        assert cell["a"]["ci90"][0] < cell["a"]["return"] < cell["a"]["ci90"][1]


def test_terminal_liquidation_costs_the_final_gross_notional():
    ledger = pd.DataFrame({"holding_period_start": pd.to_datetime(["2026-08-29", "2026-08-30", "2026-08-30"], utc=True),
                           "notional_after": [999.0, 1000.0, -500.0]})
    out = terminal_liquidation(ledger, 10.0, days=1883)
    assert out["final_gross_notional"] == 1500.0 and np.isclose(out["liquidation_cost"], 1.5)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
