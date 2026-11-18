"""The registered reporting supplement (pre-registration Part 7), computed from the SEALED registered run.

Reads only the stored results (verified against SHA256SUMS first) and reruns no strategy. The one
recomputation, the per-speed forecast diagnostics, rebuilds the signal from the frozen code and checks it
against the sealed decisions before using it. Explanatory and descriptive: nothing here is a second
definition of success, and nothing here can change H1 or H2.
"""
import json
from math import erf, exp, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio.freeze as fz
from portfolio.inference import (_bootstrap_means, _interval, alpha_beta, load_inference_config, newey_west_alpha,
                                 _require_returns)

EULER_GAMMA = 0.5772156649015329
ROOT = Path(__file__).resolve().parent.parent


def _phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _phi_inv(p):
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    lo = 0.02425
    if p < lo:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - lo:
        return -_phi_inv(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ------------------------------------------------------------------ drawdown under both conventions
def drawdowns(r: pd.Series) -> dict:
    """Worst peak-to-trough loss two ways. Fixed base (the primary accounting): cumulative P&L in units of
    the starting capital, which is topped up to the base at every monthly reset, so losses can exceed 100%.
    Compounded (display only): growth of 1 with gains and losses reinvested."""
    cum = r.cumsum()
    dd = cum - cum.cummax()
    trough = dd.idxmin()
    peak = cum[:trough].idxmax()
    growth = (1 + r).cumprod()
    return {"fixed_base_drawdown": float(dd.min()), "fixed_base_peak": str(peak.date()), "fixed_base_trough": str(trough.date()),
            "fixed_base_total_pnl": float(cum.iloc[-1]),
            "compounded_drawdown": float((growth / growth.cummax() - 1).min()), "compounded_end": float(growth.iloc[-1])}


# ------------------------------------------------------------------ alpha, appraisal, convexity, two-factor
def _ols_nw(y: np.ndarray, X: np.ndarray, lags: int) -> tuple:
    inv = np.linalg.inv(X.T @ X)
    coef = inv @ X.T @ y
    u = X * (y - X @ coef)[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        g = u[lag:].T @ u[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    se = np.sqrt(np.diag(inv @ S @ inv))
    resid = y - X @ coef
    return coef, se, resid


def appraisal(strategy: pd.Series, benchmark: pd.Series, annualisation: int = 365) -> dict:
    """alpha (annual), beta, residual volatility (annual) and the appraisal ratio alpha / residual vol."""
    y, x = _require_returns("strategy", strategy), _require_returns("benchmark", benchmark)
    a, b = alpha_beta(y, x)
    resid_vol = float(np.std(y - a - b * x, ddof=2) * sqrt(annualisation))
    return {"alpha_annual": a * annualisation, "beta": b, "residual_vol_annual": resid_vol,
            "appraisal_ratio": a * annualisation / resid_vol}


def monthly(r: pd.Series) -> pd.Series:
    """Non-overlapping calendar months: the sum of daily returns (fixed base)."""
    return r.groupby(r.index.to_period("M")).sum()


def convexity(strategy: pd.Series, benchmark: pd.Series, lags: int, periods_per_year: int) -> dict:
    """Treynor-Mazuy with a centred square: r_s = a_c + b r_m + g [r_m^2 - mean(r_m^2)] + e, Newey-West
    standard errors. g > 0 means the strategy gains more in large market moves of either sign."""
    y, x = strategy.to_numpy(float), benchmark.to_numpy(float)
    X = np.column_stack([np.ones(len(x)), x, x ** 2 - np.mean(x ** 2)])
    coef, se, _ = _ols_nw(y, X, lags)
    return {"alpha_c_annual": float(coef[0] * periods_per_year), "beta": float(coef[1]), "gamma": float(coef[2]),
            "gamma_t": float(coef[2] / se[2]), "alpha_c_t": float(coef[0] / se[0]), "n": len(y), "nw_lags": lags}


def two_factor_alpha(strategy: pd.Series, benchmark: pd.Series, btc_trend: pd.Series, lags: int = 21,
                     annualisation: int = 365) -> dict:
    """S15: alpha beyond both passive basket beta and BTC-only trend (secondary, never gates the headline)."""
    X = np.column_stack([np.ones(len(strategy)), benchmark.to_numpy(float), btc_trend.to_numpy(float)])
    coef, se, _ = _ols_nw(strategy.to_numpy(float), X, lags)
    t = coef[0] / se[0]
    return {"alpha_annual": float(coef[0] * annualisation), "alpha_t": float(t), "p_one_sided": float(1 - _phi(t)),
            "beta_benchmark": float(coef[1]), "beta_btc_trend": float(coef[2]), "nw_lags": lags}


# ------------------------------------------------------------------ deflated Sharpe
def deflated_sharpe(returns: pd.Series, n_trials: int, annualisation: int = 365, sr_variance: float | None = None) -> dict:
    """Bailey and Lopez de Prado (2014). SR0 is the Sharpe the best of `n_trials` independent no-skill trials
    would reach by luck; DSR is the probability the observed Sharpe exceeds it, allowing for skew and fat
    tails. `sr_variance` (per-period units) defaults to the null sampling variance 1/(T-1), an assumption:
    the variance of Sharpe ratios across the actual trials is not known. Partial lower-bound correction."""
    r = _require_returns("returns", returns)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4))
    v = 1.0 / (T - 1) if sr_variance is None else sr_variance
    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = sqrt(v) * ((1 - EULER_GAMMA) * _phi_inv(1 - 1 / n_trials) + EULER_GAMMA * _phi_inv(1 - 1 / (n_trials * exp(1))))
    denom = sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    psr = lambda target: _phi((sr - target) * sqrt(T - 1) / denom)
    return {"n_trials": n_trials, "sharpe_annual": float(sr * sqrt(annualisation)), "sr0_annual": float(sr0 * sqrt(annualisation)),
            "psr_vs_zero": float(psr(0.0)), "dsr": float(psr(sr0)), "skew": skew, "kurtosis": kurt, "T": T,
            "sr_variance_assumption": "null sampling variance 1/(T-1)" if sr_variance is None else "supplied"}


# ------------------------------------------------------------------ calendar years with intervals
def calendar_year_intervals(frame: pd.DataFrame, cfg, seed: int, reps: int | None = None) -> dict:
    """Per calendar year: the year's arithmetic return (sum of daily returns) with a 90% interval from the
    stationary block bootstrap within that year. Descriptive, not tests (protocol Part 7, item 5)."""
    reps = cfg.replications if reps is None else reps
    out = {}
    for year, block in frame.groupby(frame.index.year):
        days = len(block)
        boots = _bootstrap_means({c: block[c] for c in block}, cfg, seed, f"year:{year}", cfg.mean_block_days, reps)
        out[int(year)] = {"days": days, **{c: {"return": float(block[c].sum()),
                                                "ci90": tuple(float(v) * days / cfg.annualisation for v in
                                                              np.quantile(boots[c], [cfg.alpha_level, 1 - cfg.alpha_level]))}
                                           for c in block}}
    return out


# ------------------------------------------------------------------ terminal liquidation, costs, H2, coverage, forecasts
def terminal_liquidation(ledger: pd.DataFrame, cost_bp: float, days: int, capital: float = 100_000.0) -> dict:
    """Cost of closing every position at the final mark, as a sensitivity beside mark-to-final-close."""
    last = ledger[ledger.holding_period_start == ledger.holding_period_start.max()]
    gross = float(last.notional_after.abs().sum())
    cost = gross * cost_bp / 10_000.0
    return {"final_gross_notional": gross, "liquidation_cost": cost, "cost_share_of_capital": cost / capital,
            "annual_return_effect": cost / capital * 365 / days}


def cost_comparison(folder: Path) -> dict:
    out = {}
    for name, bp in (("basket_buffered_5", 5), ("basket_buffered_10", 10), ("basket_buffered_20", 20)):
        p = pd.read_csv(folder / name / "paired.csv.gz", index_col=0, parse_dates=True)
        nw = newey_west_alpha(p.strategy_net, p.benchmark_net, 21, 365)
        r = p.strategy_net
        out[bp] = {"net_annual": float(r.mean() * 365), "sharpe": float(r.mean() * 365 / (r.std() * sqrt(365))),
                   "alpha_annual": nw["alpha_annual"], "alpha_t": nw["t"]}
    return out


def h2_diagnostics_sealed(folder: Path) -> dict:
    basket = pd.read_csv(folder / "basket_buffered_10" / "daily.csv.gz", index_col=0, parse_dates=True)
    btc = pd.read_csv(folder / "btc_buffered_10" / "daily.csv.gz", index_col=0, parse_dates=True)
    sb, st = (json.loads((folder / n / "stats.json").read_text()) for n in ("basket_buffered_10", "btc_buffered_10"))
    vol = lambda r: float(r.std() * sqrt(365))
    v_b, v_t = vol(basket.net_return), vol(btc.net_return)
    gap = basket.net_return - btc.net_return
    matched = basket.net_return - btc.net_return * (v_b / v_t)
    return {"gap_annual": float(gap.mean() * 365), "realised_vol": {"basket": v_b, "btc": v_t},
            "mean_abs_exposure": {"basket": sb["mean_abs_target_exposure"], "btc": st["mean_abs_target_exposure"]},
            "turnover_per_year": {"basket": sb["turnover_per_year"], "btc": st["turnover_per_year"]},
            "ex_post_vol_matched_gap_annual": float(matched.mean() * 365),
            "strategy_correlation": float(np.corrcoef(basket.net_return, btc.net_return)[0, 1])}


def coverage(start, end) -> dict:
    """Share of the true top-10 market cap (stablecoins and wrapped assets excluded) with no tradable Binance
    perp on each monthly reconstitution date. Universe data only."""
    from data.universe.selector import EXCLUDED_CMC_IDS, UniverseSelector
    from portfolio.basket import reconstitution_dates
    from datetime import timedelta
    selector = UniverseSelector(min_history_days=0)
    rows = []
    for trade_date in reconstitution_dates(end):
        if trade_date < start:
            continue
        day = trade_date - timedelta(days=1)
        top = [r for r in selector._by_date[day].itertuples() if r.cmc_id not in EXCLUDED_CMC_IDS][:10]
        no_perp = [r for r in top if selector.screen(r, day)[0] in ("outside_pool", "no_contract", "not_trading")]
        total = sum(r.market_cap for r in top)
        rows.append((trade_date, sum(r.market_cap for r in no_perp) / total, len(no_perp)))
    shares = pd.Series({d: s for d, s, _ in rows})
    return {"dates": len(rows), "mean_share": float(shares.mean()), "max_share": float(shares.max()),
            "dates_with_any": int(sum(n > 0 for _, _, n in rows))}


def forecast_diagnostics(folder: Path) -> dict:
    """Mean |forecast| per speed and for the blend, and the share of days at the +-20 cap, over the scored
    days. Rebuilds the signal with the frozen code and checks it against the sealed decisions first."""
    from portfolio.basket import build_inputs
    from portfolio.trend_signal import basket_forecast, load_signal_config
    decisions = pd.read_csv(folder / "basket_buffered_10" / "decisions.csv.gz", index_col=0, parse_dates=True)
    _, _, _, _, index = build_inputs()
    signal = basket_forecast(index, load_signal_config())
    signal.index = pd.to_datetime(signal.index)
    rebuilt = signal.loc[decisions.index]
    if not np.allclose(rebuilt.forecast.to_numpy(), decisions.forecast.to_numpy(), rtol=1e-10, atol=1e-10):
        raise ValueError("the rebuilt signal does not match the sealed decisions")
    cols = [c for c in signal.columns if c.startswith("f_")]
    return {"mean_abs": {c[2:]: float(rebuilt[c].abs().mean()) for c in cols} | {"blend": float(rebuilt.forecast.abs().mean())},
            "share_at_cap": {c[2:]: float((rebuilt[c].abs() >= 20.0).mean()) for c in cols} | {"blend": float(rebuilt.forecast_cap_hit.mean())},
            "vol_floor_bind_rate": float(rebuilt.vol_floor_bound.mean()), "x_star_cap_hit_rate": float(rebuilt.x_star_cap_hit.mean())}


def supplement(folder: Path | None = None, reps: int | None = None) -> dict:
    folder = folder or next(fz.DEFAULT_STORE.glob("*"))
    if not fz.verify(folder):
        raise ValueError("the stored results no longer match SHA256SUMS")
    manifest = json.loads((folder / "manifest.json").read_text())
    cfg, seed = load_inference_config(), manifest["seed"]
    basket = pd.read_csv(folder / "basket_buffered_10" / "paired.csv.gz", index_col=0, parse_dates=True)
    btc = pd.read_csv(folder / "btc_buffered_10" / "paired.csv.gz", index_col=0, parse_dates=True)
    s, b, t = basket.strategy_net, basket.benchmark_net, btc.strategy_net
    ledger = pd.read_csv(folder / "basket_buffered_10" / "ledger.csv.gz", parse_dates=["holding_period_start"])
    window = [pd.Timestamp(d).date() for d in manifest["window"]]
    trials = sorted({manifest["trial_counts"][k] for k in ("low", "central", "high")} | {20, 35, 60})
    return {
        "drawdowns": {"strategy": drawdowns(s), "benchmark": drawdowns(b), "btc_strategy": drawdowns(t)},
        "appraisal": appraisal(s, b),
        "convexity": {"daily": convexity(s, b, lags=21, periods_per_year=365),
                      "monthly": convexity(monthly(s), monthly(b), lags=3, periods_per_year=12)},
        "two_factor_S15": two_factor_alpha(s, b, t),
        "deflated_sharpe": {n: deflated_sharpe(s, n) for n in trials},
        "calendar_years": calendar_year_intervals(pd.DataFrame({"strategy": s, "benchmark": b, "btc_strategy": t,
                                                                "strategy_minus_benchmark": s - b}), cfg, seed, reps),
        "terminal_liquidation": terminal_liquidation(ledger, 10.0, len(s)),
        "cost_comparison": cost_comparison(folder),
        "h2_diagnostics": h2_diagnostics_sealed(folder),
        "coverage": coverage(window[0], window[1]),
        "forecast_diagnostics": forecast_diagnostics(folder),
        "luna": "LUNAUSDT was never held in the registered monthly basket (basket_event_inventory.md): its contribution "
                "is zero and the settlement counterfactual does not apply.",
        "not_computable_here": ["S13/S14 measured-cost runs (Abdi-Ranaldo, aggTrades): not implemented",
                                "the registered secondary strategy runs S2-S10 and S16 were not executed in the sealed run"],
        "record": {"source": str(folder.relative_to(ROOT)), "seed": seed, "reps": reps or cfg.replications,
                   "convexity_nw_lags": "daily 21, monthly 3 (a reporting choice, not registered)"},
    }


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    out = supplement()
    path = ROOT / "docs/september/reporting_supplement_2026-10-04.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("written", path.relative_to(ROOT))
