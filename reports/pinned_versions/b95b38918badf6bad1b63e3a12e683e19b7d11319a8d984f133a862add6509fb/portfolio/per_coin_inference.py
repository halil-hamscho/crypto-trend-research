"""The per-coin registered analysis: the inference adapter, the identity checks and the verdict.

Design: docs/october/per_coin_run_driver_design_2026-10-24.md §3-4.3; protocol Part 9 read through
Clarification 4. The sealed statistics are reused (`bootstrap_alpha`, `newey_west_alpha`, `gap_test`,
`sharpe_intervals`, `alpha_attribution`, `classify_full_vs_buffered`, `gross_tripwire`,
`deflated_sharpe`) with a per-coin InferenceConfig built from the per-coin YAML. The sealed
`evaluate()`, `classify_h1` and `check_experiment_identity` encode the basket study and are not used.
"""
from math import isfinite

import numpy as np
import pandas as pd

from portfolio.inference import (InferenceConfig, alpha_attribution, bootstrap_alpha, classify_full_vs_buffered,
                                 gap_test, gross_tripwire, newey_west_alpha, sharpe_intervals)
from portfolio.per_coin_attribution import validity
from portfolio.per_coin_runs import BY_NAME, CONFIG
from portfolio.per_coin_sizing import idm
from portfolio.reporting import deflated_sharpe

GAP_HURDLE_FIELDS = ("clears_hurdle", "rules_out_hurdle", "hurdle")
FAIL, INCONCLUSIVE, H1_PASSES = "FAIL", "INCONCLUSIVE", "H1 PASSES"
FRAGILE, NOT_PER_COIN, COLLAPSED, LARGE_CLAIM_REJECTED = (
    "FRAGILE", "NOT PER-COIN", "COLLAPSED TO MARKET TIMING", "LARGE CLAIM REJECTED")
BLOCKING = (FRAGILE, NOT_PER_COIN, COLLAPSED)


class IdentityError(ValueError):
    """Two compared books differ in something their comparison does not allow."""


# ------------------------------------------------------------------ §3.1 the adapter
def load_per_coin_inference_config(raw: dict | None = None) -> InferenceConfig:
    """The registered settings from the per-coin YAML. The 5% level comes from the pass rule
    (p05_alpha_gt_0); the stress cost is the 20 bp of the verdict's stress test (30 bp is a
    sensitivity); there is no gap hurdle in this study (NaN, and stripped from every gap report)."""
    raw = CONFIG.raw if raw is None else raw
    inf = raw["inference"]
    for key, want in (("bootstrap", "stationary"), ("pass_rule", "p05_alpha_gt_0"), ("primary", "alpha_vs_K1_net")):
        if inf.get(key) != want:
            raise ValueError(f"inference.{key} is {inf.get(key)!r}; this adapter implements {want!r}")
    cfg = InferenceConfig(alpha_level=0.05, annualisation=int(raw["risk"]["annualisation_days"]),
                          replications=int(inf["replications"]), mean_block_days=int(inf["mean_block_days"]),
                          sensitivity_block_days=tuple(int(b) for b in inf["sensitivity_block_days"]),
                          newey_west_lags=int(inf["newey_west_lags"]), hurdle_annual=float("nan"),
                          primary_bp=float(raw["costs"]["primary_bp_per_side"]),
                          stress_bp=float(min(raw["costs"]["stress_bp_per_side"])),
                          cash_rate=float(raw["risk"]["cash_rate"]))
    if cfg.cash_rate != 0.0 or not {40, 60} <= set(cfg.sensitivity_block_days):
        raise ValueError("the registered cash rate is 0 and the 40- and 60-day blocks are required")
    return cfg


def _gap(gap: pd.Series, cfg, seed, reps, name) -> dict:
    out = gap_test(gap.rename(name), cfg, seed, reps=reps)
    return {k: v for k, v in out.items() if k not in GAP_HURDLE_FIELDS}


def _alpha(s, b, cfg, seed, reps, name, block=None) -> dict:
    return {k: v for k, v in bootstrap_alpha(s, b, cfg, seed, block, reps, name=name).items() if k != "alphas"}


def finite(name: str, value) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} is not finite ({value}): an undefined statistic cannot enter the verdict")
    return value


# ------------------------------------------------------------------ §3.3 the verdict
def verdict(alpha, p_low, p_low_40, p_low_60, p_low_stress, h2_p_low, two_sided_share, sharpe_p95) -> dict:
    """Clarification 4. A headline from H1 at the primary settings, and every warning that holds,
    each evaluated on its own. Thresholds exactly as written: '> 0', '< 20%' and '< 1.0' are strict."""
    v = {k: finite(k, x) for k, x in dict(alpha=alpha, p_low=p_low, p_low_40=p_low_40, p_low_60=p_low_60,
                                          p_low_stress=p_low_stress, h2_p_low=h2_p_low,
                                          two_sided_share=two_sided_share, sharpe_p95=sharpe_p95).items()}
    headline = FAIL if v["alpha"] <= 0 else INCONCLUSIVE if v["p_low"] <= 0 else H1_PASSES
    min_share = float(CONFIG.raw["inference"]["validity"]["min_share_of_two_sided_days"])
    large = float(CONFIG.raw["inference"]["large_claim_sharpe"])
    warnings = [w for w, holds in (
        (FRAGILE, not (v["p_low_stress"] > 0 and v["p_low_40"] > 0 and v["p_low_60"] > 0)),
        (NOT_PER_COIN, not v["h2_p_low"] > 0),
        (COLLAPSED, v["two_sided_share"] < min_share),
        (LARGE_CLAIM_REJECTED, v["sharpe_p95"] < large)) if holds]
    return {"headline": headline, "warnings": warnings,
            "pass": headline == H1_PASSES and not any(w in BLOCKING for w in warnings), "inputs": v}


# ------------------------------------------------------------------ §4.3 identity
COMPARED_PARAMETERS = ("universe", "forecast", "sizing", "lag", "clock", "cost_bp", "band")
FORECAST_HASH = "forecast"


def _fields(result) -> dict:
    r = result.record
    out = {f"parameter:{k}": r["parameters"][k] for k in COMPARED_PARAMETERS}
    out.update({"window": tuple(r["window"]), "capital": r["capital"], "contracts": tuple(r["contracts"])})
    out.update({f"hash:{k}": v for k, v in r["hashes"].items()})
    return out


def comparison_rule(kind: str) -> tuple:
    """(fields that MUST differ, fields that MAY differ besides them). Everything else must be equal."""
    if kind == "twin":                        # H1-type: the forecast only
        return {"parameter:forecast", f"hash:{FORECAST_HASH}"}, set()
    if kind == "forecast":                    # H3 vs K3, H4 vs K5
        return {"parameter:forecast", f"hash:{FORECAST_HASH}"}, set()
    if kind == "btc":                         # H3 vs K2: universe and forecast, and so every derived input
        return {"parameter:universe", "parameter:forecast"}, {"contracts", *(f"hash:{k}" for k in HASH_KEYS)}
    if kind == "buffer":                      # full vs buffered: the band only
        return {"parameter:band"}, set()
    raise ValueError(f"unknown comparison kind {kind!r}")


HASH_KEYS = ("forecast", "sigma", "price", "mark_price", "rho_bar", "eligibility", "fill_prices",
             "decision_prices", "tradable", "settlements")


def check_identity(a, b, kind: str) -> None:
    """Raises IdentityError unless `a` and `b` differ in exactly what `kind` allows (§4.3)."""
    must, may = comparison_rule(kind)
    fa, fb = _fields(a), _fields(b)
    differing = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
    unexpected, missing = differing - must - may, must - differing
    if unexpected or missing:
        raise IdentityError(f"{a.spec.name} vs {b.spec.name} ({kind}): differs in {sorted(unexpected)}"
                            if unexpected else f"{a.spec.name} vs {b.spec.name} ({kind}): does not differ in "
                                                f"{sorted(missing)}, which this comparison must change")
    if not a.daily.index.equals(b.daily.index):
        raise IdentityError(f"{a.spec.name} vs {b.spec.name}: not the same days")


def comparisons(results: dict) -> list:
    """Every registered comparison among the books present: (a, b, kind)."""
    out = [(r.name, r.twin, "twin") for r in (BY_NAME[n] for n in results) if r.twin and r.twin in results]
    out += [(a, b, kind) for a, b, kind in (("strategy", "k2", "btc"), ("strategy", "k3", "forecast"),
                                            ("strategy", "k5", "forecast"), ("strategy", "full_rebalance", "buffer"))
            if a in results and b in results]
    return out


def check_all_identities(results: dict) -> list:
    """Every registered comparison among the books present; returns what was checked."""
    done = []
    for a, b, kind in comparisons(results):
        check_identity(results[a], results[b], kind)
        done.append(f"{a} vs {b}: {kind}")
    return done


# ------------------------------------------------------------------ §3.2-3.4 the analysis
def _layers(r) -> pd.DataFrame:
    return r.daily[["price_return", "funding_return", "cost_return", "net_return"]]


def _annual(series: pd.Series, cfg) -> float:
    return float(series.mean() * cfg.annualisation)


def _always_reported(results: dict, cfg) -> dict:
    s = results["strategy"]
    rows = s.rows
    ladder = {name: {layer: _annual(results[name].daily[f"{layer}_return"], cfg) for layer in ("price", "gross", "net")}
              for name in ("strategy", "full_rebalance") if name in results}
    turnover = {k: _annual(s.daily[k], cfg) for k in ("turnover", "turnover_membership", "turnover_signal",
                                                     "turnover_instrument_end")}
    cap = s.record["capital"]
    by_coin = (rows.groupby("contract").net_pnl.sum() / cap).to_dict()
    by_year = (s.daily.net_return.groupby(pd.Index([d.year for d in s.daily.index])).sum()).to_dict()
    luna = [c for c in s.record["contracts"] if "LUNA" in c]
    t = s.tables
    eligible_count = t.eligible.sum(axis=1)
    idm_path = pd.Series([idm(int(n), float(t.rho_bar.get(x, np.nan))) for x, n in eligible_count.items()],
                         index=eligible_count.index)
    fc = t.forecast.where(t.eligible)
    cap_hits = {c: {"long": float((fc[c] >= 20.0).sum() / max(fc[c].notna().sum(), 1)),
                    "short": float((fc[c] <= -20.0).sum() / max(fc[c].notna().sum(), 1))} for c in fc.columns}
    net = s.daily.net_return
    return {
        "ladder": ladder, "turnover": turnover,
        "funding_by_leg": {k: float(s.daily[k].sum() / cap) for k in ("funding_long", "funding_short")},
        "split": {k: _annual(s.split[k], cfg) for k in ("price_market", "price_coin", "price_ineligible",
                                                        "funding_market", "funding_coin", "funding_ineligible",
                                                        "costs", "coin_net_of_all_costs")},
        "per_coin": by_coin, "per_year": {int(k): float(v) for k, v in by_year.items()},
        "luna": {"contracts": luna, "contribution": float(sum(by_coin.get(c, 0.0) for c in luna)),
                 "evidence": "no LUNA contract among the book's contracts" if not luna else "measured"},
        "idm_path": {"median": float(idm_path.median()), "min": float(idm_path.min()), "max": float(idm_path.max())},
        "rho_bar_path": {"median": float(t.rho_bar.median()), "min": float(t.rho_bar.min()), "max": float(t.rho_bar.max())},
        "eligible_count": {"median": float(eligible_count.median()), "min": int(eligible_count.min()),
                           "max": int(eligible_count.max())},
        "returns": {"arithmetic_annual": _annual(net, cfg),
                    "compounded_annual": float((1 + net).prod() ** (cfg.annualisation / len(net)) - 1)},
        "execution_notes": {"model": len(s.record["model_notes"]), "ledger": len(s.record["ledger_notes"])},
        "forecast_cap_hits": cap_hits,
    }


def evaluate_per_coin(results: dict, seed: int, cfg: InferenceConfig | None = None, reps: int | None = None,
                      k4_reference: dict | None = None) -> dict:
    """The registered analysis (§3) on stage-A results, after every identity check (§4.3). `reps`
    overrides the replications for tests only, and the record says so."""
    cfg = cfg or load_per_coin_inference_config()
    reps = cfg.replications if reps is None else reps
    identities = check_all_identities(results)
    net = {n: r.daily.net_return for n, r in results.items()}
    s, k1 = net["strategy"], net["k1"]

    h1 = _alpha(s, k1, cfg, seed, reps, "h1", cfg.mean_block_days)
    blocks = {b: _alpha(s, k1, cfg, seed, reps, "h1", b) for b in cfg.sensitivity_block_days}
    stress = _alpha(net["cost_20bp"], net["cost_20bp_k1"], cfg, seed, reps, "h1stress")
    h2 = _gap(results["strategy"].split.price_coin, cfg, seed, reps, "h2")
    h2_net = _gap(results["strategy"].split.coin_net_of_all_costs, cfg, seed, reps, "h2net")
    val = validity(results["strategy"].split.two_sided)
    sharpe = sharpe_intervals({n: net[n] for n in net}, cfg, seed, reps=reps)
    out = {
        "h1": h1, "h1_blocks": blocks, "h1_stress_20bp": stress,
        "newey_west": newey_west_alpha(s, k1, cfg.newey_west_lags, cfg.annualisation),
        "h2": h2, "h2_net_of_all_costs": h2_net, "validity": val, "sharpe": sharpe,
        "h3": {k: _alpha(s, net[k], cfg, seed, reps, f"h3{k}") for k in ("k2", "k3") if k in net},
        "h4": _gap(s - net["k5"], cfg, seed, reps, "h4") if "k5" in net else None,
        "attribution": alpha_attribution(_layers(results["strategy"]), _layers(results["k1"]), cfg.annualisation),
        "deflated_sharpe": {n: deflated_sharpe(s, n, cfg.annualisation)
                            for n in (int(CONFIG.raw["inference"]["trials_for_dsr"]),
                                      *CONFIG.raw["inference"]["trials_bracket"])},
        "sensitivities": {r: {"alpha_vs_twin": _alpha(net[r], net[BY_NAME[r].twin], cfg, seed, reps, f"sens:{r}"),
                              "sharpe": sharpe[r]["sharpe"], "twin_sharpe": sharpe[BY_NAME[r].twin]["sharpe"]}
                          for r in results if BY_NAME[r].counted},
    }
    if "full_rebalance" in results:
        full = out["sensitivities"]["full_rebalance"]["alpha_vs_twin"]["alpha_annual"]
        out["full_vs_buffered"] = {
            "alpha_full": full, "alpha_buffered": h1["alpha_annual"],
            "reading": classify_full_vs_buffered(full, h1["alpha_annual"]),
            "gross_tripwire_audit": gross_tripwire(results["strategy"].daily.gross_return.mean(),
                                                   results["full_rebalance"].daily.gross_return.mean())}
    out["verdict"] = verdict(h1["alpha_annual"], h1["alpha_p_low"], blocks[40]["alpha_p_low"], blocks[60]["alpha_p_low"],
                             stress["alpha_p_low"], h2["p_low"], val["two_sided_share"],
                             sharpe["strategy"]["ci90"][1])
    out["always_reported"] = _always_reported(results, cfg)
    out["k4_reference"] = k4_reference
    out["record"] = {"seed": seed, "replications": reps, "registered_replications": cfg.replications,
                     "identity_checks": identities, "runs": sorted(results),
                     "not_registered_because": [] if reps == cfg.replications
                     else [f"replications {reps} != {cfg.replications}"]}
    return out
