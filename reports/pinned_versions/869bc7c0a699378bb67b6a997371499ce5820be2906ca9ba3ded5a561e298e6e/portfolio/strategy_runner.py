"""The registered strategy runner: the signal's ideal exposure, executed through the benchmark's ledger.

An assembler, not new accounting. Every day: the frozen signal gives the ideal exposure x*, an
execution policy (`full` or `buffered`) turns x* and the exposure currently held into today's target,
and `run_ledger` (notional mode) books it exactly as it books the benchmark. Returns are paired with
the same-policy benchmark over EVERY scored day, flat days included. Parameters come from the frozen
config only; the run writes a record that reproduces it.
"""
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.basket import (ANNUALISATION, CAPITAL_POLICIES, INSTRUMENTS, SCORED_END, SCORED_START, benchmark,
                              build_inputs, ladder, run_ledger)
from portfolio.trend_signal import SignalConfig, basket_forecast, load_signal_config

POLICIES = ("full", "buffered")
TRADE_EPSILON = 1e-9      # a leg-day counts as a trade above this share of the capital base


def buffered_target(held: float, x_star: float, standard: float, band_half_width: float, cap: float) -> float:
    """The registered no-trade band: hold while within +-band of the ideal, else trade to the nearest edge.

    band = band_half_width x standard exposure, centred on x*. All in active-allocation units.
    The result is capped again, because the held exposure can itself sit outside the cap.
    """
    band = band_half_width * standard
    if abs(held - x_star) <= band:
        target = held
    elif held > x_star + band:
        target = x_star + band
    else:
        target = x_star - band
    return float(np.clip(target, -cap, cap))


def full_target(x_star: float, cap: float) -> float:
    return float(np.clip(x_star, -cap, cap))


def make_exposure_fn(signal: pd.DataFrame, config: SignalConfig, policy: str, decisions: list,
                     signal_lag_days: int = 0):
    """The ledger's `exposure_fn`, closed over the signal table and a decision log.

    It reads the signal at the decision day (or `signal_lag_days` rows earlier: the one-day-lag timing
    sensitivity) and nothing else from the future. It raises if the signal is missing on a scored day.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if signal_lag_days < 0:
        raise ValueError("signal_lag_days must be >= 0")
    position = {d: i for i, d in enumerate(signal.index)}

    def exposure_fn(day, held, base):
        i = position[day] - signal_lag_days
        if i < 0:
            raise ValueError(f"{day}: no signal {signal_lag_days} day(s) earlier")
        row = signal.iloc[i]
        x_star, standard = float(row.x_star), float(row.standard)
        if not (np.isfinite(x_star) and np.isfinite(standard)):
            raise ValueError(f"{day}: signal is not available (warm-up or invalid), x*={x_star}")
        target = (full_target(x_star, config.leverage_cap) if policy == "full" else
                  buffered_target(held, x_star, standard, config.band_half_width, config.leverage_cap))
        decisions.append({"day": day, "signal_day": signal.index[i], "held": held, "base": base, "x_star": x_star,
                          "standard": standard, "target": target, "forecast": float(row.forecast)})
        return target

    return exposure_fn


def daily_returns_over(ledger: pd.DataFrame, days: list, capital: float, capital_policy: str) -> pd.DataFrame:
    """One row per scored day, flat days included (the ledger writes none for a book that is flat
    and stays flat). A flat day has zero P&L and a base that continues: the fixed base, or under
    'equity' the previous base plus the previous day's net P&L."""
    by_day = ledger.groupby("holding_period_start").agg(
        price=("price_pnl", "sum"), funding=("funding_pnl", "sum"), fee=("fee", "sum"),
        slippage=("slippage", "sum"), net=("net_pnl", "sum"), base=("capital_base", "first"),
        exposure=("notional_after", "sum"))
    by_day.index = pd.DatetimeIndex(by_day.index).tz_convert(None).date
    rows, equity = [], capital
    for d in days:
        expected = capital if capital_policy == "fixed" else equity
        if d in by_day.index:
            row = by_day.loc[d].to_dict()
            if not np.isclose(row["base"], expected, rtol=1e-9):
                raise ValueError(f"{d}: ledger base {row['base']} differs from the continued base {expected}")
        else:
            row = {k: 0.0 for k in ("price", "funding", "fee", "slippage", "net", "exposure")} | {"base": expected}
        equity += row["net"]
        rows.append(row)
    daily = pd.DataFrame(rows, index=days)
    daily["cost"] = daily.fee + daily.slippage
    for layer in ("price", "funding", "cost", "net"):
        daily[f"{layer}_return"] = daily[layer] / daily.base
    daily["gross_return"] = daily.price_return + daily.funding_return
    return daily


@dataclass
class StrategyResult:
    ledger: pd.DataFrame
    decisions: pd.DataFrame      # one row per scored day: held, x*, standard, target (flat days included)
    daily: pd.DataFrame          # strategy, one row per scored day
    paired: pd.DataFrame         # strategy and same-policy benchmark, by day
    ladder: pd.DataFrame
    stats: dict
    record: dict
    policy: str
    capital_policy: str


def scored_days(marks: pd.DataFrame, start: date, end: date) -> list:
    return [d for d in marks.index if start <= d <= end]


def _fingerprint(inputs, config: SignalConfig) -> dict:
    members, marks, settlements, weights, index = inputs

    def digest(*frames):
        h = hashlib.sha256()
        for f in frames:
            h.update(pd.util.hash_pandas_object(f, index=True).to_numpy().tobytes())
        return h.hexdigest()

    return {
        "config_sha256": config.config_hash,
        "marks": digest(marks.fillna(-1.0)),
        "settlements": digest(*[settlements[c].drop(columns=["stamp"]).assign(
            stamp=settlements[c].stamp.astype("int64")) for c in sorted(settlements)]),
        "members": digest(*[members[d].assign(date=str(d)) for d in sorted(members)]),
        "reference_index": digest(index.to_frame("index").astype(float)),
    }


def _git() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    try:
        return {"commit": run("rev-parse", "HEAD").stdout.strip(),
                "dirty": bool(run("status", "--porcelain").stdout.strip())}
    except OSError:
        return {"commit": None, "dirty": None}


def run_strategy(policy: str = "buffered", capital_policy: str = "fixed", capital: float = 100_000.0,
                 start: date = SCORED_START, end: date = SCORED_END, inputs=None, config: SignalConfig | None = None,
                 signal_lag_days: int = 0, instrument: str = "basket",
                 cost_bp_per_side: float = 10.0, signal_override=None) -> StrategyResult:
    """One strategy run and its same-policy benchmark. `signal_lag_days=1` is the timing sensitivity.
    `instrument='btc'` is the H2 comparator: the same strategy on BTC alone (see build_inputs).
    `signal_override(signal, index) -> signal` is for TESTING ONLY (planting a deliberate leak to prove the
    pipeline can see one). A registered run never passes it; the record stores it and the run is never
    called registered."""
    if instrument not in INSTRUMENTS:
        raise ValueError(f"instrument must be one of {INSTRUMENTS}, got {instrument!r}")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if capital_policy not in CAPITAL_POLICIES:
        raise ValueError(f"capital_policy must be one of {CAPITAL_POLICIES}, got {capital_policy!r}")
    config = config or load_signal_config()
    inputs = inputs if inputs is not None else build_inputs(end, instrument)
    members, marks, settlements, weights, index = inputs
    signal = basket_forecast(index, config)
    if signal_override is not None:
        signal = signal_override(signal.copy(), index)

    decisions: list = []
    fn = make_exposure_fn(signal, config, policy, decisions, signal_lag_days)
    ledger = run_ledger({d: m for d, m in members.items() if d > start}, marks, settlements, weights, start, end,
                        capital=capital, mode="notional", capital_policy=capital_policy, exposure_fn=fn,
                        cost_bp_per_side=cost_bp_per_side)

    days = scored_days(marks, start, end)
    decision_table = pd.DataFrame(decisions).set_index("day")
    missing = [d for d in days if d not in decision_table.index]
    if missing:
        raise ValueError(f"no decision recorded on {len(missing)} scored day(s), first {missing[0]}")
    daily = daily_returns_over(ledger, days, capital, capital_policy)
    bench = benchmark(capital_policy, capital, start, end, inputs=inputs, cost_bp_per_side=cost_bp_per_side)
    paired = pd.DataFrame({
        **{f"strategy_{k}": daily[f"{k}_return"] for k in ("price", "gross", "net")},
        **{f"benchmark_{k}": bench.daily[f"{k}_return"] for k in ("price", "gross", "net")}})
    if paired.isna().any().any():
        raise ValueError("strategy and benchmark returns do not cover the same days")

    years = len(days) / ANNUALISATION
    traded = ledger[ledger.trade.abs() > TRADE_EPSILON * ledger.capital_base]
    scored_signal = signal.loc[[d for d in days]]
    stats = {
        "days": len(days), "flat_days": int((daily.exposure.abs() < 1e-12).sum()),
        "turnover_per_year": float((ledger.trade.abs() / ledger.capital_base).sum() / years),
        "trade_count": int(len(traded)), "trade_days": int(traded.holding_period_start.nunique()),
        "mean_abs_target_exposure": float(decision_table.loc[days, "target"].abs().mean()) if len(days) else 0.0,
        "mean_abs_forecast": float(scored_signal.forecast.abs().mean()),
        "forecast_cap_hit_rate": float(scored_signal.forecast_cap_hit.mean()),
        "x_star_cap_hit_rate": float(scored_signal.x_star_cap_hit.mean()),
        "leverage_cap_bind_rate": float((decision_table.loc[days, "target"].abs() >= config.leverage_cap - 1e-12).mean()),
        "vol_floor_bind_rate": float(scored_signal.vol_floor_bound.mean()),
        "funding_pnl_long_days": float(ledger[ledger.quantity_after > 0].funding_pnl.sum()),
        "funding_pnl_short_days": float(ledger[ledger.quantity_after < 0].funding_pnl.sum()),
        "identity_max_residual": float((ledger.net_pnl - (ledger.price_pnl + ledger.funding_pnl
                                                         - ledger.fee - ledger.slippage)).abs().max()),
    }
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {"policy": policy, "capital_policy": capital_policy, "capital": capital,
                       "start": str(start), "end": str(end), "signal_lag_days": signal_lag_days, "instrument": instrument, "cost_bp_per_side": cost_bp_per_side,
                       "signal_override": signal_override is not None,
                       "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()}},
        "git": _git(), "fingerprints": _fingerprint(inputs, config),
        "result_fingerprint": hashlib.sha256(pd.util.hash_pandas_object(
            ledger.drop(columns=["event"]).assign(holding_period_start=ledger.holding_period_start.astype("int64")),
            index=False).to_numpy().tobytes()).hexdigest(),
    }
    return StrategyResult(ledger, decision_table, daily, paired, ladder(daily), stats, record, policy, capital_policy)


def write_run_record(result: StrategyResult, directory: Path) -> Path:
    """Save the run record next to nothing else: parameters, git commit, data fingerprints, result hash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{result.policy}_{result.capital_policy}_{result.record['result_fingerprint'][:12]}.json"
    path = directory / name
    path.write_text(json.dumps(result.record, indent=2, sort_keys=True, default=str))
    return path


def h2_diagnostics(basket: StrategyResult, btc: StrategyResult) -> dict:
    """The registered H2 diagnostics for one policy: the paired daily gap between the basket strategy and
    the BTC-only strategy (same policy, same capital policy, same days), realised vol, mean absolute
    exposure, turnover, and the gap after scaling BTC's returns to the basket's realised vol (ex post).
    Descriptive only: the hurdle test and its bootstrap belong to the inference milestone."""
    if (basket.policy, basket.capital_policy) != (btc.policy, btc.capital_policy):
        raise ValueError("H2 compares like with like: same execution policy and capital policy")
    if list(basket.daily.index) != list(btc.daily.index):
        raise ValueError("H2 needs the same scored days for both strategies")
    gap = pd.DataFrame({k: basket.daily[f"{k}_return"] - btc.daily[f"{k}_return"] for k in ("price", "gross", "net")})
    vol = lambda r: float(r.std() * np.sqrt(ANNUALISATION))
    v_b, v_t = vol(basket.daily.net_return), vol(btc.daily.net_return)
    matched = basket.daily.net_return - btc.daily.net_return * (v_b / v_t)
    return {
        "gap_daily": gap,
        "gap_mean_annual": {k: float(gap[k].mean() * ANNUALISATION) for k in gap},
        "realised_vol": {"basket": v_b, "btc": v_t},
        "mean_abs_exposure": {"basket": basket.stats["mean_abs_target_exposure"], "btc": btc.stats["mean_abs_target_exposure"]},
        "turnover_per_year": {"basket": basket.stats["turnover_per_year"], "btc": btc.stats["turnover_per_year"]},
        "ex_post_vol_matched_gap_annual": float(matched.mean() * ANNUALISATION),
    }
