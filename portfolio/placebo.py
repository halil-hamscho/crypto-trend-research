"""End-to-end placebo, planted-edge and look-ahead-canary worlds.

Synthetic prices where the right answer is known, run through the SAME runner and inference a real run
uses. No real strategy return is involved. A world is a small basket (three assets, a common factor with
fat-tailed volatility-clustered returns) with an optional persistent drift planted in the factor:

  strength 0     no trend to find: a placebo (the pipeline must not find an edge)
  strength > 0   a slowly varying drift: trend-following genuinely has an edge, growing with strength
  meanrev        negative autocorrelation: trend-following should lose
"""
import zlib

import numpy as np
import pandas as pd

from portfolio.basket import reference_basket
from portfolio.calibration import _t
from portfolio.inference import evaluate
from portfolio.strategy_runner import run_strategy

N_DAYS = 2396               # 2020-02-09 .. 2026-08-31, as the real index
WARMUP = 512
FUNDING_PER_DAY = 3e-4      # ~11%/yr, paid by longs


def world(kind: str, strength: float, seed: int, n: int = N_DAYS):
    """(inputs, start, end) for one synthetic basket."""
    rng = np.random.default_rng([seed, zlib.crc32(kind.encode()), int(strength * 1000)])
    days = [d.date() for d in pd.date_range("2020-02-09", periods=n)]
    z, var, alpha, beta = _t(rng, n), 1.0, 0.08, 0.90
    factor = np.zeros(n)
    for t in range(n):                                                 # volatility clustering, fat tails
        factor[t] = np.sqrt(var) * z[t]
        var = (1 - alpha - beta) + alpha * factor[t] ** 2 + beta * var
    factor = factor / factor.std() * 0.032
    if kind == "meanrev":
        for t in range(1, n):
            factor[t] += -0.10 * factor[t - 1]
    if strength > 0 and kind == "trend":
        drift, phi = np.zeros(n), 0.5 ** (1 / 60.0)                    # persistent drift, 60-day half-life
        shocks = rng.normal(size=n)
        for t in range(1, n):
            drift[t] = phi * drift[t - 1] + np.sqrt(1 - phi ** 2) * shocks[t]
        factor = factor + strength * 0.003 * drift
    betas = {"A": 1.0, "B": 0.9, "C": 1.1}
    marks = pd.DataFrame({c: 100 * np.exp(np.cumsum(b * factor + rng.normal(0, 0.012, n) - 0.5 * (b * 0.032) ** 2))
                          for c, b in betas.items()}, index=days)
    members = {d: pd.DataFrame({"contract": ["A", "B", "C"], "weight": [0.5, 0.3, 0.2]}) for d in days if d.day == 1}
    members[days[0]] = pd.DataFrame({"contract": ["A", "B", "C"], "weight": [0.5, 0.3, 0.2]})
    index, weights = reference_basket(members, marks)
    stamps = pd.to_datetime([pd.Timestamp(d, tz="UTC") for d in days], utc=True)
    settlements = {c: pd.DataFrame({"stamp": stamps, "rate": FUNDING_PER_DAY, "mark": marks[c].values,
                                    "interval_hours": 24.0, "exchange_skipped": False}) for c in marks}
    return (members, marks, settlements, weights, index), days[WARMUP], days[-2]


def one_day_leak(signal: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    """The subtle bug: today's decision uses TOMORROW's signal (an off-by-one in the timing)."""
    leaked = pd.concat([signal.iloc[1:], signal.iloc[[-1]]])           # the last day has no tomorrow: hold it
    leaked.index = signal.index
    return leaked


def foresight(signal: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    """The blatant cheat: hold the sign of tomorrow's index return, at full size."""
    nxt = index.pct_change().shift(-1).fillna(0.0)
    out = signal.copy()
    out["x_star"] = 1.5 * np.sign(nxt.to_numpy())
    out["standard"] = 1.0
    return out


def evaluate_world(task) -> dict:
    """One world through the runner and the inference; returns the verdict and the alpha."""
    import logging
    logging.disable(logging.WARNING)                                   # the forecast-cap warning fires on every synthetic run
    kind, strength, seed, leak, reps = task
    inputs, start, end = world(kind, strength, seed)
    override = {None: None, "one_day": one_day_leak, "foresight": foresight}[leak]
    runs = {"buffered": run_strategy("buffered", inputs=inputs, start=start, end=end, signal_override=override),
            "buffered_stress": run_strategy("buffered", inputs=inputs, start=start, end=end, cost_bp_per_side=20.0,
                                            signal_override=override)}
    out = evaluate(runs, seed=seed, reps=reps)
    return {"kind": kind, "strength": strength, "leak": leak, "seed": seed, "class": out["classification"],
            "alpha": out["h1"]["alpha_annual"], "p_low": out["h1"]["alpha_p_low"], "gate": bool(out["h1"]["passes"]),
            "nw_p": out["newey_west"]["p_value_one_sided"], "registered": out["record"]["registered_run"]}
