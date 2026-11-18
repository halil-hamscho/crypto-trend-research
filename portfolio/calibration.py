"""Does the registered bootstrap test have the error rate it claims?

Simulates strategies that have NO skill (true alpha zero) against the real benchmark's market returns,
under several kinds of persistence, and measures how often the registered gate (5th percentile of
bootstrapped alpha above zero) wrongly passes, and how often the 90% interval covers the truth.
Market returns x are the benchmark's own (a series that has no strategy in it); only the strategy side
is simulated. Nothing here touches a real strategy return.
"""
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from portfolio.inference import bootstrap_alpha, load_inference_config

RESIDUAL_SD = 0.0147                 # 28%/yr, the residual volatility the protocol's power calculation assumes
T_DF = 4                             # fat tails


def _t(rng, n):
    return rng.standard_t(T_DF, n) / np.sqrt(T_DF / (T_DF - 2.0))        # unit variance


def residual_ar1(n, rho, rng):
    e, z = np.zeros(n), _t(rng, n)
    e[0] = z[0]
    for t in range(1, n):
        e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * z[t]
    return e * RESIDUAL_SD


def residual_garch(n, rng, alpha=0.08, beta=0.90):
    z, e, var = _t(rng, n), np.zeros(n), 1.0
    omega = 1.0 - alpha - beta
    for t in range(n):
        e[t] = np.sqrt(var) * z[t]
        var = omega + alpha * e[t] ** 2 + beta * var
    return e / e.std() * RESIDUAL_SD


def random_exposure(n, half_life, rng, mean=0.5, sd=0.4):
    """A no-skill trend follower's exposure: persistent, independent of the market's returns."""
    phi = 0.5 ** (1.0 / half_life)
    u, z = np.zeros(n), rng.normal(size=n)
    u[0] = z[0]
    for t in range(1, n):
        u[t] = phi * u[t - 1] + np.sqrt(1 - phi ** 2) * z[t]
    return np.clip(mean + sd * u, -1.0, 2.0)


def make_dgp(name: str):
    """(strategy returns from x, rng) -> y, for a named no-skill process. True alpha is zero for all."""
    kind, _, arg = name.partition(":")
    if kind == "iid":
        return lambda x, rng: 0.5 * x + residual_ar1(len(x), 0.0, rng)
    if kind == "ar1":
        return lambda x, rng: 0.5 * x + residual_ar1(len(x), float(arg), rng)
    if kind == "garch":
        return lambda x, rng: 0.5 * x + residual_garch(len(x), rng)
    if kind == "exposure":
        return lambda x, rng: random_exposure(len(x), float(arg), rng) * x
    raise ValueError(f"unknown process {name!r}")


DGPS = ("iid", "ar1:0.1", "ar1:0.3", "ar1:0.5", "ar1:0.7", "garch", "exposure:5", "exposure:20", "exposure:60")


def _one(task):
    x, name, block, reps, seed, i, planted_annual = task
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), i])
    y = make_dgp(name)(x, rng) + planted_annual / 365.0
    days = pd.RangeIndex(len(x))
    r = bootstrap_alpha(pd.Series(y, index=days), pd.Series(x, index=days), CFG, seed=int(rng.integers(1 << 31)),
                        mean_block=block, reps=reps, name="calibration")
    truth = planted_annual
    return (bool(r["passes"]), bool(r["alpha_p_low"] <= truth <= r["alpha_p_high"]), r["alpha_p_high"] - r["alpha_p_low"],
            r["alpha_annual"])


CFG = load_inference_config()


def run_cell(x: np.ndarray, name: str, block: float, reps: int, n_sims: int, seed: int, planted_annual: float = 0.0,
             workers: int = 8) -> dict:
    """Pass rate (size when planted_annual is 0, power otherwise), 90% interval coverage of the true alpha, and
    mean interval width, over `n_sims` simulated datasets."""
    tasks = [(x, name, block, reps, seed, i, planted_annual) for i in range(n_sims)]
    with ProcessPoolExecutor(workers) as pool:
        out = list(pool.map(_one, tasks, chunksize=max(1, n_sims // (workers * 4))))
    passes = np.array([o[0] for o in out])
    covers = np.array([o[1] for o in out])
    p = passes.mean()
    return {"dgp": name, "block": block, "reps": reps, "n_sims": n_sims, "planted_annual": planted_annual,
            "pass_rate": float(p), "pass_rate_se": float(np.sqrt(p * (1 - p) / n_sims)),
            "coverage90": float(covers.mean()), "mean_width": float(np.mean([o[2] for o in out])),
            "mean_alpha_hat": float(np.mean([o[3] for o in out]))}
