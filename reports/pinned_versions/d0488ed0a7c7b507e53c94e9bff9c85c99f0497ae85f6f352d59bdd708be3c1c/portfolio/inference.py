"""Statistical inference for the registered tests (protocol Parts 4 and 5).

Given the daily return series the runner produces, this answers how much of an edge could be luck:
H1 (alpha over the same-policy benchmark), H2 (basket trend against BTC-only trend), Sharpe intervals,
the exact alpha attribution, and the result classification fixed in advance. It does no accounting:
it takes paired daily returns and nothing else.

Method (frozen_config.yaml, tests.*): a stationary block bootstrap (Politis and Romano 1994) on PAIRED
daily returns, 10,000 replications, mean block 20 days, alpha and beta re-estimated in every replication;
Newey-West standard errors (21 lags) as an analytical cross-check. The bootstrap resamples our own
days, so it measures this sample's noise, not what a market with no edge would do.
"""
import zlib
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from portfolio.trend_signal import FROZEN_CONFIG, _expect, _need

CHUNK = 1000    # replications per vectorised batch


@dataclass(frozen=True)
class InferenceConfig:
    alpha_level: float
    annualisation: int
    replications: int
    mean_block_days: int
    sensitivity_block_days: tuple
    newey_west_lags: int
    hurdle_annual: float
    primary_bp: float
    stress_bp: float
    cash_rate: float


def load_inference_config(path: Path = FROZEN_CONFIG, raw: dict | None = None) -> InferenceConfig:
    """The registered test settings. Missing keys raise, and so does any method this module does not implement."""
    raw = yaml.safe_load(Path(path).read_text()) if raw is None else raw
    _expect(raw, "stationary_block_bootstrap", "tests", "inference", "method")
    _expect(raw, True, "tests", "inference", "paired")
    _expect(raw, "daily_return_pairs", "tests", "inference", "resample_object")
    _expect(raw, True, "tests", "inference", "reestimate_within_replication")
    _expect(raw, "newey_west", "tests", "inference", "cross_check")
    _expect(raw, "one", "tests", "h1", "sided")
    _expect(raw, "paired_daily_gap_mean", "tests", "h2", "statistic")
    stress = [float(v) for v in _need(raw, "costs", "stress_bp_per_side")]
    cfg = InferenceConfig(
        alpha_level=float(_need(raw, "tests", "h1", "alpha_level")),
        annualisation=int(_need(raw, "tests", "h1", "annualisation")),
        replications=int(_need(raw, "tests", "inference", "replications")),
        mean_block_days=int(_need(raw, "tests", "inference", "mean_block_days")),
        sensitivity_block_days=tuple(int(v) for v in _need(raw, "tests", "inference", "sensitivity_block_days")),
        newey_west_lags=int(_need(raw, "tests", "inference", "newey_west_lags")),
        hurdle_annual=float(_need(raw, "tests", "h2", "hurdle_delta_annual_return")),
        primary_bp=float(_need(raw, "costs", "primary_bp_per_side")),
        stress_bp=max(stress), cash_rate=float(_need(raw, "risk", "cash_rate")))
    problems = []
    if not (0 < cfg.alpha_level < 0.5):
        problems.append("alpha_level must be in (0, 0.5)")
    if cfg.replications < 100 or cfg.mean_block_days < 1 or cfg.newey_west_lags < 0 or cfg.annualisation <= 0:
        problems.append("replications >= 100, mean block >= 1, lags >= 0, annualisation > 0 are required")
    if not all(b >= 1 for b in cfg.sensitivity_block_days) or cfg.cash_rate != 0.0:
        problems.append("sensitivity blocks must be >= 1 and the registered cash rate is 0")
    if problems:
        raise ValueError("frozen config has invalid inference settings: " + "; ".join(problems))
    return cfg


def rng_for(seed: int, name: str) -> np.random.Generator:
    """A reproducible generator per analysis: the same seed and name give the same draws, and different
    names are independent of each other and of the order they run in."""
    return np.random.default_rng([int(seed), zlib.crc32(name.encode())])


def stationary_bootstrap_indices(n: int, mean_block: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: (reps, n) positions. Each step starts a new block at a uniform
    random day with probability 1/mean_block, otherwise continues to the next day (wrapping around the end),
    so block lengths are geometric with the given mean and the resampled series is stationary."""
    if n < 2 or mean_block < 1 or reps < 1:
        raise ValueError("need n >= 2, mean_block >= 1 and reps >= 1")
    new = rng.random((reps, n)) < 1.0 / mean_block
    new[:, 0] = True
    starts = rng.integers(0, n, size=(reps, n))
    position = np.arange(n)
    last_new = np.maximum.accumulate(np.where(new, position, 0), axis=1)
    origin = np.take_along_axis(starts, last_new, axis=1)
    return (origin + (position - last_new)) % n


def alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple:
    """OLS of y on a constant and x: (alpha, beta), daily."""
    xc = x - x.mean()
    beta = float((xc * (y - y.mean())).sum() / (xc * xc).sum())
    return float(y.mean() - beta * x.mean()), beta


def _batch_alpha_beta(y: np.ndarray, x: np.ndarray, idx: np.ndarray) -> tuple:
    yy, xx = y[idx], x[idx]
    xc = xx - xx.mean(axis=1, keepdims=True)
    beta = (xc * (yy - yy.mean(axis=1, keepdims=True))).sum(axis=1) / (xc * xc).sum(axis=1)
    return yy.mean(axis=1) - beta * xx.mean(axis=1), beta


MIN_DAYS = 30


def _require_returns(name: str, series: pd.Series) -> np.ndarray:
    """A daily return series the statistics can use: finite on every day, one row per date, dates in order."""
    if not series.index.is_unique:
        raise ValueError(f"{name}: duplicate dates")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates must be in increasing order")
    values = series.to_numpy(dtype=float)
    if len(values) < MIN_DAYS:
        raise ValueError(f"{name}: needs at least {MIN_DAYS} days, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: returns must be finite on every day")
    return values


def _paired_arrays(strategy: pd.Series, benchmark: pd.Series) -> tuple:
    if not strategy.index.equals(benchmark.index):
        raise ValueError("strategy and benchmark returns must cover exactly the same days")
    y, x = _require_returns("strategy", strategy), _require_returns("benchmark", benchmark)
    if np.ptp(x) == 0.0:
        raise ValueError("benchmark has no variance: beta is undefined")
    return y, x


def bootstrap_alpha(strategy: pd.Series, benchmark: pd.Series, cfg: InferenceConfig, seed: int,
                    mean_block: float | None = None, reps: int | None = None, name: str = "h1") -> dict:
    """H1: r_strategy = alpha + beta r_benchmark (cash rate 0), alpha annualised. Pass = the 5th percentile
    of the bootstrapped alpha exceeds zero (the registered rule), with alpha and beta re-estimated in
    every replication. Also reports the bootstrap p-value (share of replications with alpha <= 0)."""
    y, x = _paired_arrays(strategy, benchmark)
    mean_block = cfg.mean_block_days if mean_block is None else mean_block
    reps = cfg.replications if reps is None else reps
    rng = rng_for(seed, f"{name}:{mean_block}")
    alphas, betas = [], []
    for done in range(0, reps, CHUNK):
        idx = stationary_bootstrap_indices(len(y), mean_block, min(CHUNK, reps - done), rng)
        a, b = _batch_alpha_beta(y, x, idx)
        alphas.append(a * cfg.annualisation)
        betas.append(b)
    alphas, betas = np.concatenate(alphas), np.concatenate(betas)
    a_hat, b_hat = alpha_beta(y, x)
    lo = cfg.alpha_level
    return {"alpha_annual": a_hat * cfg.annualisation, "beta": b_hat, "mean_block": mean_block, "reps": reps,
            "seed": seed, "alpha_p_low": float(np.quantile(alphas, lo)), "alpha_p_high": float(np.quantile(alphas, 1 - lo)),
            "alpha_ci95": (float(np.quantile(alphas, 0.025)), float(np.quantile(alphas, 0.975))),
            "p_value_one_sided": float((alphas <= 0).mean()), "beta_ci95": (float(np.quantile(betas, 0.025)),
                                                                          float(np.quantile(betas, 0.975))),
            "passes": bool(np.quantile(alphas, lo) > 0), "alphas": alphas}


def newey_west_alpha(strategy: pd.Series, benchmark: pd.Series, lags: int, annualisation: int) -> dict:
    """Analytical cross-check: OLS alpha with Newey-West (1987) HAC standard error, Bartlett kernel,
    no small-sample correction, one-sided normal p-value. Annualised like the bootstrap."""
    y, x = _paired_arrays(strategy, benchmark)
    X = np.column_stack([np.ones(len(y)), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    u = X * (y - X @ coef)[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        gamma = u[lag:].T @ u[:-lag]
        S += (1.0 - lag / (lags + 1.0)) * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv
    se, t = sqrt(cov[0, 0]), coef[0] / sqrt(cov[0, 0])
    return {"alpha_annual": float(coef[0] * annualisation), "se_annual": float(se * annualisation), "t": float(t),
            "p_value_one_sided": float(0.5 * (1.0 - erf(t / sqrt(2.0)))), "lags": lags}


def sharpe(returns: np.ndarray, annualisation: int = 365) -> np.ndarray:
    """Registered: annualised arithmetic mean over annualised volatility, over the last axis."""
    return returns.mean(axis=-1) * annualisation / (returns.std(axis=-1, ddof=1) * sqrt(annualisation))


def sharpe_intervals(series: dict, cfg: InferenceConfig, seed: int, mean_block: float | None = None,
                     reps: int | None = None) -> dict:
    """Point Sharpe and percentile intervals for several series at once. All series are resampled with the
    SAME days in each replication, so their comparisons are paired."""
    for name, one in series.items():
        _require_returns(name, one)
    frame = pd.DataFrame(series)
    if frame.isna().any().any():
        raise ValueError("Sharpe intervals need the same days for every series")
    arrays = {k: frame[k].to_numpy(dtype=float) for k in frame}
    if any(np.ptp(a) == 0.0 for a in arrays.values()):
        raise ValueError("a series has no variance: its Sharpe ratio is undefined")
    mean_block = cfg.mean_block_days if mean_block is None else mean_block
    reps = cfg.replications if reps is None else reps
    rng = rng_for(seed, f"sharpe:{mean_block}")
    draws = {k: [] for k in arrays}
    for done in range(0, reps, CHUNK):
        idx = stationary_bootstrap_indices(len(frame), mean_block, min(CHUNK, reps - done), rng)
        for k, a in arrays.items():
            draws[k].append(sharpe(a[idx], cfg.annualisation))
    out = {}
    for k, a in arrays.items():
        d = np.concatenate(draws[k])
        out[k] = {"sharpe": float(sharpe(a, cfg.annualisation)), "ci95": (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))),
                  "ci90": (float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95)))}
    return out


def gap_test(gap: pd.Series, cfg: InferenceConfig, seed: int, mean_block: float | None = None,
             reps: int | None = None) -> dict:
    """H2: the paired daily gap (basket trend minus BTC-only trend). H0: E[gap] <= 0, one-sided 5%.
    `clears_hurdle`: the 5th percentile exceeds the 5%/yr basket-superiority hurdle. `rules_out_hurdle`:
    the 95th percentile is below it (BTC operational sufficiency, not equivalence). The hurdle is
    not an equivalence margin, and per Amendment 3 it sits below the minimum detectable effect."""
    values = _require_returns("the gap", gap)
    mean_block = cfg.mean_block_days if mean_block is None else mean_block
    reps = cfg.replications if reps is None else reps
    rng = rng_for(seed, f"h2:{mean_block}")
    means = np.concatenate([values[stationary_bootstrap_indices(len(values), mean_block, min(CHUNK, reps - d), rng)].mean(axis=1)
                            for d in range(0, reps, CHUNK)]) * cfg.annualisation
    lo = cfg.alpha_level
    p_low, p_high = float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))
    return {"gap_annual": float(values.mean() * cfg.annualisation), "p_low": p_low, "p_high": p_high,
            "rejects_h0": p_low > 0, "clears_hurdle": p_low > cfg.hurdle_annual, "rules_out_hurdle": p_high < cfg.hurdle_annual,
            "hurdle": cfg.hurdle_annual, "mean_block": mean_block, "reps": reps, "seed": seed}


def alpha_attribution(strategy: pd.DataFrame, benchmark: pd.DataFrame, annualisation: int = 365) -> dict:
    """Exact decomposition at ONE beta, estimated in the net regression:
    alpha_net = alpha_price + alpha_funding - alpha_cost, each term mean(r_s,layer - beta r_b,layer) x 365 on
    the identical sample (cost is a positive drag). The terms sum to the headline by construction; layer-
    by-layer regressions with their own betas are diagnostics, never attribution. Raises if either frame
    does not satisfy net = price + funding - cost."""
    for name, frame in (("strategy", strategy), ("benchmark", benchmark)):
        gap = (frame.net_return - (frame.price_return + frame.funding_return - frame.cost_return)).abs().max()
        if gap > 1e-9:
            raise ValueError(f"{name} layers do not add up to net (max gap {gap:.2e})")
    if not strategy.index.equals(benchmark.index):
        raise ValueError("strategy and benchmark must cover the same days")
    _, beta = alpha_beta(strategy.net_return.to_numpy(float), benchmark.net_return.to_numpy(float))
    term = lambda col: float((strategy[col] - beta * benchmark[col]).mean() * annualisation)
    out = {"beta": beta, "alpha_net": term("net_return"), "alpha_price": term("price_return"),
           "alpha_funding": term("funding_return"), "alpha_cost": term("cost_return")}
    out["identity_residual"] = abs(out["alpha_net"] - (out["alpha_price"] + out["alpha_funding"] - out["alpha_cost"]))
    if out["identity_residual"] > 1e-9:
        raise ValueError(f"attribution identity broken by {out['identity_residual']:.2e}")
    return out


# ------------------------------------------------------------------ classification, fixed in advance (4.4, 4.5, 5)
SUPPORTED = "statistically supported historical alpha (not out-of-sample proof)"
INCONCLUSIVE = "economically positive, statistically inconclusive"
FAILS = "the primary hypothesis fails"
FRAGILE = "fragile"


def classify_h1(alpha: float, p_low_20: float, p_low_40: float, p_low_60: float, p_low_stress: float) -> str:
    """The four cases of protocol 4.4. `p_low_*` are the 5th percentiles of bootstrapped alpha: at 20, 40 and
    60-day blocks (10 bp) and at 20-day blocks under the stress cost."""
    inputs = (alpha, p_low_20, p_low_40, p_low_60, p_low_stress)
    if not all(np.isfinite(v) for v in inputs):
        raise ValueError(f"cannot classify an undefined statistic: {inputs}")
    if alpha <= 0:
        return FAILS
    if p_low_20 <= 0:
        return INCONCLUSIVE
    return SUPPORTED if (p_low_40 > 0 and p_low_60 > 0 and p_low_stress > 0) else FRAGILE


def classify_full_vs_buffered(alpha_full: float, alpha_buffered: float) -> str:
    """Protocol 4.5."""
    if alpha_full > 0 and alpha_buffered > 0:
        return "survives even naive trading"
    if alpha_full <= 0 and alpha_buffered > 0:
        return "exists only with execution discipline"
    if alpha_full <= 0 and alpha_buffered <= 0:
        return "fails"
    return "suspicious: audit the buffer before reporting"


def gross_tripwire(gross_buffered_mean: float, gross_full_mean: float) -> bool:
    """G-buf above G-full means the buffer may be seeing information it should not: audit, not a verdict."""
    return gross_buffered_mean > gross_full_mean


def h2_conclusion(h1_passes: bool, h2_rejects_h0: bool) -> str:
    """Protocol Part 5, fixed-sequence gatekeeping: H2 can support the combined claim only if H1 passes."""
    return {(True, True): "basket trend adds value beyond passive beta and BTC trend",
            (True, False): "trend works; incremental basket value unproven",
            (False, True): "basket may improve on BTC trend, but active trend does not beat passive beta",
            (False, False): "headline claim unsupported"}[(h1_passes, h2_rejects_h0)]


# ------------------------------------------------------------------ funding and cost drag (the primary descriptive objective)
LAYER_COLUMNS = ("price_return", "funding_return", "cost_return", "net_return")


def _interval(draws: np.ndarray, point: float, cfg: InferenceConfig) -> dict:
    lo = cfg.alpha_level
    return {"estimate": float(point), "ci90": (float(np.quantile(draws, lo)), float(np.quantile(draws, 1 - lo))),
            "ci95": (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))}


def _bootstrap_means(series: dict, cfg: InferenceConfig, seed: int, name: str, mean_block: float, reps: int) -> dict:
    """Annualised bootstrap means of several daily series resampled on the SAME days in each replication,
    so quantities that add up still add up inside every replication."""
    frame = pd.DataFrame(series)
    arrays = {k: _require_returns(k, frame[k]) for k in frame}
    rng = rng_for(seed, f"{name}:{mean_block}")
    draws = {k: [] for k in arrays}
    for done in range(0, reps, CHUNK):
        idx = stationary_bootstrap_indices(len(frame), mean_block, min(CHUNK, reps - done), rng)
        for k, a in arrays.items():
            draws[k].append(a[idx].mean(axis=1) * cfg.annualisation)
    return {k: np.concatenate(v) for k, v in draws.items()}


DRAG_LONG_BLOCKS = (60, 120)    # funding and exposure persist for months, far beyond 20 days


def _drag_intervals(series: dict, cfg: InferenceConfig, seed: int, name: str, mean_block: float, reps: int) -> dict:
    """Interval for each series at the registered block length, plus at longer blocks and their envelope.

    Funding drag follows exposure and funding regimes, which persist for months, so 20-day blocks give
    intervals that are too narrow: in simulation the 90% interval covered the truth about 72% of the
    time at the real sample length, 81% with 60-day blocks and 83% with 120-day blocks. `ci90` is the
    registered-block interval; `ci90_conservative` is the envelope over all block lengths and is the one
    to quote."""
    blocks = (mean_block,) + tuple(b for b in DRAG_LONG_BLOCKS if b != mean_block)
    boots = {b: _bootstrap_means(series, cfg, seed, name, b, reps) for b in blocks}
    out = {}
    for k, one in series.items():
        point = one.mean() * cfg.annualisation
        cell = _interval(boots[mean_block][k], point, cfg)
        per_block = {b: _interval(boots[b][k], point, cfg)["ci90"] for b in blocks}
        cell["by_block_ci90"] = per_block
        cell["ci90_conservative"] = (min(v[0] for v in per_block.values()), max(v[1] for v in per_block.values()))
        out[k] = cell
    return out


def check_layers(layers: pd.DataFrame, name: str = "run") -> None:
    """Raises unless net = price + funding - cost on every day (cost is a positive drag)."""
    missing = [c for c in LAYER_COLUMNS if c not in layers.columns]
    if missing:
        raise ValueError(f"{name}: layers lack {missing}")
    gap = (layers.net_return - (layers.price_return + layers.funding_return - layers.cost_return)).abs().max()
    if not gap <= 1e-9:
        raise ValueError(f"{name}: layers do not add up to net (max gap {gap:.2e})")


def drag_table(layers: pd.DataFrame, cfg: InferenceConfig, seed: int, name: str, side: pd.Series | None = None,
               mean_block: float | None = None, reps: int | None = None) -> dict:
    """Annualised funding drag and execution-cost drag of one run, with 90% and 95% bootstrap intervals.

    Not a rerun without funding or costs: the layers of ONE run share positions and days, so the drags are
    paired same-day differences and far tighter than any Sharpe. funding_drag is positive when funding is
    paid (longs pay) and negative when received; cost_drag is positive. If `side` (+1 long, -1 short, 0 flat,
    by day) is given, funding is also split into long-day and short-day contributions that add to the total.
    Conditional on the cost model (fee plus slippage per side), not a measured cost."""
    check_layers(layers, name)
    mean_block = cfg.mean_block_days if mean_block is None else mean_block
    reps = cfg.replications if reps is None else reps
    series = {"price": layers.price_return, "funding_drag": -layers.funding_return, "cost_drag": layers.cost_return,
              "net": layers.net_return}
    series["gross"] = series["price"] - series["funding_drag"]
    series["total_drag"] = series["funding_drag"] + series["cost_drag"]
    if side is not None:
        if not side.index.equals(layers.index):
            raise ValueError(f"{name}: position side must cover the same days as the layers")
        series["funding_drag_long_days"] = series["funding_drag"].where(side > 0, 0.0)
        series["funding_drag_short_days"] = series["funding_drag"].where(side < 0, 0.0)
        series["funding_drag_flat_days"] = series["funding_drag"].where(side == 0, 0.0)
    out = _drag_intervals(series, cfg, seed, f"drag:{name}", mean_block, reps)
    out["days"] = len(layers)
    return out


def drag_gap(a: pd.DataFrame, b: pd.DataFrame, cfg: InferenceConfig, seed: int, name: str,
             mean_block: float | None = None, reps: int | None = None) -> dict:
    """a minus b, day by day: how much more (positive) or less funding and cost drag a bears than b."""
    check_layers(a, f"{name} (a)")
    check_layers(b, f"{name} (b)")
    if not a.index.equals(b.index):
        raise ValueError(f"{name}: the two runs must cover the same days")
    mean_block = cfg.mean_block_days if mean_block is None else mean_block
    reps = cfg.replications if reps is None else reps
    series = {"funding_drag": -(a.funding_return - b.funding_return), "cost_drag": a.cost_return - b.cost_return}
    series["total_drag"] = series["funding_drag"] + series["cost_drag"]
    return _drag_intervals(series, cfg, seed, f"draggap:{name}", mean_block, reps)


def evaluate_drags(results: dict, seed: int, cfg: InferenceConfig | None = None, reps: int | None = None) -> dict:
    """The registered primary descriptive objective, from runner outputs: funding and cost drag with intervals
    for the strategy, its benchmark and (if given) BTC-only trend; the strategy-minus-benchmark gap; and the
    cost drag under the stress cost. Needs 'buffered' and 'buffered_stress'; 'btc' is optional."""
    cfg = cfg or load_inference_config()
    reps = cfg.replications if reps is None else reps
    for key in ("buffered", "buffered_stress"):
        if key not in results:
            raise KeyError(f"evaluate_drags needs the '{key}' run")
    buf, stress = results["buffered"], results["buffered_stress"]
    bench = _benchmark_frame(buf)
    side = np.sign(buf.daily.exposure.round(9))
    out = {"strategy": drag_table(buf.daily, cfg, seed, "strategy", side=side, reps=reps),
           "benchmark": drag_table(bench, cfg, seed, "benchmark", reps=reps),
           "strategy_minus_benchmark": drag_gap(buf.daily, bench, cfg, seed, "strategy-benchmark", reps=reps),
           "cost_stress": {"bp": cfg.stress_bp, **{k: v for k, v in drag_table(stress.daily, cfg, seed, "strategy-stress", reps=reps).items()
                                                     if k in ("cost_drag", "total_drag", "net")}}}
    if "btc" in results:
        btc = results["btc"]
        out["btc_strategy"] = drag_table(btc.daily, cfg, seed, "btc", side=np.sign(btc.daily.exposure.round(9)), reps=reps)
        out["btc_benchmark"] = drag_table(_benchmark_frame(btc), cfg, seed, "btc-benchmark", reps=reps)
    return out


# ------------------------------------------------------------------ is this the registered experiment?
REGISTERED_PRIMARY = {"policy": "buffered", "capital_policy": "fixed", "instrument": "basket", "signal_lag_days": 0,
                      "start": "2021-07-05", "end": "2026-08-30", "signal_override": False}
FINGERPRINTS = ("marks", "settlements", "members", "reference_index")


def check_experiment_identity(results: dict, cfg: InferenceConfig) -> dict:
    """The runs handed to `evaluate` must be one experiment, varied only as the protocol varies it.

    buffered_stress differs from buffered ONLY in the cost (primary vs stress bp); `full` ONLY in the
    execution policy; `btc` ONLY in the instrument. Every run must share the capital policy, capital,
    window, timing, configuration, and (for basket runs) the same input data. A mismatch raises: the
    analysis would otherwise label the wrong comparison. Returns {'registered': bool, 'reasons': [...]},
    where `registered` is True only for the exact registered primary on clean committed code with the
    current frozen configuration (reasons lists every way it is not).
    """
    for key in ("buffered", "buffered_stress"):
        if key not in results:
            raise KeyError(f"evaluate needs the '{key}' run")
    from portfolio.trend_signal import load_signal_config
    base = results["buffered"].record["parameters"]
    others = {"buffered_stress": {"cost_bp_per_side": cfg.stress_bp},
              "full": {"policy": "full"}, "btc": {"instrument": "btc"}}
    if base["cost_bp_per_side"] != cfg.primary_bp:
        raise ValueError(f"the primary run must use the registered {cfg.primary_bp:g} bp, got {base['cost_bp_per_side']}")
    for key, change in others.items():
        if key not in results:
            continue
        expected = {**base, **change}
        found = results[key].record["parameters"]
        differing = sorted(k for k in set(expected) | set(found) if expected.get(k) != found.get(k))
        if differing:
            raise ValueError(f"'{key}' is not the same experiment as 'buffered' apart from {sorted(change)}: "
                             f"differs in {differing}")
        if key != "btc":
            for fp in FINGERPRINTS:
                if results[key].record["fingerprints"][fp] != results["buffered"].record["fingerprints"][fp]:
                    raise ValueError(f"'{key}' was run on different input data ({fp} fingerprint differs)")
        if not results[key].paired.index.equals(results["buffered"].paired.index):
            raise ValueError(f"'{key}' does not cover the same days as 'buffered'")
    reasons = [f"{k}={base.get(k)!r}, registered {v!r}" for k, v in REGISTERED_PRIMARY.items() if base.get(k) != v]
    current = load_signal_config()
    if base["config"].get("config_hash") != current.config_hash:
        reasons.append("run used a configuration that differs from the current frozen_config.yaml")
    git = results["buffered"].record["git"]
    if git.get("dirty") is not False:
        reasons.append("code was not clean and committed when the run was made")
    return {"registered": not reasons, "reasons": reasons}


def evaluate(results: dict, seed: int, cfg: InferenceConfig | None = None, reps: int | None = None) -> dict:
    """The registered analysis from runner outputs. `results` needs 'buffered' (the primary: basket, buffered,
    primary cost) and 'buffered_stress' (the same at the stress cost); optionally 'full' (full rebalancing,
    primary cost) and 'btc' (the BTC-only strategy, buffered, primary cost). `reps` overrides the registered
    replications for tests only; the record says so."""
    cfg = cfg or load_inference_config()
    reps = cfg.replications if reps is None else reps
    identity = check_experiment_identity(results, cfg)
    buf, stress = results["buffered"], results["buffered_stress"]
    s, b = buf.paired.strategy_net, buf.paired.benchmark_net

    primary = bootstrap_alpha(s, b, cfg, seed, cfg.mean_block_days, reps)
    sens = {m: bootstrap_alpha(s, b, cfg, seed, m, reps) for m in cfg.sensitivity_block_days}
    stressed = bootstrap_alpha(stress.paired.strategy_net, stress.paired.benchmark_net, cfg, seed,
                               cfg.mean_block_days, reps, name="h1stress")
    if 40 not in sens or 60 not in sens:
        raise ValueError("the registered classification needs the 40 and 60-day block sensitivities")
    label = classify_h1(primary["alpha_annual"], primary["alpha_p_low"], sens[40]["alpha_p_low"],
                        sens[60]["alpha_p_low"], stressed["alpha_p_low"])
    out = {
        "h1": {k: v for k, v in primary.items() if k != "alphas"},
        "h1_block_sensitivity": {m: {k: v for k, v in r.items() if k != "alphas"} for m, r in sens.items()},
        "h1_stress_cost": {k: v for k, v in stressed.items() if k != "alphas"},
        "newey_west": newey_west_alpha(s, b, cfg.newey_west_lags, cfg.annualisation),
        "attribution": alpha_attribution(buf.daily, _benchmark_frame(buf), cfg.annualisation),
        "sharpe": sharpe_intervals({"strategy": s, "benchmark": b, **({"btc_strategy": results["btc"].paired.strategy_net}
                                                                     if "btc" in results else {})}, cfg, seed, reps=reps),
        "classification": label,
        "drags": evaluate_drags(results, seed, cfg, reps),
        "record": {"seed": seed, "replications": reps, "registered_replications": cfg.replications,
                   "registered_run": identity["registered"] and reps == cfg.replications,
                   "not_registered_because": identity["reasons"] + ([] if reps == cfg.replications
                                                                    else [f"replications {reps} != {cfg.replications}"]),
                   "mean_block": cfg.mean_block_days, "stress_bp": cfg.stress_bp, "primary_bp": cfg.primary_bp},
    }
    if "full" in results:
        full = results["full"]
        a_full = bootstrap_alpha(full.paired.strategy_net, full.paired.benchmark_net, cfg, seed, reps=reps, name="h1full")
        out["full_vs_buffered"] = {"alpha_full": a_full["alpha_annual"], "alpha_buffered": primary["alpha_annual"],
                                   "reading": classify_full_vs_buffered(a_full["alpha_annual"], primary["alpha_annual"]),
                                   "gross_tripwire_audit": gross_tripwire(buf.paired.strategy_gross.mean(),
                                                                          full.paired.strategy_gross.mean())}
    if "btc" in results:
        gap = buf.paired.strategy_net - results["btc"].paired.strategy_net
        h2 = gap_test(gap, cfg, seed, reps=reps)
        out["h2"] = {**h2, "gated_on_h1": True, "h1_passes": primary["passes"],
                     "conclusion": h2_conclusion(primary["passes"], h2["rejects_h0"])}
    return out


def _benchmark_frame(result) -> pd.DataFrame:
    """The benchmark's layers as a daily frame with the strategy's column names, from a StrategyResult."""
    p = result.paired
    funding = p.benchmark_gross - p.benchmark_price
    return pd.DataFrame({"price_return": p.benchmark_price, "funding_return": funding,
                         "cost_return": p.benchmark_gross - p.benchmark_net, "net_return": p.benchmark_net})
