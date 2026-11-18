"""The registered secondary runs S2-S8, under their own disclosed freeze.

Explanatory runs the protocol committed to executing whatever they show (pre-registration Part 6). None can
rescue a failed H1; each is a strategy configuration and counts as a trial in the deflated Sharpe. Same
guard, sealed store and attempt log as the primary freeze, in a separate store so the two runs cannot be
confused. S9, S10 and S16 (and the measured-cost runs S13, S14) are recorded as deferred, with reasons.
"""
import json
import traceback
from pathlib import Path

import portfolio.freeze as fz
from portfolio.basket import SCORED_END, SCORED_START, build_inputs
from portfolio.inference import bootstrap_alpha, load_inference_config, newey_west_alpha, sharpe_intervals
from portfolio.strategy_runner import run_strategy
from portfolio.trend_signal import load_signal_config, single_speed_config

STORE = fz.ROOT / "reports" / "registered_secondaries"
SEED = 20261004                     # chosen at this freeze, before any secondary result was seen
BASE = {"instrument": "basket", "policy": "buffered", "cost_bp": 10.0, "capital_policy": "fixed", "lag": 0,
        "btc_cap": None, "speed": None, "signal_variant": None}
SECONDARIES = {
    "S2_btc_cap_30": {**BASE, "btc_cap": 0.30},
    "S3_btc_cap_35": {**BASE, "btc_cap": 0.35},
    "S4_speed_8_32": {**BASE, "speed": [8, 32]},
    "S5_speed_16_64": {**BASE, "speed": [16, 64]},
    "S6_speed_32_128": {**BASE, "speed": [32, 128]},
    "S7_speed_64_256": {**BASE, "speed": [64, 256]},
    "S8_sign_only": {**BASE, "signal_variant": "sign_only"},
}
DEFERRED = {
    "S9 weekly reconstitution": "needs the delisting path (LUNA settles in the week of 2022-05-09) wired into the daily loop; not built",
    "S10 daily reconstitution": "a different, higher-turnover instrument; needs the same lifecycle work as S9; not built",
    "S16 per-coin TSMOM": "a different hypothesis that needs its own engine; the protocol runs it last, after the primary is locked",
    "S13/S14 measured costs": "Abdi-Ranaldo and aggTrades spread data are not built",
}
INTERPRETATIONS = {
    "S4-S7": "a single speed keeps its registered scalar and uses a diversification multiplier of 1 (FDM 1.15 belongs to the blend)",
    "S8": "sign-only means the blend's direction at the average forecast size (+-10), with volatility targeting and caps unchanged",
    "S2-S3": "the cap applies at each reconstitution to the index, the books and the benchmark alike; weights drift between resets",
}
CODE_FILES = fz.CODE_FILES + ("portfolio/secondaries.py",)


def build_manifest(primary_manifest_hash: str, seed: int = SEED) -> dict:
    return {
        "purpose": "registered secondary runs S2-S8 (explanatory; cannot rescue H1)",
        "references_primary_manifest": primary_manifest_hash,
        "disclosure": "frozen after the primary result had been seen; the primary freeze listed none of these runs",
        "seed": seed, "window": [str(SCORED_START), str(SCORED_END)], "analyses": SECONDARIES,
        "deferred": DEFERRED, "interpretations": INTERPRETATIONS,
        "frozen_config_sha256": load_signal_config().config_hash,
        "code_sha256": {f: fz.sha256_file(fz.ROOT / f) for f in CODE_FILES},
        "git": fz.git_state(),
    }


def secondary_stats(result, cfg, seed: int, reps: int | None = None) -> dict:
    """Per run: net return, Sharpe with a bootstrap interval, alpha over the run's own same-policy benchmark
    with its bootstrap interval, Newey-West, turnover and mean exposure. Descriptive per registry row."""
    p = result.paired
    boot = bootstrap_alpha(p.strategy_net, p.benchmark_net, cfg, seed, reps=reps, name="secondary")
    nw = newey_west_alpha(p.strategy_net, p.benchmark_net, cfg.newey_west_lags, cfg.annualisation)
    sharpe = sharpe_intervals({"strategy": p.strategy_net, "benchmark": p.benchmark_net}, cfg, seed, reps=reps)
    return {"net_annual": float(p.strategy_net.mean() * cfg.annualisation),
            "benchmark_net_annual": float(p.benchmark_net.mean() * cfg.annualisation),
            "sharpe": sharpe, "alpha": {k: v for k, v in boot.items() if k != "alphas"}, "newey_west": nw,
            "turnover_per_year": result.stats["turnover_per_year"],
            "mean_abs_target_exposure": result.stats["mean_abs_target_exposure"]}


def run_secondaries(manifest: dict, store: Path = STORE, rerun_reason: str | None = None, check_git: bool = True,
                    inputs_provider=None, reps: int | None = None, window: tuple | None = None) -> Path:
    """One entry point, guarded and logged exactly like the primary freeze."""
    path = fz.write_manifest(manifest, store)
    folder = path.parent
    notes = fz.check_guard(manifest, folder, rerun_reason, check_git)
    overridden = inputs_provider is not None or reps is not None or window is not None
    if overridden:
        notes.append("test overrides in use: this run is NOT the registered secondary run")
    fz.log_attempt(folder, "started", manifest_sha256=fz.sha256_file(path), rerun_reason=rerun_reason, notes=notes,
                   git=fz.git_state())
    start, end = (SCORED_START, SCORED_END) if window is None else window
    provider = inputs_provider or (lambda cap: build_inputs(end, "basket", btc_cap=cap))
    cfg, base_config = load_inference_config(), load_signal_config()
    try:
        cache, stats = {}, {}
        for name, spec in manifest["analyses"].items():
            cap = spec["btc_cap"]
            cache.setdefault(cap, provider(cap))
            config = single_speed_config(base_config, spec["speed"]) if spec["speed"] else base_config
            result = run_strategy(spec["policy"], spec["capital_policy"], inputs=cache[cap], start=start, end=end,
                                  config=config, signal_lag_days=spec["lag"], instrument=spec["instrument"],
                                  cost_bp_per_side=spec["cost_bp"], signal_variant=spec["signal_variant"])
            fz._save_run(folder, name, result)
            stats[name] = secondary_stats(result, cfg, manifest["seed"], reps)
        (folder / "secondaries.json").write_text(json.dumps(stats, indent=2, sort_keys=True, default=fz._jsonable))
        digest = fz.seal(folder)
    except Exception as error:
        fz.log_attempt(folder, "failed", error=repr(error), trace=traceback.format_exc(limit=6))
        raise
    fz.log_attempt(folder, "completed", sums_sha256=digest, registered=not overridden and check_git, notes=notes)
    return folder

