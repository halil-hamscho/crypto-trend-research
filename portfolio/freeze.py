"""The freeze: lock in exactly what the registered run is, run it once through one guarded entry point,
and store everything immutably.

  manifest   what will be run: analyses, seed, trial counts, configuration hash, code hashes, git state
  guard      the run refuses unless the tree is clean, the configuration matches the manifest, and no
             completed run exists (a rerun needs a stated reason and is logged)
  store      every analysis saved as it ran (ledger, decisions, daily and paired returns, stats, records),
             the inference output, a SHA256SUMS file, then made read-only
  log        an append-only attempt log, written BEFORE the run starts and after it ends or fails

Nothing here changes a result: it only fixes what is run and keeps what came out.
"""
import hashlib
import json
import os
import platform
import stat
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.basket import SCORED_END, SCORED_START, build_inputs
from portfolio.inference import evaluate, load_inference_config
from portfolio.strategy_runner import run_strategy
from portfolio.trend_signal import FROZEN_CONFIG, load_signal_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "reports" / "registered_run"
SEED = 20261003                     # chosen at the freeze, before any strategy result was seen
TRIAL_COUNTS = {"low": 13, "central": 21, "high": 35, "headline": 35,
                "source": "docs/september/trial_inventory_2026-10-03.md (strategy-only counts plus 10 registered configurations)"}

# name -> run_strategy arguments. 'registered' ones feed the registered tests; 'sensitivity' ones are
# labelled, unregistered, and only stored.
ANALYSES = {
    "basket_buffered_10": {"role": "registered", "instrument": "basket", "policy": "buffered", "cost_bp": 10.0, "capital_policy": "fixed", "lag": 0},
    "basket_buffered_20": {"role": "registered", "instrument": "basket", "policy": "buffered", "cost_bp": 20.0, "capital_policy": "fixed", "lag": 0},
    "basket_full_10": {"role": "registered", "instrument": "basket", "policy": "full", "cost_bp": 10.0, "capital_policy": "fixed", "lag": 0},
    "btc_buffered_10": {"role": "registered", "instrument": "btc", "policy": "buffered", "cost_bp": 10.0, "capital_policy": "fixed", "lag": 0},
    "basket_buffered_5": {"role": "sensitivity", "instrument": "basket", "policy": "buffered", "cost_bp": 5.0, "capital_policy": "fixed", "lag": 0},
    "basket_buffered_10_equity": {"role": "sensitivity", "instrument": "basket", "policy": "buffered", "cost_bp": 10.0, "capital_policy": "equity", "lag": 0},
    "basket_buffered_10_lag1": {"role": "sensitivity", "instrument": "basket", "policy": "buffered", "cost_bp": 10.0, "capital_policy": "fixed", "lag": 1},
}
EVALUATE_KEYS = {"buffered": "basket_buffered_10", "buffered_stress": "basket_buffered_20",
                 "full": "basket_full_10", "btc": "btc_buffered_10"}
CODE_FILES = ("portfolio/basket.py", "portfolio/trend_signal.py", "portfolio/strategy_runner.py",
              "portfolio/inference.py", "portfolio/freeze.py", "strategies/ewma_crossover.py",
              "strategies/base.py", "data/volatility.py", "data/universe/selector.py")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def git_state() -> dict:
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))}


def build_manifest(seed: int = SEED, window: tuple = (SCORED_START, SCORED_END)) -> dict:
    """Everything that defines the run. Deterministic given the repository state."""
    cfg = load_inference_config()
    return {
        "purpose": "the registered historical run of the basket trend strategy (protocol Parts 4-5)",
        "seed": seed, "window": [str(window[0]), str(window[1])],
        "analyses": ANALYSES, "evaluate_keys": EVALUATE_KEYS, "trial_counts": TRIAL_COUNTS,
        "inference": {"replications": cfg.replications, "mean_block_days": cfg.mean_block_days,
                      "sensitivity_blocks": list(cfg.sensitivity_block_days), "newey_west_lags": cfg.newey_west_lags,
                      "hurdle_annual": cfg.hurdle_annual, "stress_bp": cfg.stress_bp},
        "frozen_config_sha256": load_signal_config().config_hash,
        "code_sha256": {f: sha256_file(ROOT / f) for f in CODE_FILES},
        "git": git_state(),
        "python": platform.python_version(), "packages": {p: _version(p) for p in ("pandas", "numpy", "yaml")},
    }


def _version(name):
    module = __import__(name)
    return getattr(module, "__version__", "unknown")


def manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()


def write_manifest(manifest: dict, store: Path = DEFAULT_STORE) -> Path:
    """Write once. A different manifest for the same store directory name is impossible (the name is its hash)."""
    folder = Path(store) / manifest_hash(manifest)[:12]
    path = folder / "manifest.json"
    if path.exists():
        if path.read_text() != json.dumps(manifest, indent=2, sort_keys=True, default=str):
            raise ValueError(f"{path} exists with different content")
        return path
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path


def log_attempt(folder: Path, status: str, **fields) -> None:
    """Append-only. Written before the run starts, and again when it completes or fails."""
    entry = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": status, **fields}
    with open(Path(folder) / "ATTEMPTS.jsonl", "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def read_attempts(folder: Path) -> list:
    path = Path(folder) / "ATTEMPTS.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def check_guard(manifest: dict, folder: Path, rerun_reason: str | None, check_git: bool) -> list:
    """Raises unless this run may start; returns the notes to log (e.g. that the git check was skipped)."""
    notes = []
    if manifest["frozen_config_sha256"] != load_signal_config().config_hash:
        raise ValueError("frozen_config.yaml differs from the manifest")
    changed = [f for f, h in manifest["code_sha256"].items() if sha256_file(ROOT / f) != h]
    if changed:
        raise ValueError(f"code changed since the manifest was written: {changed}")
    if check_git:
        state = git_state()
        if state["dirty"]:
            raise ValueError("the working tree has uncommitted changes")
        if state["commit"] != manifest["git"]["commit"] and not _is_doc_only_after(manifest["git"]["commit"]):
            raise ValueError("HEAD differs from the manifest's commit by more than documents")
    else:
        notes.append("git check skipped: this run is NOT registered")
    if any(a["status"] == "completed" for a in read_attempts(folder)) and not rerun_reason:
        raise ValueError("a completed run already exists; a rerun needs a stated reason")
    return notes


def _is_doc_only_after(commit: str) -> bool:
    changed = _git("diff", "--name-only", f"{commit}..HEAD").splitlines()
    return all(f.startswith("docs/") or f.startswith("reports/") for f in changed)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _save_run(folder: Path, name: str, result) -> None:
    out = Path(folder) / name
    out.mkdir(parents=True, exist_ok=True)
    result.ledger.to_csv(out / "ledger.csv.gz", index=False)
    result.decisions.to_csv(out / "decisions.csv.gz")
    result.daily.to_csv(out / "daily.csv.gz")
    result.paired.to_csv(out / "paired.csv.gz")
    (out / "stats.json").write_text(json.dumps(result.stats, indent=2, sort_keys=True, default=_jsonable))
    (out / "record.json").write_text(json.dumps(result.record, indent=2, sort_keys=True, default=_jsonable))


def seal(folder: Path) -> str:
    """SHA256SUMS over every stored file, then make the folder read-only. Returns the digest of the sums."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "ATTEMPTS.jsonl"))
    sums = "".join(f"{sha256_file(p)}  {p.relative_to(folder)}\n" for p in files)
    sums_path = folder / "SHA256SUMS"
    if sums_path.exists():
        sums_path.chmod(stat.S_IWUSR | stat.S_IRUSR)
    sums_path.write_text(sums)
    for p in files + [sums_path]:
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return hashlib.sha256(sums.encode()).hexdigest()


def verify(folder: Path) -> bool:
    """True iff every stored file still matches SHA256SUMS."""
    folder = Path(folder)
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        digest, _, rel = line.partition("  ")
        if sha256_file(folder / rel) != digest:
            return False
    return True


def run_registered(manifest: dict, store: Path = DEFAULT_STORE, rerun_reason: str | None = None,
                   check_git: bool = True, inputs_provider=None, reps: int | None = None,
                   window: tuple | None = None) -> Path:
    """The one entry point. Writes the manifest, logs the attempt, runs exactly the listed analyses, runs the
    registered inference, stores everything and seals it. `inputs_provider(instrument)` and `reps`/`window`
    exist so the machinery can be tested on synthetic data; any such override is logged and makes the run
    unregistered."""
    manifest_path = write_manifest(manifest, store)
    folder = manifest_path.parent
    notes = check_guard(manifest, folder, rerun_reason, check_git)
    overridden = inputs_provider is not None or reps is not None or window is not None
    if overridden:
        notes.append("test overrides in use: this run is NOT registered")
    reruns = [d for d in folder.glob("rerun_*") if d.is_dir()]
    target = folder if not any(a["status"] == "completed" for a in read_attempts(folder)) else folder / f"rerun_{len(reruns) + 1}"
    log_attempt(folder, "started", manifest_sha256=sha256_file(manifest_path), rerun_reason=rerun_reason, notes=notes,
                git=git_state(), target=str(target.relative_to(folder)) if target != folder else ".")
    start, end = (SCORED_START, SCORED_END) if window is None else window
    provider = inputs_provider or (lambda instrument: build_inputs(end, instrument))
    try:
        runs, cache = {}, {}
        for name, spec in manifest["analyses"].items():
            instrument = spec["instrument"]
            cache.setdefault(instrument, provider(instrument))
            runs[name] = run_strategy(spec["policy"], spec["capital_policy"], inputs=cache[instrument], start=start,
                                      end=end, signal_lag_days=spec["lag"], instrument=instrument,
                                      cost_bp_per_side=spec["cost_bp"])
            _save_run(target, name, runs[name])
        results = {key: runs[name] for key, name in manifest["evaluate_keys"].items()}
        analysis = evaluate(results, seed=manifest["seed"], reps=reps)
        (target / "inference.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, default=_jsonable))
        digest = seal(target)
    except Exception as error:
        log_attempt(folder, "failed", error=repr(error), trace=traceback.format_exc(limit=6))
        raise
    log_attempt(folder, "completed", target=str(target.relative_to(folder)) if target != folder else ".", sums_sha256=digest, registered=bool(analysis["record"]["registered_run"] and not overridden),
                not_registered_because=analysis["record"]["not_registered_because"] + notes)
    return target
