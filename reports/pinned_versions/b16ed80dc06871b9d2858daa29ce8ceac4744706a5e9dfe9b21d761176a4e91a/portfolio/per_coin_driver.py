"""The per-coin run driver: the run manifest, the guarded attempts, the sealed store and blind mode.

Design: docs/october/per_coin_run_driver_design_2026-10-24.md §5-6.

  manifest    runs and permitted files, the dataset manifest (§4.1), the expected membership and K5
              hashes, seed, trials, config and code hashes, git state; written once, named by its hash
  attempts    every attempt gets its own directory, attempt_<n>/, never reused; a reason is required
              after ANY earlier attempt of the stage (started, failed or completed, under any manifest);
              the study-wide INVENTORY.jsonl, append-only, lists them all, blind runs included
  store       each attempt's ledgers, daily tables, attributions, records and inference.json, sealed
              read-only with SHA256SUMS
  blind       a real-data smoke test whose child process discloses only a validated typed record:
              enumerated checks (passed / failed / not_run), the planned totals, and an enumerated error
              category. Controlled performance disclosure: its duration and resource use stay observable

Run from the repo root:
  python -m portfolio.per_coin_driver blind          # D9: stage A on real data, blind
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from portfolio.freeze import _jsonable, git_state, manifest_hash, seal, sha256_file, _is_doc_only_after
from portfolio.per_coin_inputs import (PER_COIN_CONFIG, SEPTEMBER_CONFIG, CheckedLoader, DataManifestError,
                                       PermittedSubsetError, dataset_manifest, frame_hash, membership_hash,
                                       stage_a_inventory)
from portfolio.per_coin_runs import BY_NAME, CAPITAL, WINDOW, build_run, check_run_list, contracts_of, runs_for

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "reports" / "per_coin_run"
SEED = 20261007
K4_REFERENCE = "reports/registered_run/c3e68cb1f90a"
STAGE_B_SPEC = "docs/october/per_coin_stage_b_specification.md"
CODE_FILES = (
    "portfolio/per_coin_driver.py", "portfolio/per_coin_runs.py", "portfolio/per_coin_inputs.py",
    "portfolio/per_coin_inference.py", "portfolio/per_coin_universe.py", "portfolio/per_coin_signal.py",
    "portfolio/per_coin_sizing.py", "portfolio/per_coin_model.py", "portfolio/per_coin_ledger.py",
    "portfolio/per_coin_attribution.py", "portfolio/inference.py", "portfolio/reporting.py", "portfolio/freeze.py",
    "portfolio/basket.py", "portfolio/trend_signal.py", "strategies/base.py", "strategies/ewma_crossover.py",
    "data/volatility.py", "data/universe/selector.py", "data/universe/coinmarketcap/new_fetcher/panel.py",
    "data/universe/coinmarketcap/new_fetcher/normalize.py", "data/universe/coinmarketcap/new_fetcher/contract_map.py")

CHECKS = ("data_manifest", "permitted_subset", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "attribution",
          "identity", "inference")
STATUSES = ("passed", "failed", "not_run")
ERROR_CATEGORIES = ("data_manifest", "identity", "validation", "reconciliation", "attribution", "timeout",
                    "crash", "other")
RECORD_KEYS = ("checks", "planned_runs", "planned_days", "error_category")
MAX_RECORD_BYTES = 4096


# ================================================================== the run manifest
def planned_days(window=WINDOW) -> int:
    return (window[1] - window[0]).days + 1


def build_run_manifest(stage: str = "A", source=None, dataset: dict | None = None,
                       stage_b_spec: str = STAGE_B_SPEC) -> dict:
    """Everything that defines stage A. The dataset manifest is hashed first; membership and the K5
    reference are then computed THROUGH the checked loader, so their expected hashes come from frozen
    bytes, before any book runs."""
    if stage != "A":
        raise NotImplementedError("stage B waits on ledger steps 7-8 and the event register")
    from portfolio.per_coin_runs import real_source
    check_run_list()
    dataset = dataset if dataset is not None else dataset_manifest(stage_a_inventory())
    source = source or real_source()
    window = source.window
    loader = CheckedLoader(dataset)
    with loader.active():
        with loader.scope("membership"):
            members = source.members("top10")
        with loader.scope("k5"):
            k5 = source.k5_forecast(members)
            k5_index = frame_hash(source.k5_index(members)) if source.k5_index else None
        permitted = {r.name: source.permitted(r.universe) if source.permitted else None for r in runs_for(stage)}
    runs = runs_for(stage)
    return {
        "purpose": "the registered per-coin run, stage A (protocol Part 9, Clarification 4)",
        "stage": stage, "seed": SEED, "window": [str(window[0]), str(window[1])], "capital": CAPITAL,
        "runs": {r.name: {"role": r.role, "twin": r.twin, "counted": r.counted, **r.parameters()} for r in runs},
        "permitted_files": permitted, "dataset": dataset,
        "expected": {"membership": membership_hash(members), "k5_index": k5_index, "k5_forecast": frame_hash(k5),
                     "membership_contracts": list(contracts_of(members))},
        "trials": {"for_dsr": 58, "bracket": [30, 80]},
        "configs": {PER_COIN_CONFIG: sha256_file(ROOT / PER_COIN_CONFIG),
                    SEPTEMBER_CONFIG: sha256_file(ROOT / SEPTEMBER_CONFIG)},
        "stage_b_specification": {"path": stage_b_spec,
                                  "sha256": sha256_file(ROOT / stage_b_spec) if (ROOT / stage_b_spec).exists() else None},
        "k4_reference": K4_REFERENCE,
        "code_sha256": {f: sha256_file(ROOT / f) for f in CODE_FILES},
        "git": git_state(), "python": platform.python_version(),
    }


def write_manifest(manifest: dict, store: Path = STORE) -> Path:
    folder = Path(store) / manifest_hash(manifest)[:12]
    path = folder / "manifest.json"
    text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"{path} exists with different content")
        return path
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o444)
    return path


# ================================================================== the study-wide inventory
def inventory_path(store: Path = STORE) -> Path:
    return Path(store) / "INVENTORY.jsonl"


def log_inventory(store: Path, **entry) -> None:
    """Append-only, one line per event: every attempt and every blind run, under every manifest."""
    line = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    Path(store).mkdir(parents=True, exist_ok=True)
    with open(inventory_path(store), "a") as handle:
        handle.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def read_inventory(store: Path = STORE) -> list:
    path = inventory_path(store)
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# ================================================================== the guard
def check_guard(manifest: dict, store: Path, reason: str | None, check_git: bool) -> list:
    """Raises unless an attempt may start. Returns notes (e.g. that the git check was skipped)."""
    notes = []
    for path, digest in manifest["configs"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"{path} differs from the manifest")
    changed = [f for f, h in manifest["code_sha256"].items() if sha256_file(ROOT / f) != h]
    if changed:
        raise ValueError(f"code changed since the manifest was written: {changed}")
    spec = manifest["stage_b_specification"]
    if manifest["stage"] == "A" and (spec["sha256"] is None or not (ROOT / spec["path"]).exists()
                                     or sha256_file(ROOT / spec["path"]) != spec["sha256"]):
        raise ValueError("stage B must be specified, committed and unchanged before stage A runs")
    if check_git:
        state = git_state()
        if state["dirty"]:
            raise ValueError("the working tree has uncommitted changes")
        if state["commit"] != manifest["git"]["commit"] and not _is_doc_only_after(manifest["git"]["commit"]):
            raise ValueError("HEAD differs from the manifest's commit by more than documents")
    else:
        notes.append("git check skipped: this attempt is NOT registered")
    earlier = [e for e in read_inventory(store) if e.get("kind") == "attempt" and e.get("stage") == manifest["stage"]]
    if earlier and not (reason and reason.strip()):
        raise ValueError(f"{len(earlier)} earlier attempt event(s) exist for stage {manifest['stage']}: "
                         "a new attempt needs a stated reason")
    return notes


def next_attempt_dir(folder: Path) -> Path:
    n = 1 + max([int(p.name.split("_")[1]) for p in Path(folder).glob("attempt_*")] or [0])
    target = Path(folder) / f"attempt_{n}"
    target.mkdir(exist_ok=False)
    return target


# ================================================================== running the family
def check_expected(manifest: dict, members, k5, k5_index=None) -> None:
    """The derived inputs fixed at the freeze: a run computing anything else stops."""
    from portfolio.per_coin_inference import IdentityError
    got = {"membership": membership_hash(members), "k5_forecast": frame_hash(k5),
           "k5_index": None if k5_index is None else frame_hash(k5_index)}
    for k, v in got.items():
        if manifest["expected"][k] != v:
            raise IdentityError(f"{k} differs from the hash fixed at the freeze")


def run_books(manifest: dict, source) -> dict:
    """Every book of the manifest, each inside its own checked-loader scope: a read of a file outside
    the dataset manifest, with a different hash, or outside the book's permitted subset raises."""
    loader = CheckedLoader(manifest["dataset"])
    results = {}
    with loader.active():
        with loader.scope("membership"):
            members = source.members("top10")
        with loader.scope("k5"):
            k5 = source.k5_forecast(members)
            k5_index = source.k5_index(members) if source.k5_index else None
        check_expected(manifest, members, k5, k5_index)
        for name in manifest["runs"]:
            spec = BY_NAME[name]
            with loader.scope(name, manifest["permitted_files"][name]):
                results[name] = build_run(spec, source, k5=k5 if spec.forecast == "k5" else None)
            results[name].record["files_read"] = loader.reads.get(name, {})
    return results


BOOK_CHECKS = ("data_manifest", "permitted_subset", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "attribution")


def run_family(manifest: dict, source, reps: int | None = None, progress=None) -> tuple:
    """Every book, then the identity checks and the analysis. `progress(check)` is called as each
    enumerated check has completed on every book."""
    from portfolio.per_coin_inference import evaluate_per_coin
    progress = progress or (lambda check: None)
    results = run_books(manifest, source)
    for check in BOOK_CHECKS:
        progress(check)
    analysis = evaluate_per_coin(results, manifest["seed"], reps=reps, k4_reference={"path": manifest["k4_reference"]})
    progress("identity")
    progress("inference")
    return results, analysis


def _save(target: Path, results: dict, analysis: dict) -> None:
    for name, r in results.items():
        out = target / name
        out.mkdir()
        r.rows.to_csv(out / "rows.csv.gz", index=False)
        r.daily.to_csv(out / "daily.csv.gz")
        r.split.to_csv(out / "attribution.csv.gz")
        (out / "record.json").write_text(json.dumps(r.record, indent=2, sort_keys=True, default=_jsonable))
    (target / "inference.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, default=_jsonable))


def run_attempt(manifest: dict, store: Path = STORE, reason: str | None = None, check_git: bool = True,
                source=None, reps: int | None = None) -> Path:
    """The one entry point for a stage-A attempt. Any override (a synthetic source, fewer replications,
    no git check) is logged and makes the attempt unregistered."""
    from portfolio.per_coin_runs import real_source
    folder = write_manifest(manifest, store).parent
    notes = check_guard(manifest, store, reason, check_git)
    if source is not None or reps is not None:
        notes.append("test overrides in use: this attempt is NOT registered")
    target = next_attempt_dir(folder)
    common = {"kind": "attempt", "stage": manifest["stage"], "manifest": folder.name, "attempt": target.name}
    log_inventory(store, **common, status="started", reason=reason, notes=notes, git=git_state())
    try:
        results, analysis = run_family(manifest, source or real_source(), reps=reps)
        analysis["record"]["not_registered_because"] += notes
        _save(target, results, analysis)
        digest = seal(target)
    except Exception as error:
        log_inventory(store, **common, status="failed", error=repr(error), trace=traceback.format_exc(limit=6))
        raise
    log_inventory(store, **common, status="completed", sums_sha256=digest,
                  registered=not analysis["record"]["not_registered_because"])
    return target


# ================================================================== blind mode (§6)
def error_category(error: BaseException) -> tuple:
    """(category, failed check or None), from the exception's type and the engine's own prefixes."""
    from portfolio.per_coin_inference import IdentityError
    text = str(error)
    if isinstance(error, PermittedSubsetError):
        return "data_manifest", "permitted_subset"
    if isinstance(error, DataManifestError):
        return "data_manifest", "data_manifest"
    if isinstance(error, IdentityError):
        return "identity", "identity"
    if text.startswith("reconcile R") and text[11:12].isdigit():
        return "reconciliation", f"R{text[11]}"
    if text.startswith("reconcile"):
        return "reconciliation", None
    if text.startswith("attribution"):
        return "attribution", "attribution"
    if isinstance(error, (ValueError, NotImplementedError)):
        return "validation", None
    return "other", None


def empty_record(manifest: dict, category: str | None) -> dict:
    return {"checks": {c: "not_run" for c in CHECKS}, "planned_runs": len(manifest["runs"]),
            "planned_days": planned_days(tuple(map(_date, manifest["window"]))), "error_category": category}


def _date(text):
    from datetime import date
    return date.fromisoformat(text)


def validate_record(raw: bytes, manifest: dict) -> dict:
    """The parent accepts only this exact shape; anything else becomes error_category 'other'."""
    fallback = empty_record(manifest, "other")
    if len(raw) > MAX_RECORD_BYTES:
        return fallback
    try:
        record = json.loads(raw.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return fallback
    if not isinstance(record, dict) or set(record) != set(RECORD_KEYS):
        return fallback
    checks = record["checks"]
    if not isinstance(checks, dict) or set(checks) != set(CHECKS) or any(v not in STATUSES for v in checks.values()):
        return fallback
    planned = empty_record(manifest, None)
    if record["planned_runs"] != planned["planned_runs"] or record["planned_days"] != planned["planned_days"]:
        return fallback
    if record["error_category"] is not None and record["error_category"] not in ERROR_CATEGORIES:
        return fallback
    if record["error_category"] is None and any(v != "passed" for v in checks.values()):
        return fallback
    return {"checks": {c: checks[c] for c in CHECKS}, "planned_runs": planned["planned_runs"],
            "planned_days": planned["planned_days"], "error_category": record["error_category"]}


def _restrict_writes(allowed: Path) -> None:
    """Inside the child: any Python-level write outside `allowed` raises (an audit hook cannot be
    removed once added). Reads are unaffected."""
    allowed = str(allowed.resolve())
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def inside(path) -> bool:
        if isinstance(path, int):
            return True                                             # an already-open descriptor
        p = os.path.realpath(os.fsdecode(path))
        return p == allowed or p.startswith(allowed + os.sep) or p == os.devnull

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(m in mode for m in "wax+")) or \
                      (mode is None and isinstance(flags, int) and flags & write_flags)
            if writing and path is not None and not inside(path):
                raise PermissionError("blind mode: write outside the run directory")
        elif event in ("os.mkdir", "os.rename", "os.replace", "os.remove", "os.rmdir", "os.symlink", "os.link",
                       "os.truncate", "os.chmod", "os.utime", "shutil.copyfile", "shutil.rmtree", "shutil.move"):
            paths = [a for a in args if isinstance(a, (str, bytes, os.PathLike))]
            if any(not inside(p) for p in paths):
                raise PermissionError(f"blind mode: {event} outside the run directory")

    sys.addaudithook(hook)


def _silence() -> None:
    import logging
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.CRITICAL)
    logging.disable(logging.CRITICAL)


def blind_child(manifest_path: str, out_dir: str, target: str = "real") -> None:
    """The child process: run the family with every channel closed except one typed record, written
    into the parent's directory. `target` other than 'real' names a synthetic test scenario."""
    out = Path(out_dir)
    _silence()
    _restrict_writes(out)
    manifest = json.loads(Path(manifest_path).read_text())
    record = empty_record(manifest, None)
    try:
        if target == "real":
            from portfolio.per_coin_runs import real_source
            run_family(manifest, real_source(), reps=200, progress=lambda c: record["checks"].__setitem__(c, "passed"))
        else:
            from portfolio import test_per_coin_driver as scenarios
            scenarios.blind_scenario(target, manifest, record)
    except BaseException as error:                                   # noqa: BLE001 - every failure is categorised
        category, failed = error_category(error)
        record = empty_record(manifest, category)
        if failed:
            record["checks"][failed] = "failed"
    (out / "result.json").write_text(json.dumps(record))


SANDBOX = "/usr/bin/sandbox-exec"


def _sandbox_prefix(work: Path) -> list:
    """The operating-system layer (macOS): every file write outside `work` is denied to the child and
    everything it starts. Absent on other systems, where the in-process audit hook still applies."""
    if not Path(SANDBOX).exists():
        return []
    profile = f'(version 1)(allow default)(deny file-write* (subpath "/"))(allow file-write* (subpath "{work}"))'
    return [SANDBOX, "-p", profile]


def run_blind(manifest_path: Path, store: Path = STORE, timeout: float = 4 * 3600, target: str = "real") -> dict:
    """The parent: a fresh directory it owns, a child with no output channel but that directory, a
    validated record, and cleanup after any exit. The inventory gets only the validated record."""
    manifest = json.loads(Path(manifest_path).read_text())
    work = Path(tempfile.mkdtemp(prefix="per_coin_blind_")).resolve()
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "VIRTUAL_ENV")}
    env.update({"TMPDIR": str(work), "MPLCONFIGDIR": str(work), "XDG_CACHE_HOME": str(work),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)})

    def no_core_dumps():
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    try:
        try:
            done = subprocess.run([*_sandbox_prefix(work), sys.executable, "-m", "portfolio.per_coin_driver", "blind-child",
                                   str(Path(manifest_path).resolve()), str(work), target],
                                  cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=timeout, preexec_fn=no_core_dumps)
            result = work / "result.json"
            if done.returncode != 0:
                record = empty_record(manifest, "crash")
            elif not result.is_file() or result.is_symlink():
                record = empty_record(manifest, "other")
            else:
                record = validate_record(result.read_bytes()[:MAX_RECORD_BYTES + 1], manifest)
        except subprocess.TimeoutExpired:
            record = empty_record(manifest, "timeout")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log_inventory(store, kind="blind", stage=manifest["stage"], manifest=Path(manifest_path).parent.name,
                  target=target, boundary=["audit_hook"] + (["sandbox_exec"] if _sandbox_prefix(work) else []),
                  record=record)
    return record


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "blind-child":
        blind_child(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "real")
    elif len(sys.argv) >= 2 and sys.argv[1] == "blind":
        raise SystemExit("D9 needs a written stage-A manifest and the researcher's approval; not run here")
    else:
        raise SystemExit(__doc__)
