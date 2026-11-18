"""Run from the repo root: python -m portfolio.test_per_coin_driver

The per-coin run driver (design: docs/october/per_coin_run_driver_design_2026-10-24.md §8, tests D1-D8).
Synthetic data only: no real-data return is computed. D9 (stage A on real data, blind) is run
separately, after the researcher approves it. `blind_scenario` holds the hostile child scenarios D5
runs through the real blind boundary.
"""
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio.test_per_coin_model as M
from portfolio.per_coin_driver import (CHECKS, ROOT, build_run_manifest, check_guard, error_category, read_inventory,
                                       run_attempt, run_blind, run_books, run_family, validate_record, write_manifest)
from portfolio.per_coin_inference import (COLLAPSED, FAIL, FRAGILE, H1_PASSES, INCONCLUSIVE, LARGE_CLAIM_REJECTED,
                                          NOT_PER_COIN, IdentityError, check_identity, evaluate_per_coin, verdict)
from portfolio.per_coin_inputs import DataManifestError, PermittedSubsetError, dataset_manifest, relative
from portfolio.per_coin_runs import RUNS, BY_NAME, Source, btc_members, check_run_list, runs_for

WINDOW = (M.DAYS[150], M.DAYS[400])
CORE = ("strategy", "k1", "cost_20bp", "cost_20bp_k1")         # the books evaluate_per_coin cannot do without
STAGE_B_STAND_IN = "docs/october/per_coin_protocol_2026-10-07.md"   # any committed file, for tests
LEAK_PROBE = ROOT / "reports" / "per_coin_blind_leak_probe.txt"
REPS = 200


def assert_raises(call, kind=ValueError, *fragments):
    try:
        call()
    except kind as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return error
    raise AssertionError(f"expected {kind.__name__}")


# ------------------------------------------------------------------ a synthetic world as a Source
def hourly_marks(bars):
    out = {}
    for c, b in bars.items():
        idx = pd.date_range(pd.Timestamp(M.DAYS[0], tz="UTC"), periods=24 * len(M.DAYS), freq="h")
        drift = 1.0 + 0.001 * np.tile(np.arange(24) / 24.0, len(M.DAYS))
        out[c] = pd.DataFrame({"open": np.repeat(b.open.to_numpy(), 24) * drift}, index=idx)
    return out


def synthetic_source(folder: Path | None = None, seed=5, permitted=True, k5_shift=0.0) -> Source:
    """The model tests' four-coin world. With `folder`, daily bars are written there and read back
    through pandas, so the checked loader sees them; otherwise everything stays in memory."""
    bars, counts = M.world(seed=seed)
    settlements, marks = M.settlements(bars), hourly_marks(bars)
    members = {M.D0: pd.DataFrame({"contract": list(M.COINS), "weight": 1.0 / len(M.COINS)})}
    k5 = pd.Series(np.sin(np.arange(len(M.DAYS)) / 9.0) * 12.0 + k5_shift, index=M.DAYS)
    files = {}
    if folder is not None:
        for c, b in bars.items():
            files[c] = Path(folder) / f"{c}_1d.csv"
            if not files[c].exists():
                b.to_csv(files[c], index=False)

    def load_bars(contracts):
        if folder is None:
            return {c: bars[c] for c in contracts}
        out = {}
        for c in contracts:
            frame = pd.read_csv(files[c])
            frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
            out[c] = frame
        return out

    def universe(name):
        return members if name == "top10" else btc_members(members)

    def allowed(name):
        if not permitted or folder is None:
            return None
        coins = list(M.COINS) if name == "top10" else (["BTCUSDT"] if permitted != "deny_btc" else [])
        return sorted(relative(files[c]) for c in coins)

    return Source(members=universe, bars=load_bars, mark_counts=lambda cs: {c: counts[c] for c in cs},
                  k5_forecast=lambda m: k5, settlements=lambda cs: {c: settlements[c] for c in cs},
                  marks=lambda cs: {c: marks[c] for c in cs}, permitted=allowed, window=WINDOW)


def manifest_for(source, runs=CORE, folder=None) -> dict:
    dataset = dataset_manifest(sorted(Path(folder).glob("*.csv"))) if folder else {}
    manifest = build_run_manifest(source=source, dataset=dataset, stage_b_spec=STAGE_B_STAND_IN)
    manifest["runs"] = {k: v for k, v in manifest["runs"].items() if k in runs}
    manifest["permitted_files"] = {k: v for k, v in manifest["permitted_files"].items() if k in runs}
    return manifest


_CACHE = {}


def family(runs=CORE):
    """Books on the in-memory world, computed once per run list."""
    if runs not in _CACHE:
        source = synthetic_source()
        _CACHE[runs] = run_books(manifest_for(source, runs), source)
    return _CACHE[runs]


# ------------------------------------------------------------------ D1
def test_D1_the_run_list_matches_the_protocol_and_config():
    check_run_list()
    assert len(runs_for("A")) == 17 and len(runs_for("B")) == 10
    assert sum(1 for r in RUNS if r.counted) == 11
    import portfolio.per_coin_runs as R
    broken = tuple(r for r in RUNS if r.name != "top_50")                       # a counted sensitivity missing
    assert_raises(lambda: check_run_list(broken), ValueError, "counted runs")
    original = R.BY_NAME["same_bar_k1"]
    R.BY_NAME["same_bar_k1"] = replace(original, cost_bp=20.0)                 # a twin differing in more than the forecast
    try:
        assert_raises(lambda: check_run_list(), ValueError, "twin")
    finally:
        R.BY_NAME["same_bar_k1"] = original


# ------------------------------------------------------------------ D2
def test_D2_every_stage_a_book_and_the_analysis_end_to_end_unregistered():
    store = Path(tempfile.mkdtemp())
    try:
        source = synthetic_source()
        manifest = manifest_for(source, runs=tuple(r.name for r in runs_for("A")))
        target = run_attempt(manifest, store, check_git=False, source=source, reps=REPS)
        analysis = json.loads((target / "inference.json").read_text())
        assert set(analysis["record"]["runs"]) == {r.name for r in runs_for("A")}
        assert analysis["verdict"]["headline"] in (FAIL, INCONCLUSIVE, H1_PASSES)
        assert analysis["record"]["not_registered_because"]                    # fewer reps, synthetic, no git check
        assert len(analysis["record"]["identity_checks"]) == 7 + 4             # 7 twin pairs (strategy/k1 and 6 sensitivities) + K2, K3, K5, full
        assert set(analysis["sensitivities"]) == {r.name for r in runs_for("A") if r.counted}
        from portfolio.freeze import verify
        assert verify(target) and not os.access(target / "inference.json", os.W_OK)
        status = [e["status"] for e in read_inventory(store)]
        assert status == ["started", "completed"]
    finally:
        _chmod_rm(store)


# ------------------------------------------------------------------ D3
def test_D3_a_changed_file_fails_even_when_every_book_reads_the_same_altered_file():
    folder = Path(tempfile.mkdtemp())
    try:
        source = synthetic_source(folder)
        manifest = manifest_for(source, runs=("strategy", "k1"), folder=folder)
        results = run_books(manifest, source)                                  # unaltered: passes
        assert set(results["strategy"].record["files_read"]) == set(manifest["dataset"])
        missing = {**manifest, "dataset": {k: v for k, v in manifest["dataset"].items() if "BBB" not in k}}
        assert_raises(lambda: run_books(missing, synthetic_source(folder)), DataManifestError, "not in the dataset")
        frame = pd.read_csv(folder / "AAA_1d.csv")
        frame.loc[10, "open"] *= 1.0001                                        # one pre-window price
        frame.to_csv(folder / "AAA_1d.csv", index=False)
        assert_raises(lambda: run_books(manifest, synthetic_source(folder)), DataManifestError, "AAA_1d.csv")
    finally:
        shutil.rmtree(folder)


def test_D3_permitted_subsets_k2_reads_its_own_files_and_nothing_else():
    folder = Path(tempfile.mkdtemp())
    try:
        source = synthetic_source(folder)
        manifest = manifest_for(source, runs=("strategy", "k2"), folder=folder)
        results = run_books(manifest, source)                                  # different file sets, both legitimate
        assert list(results["k2"].record["files_read"]) == [relative(folder / "BTCUSDT_1d.csv")]
        assert len(results["strategy"].record["files_read"]) == 4
        denied = manifest_for(synthetic_source(folder, permitted="deny_btc"), runs=("strategy", "k2"), folder=folder)
        assert_raises(lambda: run_books(denied, source), PermittedSubsetError, "k2")
    finally:
        shutil.rmtree(folder)


def _with(result, **changes):
    """A copy of a result whose record differs as stated: 'parameter:x' or 'hash:x' keys."""
    record = json.loads(json.dumps(result.record, default=str))
    for key, value in changes.items():
        kind, field = key.split(":")
        record["parameters" if kind == "parameter" else "hashes"][field] = value
    return replace(result, record=record)


def test_D3_each_comparison_accepts_exactly_its_listed_differences():
    books = family(("strategy", "k1", "k2", "k3", "k5"))
    s = books["strategy"]
    check_identity(s, books["k1"], "twin")
    check_identity(s, books["k3"], "forecast")
    check_identity(s, books["k5"], "forecast")
    check_identity(s, books["k2"], "btc")
    check_identity(s, _with(s, **{"parameter:band": 0.0}), "buffer")
    rejected = [
        (books["k1"], {"hash:sigma": "x"}, "twin"),                           # a twin with another vol table
        (books["k1"], {"parameter:cost_bp": 20.0}, "twin"),
        (books["k3"], {"hash:rho_bar": "x"}, "forecast"),
        (books["k5"], {"parameter:lag": 0}, "forecast"),
        (books["k2"], {"parameter:cost_bp": 20.0}, "btc"),
        (s, {"parameter:band": 0.0, "hash:forecast": "x"}, "buffer"),
    ]
    for other, change, kind in rejected:
        assert_raises(lambda: check_identity(s, _with(other, **change), kind), IdentityError, "differs in")
    assert_raises(lambda: check_identity(s, s, "twin"), IdentityError, "does not differ")
    assert_raises(lambda: check_identity(s, _with(books["k2"], **{"parameter:universe": "top10"}), "btc"),
                  IdentityError)


# ------------------------------------------------------------------ D4
def V(**kw):
    base = dict(alpha=0.05, p_low=0.01, p_low_40=0.01, p_low_60=0.01, p_low_stress=0.01, h2_p_low=0.01,
                two_sided_share=0.5, sharpe_p95=1.5)
    return verdict(**{**base, **kw})


def test_D4_the_verdict_headlines_warnings_and_exact_thresholds():
    assert V() == {**V(), "headline": H1_PASSES, "warnings": [], "pass": True}
    assert V(alpha=-0.01, p_low=-0.05)["headline"] == FAIL and V(alpha=0.0)["headline"] == FAIL
    assert V(p_low=-0.01)["headline"] == INCONCLUSIVE
    assert V(p_low=0.0)["headline"] == INCONCLUSIVE                            # exactly 0 does not pass
    for change, warning in ((dict(p_low_stress=0.0), FRAGILE), (dict(p_low_40=-1e-9), FRAGILE),
                            (dict(p_low_60=-0.1), FRAGILE), (dict(h2_p_low=0.0), NOT_PER_COIN),
                            (dict(two_sided_share=0.1999), COLLAPSED), (dict(sharpe_p95=0.9999), LARGE_CLAIM_REJECTED)):
        v = V(**change)
        assert v["warnings"] == [warning] and v["headline"] == H1_PASSES
        assert v["pass"] is (warning == LARGE_CLAIM_REJECTED)                  # only this one does not block a pass
    assert V(two_sided_share=0.20)["warnings"] == [] and V(sharpe_p95=1.0)["warnings"] == []   # strict thresholds
    both = V(p_low_stress=-0.1, h2_p_low=-0.1)
    assert both["warnings"] == [FRAGILE, NOT_PER_COIN] and not both["pass"]
    failed = V(alpha=-0.02, p_low=-0.06, two_sided_share=0.05)
    assert failed["headline"] == FAIL and COLLAPSED in failed["warnings"]       # independent of the headline
    assert V(sharpe_p95=0.4)["pass"] and LARGE_CLAIM_REJECTED in V(sharpe_p95=0.4)["warnings"]
    for bad in (float("nan"), float("inf")):
        assert_raises(lambda: V(sharpe_p95=bad), ValueError, "not finite")


# ------------------------------------------------------------------ D5 (scenarios run inside the blind child)
def blind_scenario(target, manifest, record):
    out = Path(os.environ["TMPDIR"])
    if target == "ok":
        run_family(manifest, synthetic_source(), reps=50, progress=lambda c: record["checks"].__setitem__(c, "passed"))
    elif target == "noisy":                                                    # DEBUG logs, prints, warnings: all closed
        logging.getLogger("portfolio").debug("alpha=0.0123 sharpe=1.7")
        print("net Sharpe 1.234")
        sys.stderr.write("alpha 0.0123\n")
        run_family(manifest, synthetic_source(), reps=50, progress=lambda c: record["checks"].__setitem__(c, "passed"))
    elif target == "exception_with_values":
        raise RuntimeError("alpha 0.0123 sharpe 1.7")
    elif target == "reconcile_fails":
        raise ValueError("reconcile R7: price P&L differs by 0.0123 on BTCUSDT")
    elif target == "file_logger":
        logging.getLogger("leak").addHandler(logging.FileHandler(LEAK_PROBE))
    elif target == "write_outside":
        LEAK_PROBE.write_text("alpha 0.0123")
    elif target == "crash":
        os._exit(7)
    elif target == "timeout":
        time.sleep(120)
    elif target == "malformed":
        (out / "result.json").write_text(json.dumps({**record, "alpha": 0.0123}))
        os._exit(0)
    elif target == "overlong":
        (out / "result.json").write_text(json.dumps({**record, "error_category": "x" * 10_000}))
        os._exit(0)
    elif target == "counts_carry_information":
        (out / "result.json").write_text(json.dumps({**record, "planned_days": 123}))
        os._exit(0)
    elif target == "claims_success_without_checks":
        (out / "result.json").write_text(json.dumps(record))                   # all not_run, no error
        os._exit(0)
    else:
        raise ValueError(f"unknown scenario {target}")


def test_D5_blind_mode_discloses_only_a_validated_typed_record():
    store = Path(tempfile.mkdtemp())
    before = set(Path(tempfile.gettempdir()).glob("per_coin_blind_*"))
    try:
        source = synthetic_source()
        path = write_manifest(manifest_for(source), store)
        expect = {
            "ok": None, "noisy": None, "exception_with_values": "other", "reconcile_fails": "reconciliation",
            "file_logger": "other", "write_outside": "other", "crash": "crash", "timeout": "timeout",
            "malformed": "other", "overlong": "other", "counts_carry_information": "other",
            "claims_success_without_checks": "other"}
        for target, category in expect.items():
            record = run_blind(path, store, timeout=8 if target == "timeout" else 300, target=target)
            assert set(record) == {"checks", "planned_runs", "planned_days", "error_category"}, target
            assert record["error_category"] == category, (target, record)
            assert record["planned_runs"] == len(CORE) and record["planned_days"] == (WINDOW[1] - WINDOW[0]).days + 1
            statuses = record["checks"]
            if category is None:
                assert set(statuses.values()) == {"passed"}, target
            elif target == "reconcile_fails":
                assert statuses["R7"] == "failed" and {v for k, v in statuses.items() if k != "R7"} == {"not_run"}
            else:
                assert set(statuses.values()) == {"not_run"}, (target, statuses)
            assert not LEAK_PROBE.exists(), target
        lines = [e for e in read_inventory(store) if e["kind"] == "blind"]
        assert [e["target"] for e in lines] == list(expect)
        for e in lines:
            assert set(e) == {"time_utc", "kind", "stage", "manifest", "target", "boundary", "record"}
            assert "0.0123" not in json.dumps(e) and "1.7" not in json.dumps(e)
        assert set(Path(tempfile.gettempdir()).glob("per_coin_blind_*")) == before   # the parent cleaned up
    finally:
        LEAK_PROBE.unlink(missing_ok=True)
        _chmod_rm(store)


def test_D5_the_record_validator_rejects_every_other_shape():
    manifest = {"runs": dict.fromkeys(CORE), "window": [str(WINDOW[0]), str(WINDOW[1])]}
    good = {"checks": dict.fromkeys(CHECKS, "passed"), "planned_runs": 4, "planned_days": 251, "error_category": None}
    assert validate_record(json.dumps(good).encode(), manifest) == good
    for bad in ({**good, "extra": 1}, {**good, "checks": {**good["checks"], "R1": "maybe"}},
                {**good, "error_category": "alpha_was_0.01"}, {**good, "planned_runs": 5},
                {**good, "checks": {**good["checks"], "R1": "failed"}}):
        assert validate_record(json.dumps(bad).encode(), manifest)["error_category"] == "other"
    assert validate_record(b"\xff\xfe", manifest)["error_category"] == "other"
    assert error_category(DataManifestError("x"))[0] == "data_manifest"
    assert error_category(ValueError("reconcile R3: x")) == ("reconciliation", "R3")
    assert error_category(ValueError("attribution: x")) == ("attribution", "attribution")


# ------------------------------------------------------------------ D6
def test_D6_attempts_reasons_directories_inventory_and_the_guard():
    store = Path(tempfile.mkdtemp())
    folder = Path(tempfile.mkdtemp())
    try:
        source = synthetic_source(folder)
        manifest = manifest_for(source, folder=folder)
        first = run_attempt(manifest, store, check_git=False, source=source, reps=REPS)
        assert_raises(lambda: run_attempt(manifest, store, check_git=False, source=source, reps=REPS),
                      ValueError, "stated reason")
        second = run_attempt(manifest, store, reason="test: rerun", check_git=False, source=source, reps=REPS)
        assert first.name == "attempt_1" and second.name == "attempt_2" and first.parent == second.parent
        frame = pd.read_csv(folder / "CCC_1d.csv")                             # data changed after the freeze
        frame.loc[5, "volume"] = 6.0
        frame.to_csv(folder / "CCC_1d.csv", index=False)
        assert_raises(lambda: run_attempt(manifest, store, reason="test: data", check_git=False,
                                          source=synthetic_source(folder), reps=REPS), DataManifestError)
        assert (first.parent / "attempt_3").is_dir()                           # the failed attempt keeps its own directory
        events = [(e["attempt"], e["status"]) for e in read_inventory(store) if e["kind"] == "attempt"]
        assert events == [("attempt_1", "started"), ("attempt_1", "completed"), ("attempt_2", "started"),
                          ("attempt_2", "completed"), ("attempt_3", "started"), ("attempt_3", "failed")]
        assert [e["reason"] for e in read_inventory(store) if e["status"] == "started"] == [None, "test: rerun", "test: data"]
        changed = {**manifest, "code_sha256": {**manifest["code_sha256"], "portfolio/per_coin_ledger.py": "0" * 64}}
        assert_raises(lambda: check_guard(changed, store, "r", False), ValueError, "code changed")
        config = {**manifest, "configs": {k: "0" * 64 for k in manifest["configs"]}}
        assert_raises(lambda: check_guard(config, store, "r", False), ValueError, "differs from the manifest")
        no_b = {**manifest, "stage_b_specification": {"path": "docs/october/nonexistent.md", "sha256": None}}
        assert_raises(lambda: check_guard(no_b, store, "r", False), ValueError, "stage B")
    finally:
        _chmod_rm(store)
        shutil.rmtree(folder)


# ------------------------------------------------------------------ D7
def test_D7_the_same_seed_gives_the_same_analysis():
    books = family()
    a = evaluate_per_coin(books, 20261007, reps=REPS)
    b = evaluate_per_coin(books, 20261007, reps=REPS)
    c = evaluate_per_coin(books, 1, reps=REPS)
    dump = lambda x: json.dumps(x, sort_keys=True, default=str)
    assert dump(a) == dump(b) and dump(a["h1"]) != dump(c["h1"])


# ------------------------------------------------------------------ D8
def test_D8_k2_k5_and_band_zero_differ_exactly_as_stated():
    books = family(("strategy", "k2", "k5", "full_rebalance"))
    k2 = books["k2"]
    assert k2.record["contracts"] == ["BTCUSDT"] and k2.tables.eligible.sum(axis=1).max() == 1   # N = 1, IDM = 1
    s, k5 = books["strategy"].tables, books["k5"].tables
    available = s.forecast.notna()
    k5_series = synthetic_source().k5_forecast(None)
    for c in s.forecast.columns:
        on = available[c]
        assert (k5.forecast.loc[on, c] == k5_series.reindex(s.forecast.index)[on]).all()
    check_identity(books["strategy"], books["full_rebalance"], "buffer")
    assert books["full_rebalance"].daily.turnover.sum() > books["strategy"].daily.turnover.sum()
    hold_days = (books["full_rebalance"].rows.event == "hold").sum()
    assert hold_days == 0 and (books["strategy"].rows.event == "hold").sum() > 0
    source = synthetic_source()
    manifest = manifest_for(source, runs=("strategy",))
    other = synthetic_source(k5_shift=0.5)                                     # a K5 forecast not the frozen one
    assert_raises(lambda: run_books(manifest, other), IdentityError, "k5_forecast")


# ------------------------------------------------------------------ stage B (synthetic)
SETTLE = M.DAYS[300]


def stage_b_synthetic_source() -> Source:
    """Top 25 = AAA, BBB, CCC; top 50 = all four. CCC is settled by the exchange at 09:00 on SETTLE
    and never trades again."""
    after = tuple(("CCC", d) for d in M.DAYS if d > SETTLE)
    bars, counts = M.world(halted=after)
    settlements, marks = M.settlements(bars), hourly_marks(bars)
    universes = {"top_25": ["AAA", "BBB", "CCC"], "top_50": list(M.COINS), "registry_10m": ["AAA", "BBB", "CCC", "BTCUSDT"]}
    instant = pd.Timestamp(SETTLE, tz="UTC") + pd.Timedelta(hours=9)
    register = pd.DataFrame([{"contract": "CCC", "kind": "settlement", "first_day": SETTLE,
                              "last_day": SETTLE + timedelta(days=2), "instant": instant, "evidence": "synthetic"}])
    level = float(bars["CCC"].open.iloc[M.DAYS.index(SETTLE)])

    def index_reader(contract, day):
        stamps = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=1440, freq="min")
        return pd.DataFrame({"open_time": stamps, "close": level * (1 + np.arange(1440) / 1e5)})

    # Stage D's ranking inputs: AAA > BBB > CCC by market cap, BTCUSDT unmapped. With the 10 / 50 ranks the
    # top-25 book (AAA, BBB, CCC) is one tier, so exactly stage C's; the registry book has two tiers.
    caps = pd.DataFrame([{"date": d, "cmc_id": i, "market_cap": v} for d in M.DAYS
                         for i, v in ((1, 400e9), (2, 300e9), (3, 200e9))])
    deep_map = pd.DataFrame({"cmc_id": [1, 2, 3], "contract": ["AAA", "BBB", "CCC"],
                             "claim_from": np.nan, "claim_to": np.nan})
    return Source(members=lambda u: {M.D0: pd.DataFrame({"contract": universes[u], "weight": 1.0})},
                  bars=lambda cs: {c: bars[c] for c in cs}, mark_counts=lambda cs: {c: counts[c] for c in cs},
                  k5_forecast=None, settlements=lambda cs: {c: settlements[c] for c in cs},
                  marks=lambda cs: {c: marks[c] for c in cs}, window=WINDOW, events=lambda: register,
                  index_reader=index_reader, caps=lambda: (caps, deep_map))


def test_stage_b_end_to_end_with_a_settlement_unregistered():
    store = Path(tempfile.mkdtemp())
    try:
        source = stage_b_synthetic_source()
        manifest = build_run_manifest("B", source=source, dataset={}, stage_b_spec=STAGE_B_STAND_IN)
        assert list(manifest["runs"]) == ["top_25", "top_25_k1", "top_50", "top_50_k1"]
        target = run_attempt(manifest, store, check_git=False, source=source, reps=REPS)
        a = json.loads((target / "inference.json").read_text())
        assert set(a["variants"]) == {"top_25", "top_50"} and len(a["record"]["identity_checks"]) == 2
        met = {b: [e["contract"] for e in a["books"][b]["settlements_met"]] for b in a["books"]}
        assert any("CCC" in v for v in met.values())                               # a held CCC was settled
        rows = pd.read_csv(target / "top_25" / "rows.csv.gz")
        ccc = rows[rows.contract == "CCC"]
        assert (ccc.event == "settlement").sum() == 1 and (ccc[ccc.day > str(SETTLE)].quantity_held == 0).all()
        assert "reading" in a["variants"]["top_25"] and a["record"]["not_registered_because"]
    finally:
        _chmod_rm(store)


def test_stage_b_guard_needs_the_specification_fixed_before_stage_a():
    from portfolio.per_coin_driver import STAGE_B_SPEC, STAGE_B_SPEC_SHA256
    from portfolio.freeze import sha256_file
    assert sha256_file(ROOT / STAGE_B_SPEC) == STAGE_B_SPEC_SHA256              # the committed file is unchanged
    store = Path(tempfile.mkdtemp())
    try:
        manifest = build_run_manifest("B", source=stage_b_synthetic_source(), dataset={})
        check_guard(manifest, store, None, False)
        altered = {**manifest, "stage_b_specification": {"path": STAGE_B_SPEC, "sha256": "0" * 64}}
        assert_raises(lambda: check_guard(altered, store, None, False), ValueError, "stage B")
    finally:
        _chmod_rm(store)


def test_stage_b_membership_must_match_the_freeze():
    source = stage_b_synthetic_source()
    manifest = build_run_manifest("B", source=source, dataset={}, stage_b_spec=STAGE_B_STAND_IN)
    manifest["expected"]["membership_by_universe"]["top_50"] = "0" * 64
    assert_raises(lambda: run_books(manifest, source), IdentityError, "top_50")


# ------------------------------------------------------------------ stage C (design 2026-10-27 §7)
def test_stage_c_run_list_matches_its_specification():
    from portfolio.per_coin_runs import STAGE_C_TRIALS, check_stage_c_runs
    check_stage_c_runs()
    assert len(runs_for("C")) == 8 and STAGE_C_TRIALS == 61
    check_run_list()                                                       # stages A and B unchanged by it
    bad = (replace(BY_NAME["top_25_dyn"], band=0.0), BY_NAME["top_25_dyn_k1"])
    others = tuple(r for r in runs_for("C") if not r.name.startswith("top_25_dyn"))
    assert_raises(lambda: check_stage_c_runs(others + bad), ValueError)


def sealed_counterparts(source, store: Path) -> dict:
    """Stage-B primary-sizing books on the synthetic world, sealed, standing in for the real sealed books."""
    names = ["registry_10m", "registry_10m_k1", "top_25", "top_25_k1", "top_50", "top_50_k1"]
    manifest = build_run_manifest("B", source=source, dataset={}, stage_b_spec=STAGE_B_STAND_IN, runs=names)
    target = run_attempt(manifest, store, check_git=False, source=source, reps=REPS)
    return {u: str(target / u) for u in ("registry_10m", "top_25", "top_50")}


def test_stage_c_end_to_end_unregistered():
    store, sealed_store = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    try:
        source = stage_b_synthetic_source()
        frozen = sealed_counterparts(source, sealed_store)
        manifest = build_run_manifest("C", source=source, dataset={}, frozen=frozen)
        assert manifest["stage"] == "C" and manifest["trials"]["for_dsr"] == 61 and len(manifest["runs"]) == 8
        assert all(r["sizing"] == "dynamic" for r in manifest["runs"].values())
        target = run_attempt(manifest, store, check_git=False, source=source, reps=REPS)
        a = json.loads((target / "inference.json").read_text())
        h1 = a["h1_prime"]
        assert h1["verdict"]["headline"] in (FAIL, INCONCLUSIVE, H1_PASSES) and "stress_20bp" in h1
        assert h1["verdict"]["inputs"]["p_low_stress"] == h1["stress_20bp"]["alpha_p_low"]
        assert set(h1["deflated_sharpe"]) == {"61", "30", "80"}
        assert set(a["secondaries"]) == {"top_25_dyn", "top_50_dyn"}
        assert set(a["versus_frozen_primary"]) == {"registry_10m_dyn", "top_25_dyn", "top_50_dyn"}
        assert len(a["record"]["identity_checks"]) == 4 and a["record"]["not_registered_because"]
        for name, risk in a["risk"].items():
            assert abs(risk["v_ideal_over_target"]["median"] - 1.0) < 1e-9, name    # the ideal book is at target
            assert risk["v_final_over_target"]["max"] <= 1.0 + 0.2, name             # buffer drift only, bounded
        sizing = pd.read_csv(target / "registry_10m_dyn" / "sizing.csv.gz")
        assert {"multiplier", "v_unscaled", "v_final", "gross_final", "gross_cap_bound"} <= set(sizing.columns)
        twin = pd.read_csv(target / "registry_10m_dyn_k1" / "rows.csv.gz")
        assert (twin.quantity_held >= 0).all()                                      # the twin is long-only
    finally:
        _chmod_rm(store)
        _chmod_rm(sealed_store)


def test_stage_c_guard_and_identity_against_the_sealed_books():
    from portfolio.per_coin_driver import STAGE_C_SPEC
    from portfolio.per_coin_inference import SealedBook, versus_frozen, load_per_coin_inference_config
    store, sealed_store = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    try:
        source = stage_b_synthetic_source()
        frozen = sealed_counterparts(source, sealed_store)
        manifest = build_run_manifest("C", source=source, dataset={}, frozen=frozen)
        check_guard(manifest, store, None, False)
        moved = {**manifest, "stage_c_specification": {"path": STAGE_C_SPEC, "sha256": "0" * 64}}
        assert_raises(lambda: check_guard(moved, store, None, False), ValueError, "stage C")
        swapped = json.loads(json.dumps(manifest))
        swapped["frozen_primary"]["top_25"]["sums_sha256"] = "0" * 64
        assert_raises(lambda: check_guard(swapped, store, None, False), ValueError, "top_25")
        keep = lambda d: {k: v for k, v in d.items() if k.startswith("top_25")}
        books = run_books({**manifest, "runs": keep(manifest["runs"]),
                           "permitted_files": keep(manifest["permitted_files"])}, source)
        cfg = load_per_coin_inference_config()
        versus_frozen(books["top_25_dyn"], SealedBook(frozen["top_25"]), cfg, 1, REPS)          # sizing only: accepted
        assert_raises(lambda: versus_frozen(books["top_25_dyn"], SealedBook(frozen["top_50"]), cfg, 1, REPS),
                      IdentityError)                                                          # another universe
        assert_raises(lambda: versus_frozen(books["top_25_dyn_k1"], SealedBook(frozen["top_25"]), cfg, 1, REPS),
                      IdentityError)                                                          # forecast differs too
        victim = Path(frozen["top_25"]) / "daily.csv.gz"
        os.chmod(victim, 0o644)
        victim.write_bytes(victim.read_bytes() + b"\n")
        assert_raises(lambda: SealedBook(frozen["top_25"]), IdentityError, "SHA256SUMS")
    finally:
        _chmod_rm(store)
        _chmod_rm(sealed_store)


# ------------------------------------------------------------------ stage D, trial 62 (design 2026-10-30)
def test_stage_d_run_list_matches_its_specification():
    from portfolio.per_coin_runs import STAGE_D_TRIALS, check_stage_c_runs, check_stage_d_runs
    check_stage_d_runs()
    assert len(runs_for("D")) == 8 and STAGE_D_TRIALS == 64
    check_run_list()
    check_stage_c_runs()                                                   # stages A-C unchanged by it
    assert all("weights" not in r.parameters() for r in runs_for("A") + runs_for("B") + runs_for("C"))
    bad = (replace(BY_NAME["top_25_tier"], band=0.0), BY_NAME["top_25_tier_k1"])
    others = tuple(r for r in runs_for("D") if not r.name.startswith("top_25_tier"))
    assert_raises(lambda: check_stage_d_runs(others + bad), ValueError, "more than the weights")
    equal = (replace(BY_NAME["top_25_tier"], weights="equal"), BY_NAME["top_25_tier_k1"])
    assert_raises(lambda: check_stage_d_runs(others + equal), ValueError, "size-tier")


def sealed_stage_c(source, store: Path) -> dict:
    """Stage-C books on the synthetic world, sealed (after their own sealed stage-B counterparts), standing in
    for the real sealed stage-C books."""
    frozen_b = sealed_counterparts(source, store / "b")
    manifest = build_run_manifest("C", source=source, dataset={}, frozen=frozen_b)
    target = run_attempt(manifest, store / "c", check_git=False, source=source, reps=REPS)
    return {u: str(target / f"{u}_dyn") for u in ("registry_10m", "top_25", "top_50")}


_STAGE_C = {}


def stage_c_books(source) -> dict:
    if "books" not in _STAGE_C:
        import atexit
        _STAGE_C["store"] = Path(tempfile.mkdtemp())
        atexit.register(_chmod_rm, _STAGE_C["store"])
        _STAGE_C["books"] = sealed_stage_c(source, _STAGE_C["store"])
    return _STAGE_C["books"]


def test_stage_d_end_to_end_unregistered():
    store = Path(tempfile.mkdtemp())
    try:
        source = stage_b_synthetic_source()
        frozen = stage_c_books(source)
        manifest = build_run_manifest("D", source=source, dataset={}, frozen=frozen)
        assert manifest["stage"] == "D" and manifest["trials"]["for_dsr"] == 64 and len(manifest["runs"]) == 8
        assert all(r["weights"] == "size_tiers" and r["sizing"] == "dynamic" for r in manifest["runs"].values())
        target = run_attempt(manifest, store, check_git=False, source=source, reps=REPS)
        a = json.loads((target / "inference.json").read_text())
        h1 = a["h1_double_prime"]
        assert h1["verdict"]["headline"] in (FAIL, INCONCLUSIVE, H1_PASSES) and "stress_20bp" in h1
        assert h1["verdict"]["inputs"]["p_low_stress"] == h1["stress_20bp"]["alpha_p_low"]
        assert set(h1["deflated_sharpe"]) == {"64", "30", "80"}
        assert set(a["secondaries"]) == {"top_25_tier", "top_50_tier"}
        assert set(a["versus_stage_c"]) == {"registry_10m_tier", "top_25_tier", "top_50_tier"}
        assert len(a["record"]["identity_checks"]) == 4 and a["record"]["not_registered_because"]
        # One tier (top 25: AAA, BBB, CCC all in the top 10) is stage C exactly; two tiers are not.
        assert a["versus_stage_c"]["top_25_tier"]["daily_correlation"] > 1 - 1e-12
        same = pd.read_csv(target / "top_25_tier" / "daily.csv.gz", index_col=0)
        sealed = pd.read_csv(Path(frozen["top_25"]) / "daily.csv.gz", index_col=0)
        assert same.net_return.equals(sealed.net_return)
        assert a["versus_stage_c"]["registry_10m_tier"]["daily_correlation"] < 1 - 1e-6
        tiers = a["tiers"]["registry_10m_tier"]
        assert tiers["tiers"]["2"]["share_days_empty"] == 1.0 and tiers["pnl_untiered_rows"] == 0
        assert set(tiers["pnl_by_tier_annual"]) == {"1", "3"}
        table = pd.read_csv(target / "registry_10m_tier" / "tiers.csv.gz")
        assert set(table.columns) == {"decision_day", "contract", "tier", "weight", "market_cap"}
        assert set(table[table.contract == "BTCUSDT"].weight) == {0.5}             # tier 3 alone: half the budget
        for name, risk in a["risk"].items():
            assert abs(risk["v_ideal_over_target"]["median"] - 1.0) < 1e-9, name
        twin = pd.read_csv(target / "registry_10m_tier_k1" / "rows.csv.gz")
        assert (twin.quantity_held >= 0).all()
    finally:
        _chmod_rm(store)


def test_stage_d_guard_and_identity_against_the_sealed_stage_c_books():
    from portfolio.per_coin_driver import STAGE_D_SPEC, STAGE_D_SPEC_SHA256
    from portfolio.freeze import sha256_file
    from portfolio.per_coin_inference import SealedBook, versus_frozen, load_per_coin_inference_config
    assert sha256_file(ROOT / STAGE_D_SPEC) == STAGE_D_SPEC_SHA256              # the approved text is unchanged
    store = Path(tempfile.mkdtemp())
    try:
        source = stage_b_synthetic_source()
        frozen = stage_c_books(source)
        manifest = build_run_manifest("D", source=source, dataset={}, frozen=frozen)
        check_guard(manifest, store, None, False)
        moved = {**manifest, "stage_d_specification": {"path": STAGE_D_SPEC, "sha256": "0" * 64}}
        assert_raises(lambda: check_guard(moved, store, None, False), ValueError, "stage D")
        swapped = json.loads(json.dumps(manifest))
        swapped["stage_c_books"]["registry_10m"]["sums_sha256"] = "0" * 64
        assert_raises(lambda: check_guard(swapped, store, None, False), ValueError, "registry_10m")
        keep = lambda d: {k: v for k, v in d.items() if k.startswith("registry_10m_tier") and "20bp" not in k}
        books = run_books({**manifest, "runs": keep(manifest["runs"]),
                           "permitted_files": keep(manifest["permitted_files"])}, source)
        cfg = load_per_coin_inference_config()
        sealed = SealedBook(frozen["registry_10m"])
        versus_frozen(books["registry_10m_tier"], sealed, cfg, 1, REPS, "weights")          # weights only: accepted
        assert_raises(lambda: versus_frozen(books["registry_10m_tier"], sealed, cfg, 1, REPS, "sizing"),
                      IdentityError)                                                         # not a sizing change
        assert_raises(lambda: versus_frozen(books["registry_10m_tier"], SealedBook(frozen["top_50"]), cfg, 1, REPS,
                                            "weights"), IdentityError)                       # another universe
        assert_raises(lambda: versus_frozen(books["registry_10m_tier_k1"], sealed, cfg, 1, REPS, "weights"),
                      IdentityError)                                                         # forecast differs too
        check_identity(books["registry_10m_tier"], books["registry_10m_tier_k1"], "twin")  # same tiers in the twin
    finally:
        _chmod_rm(store)


def _chmod_rm(path: Path):
    for p in Path(path).rglob("*"):
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            started = time.time()
            test()
            print(f"PASS {name} ({time.time() - started:.0f}s)")
