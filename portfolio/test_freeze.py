"""Run from the repo root: python -m portfolio.test_freeze  (synthetic data; about a minute)"""
import copy
import json
import tempfile
from pathlib import Path
from unittest import mock

import portfolio.freeze as fz
from portfolio.test_basket import assert_raises
from portfolio.test_strategy_runner import END, INPUTS, START, single_asset_inputs

PROVIDER = lambda instrument: INPUTS if instrument == "basket" else single_asset_inputs()
WINDOW = (START, END)


def synthetic_run(store, **kwargs):
    manifest = kwargs.pop("manifest", None) or fz.build_manifest()
    return fz.run_registered(manifest, store, check_git=False, inputs_provider=PROVIDER, reps=300, window=WINDOW, **kwargs)


def test_the_manifest_is_deterministic_complete_and_lists_exactly_the_registered_analyses():
    a, b = fz.build_manifest(), fz.build_manifest()
    assert a == b and fz.manifest_hash(a) == fz.manifest_hash(b)
    assert set(a["analyses"]) == {"basket_buffered_10", "basket_buffered_20", "basket_full_10", "btc_buffered_10",
                                  "basket_buffered_5", "basket_buffered_10_equity", "basket_buffered_10_lag1"}
    assert {k for k, v in a["analyses"].items() if v["role"] == "registered"} == set(a["evaluate_keys"].values())
    assert a["seed"] == 20261003 and a["trial_counts"]["headline"] == 35 and a["inference"]["replications"] == 10000
    assert len(a["frozen_config_sha256"]) == 64 and set(a["code_sha256"]) == set(fz.CODE_FILES)


def test_a_manifest_is_written_once_and_its_name_is_its_hash():
    with tempfile.TemporaryDirectory() as store:
        manifest = fz.build_manifest()
        path = fz.write_manifest(manifest, store)
        assert path.parent.name == fz.manifest_hash(manifest)[:12] and fz.write_manifest(manifest, store) == path
        path.chmod(0o644)
        path.write_text("{}")
        assert_raises(lambda: fz.write_manifest(manifest, store), "different content")


def test_the_guard_refuses_a_changed_config_changed_code_a_dirty_tree_and_a_second_run():
    manifest = fz.build_manifest()
    folder = Path(tempfile.mkdtemp())
    bad = copy.deepcopy(manifest)
    bad["frozen_config_sha256"] = "0" * 64
    assert_raises(lambda: fz.check_guard(bad, folder, None, check_git=False), "frozen_config.yaml differs")
    bad = copy.deepcopy(manifest)
    bad["code_sha256"]["portfolio/basket.py"] = "0" * 64
    assert_raises(lambda: fz.check_guard(bad, folder, None, check_git=False), "code changed")
    with mock.patch.object(fz, "git_state", return_value={"commit": manifest["git"]["commit"], "dirty": True}):
        assert_raises(lambda: fz.check_guard(manifest, folder, None, check_git=True), "uncommitted")
    with mock.patch.object(fz, "git_state", return_value={"commit": "abc", "dirty": False}), \
            mock.patch.object(fz, "_is_doc_only_after", return_value=False):
        assert_raises(lambda: fz.check_guard(manifest, folder, None, check_git=True), "HEAD differs")
    assert "NOT registered" in fz.check_guard(manifest, folder, None, check_git=False)[0]
    fz.log_attempt(folder, "completed")
    assert_raises(lambda: fz.check_guard(manifest, folder, None, check_git=False), "rerun needs a stated reason")
    assert fz.check_guard(manifest, folder, "audit", check_git=False)


def test_the_attempt_log_is_append_only_and_ordered():
    folder = Path(tempfile.mkdtemp())
    for status in ("started", "failed", "started", "completed"):
        fz.log_attempt(folder, status, n=len(fz.read_attempts(folder)))
    attempts = fz.read_attempts(folder)
    assert [a["status"] for a in attempts] == ["started", "failed", "started", "completed"] and [a["n"] for a in attempts] == [0, 1, 2, 3]


def test_a_synthetic_registered_run_stores_everything_seals_it_and_is_flagged_unregistered():
    with tempfile.TemporaryDirectory() as store:
        folder = synthetic_run(store)
        for name in fz.ANALYSES:
            for file in ("ledger.csv.gz", "decisions.csv.gz", "daily.csv.gz", "paired.csv.gz", "stats.json", "record.json"):
                assert (folder / name / file).exists(), (name, file)
        assert (folder / "inference.json").exists() and fz.verify(folder)
        inference = json.loads((folder / "inference.json").read_text())
        assert inference["record"]["registered_run"] is False and "drags" in inference and inference["record"]["seed"] == 20261003
        attempts = fz.read_attempts(folder)
        assert [a["status"] for a in attempts] == ["started", "completed"] and attempts[-1]["registered"] is False
        assert any("overrides" in n for n in attempts[-1]["not_registered_because"])
        (folder / "inference.json").chmod(0o644)                              # tampering is detected
        (folder / "inference.json").write_text("{}")
        assert fz.verify(folder) is False


def test_a_second_run_needs_a_reason_goes_to_its_own_folder_and_a_failure_is_logged():
    with tempfile.TemporaryDirectory() as store:
        first = synthetic_run(store)
        assert_raises(lambda: synthetic_run(store), "rerun needs a stated reason")
        second = synthetic_run(store, rerun_reason="audit of the first run")
        assert second.name == "rerun_1" and second.parent == first and fz.verify(first) and fz.verify(second)
        def broken(instrument):
            raise RuntimeError("data source down")
        manifest = fz.build_manifest(seed=7)
        try:
            fz.run_registered(manifest, store, check_git=False, inputs_provider=broken, reps=300, window=WINDOW)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected the failure to propagate")
        statuses = [a["status"] for a in fz.read_attempts(Path(store) / fz.manifest_hash(manifest)[:12])]
        assert statuses == ["started", "failed"]
        assert [a["status"] for a in fz.read_attempts(first)][-2:] == ["started", "completed"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
