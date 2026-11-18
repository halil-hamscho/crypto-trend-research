"""Run every test suite: `python run_tests.py` from the repository root (the driver suite takes minutes).

The engine's tests are synthetic, but some suites also check the engine against the real data. Without the
data (see REPLICATION.md) those tests are reported as SKIP (needs data), not as failures.
"""
import importlib
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = (ROOT / "data" / "raw", ROOT / "data" / "derived")
SUITES = ["data.universe.test_selector", "portfolio.test_per_coin", "portfolio.test_per_coin_ledger",
          "portfolio.test_per_coin_sizing", "portfolio.test_per_coin_model", "portfolio.test_per_coin_tiers",
          "portfolio.test_per_coin_attribution", "portfolio.test_inference", "portfolio.test_freeze",
          "portfolio.test_basket", "portfolio.test_trend_signal", "portfolio.test_strategy_runner",
          "portfolio.test_secondaries", "portfolio.test_placebo", "portfolio.test_reporting",
          "portfolio.test_per_coin_driver"]


def needs_data(error: BaseException) -> bool:
    """A missing file under data/raw or data/derived, anywhere in the exception chain."""
    while error is not None:
        name = getattr(error, "filename", None)
        if isinstance(error, FileNotFoundError) and name and any(str(d) in str(name) for d in DATA):
            return True
        error = error.__cause__ or error.__context__
    return False


logging.disable(logging.CRITICAL)
sys.path.insert(0, str(ROOT))
totals = {"PASS": 0, "SKIP": 0, "FAIL": 0}
for suite in SUITES:
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    start = time.time()
    try:
        module = importlib.import_module(suite)
    except Exception as error:
        status = "SKIP" if needs_data(error) else "FAIL"
        print(f"{status} {suite} (import: {type(error).__name__}: {error})")
        totals[status] += 1
        continue
    for name, test in list(vars(module).items()):
        if not (name.startswith("test_") and callable(test)):
            continue
        try:
            test()
            counts["PASS"] += 1
        except Exception as error:
            status = "SKIP" if needs_data(error) else "FAIL"
            counts[status] += 1
            if status == "FAIL":
                print(f"FAIL {suite}.{name}: {type(error).__name__}: {error}")
    for k in totals:
        totals[k] += counts[k]
    print(f"{suite}: {counts['PASS']} passed, {counts['SKIP']} skipped (needs data), {counts['FAIL']} failed "
          f"({time.time() - start:.0f}s)")
print(f"total: {totals['PASS']} passed, {totals['SKIP']} skipped (needs data), {totals['FAIL']} failed")
sys.exit(1 if totals["FAIL"] else 0)
