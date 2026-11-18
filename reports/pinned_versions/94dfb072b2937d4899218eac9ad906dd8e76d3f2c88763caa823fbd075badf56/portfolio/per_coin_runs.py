"""The per-coin run list and the building of one book.

Design: docs/october/per_coin_run_driver_design_2026-10-24.md §1-2. Every book is the same engine
(universe, signal, model, ledger, attribution) with the inputs the protocol varies. A `Source` supplies
the data: `real_source()` reads disk through the checked loader; tests pass synthetic sources.
"""
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from portfolio.per_coin_attribution import attribute
from portfolio.per_coin_inputs import tables_hashes
from portfolio.per_coin_ledger import HOURLY, PRIMARY, build_ledger_inputs, run_per_coin_ledger
from portfolio.per_coin_model import build_decision_tables, make_target_fn
from portfolio.per_coin_signal import load_per_coin_config

CONFIG = load_per_coin_config()
_raw = CONFIG.raw
BAND = float(_raw["execution"]["band_half_width"])
PRIMARY_BP = float(_raw["costs"]["primary_bp_per_side"])
STRESS_BP = tuple(float(v) for v in _raw["costs"]["stress_bp_per_side"])
CAPITAL = float(_raw["sample"]["capital"])
WINDOW = (date.fromisoformat(str(_raw["sample"]["scored_start"])), date.fromisoformat(str(_raw["sample"]["scored_end"])))
CLOCKS = {"primary": PRIMARY, "hourly_0100": HOURLY}


@dataclass(frozen=True)
class RunSpec:
    name: str
    role: str                    # primary | comparator | control | sensitivity | twin
    stage: str                   # A | B
    universe: str                # top10 | btc | top_25 | top_50 | registry_25m | registry_10m | registry_50m
    forecast: str                # own | k1 | k3 | k5
    sizing: str = "primary"      # primary | dynamic
    lag: int = 1
    clock: str = "primary"       # primary (00:00 kline open) | hourly_0100 (01:00 mark)
    cost_bp: float = PRIMARY_BP
    band: float = BAND
    twin: str | None = None      # the K1 book this variant is compared with (H1-type comparisons)
    counted: str | None = None   # its name in the config's counted_sensitivities, if a counted trial

    def parameters(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("name", "role", "stage", "twin", "counted")}


def _pair(name, stage, universe, counted, **settings):
    """A counted sensitivity and its K1 twin: identical settings, forecast own vs +10."""
    return (RunSpec(name, "sensitivity", stage, universe, "own", twin=f"{name}_k1", counted=counted, **settings),
            RunSpec(f"{name}_k1", "twin", stage, universe, "k1", **settings))


RUNS = (
    RunSpec("strategy", "primary", "A", "top10", "own", twin="k1"),
    RunSpec("k1", "comparator", "A", "top10", "k1"),
    RunSpec("k2", "control", "A", "btc", "k1"),
    RunSpec("k3", "control", "A", "top10", "k3"),
    RunSpec("k5", "control", "A", "top10", "k5"),
    *_pair("cost_20bp", "A", "top10", "cost_20bp", cost_bp=STRESS_BP[0]),
    *_pair("cost_30bp", "A", "top10", "cost_30bp", cost_bp=STRESS_BP[1]),
    *_pair("same_bar", "A", "top10", "same_bar", lag=0),
    *_pair("hourly_0100", "A", "top10", "hourly_0100", lag=0, clock="hourly_0100"),
    *_pair("full_rebalance", "A", "top10", "full_rebalance", band=0.0),
    *_pair("dynamic_rt", "A", "top10", "dynamic_risk_targeting", sizing="dynamic"),
    *_pair("top_25", "B", "top_25", "top_25"),
    *_pair("top_50", "B", "top_50", "top_50"),
    *_pair("registry_25m", "B", "registry_25m", "registry_wide_25m"),
    *_pair("registry_10m", "B", "registry_10m", "registry_wide_10m"),
    *_pair("registry_50m", "B", "registry_50m", "registry_wide_50m"),
)
BY_NAME = {r.name: r for r in RUNS}


def runs_for(stage: str) -> tuple:
    return tuple(r for r in RUNS if r.stage == stage)


def check_run_list(runs=RUNS) -> None:
    """D1: the list matches the protocol and config, or raises."""
    inference = _raw["inference"]
    counted = [r.counted for r in runs if r.counted]
    if sorted(counted) != sorted(inference["counted_sensitivities"]) or len(set(counted)) != len(counted):
        raise ValueError(f"counted runs {sorted(counted)} != config {sorted(inference['counted_sensitivities'])}")
    if 35 + 11 + 1 + len(counted) != int(inference["trials_for_dsr"]):
        raise ValueError("trial count: 35 + 11 + 1 + counted sensitivities must equal trials_for_dsr")
    names = {r.name for r in runs}
    for r in runs:
        if r.counted and r.twin not in names:
            raise ValueError(f"{r.name} has no twin")
        if r.twin:
            t = BY_NAME[r.twin]
            if t.forecast != "k1" or t.parameters() != {**r.parameters(), "forecast": "k1"} or t.stage != r.stage:
                raise ValueError(f"{r.twin} is not {r.name}'s twin: it must differ only in the forecast")
    if [len(runs_for("A")), len(runs_for("B"))] != [17, 10]:
        raise ValueError("expected 17 stage-A and 10 stage-B books")
    p = BY_NAME["strategy"]
    registered = {"universe": "top10", "forecast": "own", "sizing": "primary",
                  "lag": int(_raw["execution"]["primary_lag_days"]), "clock": "primary", "cost_bp": PRIMARY_BP,
                  "band": BAND}
    if p.parameters() != registered:
        raise ValueError(f"the primary is not the registered primary: {p.parameters()}")


# ------------------------------------------------------------------ data sources
@dataclass
class Source:
    """What a book reads. `members(universe)` -> members_by_date; `bars`, `mark_counts`, `settlements`,
    `marks` take a contract list (settlements/marks None: the ledger loads them from disk);
    `k5_forecast(members)` -> the sealed basket forecast by date, `k5_index(members)` -> the reference
    index it is computed on (None: not hashed); `permitted(universe)` -> files a book may read."""
    members: Callable
    bars: Callable
    mark_counts: Callable
    k5_forecast: Callable
    k5_index: Callable | None = None
    settlements: Callable | None = None
    marks: Callable | None = None
    permitted: Callable | None = None
    window: tuple = WINDOW
    capital: float = CAPITAL


def contracts_of(members_by_date: dict) -> tuple:
    return tuple(sorted({c for m in members_by_date.values() for c in m.contract}))


def btc_members(top10: dict) -> dict:
    """K2's universe: BTCUSDT alone on the same reset dates."""
    only = pd.DataFrame({"contract": ["BTCUSDT"], "weight": [1.0]})
    return {d: only for d in top10}


def real_source() -> Source:
    """Disk, via the engine's own loaders (each read checked once a CheckedLoader is active)."""
    from portfolio.basket import SCORED_END
    from portfolio.per_coin_universe import load_bars, load_mark_counts, membership
    from portfolio.per_coin_inputs import permitted_files
    cache = {}

    def members(universe):
        if universe not in ("top10", "btc"):
            raise NotImplementedError(f"universe {universe!r} is stage B")
        if "top10" not in cache:
            cache["top10"] = membership(SCORED_END)
        return cache["top10"] if universe == "top10" else btc_members(cache["top10"])

    def permitted(universe):
        return permitted_files(contracts_of(members(universe)))

    return Source(members=members, bars=load_bars, mark_counts=load_mark_counts, k5_forecast=sealed_k5_forecast,
                  k5_index=lambda m: sealed_k5_reference(m)[0], permitted=permitted)


def sealed_k5_reference(members: dict):
    """K5's inputs: the sealed reference index, built exactly as basket.build_inputs builds it (same
    membership, marks from BASKET_START to end + 1), and the registered basket forecast on it."""
    from portfolio.basket import BASKET_START, SCORED_END, load_marks, reference_basket
    from portfolio.trend_signal import basket_forecast, load_signal_config
    marks = load_marks(list(contracts_of(members))).loc[BASKET_START:SCORED_END + timedelta(days=1)]
    index, _ = reference_basket(members, marks)
    return index, basket_forecast(index, load_signal_config())["forecast"]


def sealed_k5_forecast(members: dict) -> pd.Series:
    return sealed_k5_reference(members)[1]


# ------------------------------------------------------------------ one book
@dataclass
class RunResult:
    spec: RunSpec
    daily: pd.DataFrame          # ledger.daily plus funding_return and cost_return
    split: pd.DataFrame          # attribution by day
    rows: pd.DataFrame           # ledger rows
    record: dict                 # parameters, hashes, notes
    tables: object = None


def decision_days(window: tuple) -> list:
    start, end = window
    return [start + timedelta(days=i) for i in range(-1, (end - start).days + 1)]


def build_run(spec: RunSpec, source: Source, k5=None) -> RunResult:
    """Run one book end to end (the driver wraps it in the spec's checked-loader scope). Raises on any
    validation, reconciliation or attribution failure."""
    from portfolio.per_coin_universe import eligibility
    members = source.members(spec.universe)
    contracts = contracts_of(members)
    start, end = source.window
    bars, counts = source.bars(contracts), source.mark_counts(contracts)
    eligible, _, _ = eligibility(members, decision_days(source.window), bars, counts, CONFIG)
    tables = build_decision_tables(bars, eligible, CONFIG, variant=spec.forecast,
                                   basket_forecast=k5 if spec.forecast == "k5" else None)
    inputs = build_ledger_inputs(
        contracts, start, end, eligible, clock=CLOCKS[spec.clock], decision_lag_days=spec.lag,
        capital=source.capital, cost_bp_per_side=spec.cost_bp, bars=bars, mark_counts=counts,
        marks=source.marks(contracts) if source.marks else None,
        settlements=source.settlements(contracts) if source.settlements else None)
    notes = []
    ledger = run_per_coin_ledger(make_target_fn(tables, spec.lag, spec.sizing, notes, spec.band), inputs)
    split = attribute(ledger, inputs)
    daily = ledger.daily.copy()
    daily["funding_return"] = daily.funding_pnl / inputs.capital
    daily["cost_return"] = (daily.fee + daily.slippage) / inputs.capital
    record = {"name": spec.name, "parameters": spec.parameters(), "window": [str(start), str(end)],
              "capital": inputs.capital, "contracts": list(contracts),
              "hashes": tables_hashes(tables, inputs),
              "ledger_fingerprint": ledger.fingerprint, "inputs_fingerprint": ledger.inputs_fingerprint,
              "model_notes": notes, "ledger_notes": list(ledger.notes)}
    return RunResult(spec, daily, split, ledger.rows, record, tables)
