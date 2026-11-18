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
    role: str                    # primary | comparator | control | sensitivity | twin | stress
    stage: str                   # A | B | C
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


def _pair_c(name, universe, role, counted, **settings):
    """A stage-C book and its risk-targeted K1 twin: dynamic sizing, identical settings, forecast own vs +10."""
    return (RunSpec(name, role, "C", universe, "own", sizing="dynamic", twin=f"{name}_k1", counted=counted, **settings),
            RunSpec(f"{name}_k1", "twin", "C", universe, "k1", sizing="dynamic", **settings))


# Stage C: per-coin trend with explicit daily portfolio risk targeting, a separately registered construction
# (docs/october/risk_targeted_trend_design_2026-10-27.md §7; protocol change log 2026-10-29). Its counted
# trials are added to the 58 here, not in the frozen config.
STAGE_C_RUNS = (
    *_pair_c("registry_10m_dyn", "registry_10m", "primary", "registry_10m_dyn"),
    *_pair_c("registry_10m_dyn_20bp", "registry_10m", "stress", None, cost_bp=STRESS_BP[0]),
    *_pair_c("top_25_dyn", "top_25", "sensitivity", "top_25_dyn"),
    *_pair_c("top_50_dyn", "top_50", "sensitivity", "top_50_dyn"),
)
STAGE_C_PRIMARY, STAGE_C_STRESS = "registry_10m_dyn", "registry_10m_dyn_20bp"
STAGE_C_TRIALS = 61
STAGE_C_FROZEN = {                    # each stage-C book's primary-sizing counterpart, sealed in stage B
    "registry_10m": "reports/per_coin_run/7cdd604ad20b/attempt_1/registry_10m",
    "top_25": "reports/per_coin_run/5498171a2afb/attempt_1/top_25",
    "top_50": "reports/per_coin_run/5498171a2afb/attempt_1/top_50",
}
ALL_RUNS = RUNS + STAGE_C_RUNS
BY_NAME = {r.name: r for r in ALL_RUNS}


def runs_for(stage: str) -> tuple:
    return tuple(r for r in ALL_RUNS if r.stage == stage)


def check_stage_c_runs(runs=STAGE_C_RUNS) -> None:
    """Stage C matches its specification (design §7), or raises: eight dynamic books in four twin pairs,
    each book identical to its stage-B counterpart except the sizing (and the stress pair's cost), three
    counted trials on top of the frozen 58."""
    counted = sorted(r.counted for r in runs if r.counted)
    if counted != ["registry_10m_dyn", "top_25_dyn", "top_50_dyn"]:
        raise ValueError(f"stage C counted runs {counted}")
    if int(_raw["inference"]["trials_for_dsr"]) + len(counted) != STAGE_C_TRIALS:
        raise ValueError("stage C trials must be the frozen 58 plus its three counted books")
    if len(runs) != 8 or any(r.stage != "C" or r.sizing != "dynamic" for r in runs):
        raise ValueError("stage C is eight dynamic-sizing books")
    names = {r.name for r in runs}
    for r in runs:
        if r.forecast == "k1":
            continue
        if r.twin not in names:
            raise ValueError(f"{r.name} has no twin")
        t = BY_NAME[r.twin]
        if t.forecast != "k1" or t.parameters() != {**r.parameters(), "forecast": "k1"} or t.stage != "C":
            raise ValueError(f"{r.twin} is not {r.name}'s twin: it must differ only in the forecast")
        frozen = BY_NAME[r.universe]
        cost = STRESS_BP[0] if r.name == STAGE_C_STRESS else PRIMARY_BP
        if r.parameters() != {**frozen.parameters(), "sizing": "dynamic", "cost_bp": cost}:
            raise ValueError(f"{r.name} differs from {frozen.name} in more than the sizing")
    if BY_NAME[STAGE_C_PRIMARY].role != "primary" or BY_NAME[STAGE_C_STRESS].role != "stress":
        raise ValueError("stage C's primary and stress books are mislabelled")


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
    events: Callable | None = None           # () -> the event register (stage B)
    suspensions: Callable | None = None      # () -> the funding-gap register (stage B)
    index_reader: Callable | None = None     # (contract, day) -> 1-minute index closes (stage B)


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


STAGE_B_NOW = ("top_25", "top_25_k1", "top_50", "top_50_k1")    # run 2026-10-26 (manifest 5498171a2afb)
STAGE_B_REGISTRY = ("registry_10m", "registry_10m_k1")           # built 2026-10-26 after that result
STAGE_B_UNIVERSES = ("top_25", "top_50", "registry_10m")
REGISTRY_FLOORS = {"registry_10m": 10e6, "registry_25m": 25e6, "registry_50m": 50e6}
SPAN_BREAK_REGISTER = "docs/october/per_coin_support/span_break_register.csv"
DEEP_START = date(2021, 7, 1)                                     # Amendment 2, A2.5
EVENT_REGISTER = "docs/october/per_coin_support/event_register.csv"
FUNDING_GAP_REGISTER = "docs/october/per_coin_support/funding_gap_register.csv"


def load_span_breaks(path=None) -> dict:
    """contract -> forced span starts (registered redenominations)."""
    from portfolio.per_coin_inputs import ROOT
    frame = pd.read_csv(path or ROOT / SPAN_BREAK_REGISTER)
    if frame.evidence.isna().any():
        raise ValueError("span-break register: every entry needs evidence")
    out = {}
    for row in frame.itertuples():
        out.setdefault(row.contract, []).append(pd.Timestamp(row.first_day))
    return out


def attach_span_breaks(bars: dict, breaks: dict) -> dict:
    """Carry each contract's registered span starts on its bars (per_coin_universe.forced_starts)."""
    from portfolio.per_coin_universe import SPAN_STARTS
    for c, frame in bars.items():
        if c in breaks:
            frame.attrs[SPAN_STARTS] = tuple(breaks[c])
    return bars


def load_funding_gap_register(path=None) -> pd.DataFrame:
    from portfolio.per_coin_inputs import ROOT
    frame = pd.read_csv(path or ROOT / FUNDING_GAP_REGISTER)
    frame["start"] = pd.to_datetime(frame.start, utc=True)
    frame["end"] = pd.to_datetime(frame.end, utc=True)
    return frame


def settlements_or_empty(contracts, register) -> dict:
    """Funding for each contract through the supplemented reader, with the funding-gap register. A
    contract with no archived marks at all (protocol Appendix B) can never be tradable (U6) and so is
    never held: it gets an empty table rather than stopping the load. A settlement whose hour has no
    mark is dropped only inside a registered `mark_hole` (which no held window may touch)."""
    from functools import partial
    from data.universe.coinmarketcap.new_fetcher import panel
    from portfolio.basket import SETTLEMENT_COLUMNS
    from portfolio.per_coin_ledger import load_settlements_registered
    markless = [c for c in contracts if not (panel.MARK_DIR / f"{c}.csv").exists()]
    holes = register[register.kind == "mark_hole"] if register is not None and "kind" in register else None
    out = load_settlements_registered([c for c in contracts if c not in markless], register,
                                      reader=partial(read_settlements_supplemented, holes=holes))
    for c in markless:
        out[c] = pd.DataFrame(columns=list(SETTLEMENT_COLUMNS))
    return out


def load_event_register(path=None) -> pd.DataFrame:
    """The committed register, typed for the ledger: dates as dates, instants as UTC timestamps."""
    from portfolio.per_coin_inputs import ROOT
    frame = pd.read_csv(path or ROOT / EVENT_REGISTER)
    frame["first_day"] = pd.to_datetime(frame.first_day).dt.date
    frame["last_day"] = pd.to_datetime(frame.last_day).dt.date
    frame["instant"] = pd.to_datetime(frame.instant, utc=True)
    return frame


def universe_members(universe: str, cache: dict | None = None) -> dict:
    """Membership of a stage-B universe: DeepSelector for the top 25/50 (Amendment 2, from
    2021-07-01); for the registry-wide books, daily U1/U3/U4 membership on every decision day over the
    frozen registry (Amendment 1)."""
    from portfolio.basket import SCORED_END
    from portfolio.per_coin_universe import (deep_membership, load_bars, load_exclusions, registry_contracts,
                                             registry_membership)
    cache = {} if cache is None else cache
    if universe not in cache:
        if universe in ("top_25", "top_50"):
            cache[universe] = deep_membership(int(universe[4:]), SCORED_END, start=DEEP_START)
        elif universe in REGISTRY_FLOORS:
            registry, excluded = registry_contracts(), load_exclusions()
            bars = load_bars([c for c in registry if c not in excluded])
            cache[universe] = registry_membership(registry, decision_days(WINDOW), bars, REGISTRY_FLOORS[universe],
                                                  CONFIG, excluded)
        else:
            raise NotImplementedError(f"universe {universe!r} is not a stage-B universe")
    return cache[universe]


def stage_b_universe_inputs(universe: str) -> tuple:
    """(members, bars with registered span breaks, mark counts) for one stage-B universe."""
    from portfolio.per_coin_universe import load_bars, load_mark_counts
    members = universe_members(universe)
    contracts = contracts_of(members)
    return members, attach_span_breaks(load_bars(contracts), load_span_breaks()), load_mark_counts(contracts)


def stage_b_source() -> Source:
    """Disk, for the stage-B universes: membership by universe_members, bars with the span-break
    register, funding with the verified ICP supplement and the funding-gap register, the event register
    and the 1-minute index files."""
    from portfolio.per_coin_inputs import permitted_files, relative
    from portfolio.per_coin_ledger import read_index_minutes
    from portfolio.per_coin_universe import load_bars, load_mark_counts
    from data.universe.coinmarketcap.new_fetcher import panel
    cache = {}

    def members(universe):
        return universe_members(universe, cache)

    def permitted(universe):
        from portfolio.per_coin_inputs import ROOT
        contracts = contracts_of(members(universe))
        events = load_event_register()
        extra = [relative(supplement_path(c)) for c in contracts if supplement_path(c).exists()]
        extra += [relative(index_file(e.contract, d)) for e in events.itertuples() if e.contract in contracts
                  for d in index_days(e.instant)]
        registers = [EVENT_REGISTER, FUNDING_GAP_REGISTER, SPAN_BREAK_REGISTER]
        if universe in REGISTRY_FLOORS:
            registers.append("docs/october/per_coin_support/exclusions.csv")
            extra.append(relative(panel.TRADED_RANGES))
        return sorted(set(permitted_files(contracts) + extra + [relative(ROOT / r) for r in registers]))

    register = {}

    def gaps():
        if "r" not in register:
            register["r"] = load_funding_gap_register()
        return register["r"]

    return Source(members=members, bars=lambda cs: attach_span_breaks(load_bars(cs), load_span_breaks()),
                  mark_counts=load_mark_counts, k5_forecast=None,
                  settlements=lambda cs: settlements_or_empty(cs, gaps()), permitted=permitted,
                  events=load_event_register, index_reader=read_index_minutes, suspensions=gaps)


def index_file(contract: str, day):
    from data.universe.coinmarketcap.new_fetcher import panel
    return panel.RAW_DIR / f"futures/um/daily/indexPriceKlines/{contract}/1m/{contract}-1m-{day}.zip"


def index_days(instant) -> list:
    """The days whose 1-minute index files settlement_price reads for an instant."""
    from portfolio.per_coin_ledger import DIAGNOSTIC_MINUTES
    t = pd.Timestamp(instant)
    return sorted({(t - pd.Timedelta(minutes=DIAGNOSTIC_MINUTES)).date(), (t - pd.Timedelta(minutes=1)).date()})


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


# ------------------------------------------------------------------ stage B: funding supplements
def supplement_path(contract: str):
    from data.universe.coinmarketcap.new_fetcher import panel
    return panel.DERIVED_DIR / "funding_supplement" / f"{contract}.csv"


def load_funding_supplemented(contract: str) -> pd.DataFrame:
    """The archive's funding table, preceded by the verified API rows where the archive starts late
    (ICPUSDT before 2022-09-01: docs/october/per_coin_support/build_icp_funding_supplement.py). The two
    must not overlap."""
    from data.universe.coinmarketcap.new_fetcher import panel
    rows = panel.load_funding(contract)
    path = supplement_path(contract)
    if not path.exists():
        return rows
    extra = panel._read_timestamped(path)
    if extra.timestamp.max() >= rows.timestamp.min():
        raise ValueError(f"{contract}: the funding supplement overlaps the archive")
    return pd.concat([extra, rows], ignore_index=True)


def read_settlements_supplemented(contract: str, holes=None) -> pd.DataFrame:
    """basket._read_settlements with the supplemented funding table; every check of the sealed reader
    (marks joined, clipped to the mark range, exchange skips added) is the same. `holes`: registered
    mark_hole entries; a settlement with no mark strictly inside one is dropped, any other raises."""
    import portfolio.basket as basket
    from data.universe.coinmarketcap.new_fetcher import panel
    rows = load_funding_supplemented(contract)
    marks = panel.load_mark_prices(contract)
    stamp = rows.timestamp.dt.floor("h")
    settlements = pd.DataFrame({"stamp": stamp, "rate": rows.funding_rate.astype(float),
                                "mark": marks.open.reindex(stamp).values,
                                "interval_hours": rows.funding_interval_hours.astype(float)})
    settlements = settlements[(settlements.stamp >= marks.index.min()) & (settlements.stamp <= marks.index.max())]
    if holes is not None and len(holes):
        mine = holes[holes.contract == contract]
        in_hole = settlements.stamp.apply(lambda t: bool(((mine.start < t) & (mine.end > t)).any()))
        settlements = settlements[~(in_hole & settlements.mark.isna())]
    if settlements.empty or settlements.mark.isna().any():
        raise ValueError(f"{contract}: settlements without a mark price, or none inside the mark range")
    return basket._add_exchange_skips(contract, settlements.reset_index(drop=True), marks)


# ------------------------------------------------------------------ one book
@dataclass
class RunResult:
    spec: RunSpec
    daily: pd.DataFrame          # ledger.daily plus funding_return and cost_return
    split: pd.DataFrame          # attribution by day
    rows: pd.DataFrame           # ledger rows
    record: dict                 # parameters, hashes, notes
    tables: object = None
    sizing: pd.DataFrame = None  # per holding day: the sizing's risk diagnostics (M, V(u), ex-ante risk, caps)


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
        settlements=source.settlements(contracts) if source.settlements else None,
        events=source.events() if source.events else None, index_reader=source.index_reader,
        suspensions=source.suspensions() if source.suspensions else None)
    notes, diagnostics = [], []
    ledger = run_per_coin_ledger(make_target_fn(tables, spec.lag, spec.sizing, notes, spec.band, diagnostics), inputs)
    split = attribute(ledger, inputs)
    daily = ledger.daily.copy()
    daily["funding_return"] = daily.funding_pnl / inputs.capital
    daily["cost_return"] = (daily.fee + daily.slippage) / inputs.capital
    record = {"name": spec.name, "parameters": spec.parameters(), "window": [str(start), str(end)],
              "capital": inputs.capital, "contracts": list(contracts),
              "hashes": tables_hashes(tables, inputs),
              "ledger_fingerprint": ledger.fingerprint, "inputs_fingerprint": ledger.inputs_fingerprint,
              "model_notes": notes, "ledger_notes": list(ledger.notes),
              "settlement_prices": {f"{c}|{d}": {k: (str(v) if k == "instant" else v) for k, v in rec.items()}
                                    for (c, d), rec in inputs.settlement_prices.items()}}
    sizing = pd.DataFrame(diagnostics).set_index("day")
    if not sizing.index.equals(pd.Index(inputs.days[:-1])):
        raise ValueError(f"{spec.name}: sizing diagnostics are not one per holding day")
    return RunResult(spec, daily, split, ledger.rows, record, tables, sizing)
