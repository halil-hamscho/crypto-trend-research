"""Run from the repo root: python -m portfolio.test_per_coin_ledger

Tests for the per-coin ledger, built step by step from
docs/october/per_coin_ledger_implementation_2026-10-13.md (plan v2). Step 1: the inputs and
their timing. Synthetic data with known answers, plus two real-data checks (the funding loader
against the sealed one, and a build over the real top-10 membership). No returns are computed.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from portfolio.per_coin_ledger import (HOURLY, PRIMARY, FillClock, build_ledger_inputs, fill_instant,
                                       load_settlements_registered)

START, END = date(2021, 3, 1), date(2021, 3, 10)
EARLY = START - timedelta(days=80)                    # history before the window, for lags and spans
CONTRACTS = ("AAA", "BBB", "CCC")
ALL_DAYS = [EARLY + timedelta(days=i) for i in range((END - EARLY).days + 3)]


def assert_raises(call, *fragments):
    try:
        call()
    except ValueError as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return
    raise AssertionError("expected ValueError")


def price(contract: str, day: date, hour: int = 0) -> float:
    """Unique by contract, day and hour, so every lookup can be checked by value."""
    return 1000.0 * (1 + CONTRACTS.index(contract)) + (day - EARLY).days + hour / 100


def synthetic(halted=(), thin_marks=(), relist=None):
    """Bars, hourly marks, mark counts and eligibility for three contracts.
    `halted`: (contract, day) with zero volume. `thin_marks`: (contract, day) with 22 marks.
    `relist`: (contract, first_day_back): no trades for 40 days before it, so a new span starts."""
    bars, marks, counts = {}, {}, {}
    for c in CONTRACTS:
        rows = []
        for d in ALL_DAYS:
            volume = 0.0 if (c, d) in halted else 5.0
            if relist and c == relist[0] and relist[1] - timedelta(days=40) <= d < relist[1]:
                volume = 0.0
            rows.append({"timestamp": pd.Timestamp(d, tz="UTC"), "open": price(c, d), "volume": volume,
                         "quote_volume": 1e9})
        bars[c] = pd.DataFrame(rows)
        hours = pd.date_range(pd.Timestamp(ALL_DAYS[0], tz="UTC"), periods=24 * len(ALL_DAYS), freq="h")
        marks[c] = pd.DataFrame({"open": [price(c, h.date(), h.hour) for h in hours]}, index=hours)
        counts[c] = pd.Series(24, index=ALL_DAYS)
        for cc, d in thin_marks:
            if cc == c:
                counts[c][d] = 22
    days = [d for d in ALL_DAYS if START - timedelta(days=1) <= d <= END]     # decision days under the lag
    eligible = pd.DataFrame(True, index=days, columns=list(CONTRACTS))
    return bars, marks, counts, eligible


def build(clock=PRIMARY, lag=1, **kw):
    bars, marks, counts, eligible = synthetic(**{k: kw.pop(k) for k in ("halted", "thin_marks", "relist") if k in kw})
    return build_ledger_inputs(CONTRACTS, START, END, eligible, clock=clock, decision_lag_days=lag,
                               bars=bars, marks=marks, mark_counts=counts, settlements={}, **kw)


# ------------------------------------------------------------------ clocks
def test_fill_instants():
    assert fill_instant(date(2022, 3, 10), PRIMARY) == pd.Timestamp("2022-03-10 00:00", tz="UTC")
    assert fill_instant(date(2022, 3, 10), HOURLY) == pd.Timestamp("2022-03-10 01:00", tz="UTC")
    assert_raises(lambda: FillClock(2, "mark_open"), "hour")
    assert_raises(lambda: FillClock(0, "close"), "price")


# ------------------------------------------------------------------ window and prices
def test_window_runs_from_start_to_one_day_past_the_end():
    inputs = build()
    assert inputs.days[0] == START and inputs.days[-1] == END + timedelta(days=1)
    assert len(inputs.days) == (END - START).days + 2
    assert list(inputs.fill_prices.index) == inputs.days and list(inputs.fill_prices.columns) == list(CONTRACTS)


def test_fill_prices_are_the_open_on_traded_days_and_blank_otherwise():
    halt = (("BBB", date(2021, 3, 5)),)
    inputs = build(halted=halt)
    assert inputs.fill_prices.at[date(2021, 3, 4), "BBB"] == price("BBB", date(2021, 3, 4))
    assert np.isnan(inputs.fill_prices.at[date(2021, 3, 5), "BBB"])
    assert inputs.fill_prices.at[date(2021, 3, 5), "AAA"] == price("AAA", date(2021, 3, 5))


def test_hourly_clock_fills_at_the_0100_mark():
    inputs = build(clock=HOURLY, lag=0)
    d = date(2021, 3, 7)
    assert inputs.fill_prices.at[d, "CCC"] == price("CCC", d, hour=1)


# ------------------------------------------------------------------ decision timing (timestamps, not values)
def test_one_day_lag_decides_on_yesterdays_0000_price():
    inputs = build(lag=1)
    for d in inputs.days:
        stamp = inputs.decision_stamps[d]
        assert stamp == pd.Timestamp(d - timedelta(days=1), tz="UTC")
        assert stamp < fill_instant(d, PRIMARY)
        for c in CONTRACTS:                               # the value is the one stamped there
            assert inputs.decision_prices.at[d, c] == price(c, stamp.date())


def test_same_bar_and_hourly_decide_on_todays_0000_price():
    for clock in (PRIMARY, HOURLY):
        inputs = build(clock=clock, lag=0)
        for d in inputs.days:
            assert inputs.decision_stamps[d] == pd.Timestamp(d, tz="UTC")
            assert inputs.decision_stamps[d] <= fill_instant(d, clock)
            assert inputs.decision_prices.at[d, "AAA"] == price("AAA", d)


def test_decision_prices_ignore_everything_after_the_decision_instant():
    """Change every price stamped after D-1 00:00: the decision prices up to D are unchanged."""
    bars, marks, counts, eligible = synthetic()
    base = build_ledger_inputs(CONTRACTS, START, END, eligible, decision_lag_days=1, bars=bars, marks=marks,
                               mark_counts=counts, settlements={})
    cut = date(2021, 3, 6)
    shocked = {c: b.assign(open=np.where(b.timestamp.dt.date > cut - timedelta(days=1), b.open * 7.0, b.open))
               for c, b in bars.items()}
    later = build_ledger_inputs(CONTRACTS, START, END, eligible, decision_lag_days=1, bars=shocked, marks=marks,
                                mark_counts=counts, settlements={})
    upto = [d for d in base.days if d <= cut]
    pd.testing.assert_frame_equal(base.decision_prices.loc[upto], later.decision_prices.loc[upto])
    assert not base.fill_prices.loc[[cut]].equals(later.fill_prices.loc[[cut]])   # the fill itself did change


# ------------------------------------------------------------------ tradability
def test_tradable_needs_a_trade_complete_marks_and_not_a_spans_first_day():
    d = date(2021, 3, 4)
    back = date(2021, 3, 8)
    inputs = build(halted=(("AAA", d),), thin_marks=(("BBB", d),), relist=("CCC", back))
    assert not inputs.tradable.at[d, "AAA"] and inputs.tradable.at[d - timedelta(days=1), "AAA"]
    assert not inputs.tradable.at[d, "BBB"] and inputs.tradable.at[d + timedelta(days=1), "BBB"]
    assert not inputs.tradable.at[back, "CCC"] and inputs.tradable.at[back + timedelta(days=1), "CCC"]
    assert not inputs.tradable.at[back - timedelta(days=1), "CCC"]


# ------------------------------------------------------------------ fail loudly
def test_eligibility_must_cover_every_decision_day():
    bars, marks, counts, eligible = synthetic()
    short = eligible.drop(index=date(2021, 3, 4))
    assert_raises(lambda: build_ledger_inputs(CONTRACTS, START, END, short, bars=bars, marks=marks,
                                              mark_counts=counts, settlements={}), "eligibility", "2021-03-04")
    no_eve = eligible.drop(index=START - timedelta(days=1))           # the lag needs the day before the window
    assert_raises(lambda: build_ledger_inputs(CONTRACTS, START, END, no_eve, bars=bars, marks=marks,
                                              mark_counts=counts, settlements={}), "eligibility")


def test_eligibility_is_stored_by_holding_day_from_the_decision_day():
    bars, marks, counts, eligible = synthetic()
    eligible.loc[date(2021, 3, 4), "AAA"] = False                     # ineligible on 03-04 only
    lag1 = build_ledger_inputs(CONTRACTS, START, END, eligible, decision_lag_days=1, bars=bars, marks=marks,
                               mark_counts=counts, settlements={})
    assert not lag1.eligible.at[date(2021, 3, 5), "AAA"] and lag1.eligible.at[date(2021, 3, 4), "AAA"]
    lag0 = build_ledger_inputs(CONTRACTS, START, END, eligible, decision_lag_days=0, bars=bars, marks=marks,
                               mark_counts=counts, settlements={})
    assert not lag0.eligible.at[date(2021, 3, 4), "AAA"] and lag0.eligible.at[date(2021, 3, 5), "AAA"]


def test_registers_need_evidence_and_known_kinds():
    ok = {"contract": "AAA", "kind": "halt", "first_day": date(2021, 3, 4), "last_day": date(2021, 3, 5),
          "instant": pd.NaT, "evidence": "announcement 123"}
    build(events=pd.DataFrame([ok]))
    assert_raises(lambda: build(events=pd.DataFrame([{**ok, "evidence": ""}])), "evidence")
    assert_raises(lambda: build(events=pd.DataFrame([{**ok, "kind": "pause"}])), "kind")
    sus = {"contract": "AAA", "start": pd.Timestamp("2021-03-04", tz="UTC"),
           "end": pd.Timestamp("2021-03-05", tz="UTC"), "evidence": ""}
    assert_raises(lambda: build(suspensions=pd.DataFrame([sus])), "evidence")


def test_negative_lag_is_refused():
    assert_raises(lambda: build(lag=-1), "decision_lag_days")


# ------------------------------------------------------------------ funding loader
def settlement_table(stamps):
    return pd.DataFrame({"stamp": pd.DatetimeIndex(stamps), "rate": 1e-4, "mark": 100.0, "interval_hours": 8.0})


def test_registered_loader_accepts_only_registered_gaps():
    stamps = pd.date_range("2021-03-01", periods=12, freq="8h", tz="UTC")
    gapped = stamps.delete(5)                                             # one missing settlement
    reader = lambda c: settlement_table(gapped)                            # noqa: E731
    assert_raises(lambda: load_settlements_registered(["AAA"], None, reader=reader), "AAA")
    covering = pd.DataFrame([{"contract": "AAA", "start": stamps[4], "end": stamps[6], "evidence": "notice 9"}])
    out = load_settlements_registered(["AAA"], covering, reader=reader)
    assert len(out["AAA"]) == 11
    short = pd.DataFrame([{"contract": "AAA", "start": stamps[5], "end": stamps[6], "evidence": "notice 9"}])
    assert_raises(lambda: load_settlements_registered(["AAA"], short, reader=reader), "AAA")
    other = pd.DataFrame([{"contract": "BBB", "start": stamps[4], "end": stamps[6], "evidence": "notice 9"}])
    assert_raises(lambda: load_settlements_registered(["AAA"], other, reader=reader), "AAA")


def test_registered_loader_equals_the_sealed_loader_with_an_empty_register():
    """Every contract the top-10 membership ever holds (24), POL among them with the sealed
    2026-06-24 exception."""
    from portfolio.basket import SCORED_END, load_settlements
    from portfolio.per_coin_universe import membership
    contracts = sorted({c for m in membership(SCORED_END).values() for c in m.contract} | {"POLUSDT"})
    sealed = load_settlements(contracts)
    ours = load_settlements_registered(contracts, None)
    for c in contracts:
        pd.testing.assert_frame_equal(sealed[c], ours[c])
    print(f"   real: registered loader == sealed loader on all {len(contracts)} contracts")


# ------------------------------------------------------------------ real data
def test_real_inputs_for_the_top_10():
    from portfolio.basket import SCORED_END, SCORED_START
    from portfolio.per_coin_signal import load_per_coin_config
    from portfolio.per_coin_universe import eligibility, load_bars, load_mark_counts, membership
    members = membership(SCORED_END)
    contracts = sorted({c for m in members.values() for c in m.contract})
    days = [SCORED_START + timedelta(days=i) for i in range(-1, (SCORED_END - SCORED_START).days + 1)]
    bars, counts = load_bars(contracts), load_mark_counts(contracts)
    ok, _, _ = eligibility(members, days, bars, counts, load_per_coin_config())
    inputs = build_ledger_inputs(tuple(contracts), SCORED_START, SCORED_END, ok, bars=bars, mark_counts=counts)
    assert inputs.days[-1] == SCORED_END + timedelta(days=1)
    assert set(inputs.settlements) == set(contracts)
    held_ok = inputs.eligible.stack()
    held_ok = held_ok[held_ok]                       # (holding day, coin) eligible at the decision
    untradable = [(d, c) for d, c in held_ok.index if not inputs.tradable.at[d, c]]
    assert not untradable, untradable[:5]
    print(f"   real: {len(contracts)} contracts x {len(inputs.days)} days; {len(held_ok)} coin-days eligible at "
          f"the decision, every one tradable at its fill")


# ================================================================== step 2: validate_targets (L9)
D = date(2021, 3, 5)                     # a holding day; its decision day is 03-04 under the lag


def vt_inputs(ineligible=(), halted=(), events=None):
    """Synthetic inputs. `ineligible`: (contract, decision_day) pairs. `halted`: (contract, day) untradable."""
    bars, marks, counts, eligible = synthetic(halted=halted)
    for c, d in ineligible:
        eligible.loc[d, c] = False
    return build_ledger_inputs(CONTRACTS, START, END, eligible, bars=bars, marks=marks, mark_counts=counts,
                               settlements={}, events=events)


def halt_event(contract, first, last, kind="halt"):
    return pd.DataFrame([{"contract": contract, "kind": kind, "first_day": first, "last_day": last,
                          "instant": pd.NaT, "evidence": "announcement 42"}])


def test_valid_targets_pass_and_missing_held_coins_become_zero():
    from portfolio.per_coin_ledger import HOLD, validate_targets
    out = validate_targets(D, {"AAA": 6_000.0, "BBB": -4_000, "CCC": HOLD}, {"CCC": 2.0}, vt_inputs())
    assert out == {"AAA": 6_000.0, "BBB": -4_000.0, "CCC": HOLD}
    out = validate_targets(D, {}, {"AAA": 1.5}, vt_inputs())               # held, no target: close it
    assert out == {"AAA": 0.0}
    out = validate_targets(D, {"BBB": HOLD}, {}, vt_inputs())              # HOLD with nothing held is flat
    assert out == {"BBB": 0.0}


def test_bad_values_and_unknown_contracts_raise_naming_coin_and_day():
    from portfolio.per_coin_ledger import validate_targets
    for bad in (float("nan"), float("inf"), "6000", None, True):
        assert_raises(lambda: validate_targets(D, {"AAA": bad}, {}, vt_inputs()), "AAA", str(D))
    assert_raises(lambda: validate_targets(D, {"ZZZ": 1.0}, {}, vt_inputs()), "ZZZ", "unknown")


def test_a_coin_ineligible_at_the_decision_may_only_go_to_zero():
    from portfolio.per_coin_ledger import HOLD, validate_targets
    inputs = vt_inputs(ineligible=(("AAA", D - timedelta(days=1)),))       # out at the 03-04 decision
    for bad in (6_000.0, -1.0, HOLD):                                       # buying more, flipping, or holding on
        assert_raises(lambda: validate_targets(D, {"AAA": bad}, {"AAA": 2.0}, inputs), "AAA", "eligible", "0")
    assert validate_targets(D, {"AAA": 0.0}, {"AAA": 2.0}, inputs) == {"AAA": 0.0}
    assert_raises(lambda: validate_targets(D, {"AAA": 50.0}, {}, inputs), "AAA", "eligible")   # not held either
    ok = validate_targets(D + timedelta(days=1), {"AAA": 50.0}, {}, inputs)  # eligible again at the 03-05 decision
    assert ok == {"AAA": 50.0}


def test_eligibility_is_read_at_the_decision_not_the_fill():
    from portfolio.per_coin_ledger import validate_targets
    inputs = vt_inputs(ineligible=(("BBB", D),))                            # ineligible on 03-05 itself
    assert validate_targets(D, {"BBB": 10.0}, {}, inputs) == {"BBB": 10.0}  # decided on 03-04: still eligible


def test_a_held_untradable_coin_needs_a_registered_event_and_is_held():
    """The decision is made the day before, so it cannot know a halt is coming. On a registered
    halt day any target for a held coin becomes HOLD (no trade is possible) and is noted; with no
    register entry the run stops, because the untradable day is unexplained."""
    from portfolio.per_coin_ledger import HOLD, validate_targets
    bare = vt_inputs(halted=(("CCC", D),))
    assert_raises(lambda: validate_targets(D, {"CCC": HOLD}, {"CCC": 1.0}, bare), "CCC", "registered")
    covered = vt_inputs(halted=(("CCC", D),), events=halt_event("CCC", D, D))
    for target in (HOLD, 0.0, 500.0):
        notes = []
        assert validate_targets(D, {"CCC": target}, {"CCC": 1.0}, covered, notes) == {"CCC": HOLD}
        assert bool(notes) == (target is not HOLD), notes
    assert validate_targets(D, {}, {"CCC": 1.0}, covered) == {"CCC": HOLD}  # missing target: held, not closed


def test_an_ineligible_held_coin_on_a_registered_halt_is_held_until_it_can_trade():
    from portfolio.per_coin_ledger import HOLD, validate_targets
    inputs = vt_inputs(ineligible=(("CCC", D - timedelta(days=1)),), halted=(("CCC", D),),
                       events=halt_event("CCC", D, D + timedelta(days=2)))
    assert validate_targets(D, {"CCC": 0.0}, {"CCC": 1.0}, inputs) == {"CCC": HOLD}   # wants out, cannot trade yet


def test_no_new_position_on_an_untradable_day():
    from portfolio.per_coin_ledger import validate_targets
    bare = vt_inputs(halted=(("BBB", D),))
    assert_raises(lambda: validate_targets(D, {"BBB": 100.0}, {}, bare), "BBB", "registered")
    assert validate_targets(D, {"BBB": 0.0}, {}, bare) == {"BBB": 0.0}
    covered = vt_inputs(halted=(("BBB", D),), events=halt_event("BBB", D, D))
    notes = []
    assert validate_targets(D, {"BBB": 100.0}, {}, covered, notes) == {"BBB": 0.0} and notes


def test_event_lookup_and_overlap():
    from portfolio.per_coin_ledger import event_on
    inputs = vt_inputs(events=halt_event("AAA", D, D + timedelta(days=2)))
    assert event_on(D + timedelta(days=1), "AAA", inputs.events).kind == "halt"
    assert event_on(D - timedelta(days=1), "AAA", inputs.events) is None
    assert event_on(D, "BBB", inputs.events) is None
    two = pd.concat([halt_event("AAA", D, D), halt_event("AAA", D, D, kind="span_end")])
    assert_raises(lambda: vt_inputs(events=two), "AAA", "overlap")


def test_a_trade_needs_a_finite_positive_fill_price():
    from portfolio.per_coin_ledger import HOLD, validate_targets
    inputs = vt_inputs()
    inputs.fill_prices.at[D, "AAA"] = np.nan                                 # tradable flag says yes, price missing
    assert_raises(lambda: validate_targets(D, {"AAA": 100.0}, {}, inputs), "AAA", "fill price")
    assert_raises(lambda: validate_targets(D, {"AAA": 0.0}, {"AAA": 1.0}, inputs), "AAA", "fill price")
    assert validate_targets(D, {"AAA": HOLD}, {"AAA": 1.0}, inputs) == {"AAA": HOLD}   # no trade, no price needed
    assert validate_targets(D, {"AAA": 0.0}, {}, inputs) == {"AAA": 0.0}
    inputs.fill_prices.at[D, "BBB"] = 0.0
    assert_raises(lambda: validate_targets(D, {"BBB": 5.0}, {}, inputs), "BBB", "fill price")


def test_the_decision_is_checked_before_execution_exceptions():
    """Buying a coin that is not eligible is an invalid decision even on a registered halt day."""
    from portfolio.per_coin_ledger import HOLD, validate_targets
    inputs = vt_inputs(ineligible=(("CCC", D - timedelta(days=1)),), halted=(("CCC", D),),
                       events=halt_event("CCC", D, D))
    for bad in (100.0, HOLD):
        assert_raises(lambda: validate_targets(D, {"CCC": bad}, {"CCC": 1.0}, inputs), "CCC", "eligible")
    assert_raises(lambda: validate_targets(D, {"CCC": 100.0}, {}, inputs), "CCC", "eligible")
    assert validate_targets(D, {"CCC": 0.0}, {"CCC": 1.0}, inputs) == {"CCC": HOLD}   # a valid exit waits


def test_decision_eligibility_ignores_activity_after_the_decision_instant():
    """Eligibility for fill day F is decided at F-1 00:00, so trading and marks on F-1 and later must
    not move it. Built through per_coin_universe.eligibility, as a real run would be."""
    from portfolio.per_coin_signal import load_per_coin_config
    from portfolio.per_coin_universe import eligibility
    config = load_per_coin_config()
    members = {EARLY: pd.DataFrame({"contract": list(CONTRACTS)})}
    days = [d for d in ALL_DAYS if START - timedelta(days=1) <= d <= END]
    cut = date(2021, 3, 6)                                                  # fill day; decided at 03-05 00:00

    def run(bars, counts):
        ok, _, _ = eligibility(members, days, bars, counts, config)
        return build_ledger_inputs(CONTRACTS, START, END, ok, bars=bars, marks=None, mark_counts=counts,
                                   settlements={})

    bars, _, counts, _ = synthetic()
    base = run(bars, counts)
    shocked_counts = {c: v.where(v.index < cut - timedelta(days=1), 22) for c, v in counts.items()}
    shocked_bars = {c: b.assign(volume=np.where(b.timestamp.dt.date >= cut, 0.0, b.volume)) for c, b in bars.items()}
    later = run(shocked_bars, shocked_counts)
    upto = [d for d in base.eligible.index if d <= cut]
    assert base.eligible.loc[upto].all().all()
    pd.testing.assert_frame_equal(base.eligible.loc[upto], later.eligible.loc[upto])
    assert not later.eligible.loc[cut + timedelta(days=1)].any()           # the next decision sees the marks


def test_a_days_open_counts_as_a_price_only_if_the_day_traded():
    """Documented convention, inherited from the sealed run: Binance's daily open is the first trade's
    price, so it exists within seconds of 00:00 when the day trades, and not at all when it does not.
    A day with no trade therefore contributes no decision price."""
    halted_day = date(2021, 3, 5)
    inputs = build(halted=(("AAA", halted_day),))
    assert np.isnan(inputs.decision_prices.at[halted_day + timedelta(days=1), "AAA"])
    assert inputs.decision_prices.at[halted_day + timedelta(days=1), "BBB"] == price("BBB", halted_day)


# ================================================================== step 3: the loop on normal days
RATE = 1e-4                                       # every synthetic settlement: longs pay 0.01%


def settlements_for(contracts, interval_hours=8, same_mark=None):
    """Funding settlements every `interval_hours` over the synthetic history, rate RATE, marked at the
    synthetic hourly price (or at `same_mark` for every contract, for the netting test)."""
    stamps = pd.date_range(pd.Timestamp(ALL_DAYS[0], tz="UTC"), pd.Timestamp(ALL_DAYS[-1], tz="UTC"),
                           freq=f"{interval_hours}h")
    out = {}
    for c in contracts:
        marks = [same_mark if same_mark is not None else price(c, s.date(), s.hour) for s in stamps]
        out[c] = pd.DataFrame({"stamp": stamps, "rate": RATE, "mark": marks, "interval_hours": float(interval_hours)})
    return out


def ledger_inputs(clock=PRIMARY, lag=1, interval_hours=8, same_mark=None):
    bars, marks, counts, eligible = synthetic()
    return build_ledger_inputs(CONTRACTS, START, END, eligible, clock=clock, decision_lag_days=lag, bars=bars,
                               marks=marks, mark_counts=counts,
                               settlements=settlements_for(CONTRACTS, interval_hours, same_mark))


def scripted(plan):
    """A target_fn that returns plan[day] (a dict), or {} on days not in the plan."""
    return lambda day, held, prices, base: dict(plan.get(day, {}))


def funding_by_hand(inputs, contract, qty, day):
    """-qty x mark x rate for every settlement in (fill instant of day, fill instant of day + 1]."""
    s = inputs.settlements[contract]
    a, b = fill_instant(day, inputs.clock), fill_instant(day + timedelta(days=1), inputs.clock)
    window = s[(s.stamp > a) & (s.stamp <= b)]
    return float(-(qty * window.mark * window.rate).sum()), len(window)


def row(ledger, day, contract):
    hit = ledger.rows[(ledger.rows.day == day) & (ledger.rows.contract == contract)]
    assert len(hit) == 1, (day, contract, len(hit))
    return hit.iloc[0]


def test_L1_hand_computed_two_coin_book():
    """Long AAA $6,000 and short BBB $4,000 from flat on day 1, hold on day 2, close on day 3."""
    from portfolio.per_coin_ledger import HOLD, run_per_coin_ledger
    inputs = ledger_inputs()
    d1, d2, d3 = START, START + timedelta(days=1), START + timedelta(days=2)
    plan = {d1: {"AAA": 6_000.0, "BBB": -4_000.0}, d2: {"AAA": HOLD, "BBB": HOLD}, d3: {"AAA": 0.0, "BBB": 0.0}}
    ledger = run_per_coin_ledger(scripted(plan), inputs)
    for c, notional in (("AAA", 6_000.0), ("BBB", -4_000.0)):
        p1, p2, p3 = (price(c, d) for d in (d1, d2, d3))
        q = notional / p1
        r1, r2, r3 = row(ledger, d1, c), row(ledger, d2, c), row(ledger, d3, c)
        f1, n1 = funding_by_hand(inputs, c, q, d1)
        f2, _ = funding_by_hand(inputs, c, q, d2)
        assert n1 == 3                                                       # 08:00, 16:00, next 00:00
        assert abs(r1.trade - notional) < 1e-9 and abs(r1.quantity_held - q) < 1e-12
        assert abs(r1.fee - 0.0005 * abs(notional)) < 1e-9 and abs(r1.slippage - 0.0005 * abs(notional)) < 1e-9
        assert abs(r1.price_pnl - q * (p2 - p1)) < 1e-9 and abs(r1.funding_pnl - f1) < 1e-12
        assert r2.event == "hold" and r2.trade == 0.0 and r2.fee == 0.0 and r2.quantity_held == r1.quantity_after
        assert abs(r2.price_pnl - q * (p3 - p2)) < 1e-9 and abs(r2.funding_pnl - f2) < 1e-12
        close = -q * p3
        assert abs(r3.trade - close) < 1e-9 and r3.quantity_held == 0.0 and r3.quantity_after == 0.0
        assert r3.price_pnl == 0.0 and r3.funding_pnl == 0.0
        assert abs(r3.fee - 0.0005 * abs(close)) < 1e-9
        total = (q * (p3 - p1) + f1 + f2 - 0.001 * (abs(notional) + abs(close)))
        got = ledger.rows[ledger.rows.contract == c].net_pnl.sum()
        assert abs(got - total) < 1e-9, (c, got, total)
    assert row(ledger, d1, "BBB").funding_pnl > 0 > row(ledger, d1, "AAA").funding_pnl   # short receives, long pays
    rows = ledger.rows
    assert (abs(rows.net_pnl - (rows.price_pnl + rows.funding_pnl - rows.fee - rows.slippage)) < 1e-9).all()


def test_L3_a_settlement_at_the_fill_instant_belongs_to_the_overnight_position():
    from portfolio.per_coin_ledger import run_per_coin_ledger
    for clock, lag in ((PRIMARY, 1), (HOURLY, 0)):
        inputs = ledger_inputs(clock=clock, lag=lag, interval_hours=1)     # a settlement every hour
        d1, d2 = START, START + timedelta(days=1)
        ledger = run_per_coin_ledger(scripted({d1: {"AAA": 5_000.0}, d2: {"AAA": 0.0}}), inputs)
        q = row(ledger, d1, "AAA").quantity_held
        expected, n = funding_by_hand(inputs, "AAA", q, d1)
        assert n == 24 and abs(row(ledger, d1, "AAA").funding_pnl - expected) < 1e-12
        s = inputs.settlements["AAA"]
        at_open = s[s.stamp == fill_instant(d1, clock)].iloc[0]           # the settlement as the position opens
        at_close = s[s.stamp == fill_instant(d2, clock)].iloc[0]          # the settlement as it closes
        inside = s[(s.stamp > fill_instant(d1, clock)) & (s.stamp <= fill_instant(d2, clock))]
        assert at_close.stamp in set(inside.stamp) and at_open.stamp not in set(inside.stamp)
        assert row(ledger, d2, "AAA").funding_pnl == 0.0                   # closed at the d2 fill: owes nothing after


def test_L4_flip_long_to_short():
    from portfolio.per_coin_ledger import run_per_coin_ledger
    inputs = ledger_inputs()
    d1, d2 = START, START + timedelta(days=1)
    ledger = run_per_coin_ledger(scripted({d1: {"AAA": 5_000.0}, d2: {"AAA": -3_000.0}}), inputs)
    r1, r2 = row(ledger, d1, "AAA"), row(ledger, d2, "AAA")
    old = r1.quantity_held * price("AAA", d2)                              # the long, marked at the d2 fill
    assert abs(r2.trade - (-3_000.0 - old)) < 1e-9 and abs(abs(r2.trade) - (3_000.0 + old)) < 1e-9
    assert abs(r2.fee + r2.slippage - 0.001 * abs(r2.trade)) < 1e-9
    assert r2.quantity_held < 0 and r2.side == "short"
    expected, _ = funding_by_hand(inputs, "AAA", r2.quantity_held, d2)
    assert abs(r2.funding_pnl - expected) < 1e-12 and r2.funding_pnl > 0  # a short receives at a positive rate


def test_L5_matched_funding_notionals_net_to_zero():
    """Equal and opposite quantity x settlement mark at every settlement, identical rates: net zero."""
    from portfolio.per_coin_ledger import run_per_coin_ledger
    inputs = ledger_inputs(same_mark=500.0)                                # both contracts marked identically
    d1 = START
    qty = 4.0
    plan = {d1: {"AAA": qty * price("AAA", d1), "BBB": -qty * price("BBB", d1)}}   # +4 and -4 units
    ledger = run_per_coin_ledger(scripted(plan), inputs)
    a, b = row(ledger, d1, "AAA"), row(ledger, d1, "BBB")
    assert abs(a.quantity_held + b.quantity_held) < 1e-12
    assert a.funding_pnl < 0 < b.funding_pnl and abs(a.funding_pnl + b.funding_pnl) < 1e-12


def test_L11_eligible_flat_coins_get_rows_with_their_return():
    from portfolio.per_coin_ledger import run_per_coin_ledger
    inputs = ledger_inputs()
    ledger = run_per_coin_ledger(scripted({START: {"AAA": 1_000.0}}), inputs)
    for d in (START, START + timedelta(days=3), END):
        r = row(ledger, d, "CCC")
        assert r.event == "flat" and r.eligible and r.quantity_held == 0.0 and r.net_pnl == 0.0
        assert abs(r.price_return - (price("CCC", d + timedelta(days=1)) / price("CCC", d) - 1)) < 1e-15
    assert len(ledger.rows) == len(CONTRACTS) * (len(inputs.days) - 1)    # every eligible coin, every day


def test_target_fn_sees_only_decision_time_information():
    from portfolio.per_coin_ledger import run_per_coin_ledger
    inputs = ledger_inputs()
    seen = []

    def spy(day, held, prices, base):
        seen.append((day, dict(held), dict(prices), base))
        return {"AAA": 1_000.0} if day == START else {}

    run_per_coin_ledger(spy, inputs)
    day, held, prices, base = seen[1]                                      # the second day: AAA held from day 1
    assert set(held) == {"AAA"} and base == inputs.capital
    assert prices == {c: price(c, day - timedelta(days=1)) for c in CONTRACTS}   # D-1 00:00, never the fill


def test_halts_are_not_built_yet():
    from portfolio.per_coin_ledger import run_per_coin_ledger
    bars, marks, counts, eligible = synthetic(halted=(("AAA", START + timedelta(days=1)),))
    events = halt_event("AAA", START + timedelta(days=1), START + timedelta(days=1))
    inputs = build_ledger_inputs(CONTRACTS, START, END, eligible, bars=bars, marks=marks, mark_counts=counts,
                                 settlements=settlements_for(CONTRACTS), events=events)
    try:
        run_per_coin_ledger(scripted({START: {"AAA": 1_000.0}}), inputs)
    except NotImplementedError as error:
        assert "AAA" in str(error) and "step" in str(error)
    else:
        raise AssertionError("expected NotImplementedError")


# ================================================================== step 8: settlements (L8, L13)
SETTLE_DAY = START + timedelta(days=2)
SETTLE_AT = pd.Timestamp(SETTLE_DAY, tz="UTC") + pd.Timedelta(hours=9)


def index_reader(contract, day):
    """Synthetic 1-minute index: close = 900 + minute of day / 100 (AAA), so every mean is known."""
    stamps = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=1440, freq="min")
    return pd.DataFrame({"open_time": stamps, "close": 900.0 + np.arange(1440) / 100.0})


def settlement_event(contract="AAA", first=SETTLE_DAY, last=END, instant=SETTLE_AT):
    return pd.DataFrame([{"contract": contract, "kind": "settlement", "first_day": first, "last_day": last,
                          "instant": instant, "evidence": "1m klines: last trade 09:00; announcement 42"}])


def settled_inputs(**kw):
    """AAA settles at 09:00 on SETTLE_DAY and never trades again."""
    after = tuple(("AAA", d) for d in ALL_DAYS if d > SETTLE_DAY)
    bars, marks, counts, eligible = synthetic(halted=after)
    return build_ledger_inputs(CONTRACTS, START, END, eligible, bars=bars, marks=marks, mark_counts=counts,
                               settlements=settlements_for(CONTRACTS), events=settlement_event(**kw),
                               index_reader=index_reader)


def test_the_settlement_price_is_the_30_minute_index_mean_before_the_instant():
    from portfolio.per_coin_ledger import settlement_price
    got = settlement_price("AAA", SETTLE_AT, index_reader)
    m = 9 * 60                                                              # minute of day at 09:00
    assert got["price"] == float(np.mean(900.0 + np.arange(m - 30, m) / 100.0))
    assert got["price_60m"] == float(np.mean(900.0 + np.arange(m - 60, m) / 100.0))
    assert got["binance_window_minutes"] == 60 and got["minutes"] == 30
    gappy = lambda c, d: index_reader(c, d).drop(index=[m - 5])            # one minute missing  # noqa: E731
    assert_raises(lambda: settlement_price("AAA", SETTLE_AT, gappy), "29 of 30")
    late = pd.Timestamp("2025-01-06 09:00", tz="UTC")
    assert settlement_price("AAA", late, index_reader)["binance_window_minutes"] == 30


def test_a_settlement_needs_an_instant_on_its_first_day():
    assert_raises(lambda: settled_inputs(instant=SETTLE_AT + pd.Timedelta(days=1)), "instant on its first day")
    assert_raises(lambda: settled_inputs(instant=pd.NaT), "instant")


def test_L8_a_held_position_is_closed_at_S_with_funding_to_T_and_the_taker_fee():
    from portfolio.per_coin_ledger import HOLD, SETTLEMENT_FEE, run_per_coin_ledger
    inputs = settled_inputs()
    ledger = run_per_coin_ledger(scripted({START: {"AAA": 1_000.0}, START + timedelta(days=1): {"AAA": HOLD},
                                           SETTLE_DAY: {"AAA": HOLD}}), inputs)
    r = row(ledger, SETTLE_DAY, "AAA")
    S = inputs.settlement_prices[("AAA", SETTLE_DAY)]["price"]
    q = 1_000.0 / price("AAA", START)
    fill = price("AAA", SETTLE_DAY)
    assert r.event == "settlement" and r.quantity_before == q == r.quantity_held and r.quantity_after == 0.0
    assert r.trade == 0.0 and r.exit_price == S and abs(r.price_pnl - q * (S - fill)) < 1e-9
    s = inputs.settlements["AAA"]
    window = s[(s.stamp > pd.Timestamp(SETTLE_DAY, tz="UTC")) & (s.stamp <= SETTLE_AT)]
    assert len(window) == 1                                                 # 08:00 only; 16:00 is after T
    assert abs(r.funding_pnl + q * float((window.mark * window.rate).sum())) < 1e-12
    assert abs(r.close_trade + q * S) < 1e-9 and abs(r.close_fee - q * S * SETTLEMENT_FEE) < 1e-12
    assert r.fee == r.close_fee and r.slippage == 0.0
    assert abs(r.net_pnl - (r.price_pnl + r.funding_pnl - r.fee)) < 1e-12
    later = ledger.rows[(ledger.rows.contract == "AAA") & (ledger.rows.day > SETTLE_DAY)]
    assert (later.quantity_held == 0.0).all() and later["flags"].str.startswith("excluded_no_price").all()
    assert abs(ledger.daily.turnover_instrument_end.sum() - q * S / inputs.capital) < 1e-15


def test_L8_an_opening_trade_on_the_settlement_day_keeps_its_slippage():
    from portfolio.per_coin_ledger import SETTLEMENT_FEE, run_per_coin_ledger
    from portfolio.basket import cost_rates
    inputs = settled_inputs()
    ledger = run_per_coin_ledger(scripted({START: {"AAA": 1_000.0}, START + timedelta(days=1): {"AAA": 1_000.0},
                                           SETTLE_DAY: {"AAA": -2_000.0}}), inputs)
    r = row(ledger, SETTLE_DAY, "AAA")
    fee, slip = cost_rates(inputs.cost_bp_per_side)
    assert abs(r.quantity_held + 2_000.0 / price("AAA", SETTLE_DAY)) < 1e-12 and r.quantity_after == 0.0
    assert abs(r.slippage - abs(r.trade) * slip) < 1e-12
    assert abs(r.fee - (abs(r.trade) * fee + abs(r.close_trade) * SETTLEMENT_FEE)) < 1e-12
    assert abs(r.close_trade - 2_000.0 / price("AAA", SETTLE_DAY) * r.exit_price) < 1e-9        # buying back a short
    assert abs(ledger.daily.turnover.sum() * inputs.capital
               - (ledger.rows.trade.abs().sum() + ledger.rows.close_trade.abs().sum())) < 1e-9


def test_L13_an_eligible_flat_coin_still_gets_its_return_to_S():
    from portfolio.per_coin_ledger import run_per_coin_ledger
    inputs = settled_inputs()
    ledger = run_per_coin_ledger(scripted({}), inputs)
    r = row(ledger, SETTLE_DAY, "AAA")
    S = inputs.settlement_prices[("AAA", SETTLE_DAY)]["price"]
    assert r.event == "settlement" and r.quantity_held == 0.0 and r.net_pnl == 0.0 and r.close_trade == 0.0
    assert r.price_return == S / price("AAA", SETTLE_DAY) - 1.0


def test_R7_recomputes_the_settlement_from_the_register_not_the_row():
    from portfolio.per_coin_ledger import HOLD, daily_table, reconcile, run_per_coin_ledger
    inputs = settled_inputs()
    ledger = run_per_coin_ledger(scripted({START: {"AAA": 1_000.0}, START + timedelta(days=1): {"AAA": HOLD},
                                           SETTLE_DAY: {"AAA": HOLD}}), inputs)
    rows = ledger.rows.copy()
    i = rows.index[(rows.day == SETTLE_DAY) & (rows.contract == "AAA")][0]
    bump = 1.01
    q = rows.at[i, "quantity_held"]
    rows.at[i, "exit_price"] *= bump                                       # a consistent but wrong S
    rows.at[i, "close_trade"] = -q * rows.at[i, "exit_price"]
    extra = q * rows.at[i, "exit_price"] - q * rows.at[i, "exit_price"] / bump
    for col in ("price_pnl", "net_pnl"):
        rows.at[i, col] += extra
    assert_raises(lambda: reconcile(rows, daily_table(rows, inputs), inputs), "R7")
    rows = ledger.rows.copy()
    rows.at[i, "funding_end"] = end = rows.at[i, "funding_end"] + pd.Timedelta(hours=8)   # funded past T,
    s = inputs.settlements["AAA"]                                                          # consistently
    w = s[(s.stamp > rows.at[i, "funding_start"]) & (s.stamp <= end)]
    paid = -q * float((w.mark * w.rate).sum())
    rows.at[i, "net_pnl"] += paid - rows.at[i, "funding_pnl"]
    rows.at[i, "funding_pnl"] = paid
    assert_raises(lambda: reconcile(rows, daily_table(rows, inputs), inputs), "R7", "past its instant")


def test_attribution_reproduces_a_book_with_a_settlement():
    from portfolio.per_coin_attribution import attribute
    from portfolio.per_coin_ledger import run_per_coin_ledger
    inputs = settled_inputs()
    ledger = run_per_coin_ledger(scripted({START: {"AAA": 1_000.0, "BBB": -800.0}}), inputs)
    daily = attribute(ledger, inputs)                                      # raises if any day fails
    assert daily.n_excluded.sum() > 0


def gap_settlements():
    """AAA's funding with the 16:00 settlement of START + 1 removed: an evidenced schedule gap."""
    s = settlements_for(CONTRACTS)
    gone = pd.Timestamp(START + timedelta(days=1), tz="UTC") + pd.Timedelta(hours=16)
    s["AAA"] = s["AAA"][s["AAA"].stamp != gone].reset_index(drop=True)
    register = pd.DataFrame([{"contract": "AAA", "start": gone - pd.Timedelta(hours=8),
                              "end": gone + pd.Timedelta(hours=8), "evidence": "archive == API"}])
    return s, register, gone


def test_registered_funding_gaps_are_accepted_and_unregistered_ones_raise():
    from portfolio.basket import funding_over
    from portfolio.per_coin_ledger import funding_registered
    s, register, gone = gap_settlements()
    a, b = pd.Timestamp(START + timedelta(days=1), tz="UTC"), pd.Timestamp(START + timedelta(days=2), tz="UTC")
    full = settlements_for(CONTRACTS)
    assert funding_registered(full, "BBB", a, b, 3.0, register) == funding_over(full, "BBB", a, b, 3.0)
    assert_raises(lambda: funding_over(s, "AAA", a, b, 3.0), "incomplete")
    got = funding_registered(s, "AAA", a, b, 3.0, register)
    w = s["AAA"][(s["AAA"].stamp > a) & (s["AAA"].stamp <= b)]
    assert len(w) == 2 and got == -float((3.0 * w.mark * w.rate).sum())            # the missing one costs 0
    assert_raises(lambda: funding_registered(s, "AAA", a, b, 3.0, register.iloc[0:0]), "incomplete")


def test_a_book_holding_through_a_registered_gap_reconciles():
    from portfolio.per_coin_ledger import HOLD, _leg, run_per_coin_ledger
    from portfolio.basket import cost_rates, leg_row
    s, register, gone = gap_settlements()
    bars, marks, counts, eligible = synthetic()
    inputs = build_ledger_inputs(CONTRACTS, START, END, eligible, bars=bars, marks=marks, mark_counts=counts,
                                 settlements=s, suspensions=register)
    ledger = run_per_coin_ledger(scripted({START: {"AAA": 1_000.0, "BBB": 500.0}, START + timedelta(days=1):
                                           {"AAA": HOLD, "BBB": 700.0}}), inputs)           # reconciles or raises
    assert row(ledger, START + timedelta(days=1), "AAA").funding_pnl != 0.0
    rates = cost_rates(10.0)
    a, b = pd.Timestamp(START, tz="UTC"), pd.Timestamp(START + timedelta(days=1), tz="UTC")
    mine = _leg("BBB", inputs, 0.4, 900.0, 1001.0, 1003.0, a, b, rates)
    sealed = leg_row("BBB", inputs.settlements, 0.4, 900.0, 1001.0, 1003.0, a, b, fee_rate=rates[0], slippage_rate=rates[1])
    assert all(mine[k] == sealed[k] for k in mine)


def test_a_mark_hole_is_never_crossed_by_a_held_window():
    from portfolio.per_coin_ledger import funding_registered
    s, register, gone = gap_settlements()
    holes = register.assign(kind="mark_hole")
    a, b = pd.Timestamp(START + timedelta(days=1), tz="UTC"), pd.Timestamp(START + timedelta(days=2), tz="UTC")
    assert_raises(lambda: funding_registered(s, "AAA", a, b, 3.0, holes), "incomplete")
    assert funding_registered(s, "AAA", a, b, 0.0, holes) == 0.0                     # nothing held: nothing to value


def test_a_registered_span_break_starts_a_new_instrument():
    from portfolio.per_coin_universe import SPAN_STARTS, span_prices
    from portfolio.per_coin_ledger import _tradable_frame
    bars, marks, counts, eligible = synthetic()
    swap = START - timedelta(days=10)
    bars["BBB"].attrs[SPAN_STARTS] = (pd.Timestamp(swap),)
    spans = span_prices(bars["BBB"], 30)
    assert len(spans) == 2 and spans[1].index[0] == swap                            # history restarts at the swap
    tradable = _tradable_frame(CONTRACTS, ALL_DAYS, bars, counts, 23, 30)
    assert not tradable.at[swap, "BBB"] and tradable.at[swap + timedelta(days=1), "BBB"]   # its first day is partial


# ================================================================== step 4: daily table, R1-R8, fingerprints
def busy_book(seed=7, inputs=None):
    """A seeded random book over every day: buys, sells, flips, holds and closes on all three coins."""
    from portfolio.per_coin_ledger import HOLD, run_per_coin_ledger
    inputs = inputs or ledger_inputs()
    rng = np.random.default_rng(seed)
    plan = {}
    for d in inputs.days[:-1]:
        day_plan = {}
        for c in CONTRACTS:
            u = rng.random()
            day_plan[c] = HOLD if u < 0.35 else 0.0 if u < 0.5 else float(rng.normal(0, 8_000))
        plan[d] = day_plan
    plan[inputs.days[2]] = {c: 0.0 for c in CONTRACTS}                     # one fully flat day
    plan[inputs.days[3]] = {}                                              # and it stays flat (nothing held)
    return inputs, run_per_coin_ledger(scripted(plan), inputs)


def test_L2_a_busy_book_passes_every_reconciliation():
    inputs, ledger = busy_book()                                           # run_per_coin_ledger reconciles
    daily = ledger.daily
    assert list(daily.index) == inputs.days[:-1]                           # every scored day, flat ones too
    flat_day = inputs.days[3]
    assert daily.at[flat_day, "net_pnl"] == 0.0 and daily.at[flat_day, "gross_exposure"] == 0.0
    rows = ledger.rows
    assert abs(daily.net_pnl.sum() - rows.net_pnl.sum()) < 1e-9
    assert np.allclose(daily.net_return, daily.net_pnl / inputs.capital, rtol=0, atol=1e-15)
    assert np.allclose(daily.funding_long + daily.funding_short, daily.funding_pnl, rtol=0, atol=1e-12)
    assert ((rows.quantity_held > 0) == (rows.side == "long")).all()
    assert (daily.n_long + daily.n_short <= len(CONTRACTS)).all() and daily.n_short.sum() > 0
    assert abs(daily.turnover.sum() - rows.trade.abs().sum() / inputs.capital) < 1e-12


def tamper(check, mutate, rebuild_daily=False):
    """Corrupt a valid ledger with `mutate(rows, daily)`; reconcile must name `check`."""
    from portfolio.per_coin_ledger import daily_table, reconcile
    inputs, ledger = busy_book()
    rows, daily = ledger.rows.copy(), ledger.daily.copy()
    daily.attrs = dict(ledger.daily.attrs)
    mutate(rows, daily)
    if rebuild_daily:
        daily = daily_table(rows, inputs)
    assert_raises(lambda: reconcile(rows, daily, inputs), check)


def first(rows, mask):
    return rows.index[mask][0]


def test_each_reconciliation_catches_its_own_corruption():
    held = lambda r: r.quantity_held != 0.0                                # noqa: E731

    def r1(rows, daily):
        rows.loc[first(rows, held(rows)), "net_pnl"] += 1.0

    def r2(rows, daily):
        rows.loc[first(rows, rows.event == "hold"), "quantity_before"] *= 1.000001

    def r3(rows, daily):
        rows.loc[first(rows, rows.event == "trade"), "trade"] += 1.0

    def r4(rows, daily):
        daily.iloc[5, daily.columns.get_loc("net_pnl")] += 1.0

    def r5(rows, daily):
        daily.iloc[5, daily.columns.get_loc("turnover")] += 0.01

    def r6(rows, daily):
        daily.iloc[5, daily.columns.get_loc("funding_long")] += 1.0

    def r7_funding_counted_twice(rows, daily):                             # identity still balances
        i = first(rows, held(rows) & (rows.funding_pnl != 0.0))
        rows.loc[i, "net_pnl"] += rows.loc[i, "funding_pnl"]
        rows.loc[i, "funding_pnl"] *= 2.0

    def r7_price_counted_twice(rows, daily):
        i = first(rows, held(rows) & (rows.price_pnl != 0.0))
        rows.loc[i, "net_pnl"] += rows.loc[i, "price_pnl"]
        rows.loc[i, "price_pnl"] *= 2.0

    def r8(rows, daily):
        daily.attrs["terminal_positions"] = {}

    tamper("R1", r1)
    tamper("R2", r2)
    tamper("R3", r3)
    tamper("R4", r4)
    tamper("R5", r5)
    tamper("R6", r6)
    tamper("R6", r7_funding_counted_twice, rebuild_daily=True)            # daily rebuilt so R4 agrees; the
                                                                           # per-side recomputation (R6) sees it first
    tamper("R7", r7_price_counted_twice, rebuild_daily=True)
    tamper("R8", r8)


def test_R7_recomputes_from_the_input_prices_not_the_rows_copies():
    """Review finding: changing a row's exit price and its P&L consistently used to pass."""
    def shift_exit(rows, daily):
        i = first(rows, (rows.quantity_held != 0.0) & (rows.price_pnl != 0.0))
        bump = 25.0
        rows.loc[i, "exit_price"] += bump
        rows.loc[i, "price_pnl"] += rows.loc[i, "quantity_held"] * bump
        rows.loc[i, "net_pnl"] += rows.loc[i, "quantity_held"] * bump

    def shift_fill(rows, daily):                                           # both fill copies and the trade
        i = first(rows, (rows.event == "hold") & (rows.quantity_held != 0.0))
        bump = 25.0
        q = rows.loc[i, "quantity_held"]
        rows.loc[i, "fill_price"] += bump
        rows.loc[i, "notional_before"] += q * bump
        rows.loc[i, "price_pnl"] -= q * bump
        rows.loc[i, "net_pnl"] -= q * bump

    tamper("R7", shift_exit, rebuild_daily=True)
    tamper("R7", shift_fill, rebuild_daily=True)


def test_R6_recomputes_each_side_from_signed_holdings():
    """Review finding: swapping the long and short funding totals used to pass."""
    def swap(rows, daily):
        both = (daily.funding_long != 0.0) & (daily.funding_short != 0.0)
        d = daily.index[both][0]
        daily.loc[d, ["funding_long", "funding_short"]] = daily.loc[d, ["funding_short", "funding_long"]].to_numpy()

    def wrong_side(rows, daily):
        rows.loc[first(rows, rows.quantity_held > 0), "side"] = "short"

    tamper("R6", swap)
    tamper("R6", wrong_side)


def test_an_eligible_coin_with_no_price_stops_the_run_unless_registered():
    """Review finding: a missing price used to give a blank return and pass, silently dropping an
    eligible coin from the attribution."""
    from portfolio.per_coin_ledger import run_per_coin_ledger
    halted = START + timedelta(days=4)
    bars, marks, counts, eligible = synthetic(halted=(("CCC", halted),))   # eligible at the decision, no price
    kw = dict(bars=bars, marks=marks, mark_counts=counts, settlements=settlements_for(CONTRACTS))
    bare = build_ledger_inputs(CONTRACTS, START, END, eligible, **kw)
    assert_raises(lambda: run_per_coin_ledger(scripted({}), bare), "CCC", "no price", "attribution")
    covered = build_ledger_inputs(CONTRACTS, START, END, eligible, events=halt_event("CCC", halted, halted), **kw)
    ledger = run_per_coin_ledger(scripted({}), covered)
    flagged = ledger.rows[ledger.rows["flags"] != ""]                      # rows.flags is a pandas attribute
    assert set(zip(flagged.day, flagged.contract)) == {(halted - timedelta(days=1), "CCC"), (halted, "CCC")}
    assert flagged["flags"].eq("excluded_no_price:halt").all() and flagged.price_return.isna().all()


def test_terminal_positions_are_marked_at_the_day_after_the_end():
    inputs, ledger = busy_book()
    terminal = ledger.daily.attrs["terminal_positions"]
    last = ledger.rows[ledger.rows.day == inputs.days[-2]]
    assert set(terminal) == set(last.contract[last.quantity_after != 0.0])
    for c in terminal:
        assert row(ledger, inputs.days[-2], c).exit_price == price(c, inputs.days[-1])


def test_turnover_from_an_exit_counts_as_membership():
    from portfolio.per_coin_ledger import run_per_coin_ledger
    bars, marks, counts, eligible = synthetic()
    out_day = START + timedelta(days=2)
    eligible.loc[out_day:, "CCC"] = False                                  # CCC leaves at the 03-03 decision
    inputs = build_ledger_inputs(CONTRACTS, START, END, eligible, bars=bars, marks=marks, mark_counts=counts,
                                 settlements=settlements_for(CONTRACTS))
    from portfolio.per_coin_ledger import HOLD
    exit_day = out_day + timedelta(days=1)                                 # sold at the 03-04 fill
    plan = {START: {"CCC": 2_000.0}, START + timedelta(days=1): {"CCC": HOLD}, out_day: {"CCC": HOLD},
            exit_day: {"CCC": 0.0}}                                         # a held coin with no target is closed
    ledger = run_per_coin_ledger(scripted(plan), inputs)
    r = row(ledger, exit_day, "CCC")
    assert r.trade < 0 and r.quantity_after == 0.0
    assert abs(ledger.daily.at[exit_day, "turnover_membership"] - abs(r.trade) / inputs.capital) < 1e-15
    assert ledger.daily.at[START, "turnover_membership"] > 0               # entering at the start is membership too


def test_L12_determinism_and_fingerprints():
    _, a = busy_book(seed=3)
    _, b = busy_book(seed=3)
    _, c = busy_book(seed=4)
    assert a.fingerprint == b.fingerprint and a.inputs_fingerprint == b.inputs_fingerprint
    assert a.fingerprint != c.fingerprint and a.inputs_fingerprint == c.inputs_fingerprint
    changed = ledger_inputs()
    changed.fill_prices.iloc[4, 0] *= 1.01
    _, d = busy_book(seed=3, inputs=changed)
    assert d.inputs_fingerprint != a.inputs_fingerprint and d.fingerprint != a.fingerprint


# ================================================================== step 5: sealed equivalence (L6)
COMPARED = ("trade", "price_pnl", "funding_pnl", "fee", "slippage", "net_pnl")


def replay_sealed(sealed: pd.DataFrame):
    """Feed a sealed basket ledger's per-leg positions to the per-coin ledger as targets.

    Each sealed row (day D, contract c) becomes the target notional_after for c on D; a contract
    with no sealed row on D is flat. Eligibility is exactly the sealed rows, decided on D itself
    (lag 0), so validation admits precisely the legs the sealed run held or traded. Prices, funding
    and costs come from the per-coin inputs, independently of the sealed run."""
    from portfolio.basket import SCORED_END, SCORED_START
    from portfolio.per_coin_ledger import run_per_coin_ledger
    sealed = sealed.copy()
    sealed["day"] = pd.to_datetime(sealed.holding_period_start, utc=True).dt.date
    contracts = tuple(sorted(sealed.contract.unique()))
    days = [SCORED_START + timedelta(days=i) for i in range((SCORED_END - SCORED_START).days + 1)]
    eligible = pd.DataFrame(False, index=days, columns=list(contracts))
    plan = {}
    for r in sealed.itertuples():
        eligible.at[r.day, r.contract] = True
        plan.setdefault(r.day, {})[r.contract] = float(r.notional_after)
    inputs = build_ledger_inputs(contracts, SCORED_START, SCORED_END, eligible, decision_lag_days=0)
    ledger = run_per_coin_ledger(scripted(plan), inputs)               # reconciles R1-R8 on the way
    ours = ledger.rows.set_index(["day", "contract"])
    theirs = sealed.set_index(["day", "contract"])
    assert set(ours.index) == set(theirs.index), "row sets differ"
    ours = ours.loc[theirs.index]
    diffs = {k: float((ours[k] - theirs[k]).abs().max()) for k in COMPARED}
    diffs["quantity"] = float((ours.quantity_held - theirs.quantity_after).abs().max())
    daily_theirs = theirs.groupby(level="day").net_pnl.sum().reindex(ledger.daily.index, fill_value=0.0)
    diffs["daily_net"] = float((ledger.daily.net_pnl - daily_theirs).abs().max())
    return ledger, diffs, len(theirs)


def test_L6_the_sealed_benchmark_ledger_is_reproduced_row_by_row():
    from portfolio.basket import benchmark
    sealed = benchmark().ledger                                         # recomputed in memory: full precision
    ledger, diffs, n = replay_sealed(sealed)
    print(f"   real: sealed benchmark, {n} rows; largest absolute differences: "
          + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()))
    assert all(v <= 1e-9 for v in diffs.values()), diffs


def test_L6_the_sealed_strategy_ledger_is_reproduced_row_by_row():
    """The sealed primary run (buffered, 10 bp): shorts, buffer holds and monthly resets. Read from the
    sealed folder, so values carry the CSV's round-trip precision."""
    from pathlib import Path
    folder = Path(__file__).resolve().parent.parent / "reports/registered_run/c3e68cb1f90a/basket_buffered_10"
    sealed = pd.read_csv(folder / "ledger.csv.gz")
    ledger, diffs, n = replay_sealed(sealed)
    shorts = int((ledger.rows.quantity_held < 0).sum())
    print(f"   real: sealed strategy, {n} rows ({shorts} short); largest absolute differences: "
          + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()))
    assert shorts > 0 and all(v <= 1e-9 for v in diffs.values()), diffs


# ================================================================== step 6: no look-ahead in the running ledger (L10)
def random_world(seed=11):
    """Random-walk prices (so a trend rule takes both sides), with marks and settlements to match."""
    bars, marks, counts, eligible = synthetic()
    rng = np.random.default_rng(seed)
    walk = {c: np.exp(np.cumsum(rng.normal(0, 0.04, len(ALL_DAYS)))) for c in CONTRACTS}
    factor = {c: dict(zip(ALL_DAYS, walk[c])) for c in CONTRACTS}
    bars = {c: b.assign(open=b.open * b.timestamp.dt.date.map(factor[c]).to_numpy()) for c, b in bars.items()}
    marks = {c: m.assign(open=m.open * [factor[c][h.date()] for h in m.index]) for c, m in marks.items()}
    settlements = settlements_for(CONTRACTS)
    for c, s in settlements.items():
        s["mark"] = s.mark * [factor[c][t.date()] for t in s.stamp]
        s["rate"] = rng.normal(5e-5, 1e-4, len(s))                          # both signs occur
    return bars, marks, counts, eligible, settlements


def scramble_after(world, instant, seed=99):
    """Change every value stamped after `instant`: daily opens and volumes of later days, hourly
    marks, and funding rates and marks. Stamps are kept (the chain stays complete) and values stay
    positive (tradability is unchanged), so any change in the ledger is look-ahead or legitimate
    dependence on the future, nothing else."""
    bars, marks, counts, eligible, settlements = world
    rng = np.random.default_rng(seed)
    after_day = lambda ts: ts > instant                                    # noqa: E731
    bars = {c: b.assign(open=np.where(after_day(b.timestamp), b.open * rng.uniform(0.5, 1.5, len(b)), b.open),
                        volume=np.where(b.timestamp >= instant.normalize(), b.volume * rng.uniform(0.1, 9, len(b)),
                                        b.volume))
            for c, b in bars.items()}
    marks = {c: m.assign(open=np.where(m.index > instant, m.open * rng.uniform(0.5, 1.5, len(m)), m.open))
             for c, m in marks.items()}
    out = {}
    for c, s in settlements.items():
        later = s.stamp > instant
        s = s.copy()
        s.loc[later, "rate"] = rng.normal(0, 3e-4, int(later.sum()))
        s.loc[later, "mark"] = s.loc[later, "mark"] * rng.uniform(0.5, 1.5, int(later.sum()))
        out[c] = s
    return bars, marks, counts, eligible, out


def trend_rule():
    """A stateful rule: long or short $3,000 by the sign of the change in the decision price since its
    last decision; HOLD when already on that side. Logs every decision it makes."""
    from portfolio.per_coin_ledger import HOLD
    seen, log = {}, {}

    def fn(day, held, prices, base):
        out = {}
        for c, p in sorted(prices.items()):
            if not np.isfinite(p):
                continue
            prev, seen[c] = seen.get(c), p
            if prev is None:
                continue
            side = float(np.sign(p - prev))
            out[c] = HOLD if c in held and np.sign(held[c]) == side else side * 3_000.0
        log[day] = ({c: ("HOLD" if v is HOLD else v) for c, v in out.items()},
                    {c: round(q, 12) for c, q in held.items()}, dict(prices))
        return out
    return fn, log


def run_world(world, clock=PRIMARY, lag=1, end=END):
    from portfolio.per_coin_ledger import run_per_coin_ledger
    bars, marks, counts, eligible, settlements = world
    inputs = build_ledger_inputs(CONTRACTS, START, end, eligible, clock=clock, decision_lag_days=lag, bars=bars,
                                 marks=marks, mark_counts=counts, settlements=settlements)
    fn, log = trend_rule()
    return run_per_coin_ledger(fn, inputs), log


def test_L10a_decisions_are_frozen_at_their_instant():
    """Scramble everything stamped after the decision instant for fill day X: every decision up to and
    including X, and the information each was given, is unchanged; later decisions do change."""
    world = random_world()
    for clock, lag in ((PRIMARY, 1), (PRIMARY, 0), (HOURLY, 0)):
        x = START + timedelta(days=5)
        instant = pd.Timestamp(x - timedelta(days=lag), tz="UTC")          # when the decision for X is stamped
        _, base = run_world(world, clock, lag)
        _, other = run_world(scramble_after(world, instant), clock, lag)
        upto = [d for d in base if d <= x]
        assert [base[d] for d in upto] == [other[d] for d in upto], (clock, lag)
        assert any(base[d] != other[d] for d in base if d > x), "the scramble did not reach later decisions"
        assert any(v != "HOLD" and v < 0 for t, _, _ in base.values() for v in t.values()), "no short tested"


def test_L10b_realised_pnl_uses_only_its_own_holding_period():
    """Scramble everything stamped after cut 00:00: every row whose holding period ends by then is
    unchanged; the row for `cut` itself (exit price at cut + 1) legitimately changes."""
    world = random_world()
    cut = START + timedelta(days=6)
    instant = pd.Timestamp(cut, tz="UTC")
    base, _ = run_world(world)
    other, _ = run_world(scramble_after(world, instant))
    a = base.rows.set_index(["day", "contract"]).sort_index()
    b = other.rows.set_index(["day", "contract"]).sort_index()
    early = [i for i in a.index if i[0] < cut]
    pd.testing.assert_frame_equal(a.loc[early], b.loc[early])
    held_at_cut = a.loc[[i for i in a.index if i[0] == cut and a.at[i, "quantity_held"] != 0.0]]
    assert len(held_at_cut) and not np.allclose(held_at_cut.price_pnl, b.loc[held_at_cut.index].price_pnl)


def test_L10a_canary_a_planted_leak_is_caught():
    """The test above must be able to fail: hand the rule the fill-day price instead of D-1's (a
    one-day look-ahead) and the scrambled future must now move decisions at or before X."""
    import dataclasses
    from portfolio.per_coin_ledger import run_per_coin_ledger

    def leaky(world):
        bars, marks, counts, eligible, settlements = world
        inputs = build_ledger_inputs(CONTRACTS, START, END, eligible, decision_lag_days=1, bars=bars, marks=marks,
                                     mark_counts=counts, settlements=settlements)
        inputs = dataclasses.replace(inputs, decision_prices=inputs.fill_prices.loc[inputs.days])
        fn, log = trend_rule()
        run_per_coin_ledger(fn, inputs)
        return log

    world = random_world()
    x = START + timedelta(days=5)
    base, other = leaky(world), leaky(scramble_after(world, pd.Timestamp(x - timedelta(days=1), tz="UTC")))
    upto = [d for d in base if d <= x]
    assert [base[d] for d in upto] != [other[d] for d in upto]


def test_truncating_the_window_leaves_earlier_rows_unchanged():
    world = random_world()
    full, _ = run_world(world)
    short, _ = run_world(world, end=END - timedelta(days=3))
    a = full.rows[full.rows.day <= END - timedelta(days=4)].reset_index(drop=True)
    b = short.rows[short.rows.day <= END - timedelta(days=4)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
