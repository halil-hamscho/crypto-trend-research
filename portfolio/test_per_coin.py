"""Run from the repo root: python -m portfolio.test_per_coin  (the real-data check takes ~1 min)

Synthetic tests with known answers for the per-coin signal and universe, plus one real-data check of
membership and eligibility. Nothing here computes a strategy or control return.
"""
import copy
from datetime import date, timedelta

import numpy as np
import pandas as pd

from portfolio.per_coin_signal import load_per_coin_config, instrument_forecast
from portfolio.per_coin_universe import (eligibility, hourly_mark_counts, members_by_day, span_prices,
                                         traded_days, traded_spans)
from portfolio.trend_signal import basket_forecast, load_signal_config

CONFIG = load_per_coin_config()
rng = np.random.default_rng(11)
DAYS = [date(2020, 1, 1) + timedelta(days=i) for i in range(700)]
PRICES = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.04, len(DAYS)))), index=DAYS)
FC = instrument_forecast(PRICES, CONFIG)


def assert_raises(call, *fragments):
    try:
        call()
    except ValueError as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return
    raise AssertionError("expected ValueError")


def bars_for(days_volume: dict) -> pd.DataFrame:
    """Daily klines from {date: volume}; open = 100."""
    return pd.DataFrame({"timestamp": [pd.Timestamp(d, tz="UTC") for d in days_volume],
                         "open": 100.0, "volume": list(days_volume.values())})


# ------------------------------------------------------------------ configuration
def test_config_matches_the_registered_signal():
    sealed = load_signal_config()
    s = CONFIG.signal
    assert (s.speeds, s.scalars, s.cap, s.vol_span) == (sealed.speeds, sealed.scalars, sealed.cap, sealed.vol_span)
    assert (s.floor_quantile, s.floor_window, s.floor_min_days, s.annualisation, s.target_vol) == (
        sealed.floor_quantile, sealed.floor_window, sealed.floor_min_days, sealed.annualisation, sealed.target_vol)
    assert s.fdm == sealed.fdm == 1.15
    assert CONFIG.fdm_by_count == {1: 1.00, 2: 1.04, 3: 1.09, 4: 1.15}
    assert (CONFIG.min_traded_days, CONFIG.span_break_days, CONFIG.min_hourly_marks) == (64, 30, 23)
    assert len(CONFIG.config_hash) == 64


def test_config_rejects_a_structure_it_does_not_implement():
    raw = copy.deepcopy(CONFIG.raw)
    raw["signal"]["ewma_adjust"] = False
    assert_raises(lambda: load_per_coin_config(raw=raw), "ewma_adjust")
    raw = copy.deepcopy(CONFIG.raw)
    raw["primary_universe"]["source"] = "binance_usdtm_registry"
    assert_raises(lambda: load_per_coin_config(raw=raw), "primary_universe.source")
    raw = copy.deepcopy(CONFIG.raw)
    del raw["signal"]["fdm_by_available_count"][3]
    assert_raises(lambda: load_per_coin_config(raw=raw), "fdm_by_available_count")


# ------------------------------------------------------------------ signal
def test_speeds_switch_on_at_twice_their_slow_span():
    n = FC.n_speeds.to_numpy()
    assert (n[:63] == 0).all() and (n[63:127] == 1).all() and (n[127:255] == 2).all()
    assert (n[255:511] == 3).all() and (n[511:] == 4).all()
    assert FC.forecast.iloc[:63].isna().all() and FC.forecast.iloc[63:].notna().all()


def test_blend_uses_the_fdm_for_the_speeds_available():
    cols = [f"f_{f}/{s}" for f, s in CONFIG.signal.speeds]
    for row in (70, 200, 400, 650):
        k = int(FC.n_speeds.iloc[row])
        expected = np.clip(FC[cols[:k]].iloc[row].mean() * CONFIG.fdm_by_count[k], -20, 20)
        assert abs(FC.forecast.iloc[row] - expected) < 1e-12, row


def test_full_history_equals_the_sealed_basket_forecast():
    sealed = basket_forecast(PRICES, load_signal_config())
    later = FC.index[511:]
    assert np.allclose(FC.loc[later, "forecast"], sealed.loc[later, "forecast"], rtol=0, atol=1e-12)
    assert np.allclose(FC.vol_annual, sealed.vol_annual, rtol=0, atol=1e-15, equal_nan=True)


def test_no_look_ahead_truncation():
    for cut in (64, 130, 300, 600):
        short = instrument_forecast(PRICES.iloc[:cut], CONFIG)
        pd.testing.assert_frame_equal(short, FC.iloc[:cut], check_exact=False, rtol=0, atol=1e-12)


def test_no_look_ahead_future_scrambled():
    scrambled = PRICES.copy()
    scrambled.iloc[400:] = scrambled.iloc[400:].sample(frac=1, random_state=3).to_numpy()
    out = instrument_forecast(scrambled, CONFIG)
    pd.testing.assert_frame_equal(out.iloc[:400], FC.iloc[:400], check_exact=False, rtol=0, atol=1e-12)


def test_signal_rejects_bad_prices():
    bad = PRICES.copy(); bad.iloc[10] = 0.0
    assert_raises(lambda: instrument_forecast(bad, CONFIG), "finite and positive")


# ------------------------------------------------------------------ universe
def test_spans_break_after_more_than_the_limit():
    d0 = pd.Timestamp("2021-01-01")
    one_gap_30 = pd.DatetimeIndex([d0, d0 + pd.Timedelta(days=31)])          # 30 idle days: same span
    one_gap_31 = pd.DatetimeIndex([d0, d0 + pd.Timedelta(days=32)])          # 31 idle days: new span
    assert len(traded_spans(one_gap_30, 30)) == 1
    assert traded_spans(one_gap_31, 30) == [(d0, d0), (d0 + pd.Timedelta(days=32), d0 + pd.Timedelta(days=32))]
    assert traded_spans(pd.DatetimeIndex([]), 30) == []


def test_zero_volume_days_are_not_traded_days_and_prices_skip_them():
    d = [date(2021, 1, 1) + timedelta(days=i) for i in range(5)]
    bars = bars_for({d[0]: 1, d[1]: 0, d[2]: 5, d[3]: 5, d[4]: 0})
    assert [x.date() for x in traded_days(bars)] == [d[0], d[2], d[3]]
    (prices,) = span_prices(bars, 30)
    assert list(prices.index) == [d[0], d[2], d[3]]


def test_hourly_mark_counts_from_column_or_index():
    stamps = pd.date_range("2021-01-01", periods=30, freq="h", tz="UTC")
    as_column = pd.DataFrame({"timestamp": stamps.astype(str), "open": 1.0})
    as_index = pd.DataFrame({"open": 1.0}, index=stamps)
    for frame in (as_column, as_index):
        counts = hourly_mark_counts(frame)
        assert counts[date(2021, 1, 1)] == 24 and counts[date(2021, 1, 2)] == 6


def test_membership_runs_from_each_reset_to_the_next():
    resets = {date(2021, 1, 1): pd.DataFrame({"contract": ["A", "B"]}),
              date(2021, 2, 1): pd.DataFrame({"contract": ["B", "C"]})}
    days = [date(2021, 1, 31), date(2021, 2, 1), date(2021, 3, 5)]
    m = members_by_day(resets, days)
    assert m.loc[date(2021, 1, 31)].to_dict() == {"A": True, "B": True, "C": False}
    assert m.loc[date(2021, 2, 1)].to_dict() == {"A": False, "B": True, "C": True}
    assert m.loc[date(2021, 3, 5)].to_dict() == {"A": False, "B": True, "C": True}


def test_eligibility_applies_each_rule_with_its_reason():
    start = date(2021, 1, 1)
    days = [start + timedelta(days=i) for i in range(120)]
    volume = {d: 1.0 for d in days}
    volume[days[90]] = 0.0                                                  # a halt day
    bars = {"A": bars_for(volume), "B": bars_for(volume)}
    full = pd.Series(24, index=days)
    thin = full.copy(); thin[days[100]] = 22                                  # incomplete marks
    marks = {"A": thin, "B": full}
    members = {start: pd.DataFrame({"contract": ["A"]}), days[110]: pd.DataFrame({"contract": ["B"]})}
    ok, why, inst = eligibility(members, days, bars, marks, CONFIG)
    # each rule reads the previous, completed day (Clarification 1), so it bites one day later
    assert why.at[days[0], "A"] == "not_traded"                              # nothing before the listing
    assert why.at[days[1], "A"] == "span_first_day"
    assert why.at[days[63], "A"] == "short_history" and ok.at[days[64], "A"]  # 64 traded days completed
    assert ok.at[days[90], "A"] and why.at[days[91], "A"] == "not_traded"    # the halt on day 90
    assert why.at[days[101], "A"] == "marks_incomplete" and ok.at[days[102], "A"]
    assert not ok.at[days[105], "B"] and pd.isna(why.at[days[105], "B"])    # not a member yet
    assert ok.at[days[110], "B"] and not ok.at[days[110], "A"] and pd.isna(why.at[days[110], "A"])
    assert inst.at[days[5], "A"] == f"A#{start}" and pd.isna(inst.at[days[90], "A"])


def test_a_relisting_restarts_history():
    start = date(2021, 1, 1)
    first = [start + timedelta(days=i) for i in range(80)]
    second = [first[-1] + timedelta(days=40 + i) for i in range(80)]         # 39 idle days: new span
    days = first + [first[-1] + timedelta(days=i) for i in range(1, 40)] + second
    bars = {"A": bars_for({**{d: 1.0 for d in first}, **{d: 1.0 for d in second}})}
    ok, why, inst = eligibility({start: pd.DataFrame({"contract": ["A"]})}, days, bars,
                                {"A": pd.Series(24, index=days)}, CONFIG)
    assert ok.at[first[70], "A"]
    assert why.at[second[0], "A"] == "not_traded" and why.at[second[1], "A"] == "span_first_day"
    assert why.at[second[30], "A"] == "short_history"
    assert ok.at[second[64], "A"] and inst.at[second[64], "A"] == f"A#{second[0]}"


def test_eligibility_on_a_day_ignores_that_days_own_activity():
    """Eligibility on day X is known at X 00:00, so changing X's trading or marks must not move it."""
    start = date(2021, 1, 1)
    days = [start + timedelta(days=i) for i in range(100)]
    x = days[80]
    members = {start: pd.DataFrame({"contract": ["A"]})}
    base_bars = {"A": bars_for({d: 1.0 for d in days})}
    base_marks = {"A": pd.Series(24, index=days)}
    shocked_bars = {"A": bars_for({d: (0.0 if d >= x else 1.0) for d in days})}
    shocked_marks = {"A": pd.Series([22 if d >= x else 24 for d in days], index=days)}
    ok0, _, _ = eligibility(members, days, base_bars, base_marks, CONFIG)
    ok1, _, _ = eligibility(members, days, shocked_bars, shocked_marks, CONFIG)
    assert ok0.loc[:x].equals(ok1.loc[:x]) and ok0.at[x, "A"]
    assert not ok1.at[days[81], "A"]                                          # the next day sees it


# ------------------------------------------------------------------ real data (membership only)
def test_real_membership_and_eligibility():
    from portfolio.basket import SCORED_END, SCORED_START
    from portfolio.per_coin_universe import load_bars, load_mark_counts, membership
    members = membership(SCORED_END)
    contracts = sorted({c for m in members.values() for c in m.contract})
    days = [SCORED_START + timedelta(days=i) for i in range((SCORED_END - SCORED_START).days + 1)]
    ok, why, _ = eligibility(members, days, load_bars(contracts), load_mark_counts(contracts), CONFIG)
    member = members_by_day(members, days)
    assert (member.sum(axis=1) == 10).all()
    held = member.loc[:, member.any()].columns
    counts = why.stack().value_counts().to_dict()
    print(f"   real: {len(days)} days, {len(held)} contracts ever members; eligible members per day "
          f"min {ok.sum(axis=1).min()}, median {int(ok.sum(axis=1).median())}; ineligible member-days {counts}")
    assert ok.sum(axis=1).min() >= 5


def test_real_lagged_holding_periods_are_all_tradable():
    """Every contract-day the one-day-lag book could need: a period D held because the coin was a
    member at D-1's decision (trade at open[D], exit price open[D+1]); the exit fill of a coin that
    left at D-1's decision; and the terminal open[SCORED_END + 1]. All must be traded days with
    complete hourly marks, or a halt/delisting path could touch the primary run."""
    from portfolio.basket import SCORED_END, SCORED_START
    from portfolio.per_coin_universe import load_bars, load_mark_counts, membership
    members = membership(SCORED_END)
    contracts = sorted({c for m in members.values() for c in m.contract})
    first, last = SCORED_START - timedelta(days=2), SCORED_END + timedelta(days=1)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    member = members_by_day(members, days)
    bars, marks = load_bars(contracts), load_mark_counts(contracts)
    needed, exits = set(), 0
    for c in contracts:
        for i in range(2, len(days) - 1):
            d = days[i]
            if d < SCORED_START or d > SCORED_END:
                continue
            if member.at[days[i - 1], c]:
                needed |= {(c, d), (c, days[i + 1])}
            elif member.at[days[i - 2], c]:
                needed.add((c, d)); exits += 1
    traded = {c: {x.date() for x in traded_days(bars[c])} for c in contracts}
    bad = [(c, d) for c, d in sorted(needed)
           if d not in traded[c] or int(marks[c].get(d, 0)) < CONFIG.min_hourly_marks]
    print(f"   real: {len(needed)} contract-days needed by lagged holdings ({exits} exit fills, "
          f"terminal {last}); untradable or incomplete marks: {len(bad)}")
    assert not bad, bad[:10]


def test_deep_selector_with_the_sealed_map_reproduces_the_sealed_membership():
    from data.universe.coinmarketcap.new_fetcher import contract_map
    from portfolio.basket import SCORED_END, reconstitution_dates
    from portfolio.per_coin_universe import DeepSelector, membership
    sealed = membership(SCORED_END)
    same = DeepSelector(min_history_days=0, n=10, table=contract_map.MAP_TABLE, depth=contract_map.POOL_DEPTH)
    again = same.select_buffered(reconstitution_dates(SCORED_END))
    assert list(sealed) == list(again)
    assert all(list(sealed[d].contract) == list(again[d].contract) for d in sealed)


def test_deep_memberships_are_fieldable_from_the_scored_start():
    from portfolio.basket import SCORED_END
    from portfolio.per_coin_universe import deep_membership
    for n in (25, 50):
        members = deep_membership(n, SCORED_END, start=date(2021, 7, 1))
        assert min(members) == date(2021, 7, 1) and all(len(m) == n for m in members.values())
        print(f"   top-{n}: {len(members)} monthly resets, "
              f"{len({c for m in members.values() for c in m.contract})} contracts ever members")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
