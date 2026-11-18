"""Run from the repo root: python -m portfolio.test_per_coin_model

The per-coin portfolio model: decision tables and target_fn, on synthetic random-walk worlds, run
through the real ledger. The look-ahead tests REBUILD every precomputed input (bars, eligibility,
decision tables, ledger inputs) from truncated or scrambled data. No real-data return is computed.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from portfolio.per_coin_ledger import HOLD, build_ledger_inputs, run_per_coin_ledger
from portfolio.per_coin_model import build_decision_tables, make_target_fn
from portfolio.per_coin_signal import instrument_forecast, load_per_coin_config
from portfolio.per_coin_sizing import primary_positions
from portfolio.per_coin_universe import eligibility, span_prices

CONFIG = load_per_coin_config()
COINS = ("AAA", "BBB", "CCC", "BTCUSDT")
D0 = date(2020, 1, 1)
DAYS = [D0 + timedelta(days=i) for i in range(420)]
START, END = DAYS[300], DAYS[340]


def assert_raises(call, *fragments):
    try:
        call()
    except ValueError as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return
    raise AssertionError("expected ValueError")


def world(seed=5, halted=(), gap=None, upto=None):
    """Bars and mark counts for four coins, all members from D0. `halted`: (coin, day) with zero
    volume. `gap`: (coin, first day back) with no trades for 40 days before it, so a new span starts.
    `upto`: keep only data stamped by that day's 00:00 (bars through that day, whose open is the
    00:00 price, and mark counts through the day before)."""
    rng = np.random.default_rng(seed)
    bars, counts = {}, {}
    for k, c in enumerate(COINS):
        opens = 100.0 * (k + 1) * np.exp(np.cumsum(rng.normal(0.0, 0.04, len(DAYS))))
        volume = np.full(len(DAYS), 5.0)
        for cc, d in halted:
            if cc == c:
                volume[DAYS.index(d)] = 0.0
        if gap and gap[0] == c:
            back = DAYS.index(gap[1])
            volume[back - 40:back] = 0.0
        frame = pd.DataFrame({"timestamp": [pd.Timestamp(d, tz="UTC") for d in DAYS], "open": opens,
                              "volume": volume, "quote_volume": 1e9})
        series = pd.Series(24, index=DAYS)
        if upto is not None:
            frame = frame[frame.timestamp.dt.date <= upto]
            series = series[series.index < upto]
        bars[c], counts[c] = frame.reset_index(drop=True), series
    return bars, counts


def calendar_eligibility(bars, counts, last=None):
    days = [d for d in DAYS if last is None or d <= last]
    ok, _, _ = eligibility({D0: pd.DataFrame({"contract": list(COINS)})}, days, bars, counts, CONFIG)
    return ok


def settlements(bars, seed=3):
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(pd.Timestamp(DAYS[0], tz="UTC"), pd.Timestamp(DAYS[-1], tz="UTC"), freq="8h")
    out = {}
    for c, b in bars.items():
        opens = pd.Series(b.open.to_numpy(), index=b.timestamp.dt.date.to_numpy())
        out[c] = pd.DataFrame({"stamp": stamps, "rate": rng.normal(5e-5, 1e-4, len(stamps)),
                               "mark": [float(opens.get(s.date(), opens.iloc[-1])) for s in stamps],
                               "interval_hours": 8.0})
    return out


def setup(variant="own", lag=1, seed=5, **kw):
    bars, counts = world(seed=seed, **kw)
    elig = calendar_eligibility(bars, counts)
    tables = build_decision_tables(bars, elig, CONFIG, variant=variant)
    inputs = build_ledger_inputs(COINS, START, END, elig, decision_lag_days=lag, bars=bars, mark_counts=counts,
                                 settlements=settlements(bars))
    return bars, counts, elig, tables, inputs


def logged(fn):
    log = {}

    def g(day, held, prices, base):
        out = fn(day, held, prices, base)
        log[day] = {c: ("HOLD" if v is HOLD else v) for c, v in out.items()}
        return out
    return g, log


# ------------------------------------------------------------------ decision tables
def test_tables_hold_each_spans_forecast_vol_and_price():
    bars, counts, elig, tables, _ = setup()
    x = DAYS[320]
    for c in COINS:
        (prices,) = span_prices(bars[c], CONFIG.span_break_days)
        fc = instrument_forecast(prices, CONFIG)
        assert tables.forecast.at[x, c] == fc.forecast[x] and tables.sigma.at[x, c] == fc.vol_annual[x]
        assert tables.price.at[x, c] == prices[x] == tables.mark_price.at[x, c]


def test_a_day_without_a_trade_uses_the_latest_row_of_the_span():
    x = DAYS[320]
    bars, counts, elig, tables, _ = setup(halted=(("BBB", x),))
    assert np.isnan(tables.price.at[x, "BBB"])
    assert tables.forecast.at[x, "BBB"] == tables.forecast.at[x - timedelta(days=1), "BBB"]
    assert tables.mark_price.at[x, "BBB"] == tables.price.at[x - timedelta(days=1), "BBB"]


def test_a_relisting_restarts_the_forecast_and_does_not_carry_prices_across():
    back = DAYS[250]
    bars, counts, elig, tables, _ = setup(gap=("CCC", back))
    assert tables.forecast.loc[back:back + timedelta(days=62), "CCC"].isna().all()   # < 64 traded days
    assert tables.forecast.loc[back + timedelta(days=63):, "CCC"].notna().all()
    pause = tables.mark_price.loc[back - timedelta(days=40):back - timedelta(days=1), "CCC"]
    last_old = tables.price.at[back - timedelta(days=41), "CCC"]
    assert (pause == last_old).all()                                       # held through the pause: old price
    assert not elig.loc[back - timedelta(days=39):back, "CCC"].any()       # never eligible during the pause
    assert tables.mark_price.at[back, "CCC"] == tables.price.at[back, "CCC"] != last_old   # new span: its own


def test_rho_bar_is_the_mean_pairwise_correlation_over_the_window():
    bars, counts, elig, tables, _ = setup()
    x = DAYS[330]
    names = [c for c in COINS if elig.at[x, c]]
    opens = pd.DataFrame({c: pd.Series(bars[c].open.to_numpy(), index=bars[c].timestamp.dt.date.to_numpy())
                          for c in names})
    block = opens.pct_change().loc[x - timedelta(days=179):x]
    corr = block.corr(min_periods=90).to_numpy()
    assert abs(tables.rho_bar[x] - corr[np.triu_indices(len(names), 1)].mean()) < 1e-12
    assert np.isnan(tables.rho_bar[DAYS[70]])                               # fewer than 90 shared days
    single = elig.copy()
    single.loc[:, ["AAA", "BBB", "CCC"]] = False
    assert np.isnan(build_decision_tables(bars, single, CONFIG).rho_bar[x])  # one coin: undefined


def test_every_table_row_equals_its_rebuild_from_data_known_at_that_instant():
    """Rebuild bars, eligibility and tables from only the data stamped by X 00:00: row X is identical."""
    bars, counts = world(halted=(("BBB", DAYS[331]),), gap=("CCC", DAYS[250]))
    full = build_decision_tables(bars, calendar_eligibility(bars, counts), CONFIG)
    for x in (DAYS[200], DAYS[251], DAYS[315], DAYS[331], DAYS[339]):
        tb, tc = world(halted=(("BBB", DAYS[331]),), gap=("CCC", DAYS[250]), upto=x)
        part = build_decision_tables(tb, calendar_eligibility(tb, tc, last=x), CONFIG)
        for name in ("forecast", "sigma", "price", "mark_price", "eligible"):
            pd.testing.assert_series_equal(getattr(full, name).loc[x], getattr(part, name).loc[x], check_names=False)
        assert (full.rho_bar[x] == part.rho_bar[x]) or (np.isnan(full.rho_bar[x]) and np.isnan(part.rho_bar[x]))


# ------------------------------------------------------------------ target_fn
def test_target_fn_matches_the_sizing_arithmetic_by_hand():
    bars, counts, elig, tables, inputs = setup()
    d = DAYS[320]
    x = d - timedelta(days=1)
    held = {"AAA": 3.0, "CCC": -1.5}
    names = [c for c in COINS if elig.at[x, c]]
    h = np.array([held.get(c, 0.0) * tables.price.at[x, c] / 100_000 for c in names])
    p, keep = primary_positions(tables.forecast.loc[x, names].to_numpy(), tables.sigma.loc[x, names].to_numpy(),
                                tables.rho_bar[x], h)
    prices = {c: float(v) for c, v in inputs.decision_prices.loc[d].items()}
    out = make_target_fn(tables, lag=1)(d, held, prices, 100_000)
    for c, value, kept, hc in zip(names, p, keep, h):
        expected = HOLD if (kept and hc) else 0.0 if kept else value * 100_000
        assert out[c] is HOLD if expected is HOLD else abs(out[c] - expected) < 1e-9, (c, out[c], expected)


def test_a_held_coin_no_longer_eligible_gets_zero():
    bars, counts, elig, tables, inputs = setup()
    d = DAYS[320]
    elig2 = elig.copy()
    elig2.loc[d - timedelta(days=1), "BBB"] = False
    t2 = build_decision_tables(bars, elig2, CONFIG)
    prices = {c: float(v) for c, v in inputs.decision_prices.loc[d].items()}
    assert make_target_fn(t2, lag=1)(d, {"BBB": 2.0}, prices, 100_000)["BBB"] == 0.0


def test_missing_prices_are_handled_explicitly():
    x = DAYS[319]
    bars, counts, elig, tables, inputs = setup(halted=(("AAA", x),))
    d = x + timedelta(days=1)
    prices = {c: float(v) for c, v in inputs.decision_prices.loc[d].items()}
    assert np.isnan(prices["AAA"])
    notes = []
    make_target_fn(tables, lag=1, notes=notes)(d, {"AAA": 1.0}, prices, 100_000)
    assert len(notes) == 1 and "AAA" in notes[0] and "last traded" in notes[0]       # marked, and noted
    wrong = dict(prices, BBB=prices["BBB"] * 1.01)
    assert_raises(lambda: make_target_fn(tables, lag=1)(d, {}, wrong, 100_000), "disagree", "BBB")
    poisoned = build_decision_tables(bars, elig, CONFIG)
    poisoned.forecast.at[x, "CCC"] = np.nan
    assert_raises(lambda: make_target_fn(poisoned, lag=1)(d, {}, prices, 100_000), "CCC", "forecast")
    nothing = build_decision_tables(bars, elig, CONFIG)
    nothing.mark_price.at[x, "AAA"] = np.nan
    assert_raises(lambda: make_target_fn(nothing, lag=1)(d, {"AAA": 1.0}, prices, 100_000), "AAA", "no traded price")


def test_end_to_end_through_the_ledger():
    for sizing in ("primary", "dynamic"):
        bars, counts, elig, tables, inputs = setup()
        fn, log = logged(make_target_fn(tables, lag=1, sizing=sizing))
        ledger = run_per_coin_ledger(fn, inputs)                            # reconciles R1-R8 on the way
        rows = ledger.rows
        assert (rows.quantity_held > 0).any() and (rows.quantity_held < 0).any(), sizing
        assert (rows.event == "hold").any() and (rows.event == "trade").any(), sizing
        assert len(log) == len(inputs.days) - 1


# ------------------------------------------------------------------ look-ahead, with every input rebuilt
def scrambled_world(instant, seed=99, **kw):
    """Every value stamped after `instant` changed: opens of later days, volumes of the instant's day
    onward (scaled, so tradability is unchanged), and funding after it. Then EVERYTHING downstream is
    rebuilt from the scrambled data: eligibility, decision tables and ledger inputs."""
    bars, counts = world(**kw)
    rng = np.random.default_rng(seed)
    out = {}
    for c, b in bars.items():
        later = b.timestamp > instant
        same_or_later = b.timestamp >= instant.normalize()
        out[c] = b.assign(open=np.where(later, b.open * rng.uniform(0.5, 1.5, len(b)), b.open),
                          volume=np.where(same_or_later, b.volume * rng.uniform(0.2, 5, len(b)), b.volume))
    base_settlements = settlements(bars)
    for c, s in base_settlements.items():
        after = s.stamp > instant
        s.loc[after, "rate"] = rng.normal(0, 3e-4, int(after.sum()))
    return out, counts, base_settlements


def run_model(bars, counts, sett, lag, leak=False, sizing="primary", diagnostics=None):
    elig = calendar_eligibility(bars, counts)
    tables = build_decision_tables(bars, elig, CONFIG)
    if leak:                                                               # tomorrow's forecast, today
        tables.forecast.loc[:, :] = tables.forecast.shift(-1).to_numpy()
    inputs = build_ledger_inputs(COINS, START, END, elig, decision_lag_days=lag, bars=bars, mark_counts=counts,
                                 settlements=sett)
    fn, log = logged(make_target_fn(tables, lag=lag, sizing=sizing, diagnostics=diagnostics))
    run_per_coin_ledger(fn, inputs)
    return log


def test_M9_decisions_with_the_real_model_are_frozen_at_their_instant():
    for lag in (1, 0):
        x = DAYS[325]
        instant = pd.Timestamp(x - timedelta(days=lag), tz="UTC")
        bars, counts = world()
        base = run_model(bars, counts, settlements(bars), lag)
        other = run_model(*scrambled_world(instant), lag)
        upto = [d for d in base if d <= x]
        assert [base[d] for d in upto] == [other[d] for d in upto], lag
        assert any(base[d] != other[d] for d in base if d > x), "the scramble did not reach later decisions"


def test_M9_canary_a_leaky_model_is_caught():
    x = DAYS[325]
    instant = pd.Timestamp(x - timedelta(days=1), tz="UTC")
    bars, counts = world()
    base = run_model(bars, counts, settlements(bars), 1, leak=True)
    other = run_model(*scrambled_world(instant), 1, leak=True)
    upto = [d for d in base if d <= x]
    assert [base[d] for d in upto] != [other[d] for d in upto]


def test_M9_dynamic_sizing_and_its_record_are_frozen_at_their_instant():
    """Stage C: the risk-targeted decisions AND the recorded M, V(u) and rho_bar for every day up to X are
    unchanged when everything stamped after the decision instant is scrambled and every input rebuilt."""
    x = DAYS[325]
    instant = pd.Timestamp(x - timedelta(days=1), tz="UTC")
    bars, counts = world()
    rec_base, rec_other = [], []
    base = run_model(bars, counts, settlements(bars), 1, sizing="dynamic", diagnostics=rec_base)
    other = run_model(*scrambled_world(instant), 1, sizing="dynamic", diagnostics=rec_other)
    upto = [d for d in base if d <= x]
    assert [base[d] for d in upto] == [other[d] for d in upto]
    keys = ("decision_day", "n_eligible", "rho_bar", "v_unscaled", "multiplier", "v_ideal", "v_final")
    a = [{k: r[k] for k in keys} for r in rec_base if r["day"] <= x]
    b = [{k: r[k] for k in keys} for r in rec_other if r["day"] <= x]
    assert a == b and len(a) == len(upto)
    assert any(r["multiplier"] != s["multiplier"] for r, s in zip(rec_base, rec_other) if r["day"] > x)


def test_dynamic_record_is_the_sizing_of_decision_day_x_by_hand():
    """Timing and arithmetic: the record for holding day D is computed from the tables at X = D - lag."""
    from portfolio.per_coin_sizing import SIZING, ex_ante_vol
    for lag in (1, 0):
        bars, counts, elig, tables, inputs = setup(lag=lag)
        record = []
        fn = make_target_fn(tables, lag=lag, sizing="dynamic", diagnostics=record)
        d = DAYS[320]
        prices = {c: float(v) for c, v in inputs.decision_prices.loc[d].items()}
        fn(d, {}, prices, 100_000)
        x = d - timedelta(days=lag)
        names = [c for c in COINS if elig.at[x, c]]
        f, sigma = tables.forecast.loc[x, names].to_numpy(), tables.sigma.loc[x, names].to_numpy()
        u = f / 10.0 / len(names) * SIZING.target_vol / sigma
        v = ex_ante_vol(u, sigma, tables.rho_bar[x])
        r = record[0]
        assert r["day"] == d and r["decision_day"] == x and r["n_eligible"] == len(names)
        assert abs(r["v_unscaled"] - v) < 1e-15 and abs(r["multiplier"] - SIZING.target_vol / v) < 1e-12
        if lag == 1:                                   # X's row differs from D's: the record is not D's
            y = tables.forecast.loc[d, names].to_numpy()
            assert not np.array_equal(f, y)


def test_the_k1_twin_under_dynamic_sizing_is_long_only_at_target_through_the_ledger():
    bars, counts, elig, tables, inputs = setup(variant="k1")
    record = []
    ledger = run_per_coin_ledger(make_target_fn(tables, lag=1, sizing="dynamic", diagnostics=record), inputs)
    assert (ledger.rows.quantity_held >= 0).all() and (ledger.rows.quantity_held > 0).any()
    assert all(abs(r["v_ideal"] - 0.30) < 1e-12 for r in record) and len(record) == len(inputs.days) - 1


# ------------------------------------------------------------------ controls
def test_controls_produce_their_defined_forecasts():
    bars, counts = world()
    elig = calendar_eligibility(bars, counts)
    own = build_decision_tables(bars, elig, CONFIG)
    k1 = build_decision_tables(bars, elig, CONFIG, variant="k1")
    k3 = build_decision_tables(bars, elig, CONFIG, variant="k3")
    basket = {d: (-1) ** i * 7.5 for i, d in enumerate(DAYS)}
    k5 = build_decision_tables(bars, elig, CONFIG, variant="k5", basket_forecast=basket)
    x = DAYS[320]
    available = own.forecast.loc[x].notna()
    assert (k1.forecast.loc[x][available] == 10.0).all()
    for c in COINS:
        (prices,) = span_prices(bars[c], CONFIG.span_break_days)
        assert k3.forecast.at[x, c] == 10.0 * np.sign(prices.pct_change().loc[:x].mean())
    assert (k5.forecast.loc[x][available] == basket[x]).all()
    assert (k1.sigma.loc[x] == own.sigma.loc[x]).all()                       # same sizing inputs
    assert_raises(lambda: build_decision_tables(bars, elig, CONFIG, variant="k5"), "basket")


def test_K2_btc_alone_is_sized_with_one_coin_and_idm_one():
    bars, counts = world()
    elig = calendar_eligibility(bars, counts)
    only = elig.copy()
    only.loc[:, ["AAA", "BBB", "CCC"]] = False
    tables = build_decision_tables(bars, only, CONFIG, variant="k1")
    d = DAYS[320]
    x = d - timedelta(days=1)
    prices = {c: float(tables.price.at[x, c]) for c in COINS}
    out = make_target_fn(tables, lag=1)(d, {}, prices, 100_000)
    sigma = tables.sigma.at[x, "BTCUSDT"]
    assert set(out) == {"BTCUSDT"} and abs(out["BTCUSDT"] - 0.9 * 0.30 / sigma * 100_000) < 1e-6


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
