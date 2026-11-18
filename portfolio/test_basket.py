"""Run from the repo root: python -m portfolio.test_basket"""
from datetime import date
from unittest.mock import patch

import numpy as np

import pandas as pd

import portfolio.basket as basket          # names added after the first build are read off the module
from data.universe.coinmarketcap.new_fetcher import panel
from portfolio.basket import (cost_rates, held_exposure, FEE_PER_SIDE, SLIPPAGE_PER_SIDE, Book, funding_over, leg_row,
                              settle, target_notionals, load_settlements,
                              reference_basket, run_ledger, SCORED_START, SCORED_END,
                              EXCHANGE_SKIPPED, EXCHANGE_SKIP_SENSITIVITY_RATES, _add_exchange_skips, funding_gaps,
                              skipped_funding_sensitivity)

WEIGHTS = {"A": 0.5, "B": 0.5}
CAPITAL = 100.0


def drifted_book():
    """Held 1 unit each of A and B at 50; A then rises 20%."""
    return Book(quantities={"A": 1.0, "B": 1.0}), {"A": 60.0, "B": 50.0}


def test_units_mode_holds_through_price_drift():
    book, marks = drifted_book()
    out = target_notionals(book, WEIGHTS, marks, False, 1.0, CAPITAL, "units")
    assert out == {"A": 60.0, "B": 50.0}                       # no trade


def test_notional_mode_splits_the_target_by_the_reference_weights():
    book, marks = drifted_book()
    drifted = {"A": 60.0 / 110.0, "B": 50.0 / 110.0}           # what the reference basket supplies on a hold day
    out = target_notionals(book, drifted, marks, False, 1.0, CAPITAL, "notional")
    assert np.isclose(sum(out.values()), 100.0)
    assert np.isclose(out["A"] / out["B"], 60.0 / 50.0)        # drifted weights kept
    fresh = target_notionals(book, WEIGHTS, marks, False, 1.0, CAPITAL, "notional")
    assert np.allclose([fresh["A"], fresh["B"]], [50.0, 50.0])  # the reference weights are the authority


def test_notional_mode_reopens_from_flat_on_reference_weights():
    out = target_notionals(Book(), WEIGHTS, {"A": 60.0, "B": 50.0}, False, 1.0, CAPITAL, "notional")
    assert out == {"A": 50.0, "B": 50.0}


def test_reopening_skips_unpriced_contracts():
    out = target_notionals(Book(), WEIGHTS, {"A": 60.0, "B": np.nan}, False, 1.0, CAPITAL, "notional")
    assert set(out) == {"A"}


def test_idle_weight_shrinks_target_between_reconstitutions():
    book, marks = drifted_book()
    book.idle_weight = 0.2
    out = target_notionals(book, WEIGHTS, marks, False, 1.0, CAPITAL, "notional")
    assert np.isclose(sum(out.values()), 80.0)


def test_reconstitution_resets_idle_weight_and_uses_fresh_weights():
    book, marks = drifted_book()
    book.idle_weight = 0.2
    out = target_notionals(book, WEIGHTS, marks, True, 1.0, CAPITAL, "units")
    assert book.idle_weight == 0.0
    assert out == {"A": 50.0, "B": 50.0}


def test_zero_exposure_targets_zero():
    book, marks = drifted_book()
    out = target_notionals(book, WEIGHTS, marks, False, 0.0, CAPITAL, "notional")
    assert all(np.isclose(v, 0.0) for v in out.values())


def test_reconstitution_raises_on_a_member_without_a_mark():
    book, _ = drifted_book()
    for marks in ({"A": 60.0, "B": np.nan}, {"A": 60.0}):     # NaN and absent
        try:
            target_notionals(book, WEIGHTS, marks, True, 1.0, CAPITAL, "units")
        except ValueError as error:
            assert "B" in str(error)
        else:
            raise AssertionError("expected ValueError")
    assert book.idle_weight == 0.0


def utc(text):
    return pd.Timestamp(text, tz="UTC")


SETTLEMENTS = {"A": pd.DataFrame({
    "stamp": [utc("2024-01-01 00:00"), utc("2024-01-01 08:00"), utc("2024-01-01 16:00"),
              utc("2024-01-02 00:00")],
    "rate": [0.0001, 0.0001, 0.0001, 0.0001],
    "mark": [100_000.0, 101_000.0, 99_000.0, 100_000.0],
    "interval_hours": [8.0, 8.0, 8.0, 8.0],
})}


def test_long_pays_and_matches_the_worked_example():
    paid = funding_over(SETTLEMENTS, "A", utc("2024-01-01 00:00"), utc("2024-01-02 00:00"), 1.0)
    assert np.isclose(paid, -(10.10 + 9.90 + 10.00))           # 08:00, 16:00, next 00:00


def test_short_receives():
    got = funding_over(SETTLEMENTS, "A", utc("2024-01-01 00:00"), utc("2024-01-02 00:00"), -1.0)
    assert np.isclose(got, 10.10 + 9.90 + 10.00)


def test_window_is_open_at_start_and_closed_at_end():
    row = funding_over(SETTLEMENTS, "A", utc("2024-01-01 00:00"), utc("2024-01-01 08:00"), 1.0)
    assert np.isclose(row, -10.10)                              # 00:00 excluded, 08:00 included


def test_no_settlement_in_window_is_zero():
    assert funding_over(SETTLEMENTS, "A", utc("2024-01-01 08:00"), utc("2024-01-01 08:00"), 1.0) == 0.0


DAY = (utc("2024-01-01 00:00"), utc("2024-01-02 00:00"))


def test_leg_row_identity_and_components():
    row = leg_row("A", SETTLEMENTS, 0.0, 100_000.0, 100_000.0, 102_000.0, *DAY)
    assert np.isclose(row["trade"], 100_000.0)
    assert np.isclose(row["fee"], 100_000.0 * FEE_PER_SIDE)
    assert np.isclose(row["slippage"], 100_000.0 * SLIPPAGE_PER_SIDE)
    assert np.isclose(row["price_pnl"], 2_000.0)                # 1 BTC x +2,000
    assert np.isclose(row["funding_pnl"], -30.0)                # 10.10 + 9.90 + 10.00
    residual = row["net_pnl"] - (row["price_pnl"] + row["funding_pnl"] - row["fee"] - row["slippage"])
    assert abs(residual) < 1e-9


def test_leg_row_short_gains_when_price_falls_and_receives_funding():
    row = leg_row("A", SETTLEMENTS, 0.0, -100_000.0, 100_000.0, 98_000.0, *DAY)
    assert np.isclose(row["price_pnl"], 2_000.0)
    assert np.isclose(row["funding_pnl"], 30.0)


def test_leg_row_no_trade_costs_nothing():
    row = leg_row("A", SETTLEMENTS, 1.0, 100_000.0, 100_000.0, 100_000.0, *DAY)
    assert row["trade"] == 0.0 and row["fee"] == 0.0 and row["slippage"] == 0.0


def test_leg_row_costs_are_charged_on_a_flip_both_ways():
    row = leg_row("A", SETTLEMENTS, 1.0, -100_000.0, 100_000.0, 100_000.0, *DAY)
    assert np.isclose(abs(row["trade"]), 200_000.0)             # close the long AND open the short


START, DELIST = utc("2024-01-01 00:00"), utc("2024-01-01 12:00")     # intraday delisting


def settled(quantity, price=90_000.0):
    book = Book(quantities={"A": quantity, "B": 1.0})
    return book, settle(book, "A", price, 100_000.0, SETTLEMENTS, START, DELIST, weight=0.5)


def test_short_death_realises_a_gain():
    _, row = settled(-1.0)                                       # price fell 100k -> 90k
    assert np.isclose(row["price_pnl"], 10_000.0)
    assert row["net_pnl"] > 0


def test_long_death_realises_a_loss():
    _, row = settled(1.0)
    assert np.isclose(row["price_pnl"], -10_000.0)
    assert row["net_pnl"] < 0


def test_settlement_is_fee_only_and_the_identity_holds():
    _, row = settled(1.0)
    assert row["slippage"] == 0.0
    assert np.isclose(row["fee"], 90_000.0 * FEE_PER_SIDE)
    assert np.isclose(row["net_pnl"], row["price_pnl"] + row["funding_pnl"] - row["fee"])


def test_settlement_funding_stops_at_the_delisting_stamp():
    _, row = settled(1.0)
    assert np.isclose(row["funding_pnl"], -10.10)                # 08:00 in; 00:00 and 16:00 out


def test_settlement_removes_the_leg_and_idles_its_weight():
    book, row = settled(1.0)
    assert "A" not in book.quantities and "B" in book.quantities
    assert book.idle_weight == 0.5 and row["quantity_after"] == 0.0


def test_load_settlements_handles_luna_death_and_edges():
    out = load_settlements(["LUNAUSDT", "BTCUSDT", "APTUSDT", "ARBUSDT"])
    luna = out["LUNAUSDT"]
    assert luna.stamp.max() == utc("2022-05-12 16:00")         # 05-13 00:00 row is post-death tail
    assert not luna.mark.isna().any()
    assert len(out["BTCUSDT"]) == 7305                          # nothing lost on a live contract
    assert out["APTUSDT"].stamp.min() == utc("2022-10-19 08:00")
    assert out["ARBUSDT"].stamp.min() == utc("2023-03-23 16:00")


def basket_fixture(recon_days, growth=(1.0, 1.1, 1.1)):
    """A and B both rise by `growth[i]` each day; membership reset on `recon_days`."""
    days = pd.Series([pd.Timestamp(d).date() for d in ("2024-01-01", "2024-01-02", "2024-01-03")])
    level = np.cumprod(growth) * 50.0
    marks = pd.DataFrame({"A": level, "B": level}, index=days)
    members = pd.DataFrame({"contract": ["A", "B"], "weight": [0.5, 0.5]})
    return {days[i]: members for i in recon_days}, marks


def test_index_uses_the_previous_basket_value():
    """Two constituents, +10%/day, fixed quantities: 121.00, not 122.10."""
    index, _ = reference_basket(*basket_fixture([0]))
    assert np.isclose(index.iloc[-1], 121.0)


def test_reconstitution_day_keeps_the_old_baskets_move():
    """Reset on day 1, when both rise 10%: the index must rise 10%, not stay flat."""
    index, _ = reference_basket(*basket_fixture([0, 1]))
    assert np.isclose(index.iloc[1], 110.0)
    assert np.isclose(index.iloc[2], 121.0)


def test_weights_are_cap_weights_right_after_a_reset_and_sum_to_one():
    members = pd.DataFrame({"contract": ["A", "B"], "weight": [0.8, 0.2]})
    days = [pd.Timestamp(d).date() for d in ("2024-01-01", "2024-01-02")]
    marks = pd.DataFrame({"A": [10.0, 12.0], "B": [10.0, 10.0]}, index=days)
    _, weights = reference_basket({days[0]: members, days[1]: members}, marks)
    assert np.allclose(weights.loc[days[1]], [0.8, 0.2])        # reset to cap weights
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_reference_basket_raises_on_unpriced_member_at_reconstitution():
    members, marks = basket_fixture([0])
    marks.loc[marks.index[0], "B"] = np.nan
    try:
        reference_basket(members, marks)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def funding_frame(first, last, hours=8, rate=0.0001, mark=100.0):
    """A complete settlement table: one row every `hours` from `first` to `last`."""
    stamps = pd.date_range(first, last, freq=f"{hours}h", tz="UTC")
    return pd.DataFrame({"stamp": stamps, "rate": rate, "mark": mark, "interval_hours": float(hours)})


def assert_raises(call, *fragments):
    try:
        call()
    except ValueError as error:
        for fragment in fragments:
            assert fragment in str(error), f"{fragment!r} not in: {error}"
        return
    raise AssertionError("expected ValueError")


LEDGER_DAYS = [pd.Timestamp(d).date() for d in pd.date_range("2024-01-01", "2024-01-06")]


def ledger_fixture(recon_members=("A", "C")):
    """A rises 10%/day; B and C are flat. Opens 01-01 on A/B; reconstitutes 01-03."""
    marks = pd.DataFrame({"A": 100.0 * 1.1 ** np.arange(6), "B": 50.0, "C": 10.0}, index=LEDGER_DAYS)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5, "C": 0.0}, index=LEDGER_DAYS)
    members = pd.DataFrame({"contract": list(recon_members), "weight": [0.5, 0.5]})
    return dict(members_by_date={LEDGER_DAYS[2]: members}, marks=marks,
                settlements={c: funding_frame("2023-12-31 00:00", "2024-01-07 00:00") for c in "ABC"},
                reference_weights=weights, start=LEDGER_DAYS[0], end=LEDGER_DAYS[-2])


def test_ledger_identity_holds_on_every_row():
    led = run_ledger(**ledger_fixture())
    residual = led.net_pnl - (led.price_pnl + led.funding_pnl - led.fee - led.slippage)
    assert residual.abs().max() < 1e-9


def test_benchmark_trades_only_on_opening_and_reconstitution():
    led = run_ledger(**ledger_fixture())
    traded = led[led.trade.abs() > 1e-6]
    assert set(traded.event) == {"opening", "reconstitution"}


def test_opening_costs_land_on_the_first_day():
    led = run_ledger(**ledger_fixture())
    opening = led[led.event == "opening"]
    assert np.isclose(opening.trade.abs().sum(), 100_000.0)
    assert np.isclose(opening.fee.sum(), 100_000.0 * FEE_PER_SIDE)


def test_capital_is_a_fixed_base_not_compounded():
    led = run_ledger(**ledger_fixture(), capital=100_000.0)
    opening = led[led.event == "opening"]
    assert np.isclose(opening.notional_after.sum(), 100_000.0)
    recon = led[led.event == "reconstitution"]
    assert np.isclose(recon.notional_after.sum(), 100_000.0)    # reset to the base, not to P&L-grown equity
    assert (led.capital_base == 100_000.0).all()                # one base, every day, every row


def test_constituent_swap_costs_money_and_closes_the_old_leg():
    led = run_ledger(**ledger_fixture(("A", "C")))
    day = led[led.event == "reconstitution"]
    assert (day.fee + day.slippage).sum() > 0
    assert day.set_index("contract").loc["B", "quantity_after"] == 0.0
    assert "B" not in set(led[led.holding_period_start > utc("2024-01-03")].contract)


def test_held_contract_without_a_mark_raises():
    fixture = ledger_fixture()
    fixture["marks"].loc[LEDGER_DAYS[3], "A"] = np.nan
    try:
        run_ledger(**fixture)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_scored_window_is_1883_days_ending_at_the_08_30_start():
    assert SCORED_START == date(2021, 7, 5) and SCORED_END == date(2026, 8, 30)
    assert (SCORED_END - SCORED_START).days + 1 == 1883


def test_last_row_exits_at_the_open_of_the_day_after_end_and_asks_for_no_later_settlement():
    fixture = ledger_fixture()
    table = funding_frame("2023-12-31 00:00", "2024-01-06 08:00", mark=150.0)
    table.loc[table.stamp == utc("2024-01-06 08:00"), "rate"] = 1.0     # after the window: must not be paid
    fixture["settlements"]["A"] = table
    led = run_ledger(**fixture)
    last = led[(led.holding_period_start == utc("2024-01-05")) & (led.contract == "A")].iloc[0]
    marks = fixture["marks"]
    assert np.isclose(last.price_pnl, last.quantity_after * (marks.loc[LEDGER_DAYS[5], "A"]
                                                              - marks.loc[LEDGER_DAYS[4], "A"]))
    assert np.isclose(last.funding_pnl, -last.quantity_after * 150.0 * 0.0001 * 3)   # 01-06 08:00 excluded
    assert led.holding_period_start.max() == utc("2024-01-05")


def test_funding_that_ends_before_a_period_end_raises():
    fixture = ledger_fixture()
    fixture["settlements"]["A"] = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    assert_raises(lambda: run_ledger(**fixture), "funding ends")


# ---------------------------------------------------------------- funding completeness
# A held position owes every settlement in its window. Missing, NaN or unordered
# funding must stop the run: a short sum looks exactly like a cheaper strategy.

HOLE = utc("2024-01-01 16:00")


def with_hole(table, stamp=HOLE):
    return table[table.stamp != stamp].reset_index(drop=True)


def owed(table, start="2024-01-01 00:00", end="2024-01-02 00:00", quantity=1.0):
    return funding_over({"A": table}, "A", utc(start), utc(end), quantity)


def test_funding_over_raises_on_a_settlement_missing_inside_the_window():
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    assert np.isclose(owed(table), -0.03)                          # complete: three settlements
    assert_raises(lambda: owed(with_hole(table)), "A", "2024-01-01 08:00", "2024-01-02 00:00")


def test_funding_over_raises_when_every_settlement_is_missing():
    empty = funding_frame("2023-12-31 00:00", "2024-01-03 00:00").iloc[:0]
    assert_raises(lambda: owed(empty), "A")


def test_funding_over_raises_on_a_nan_rate_in_the_window():
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    table.loc[table.stamp == HOLE, "rate"] = np.nan
    assert_raises(lambda: owed(table), "A", "rate")


def test_funding_over_raises_on_a_nan_or_nonpositive_mark_in_the_window():
    for bad in (np.nan, 0.0, -5.0):
        table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
        table.loc[table.stamp == HOLE, "mark"] = bad
        assert_raises(lambda: owed(table), "A", "mark")


def test_funding_over_raises_when_four_hour_rows_are_eight_hours_apart():
    """The real 2026-06-24 pattern: two 4h rows with the 04:00 settlement absent between them."""
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00", hours=4)
    assert_raises(lambda: owed(with_hole(table, utc("2024-01-01 12:00"))), "A", "spacing")


def test_funding_over_raises_when_the_hole_is_the_first_settlement_after_the_start():
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    assert_raises(lambda: owed(with_hole(table, utc("2024-01-01 08:00"))), "A")


def test_funding_over_raises_when_the_table_ends_before_the_window_does():
    table = funding_frame("2023-12-31 00:00", "2024-01-01 16:00")
    assert_raises(lambda: owed(table), "A", "funding ends")


def test_funding_over_raises_when_the_last_owed_settlement_is_absent_but_later_rows_exist():
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    assert_raises(lambda: owed(with_hole(table, utc("2024-01-02 00:00"))), "A")


def test_funding_over_accepts_a_table_that_ends_exactly_at_the_window_end():
    table = funding_frame("2023-12-31 00:00", "2024-01-02 00:00")
    assert np.isclose(owed(table), -0.03)


def test_funding_over_raises_when_no_settlement_precedes_a_late_first_row():
    table = funding_frame("2024-01-01 16:00", "2024-01-03 00:00")     # starts 16h after the window opens
    assert_raises(lambda: owed(table), "A", "begins")


def test_funding_over_accepts_a_first_settlement_within_one_interval_of_the_start():
    table = funding_frame("2024-01-01 08:00", "2024-01-03 00:00")     # a new listing: APT's pattern
    assert np.isclose(owed(table), -0.03)


def test_funding_over_raises_when_an_intraday_window_end_hides_an_owed_settlement():
    table = with_hole(funding_frame("2023-12-31 00:00", "2024-01-03 00:00"), utc("2024-01-01 08:00"))
    assert_raises(lambda: owed(table, end="2024-01-01 12:00"), "A")


def test_funding_over_raises_on_duplicate_or_unordered_stamps():
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    assert_raises(lambda: owed(table.iloc[[0, 1, 2, 3, 4, 4, 5, 6, 7, 8]]), "A", "increasing")
    assert_raises(lambda: owed(table.iloc[::-1]), "A", "increasing")


def test_funding_over_requires_the_recorded_interval():
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00").drop(columns="interval_hours")
    assert_raises(lambda: owed(table), "A", "interval_hours")
    table = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    table.loc[table.stamp == HOLE, "interval_hours"] = np.nan
    assert_raises(lambda: owed(table), "A", "interval")


def test_funding_over_owes_nothing_and_certifies_nothing_for_a_zero_position():
    empty = funding_frame("2023-12-31 00:00", "2024-01-03 00:00").iloc[:0]
    assert owed(empty, quantity=0.0) == 0.0


def test_funding_over_ignores_holes_and_bad_rows_outside_the_window():
    table = funding_frame("2023-12-29 00:00", "2024-01-06 00:00")
    table = with_hole(table, utc("2023-12-30 08:00"))                  # long before the window
    table = with_hole(table, utc("2024-01-02 16:00"))                  # after it
    table.loc[table.stamp == utc("2024-01-03 08:00"), "rate"] = np.nan
    assert np.isclose(owed(table), -0.03)


def test_funding_over_follows_the_recorded_interval_through_every_change():
    """8 -> 4 -> 1 -> 2 -> 4 -> 8: no cadence is assumed, a change may sit on either side of a pair."""
    plan = [("2024-01-01 00:00", 8), ("2024-01-01 08:00", 8), ("2024-01-01 16:00", 4),   # gap 8: previous row's
            ("2024-01-01 20:00", 4), ("2024-01-01 21:00", 1), ("2024-01-01 22:00", 1),   # gap 1: its own
            ("2024-01-01 23:00", 1), ("2024-01-02 01:00", 2), ("2024-01-02 03:00", 2),
            ("2024-01-02 07:00", 4), ("2024-01-02 11:00", 4), ("2024-01-02 19:00", 8)]
    table = pd.DataFrame({"stamp": [utc(t) for t, _ in plan], "rate": 0.001, "mark": 100.0,
                          "interval_hours": [float(h) for _, h in plan]})
    assert len(basket.funding_gaps(table)) == 0
    assert np.isclose(owed(table, end="2024-01-02 11:00"), -0.1 * 10)   # ten settlements after the first
    bent = table.copy()
    bent.loc[7, "stamp"] = utc("2024-01-02 02:00")                      # 3h after 23:00: neither interval
    assert len(basket.funding_gaps(bent)) == 2


def test_funding_gaps_lists_each_violation_with_its_pair():
    table = with_hole(funding_frame("2023-12-31 00:00", "2024-01-02 00:00"))
    gaps = basket.funding_gaps(table)
    assert len(gaps) == 1
    assert gaps.iloc[0].previous == utc("2024-01-01 08:00") and gaps.iloc[0].stamp == utc("2024-01-02 00:00")
    assert gaps.iloc[0].spacing_hours == 16.0


def test_ledger_raises_on_a_settlement_missing_inside_a_held_period():
    fixture = ledger_fixture()
    fixture["settlements"]["A"] = with_hole(fixture["settlements"]["A"], utc("2024-01-03 08:00"))
    assert_raises(lambda: run_ledger(**fixture), "A", "2024-01-03")


def test_ledger_raises_when_a_held_contract_has_no_funding_at_all():
    fixture = ledger_fixture()
    fixture["settlements"]["B"] = fixture["settlements"]["B"].iloc[:0]
    assert_raises(lambda: run_ledger(**fixture), "B")


def test_ledger_raises_on_a_nan_funding_rate_in_a_held_period():
    fixture = ledger_fixture()
    table = fixture["settlements"]["A"]
    table.loc[table.stamp == utc("2024-01-02 16:00"), "rate"] = np.nan
    assert_raises(lambda: run_ledger(**fixture), "A", "rate")


def test_ledger_needs_no_funding_for_a_leg_it_has_already_sold():
    """B is sold at the 01-03 reconstitution; its table may end there. Only held exposure owes funding."""
    fixture = ledger_fixture()
    fixture["settlements"]["B"] = funding_frame("2023-12-31 00:00", "2024-01-03 00:00")
    led = run_ledger(**fixture)
    assert led[led.contract == "B"].holding_period_start.max() == utc("2024-01-03")


def test_ledger_runs_on_any_recorded_cadence():
    fixture = ledger_fixture()
    for contract, hours in zip("ABC", (4, 1, 2)):
        fixture["settlements"][contract] = funding_frame("2023-12-31 00:00", "2024-01-07 00:00", hours=hours)
    led = run_ledger(**fixture)
    a = led[(led.contract == "A") & (led.holding_period_start == utc("2024-01-01"))].iloc[0]
    assert np.isclose(a.funding_pnl, -a.quantity_after * 100.0 * 0.0001 * 6)    # six 4h settlements a day


# ---------------------------------------------------------------- the loader

def raw_funding(first, last, hours=8, rate=0.0001):
    """Vendor-shaped rows: a few milliseconds late, as Binance stamps them."""
    stamps = pd.date_range(first, last, freq=f"{hours}h", tz="UTC") + pd.Timedelta(milliseconds=5)
    return pd.DataFrame({"timestamp": stamps, "funding_rate": rate, "funding_interval_hours": hours})


def hourly_marks(first="2024-01-01 00:00", last="2024-01-05 23:00"):
    return pd.DataFrame({"open": 100.0}, index=pd.date_range(first, last, freq="h", tz="UTC"))


def load_with(funding, marks=None):
    marks = hourly_marks() if marks is None else marks
    with patch.object(panel, "load_funding", return_value=funding), \
            patch.object(panel, "load_mark_prices", return_value=marks):
        return load_settlements(["A"])["A"]


GOOD = ("2024-01-01 00:00", "2024-01-05 00:00")


def test_loader_carries_the_recorded_interval_and_keeps_every_row():
    loaded = load_with(raw_funding(*GOOD))
    assert len(loaded) == 13 and (loaded.interval_hours == 8.0).all()
    assert list(loaded.columns) == ["stamp", "rate", "mark", "interval_hours", "exchange_skipped"]


def test_loader_raises_on_a_nan_funding_rate():
    funding = raw_funding(*GOOD)
    funding.loc[5, "funding_rate"] = np.nan
    assert_raises(lambda: load_with(funding), "A", "rate")


def test_loader_raises_on_a_nan_interval():
    funding = raw_funding(*GOOD)
    funding.loc[5, "funding_interval_hours"] = np.nan
    assert_raises(lambda: load_with(funding), "A", "interval")


def test_loader_raises_on_empty_funding():
    empty = raw_funding(*GOOD).iloc[:0]
    assert_raises(lambda: load_with(empty), "A")


def test_loader_raises_when_no_funding_falls_inside_the_mark_range():
    before = raw_funding("2023-12-01 00:00", "2023-12-10 00:00")
    assert_raises(lambda: load_with(before), "A")


def test_loader_raises_on_a_hole_in_the_settlement_schedule_and_names_it():
    funding = raw_funding(*GOOD).drop(index=5)                     # 2024-01-02 16:00 never arrives
    assert_raises(lambda: load_with(funding), "A", "2024-01-02 08:00", "2024-01-03 00:00")


def test_loader_raises_on_duplicate_settlements():
    funding = raw_funding(*GOOD)
    funding = pd.concat([funding, funding.iloc[[4]]], ignore_index=True).sort_values("timestamp")
    assert_raises(lambda: load_with(funding), "A", "increasing")


def test_loader_accepts_interval_changes():
    plan = [("2024-01-01 00:00", 8), ("2024-01-01 08:00", 8), ("2024-01-01 16:00", 4),
            ("2024-01-01 20:00", 4), ("2024-01-01 21:00", 1), ("2024-01-01 22:00", 1),
            ("2024-01-01 23:00", 1), ("2024-01-02 01:00", 2), ("2024-01-02 03:00", 2),
            ("2024-01-02 07:00", 4), ("2024-01-02 11:00", 4), ("2024-01-02 19:00", 8)]
    funding = pd.DataFrame({"timestamp": [utc(t) + pd.Timedelta(milliseconds=5) for t, _ in plan],
                            "funding_rate": 0.0001, "funding_interval_hours": [h for _, h in plan]})
    loaded = load_with(funding)
    assert len(loaded) == 12 and list(loaded.interval_hours) == [float(h) for _, h in plan]


def test_loader_still_drops_settlements_outside_the_mark_range_without_checking_them():
    """Rule (d), unchanged: before the first mark / after the last, rows are dropped, holes in them included."""
    marks = hourly_marks("2024-01-01 03:00", "2024-01-04 23:00")      # first mark after the 00:00 settlement
    funding = raw_funding("2024-01-01 00:00", "2024-01-06 00:00").drop(index=14)   # hole in the dead tail
    loaded = load_with(funding, marks)
    assert loaded.stamp.min() == utc("2024-01-01 08:00") and loaded.stamp.max() == utc("2024-01-04 16:00")


def test_loader_carries_each_contracts_interval_on_real_data():
    out = load_settlements(["BTCUSDT", "APTUSDT"])
    assert (out["BTCUSDT"].interval_hours == 8.0).all()
    assert set(out["APTUSDT"].interval_hours) <= {8.0}


def test_loader_admits_hypes_registered_exchange_skip_and_flags_it():
    """HYPEUSDT (4h) has no 2026-06-24 04:00 row in any Binance feed; the exchange skipped it.
    The registered allowance loads it with one explicit flagged zero-rate row, not silently."""
    rows = load_settlements(["HYPEUSDT"])["HYPEUSDT"]
    assert list(rows[rows.exchange_skipped].stamp) == [utc("2026-06-24 04:00")]


def test_a_real_window_across_the_skip_is_complete_and_unregistered_holes_still_raise():
    rows = load_settlements(["HYPEUSDT"])
    assert np.isfinite(funding_over(rows, "HYPEUSDT", utc("2026-06-24 00:00"), utc("2026-06-25 00:00"), 1.0))
    ftt = {"FTTUSDT": basket._read_settlements("FTTUSDT")}           # interval-change holes, unregistered
    assert_raises(lambda: funding_over(ftt, "FTTUSDT", utc("2022-11-09 00:00"), utc("2022-11-11 00:00"), 1.0),
                  "FTTUSDT", "2022-11-10")


# ---------------------------------------------------------------- reference basket: no silent drops

def flat_pair(nan_on=None):
    days = [pd.Timestamp(d).date() for d in ("2024-01-01", "2024-01-02", "2024-01-03")]
    marks = pd.DataFrame({"A": 50.0, "B": 50.0}, index=days)
    if nan_on is not None:
        marks.loc[days[nan_on], "A"] = np.nan
    members = pd.DataFrame({"contract": ["A", "B"], "weight": [0.5, 0.5]})
    return {days[0]: members}, marks, days


def test_reference_basket_raises_on_a_nan_price_for_a_held_asset():
    """Two flat assets, one price missing on the middle day: the old code printed 100 -> 50 -> 100."""
    members, marks, days = flat_pair(nan_on=1)
    assert_raises(lambda: reference_basket(members, marks), "A", str(days[1]))


def test_reference_basket_raises_when_a_held_asset_is_unpriced_on_the_day_it_is_sold():
    days = [pd.Timestamp(d).date() for d in ("2024-01-01", "2024-01-02", "2024-01-03")]
    marks = pd.DataFrame({"A": 50.0, "B": 50.0, "C": 50.0}, index=days)
    marks.loc[days[1], "B"] = np.nan
    first = pd.DataFrame({"contract": ["A", "B"], "weight": [0.5, 0.5]})
    second = pd.DataFrame({"contract": ["A", "C"], "weight": [0.5, 0.5]})
    assert_raises(lambda: reference_basket({days[0]: first, days[1]: second}, marks), "B")


def test_reference_basket_ignores_nan_prices_of_assets_it_does_not_hold():
    days = [pd.Timestamp(d).date() for d in ("2024-01-01", "2024-01-02", "2024-01-03")]
    marks = pd.DataFrame({"A": 50.0, "B": 50.0, "C": np.nan, "D": 50.0}, index=days)
    marks.loc[days[2], "B"] = np.nan                                     # B is sold on day 1, dead on day 2
    first = pd.DataFrame({"contract": ["A", "B"], "weight": [0.5, 0.5]})
    second = pd.DataFrame({"contract": ["A", "D"], "weight": [0.5, 0.5]})
    index, weights = reference_basket({days[0]: first, days[1]: second}, marks)
    assert np.allclose(index, 100.0) and not index.isna().any()
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_reference_index_on_the_real_monthly_buffered_path_is_unchanged():
    """2020-02-09 start, monthly 10-in/13-out membership, through 2026-08-31: complete prices, no NaN."""
    from data.universe.selector import UniverseSelector
    start, final = date(2020, 2, 9), date(2026, 8, 31)
    schedule = [start, *pd.date_range("2020-03-01", "2026-08-01", freq="MS").date]
    members = UniverseSelector(min_history_days=0).select_buffered(schedule)
    contracts = sorted({c for frame in members.values() for c in frame.contract})
    marks = basket.load_marks(contracts)
    marks = marks[(marks.index >= start) & (marks.index <= final)]
    index, weights = reference_basket(members, marks)
    assert len(members) == 79 and len(contracts) == 24 and "LUNAUSDT" not in contracts
    assert len(index) == 2396 and not index.isna().any() and not weights.isna().any().any()
    assert abs(index.iloc[-1] - 696.978) < 0.01
    returns = pd.DataFrame({"basket": index.pct_change(), "btc": marks["BTCUSDT"].pct_change()}).dropna()
    assert round(returns.basket.corr(returns.btc), 3) == 0.977


# ---------------------------------------------------------------- sample endpoints

def test_ledger_raises_when_the_start_day_is_missing_from_marks():
    fixture = ledger_fixture()
    fixture["marks"] = fixture["marks"].drop(index=LEDGER_DAYS[0])
    assert_raises(lambda: run_ledger(**fixture), str(LEDGER_DAYS[0]))


def test_ledger_raises_when_the_end_day_is_beyond_the_marks():
    fixture = ledger_fixture()
    fixture["end"] = date(2024, 1, 7)                      # marks stop on 01-06
    assert_raises(lambda: run_ledger(**fixture), "2024-01-07")


def test_ledger_raises_when_the_end_day_is_the_last_mark_so_its_exit_price_is_missing():
    fixture = ledger_fixture()
    fixture["end"] = LEDGER_DAYS[-1]                       # the last period would need open[01-07]
    assert_raises(lambda: run_ledger(**fixture), "2024-01-07")


def test_ledger_raises_when_a_day_inside_the_window_is_missing():
    fixture = ledger_fixture()
    fixture["marks"] = fixture["marks"].drop(index=LEDGER_DAYS[3])
    assert_raises(lambda: run_ledger(**fixture), str(LEDGER_DAYS[3]))


def test_ledger_raises_when_start_is_after_end():
    fixture = ledger_fixture()
    fixture["start"], fixture["end"] = LEDGER_DAYS[3], LEDGER_DAYS[1]
    assert_raises(lambda: run_ledger(**fixture), "start")


def test_ledger_covers_every_requested_day():
    led = run_ledger(**ledger_fixture())
    assert sorted(set(led.holding_period_start)) == [utc(f"2024-01-0{d}") for d in range(1, 6)]


def daily_net(led):
    return led.groupby("holding_period_start").agg(net=("net_pnl", "sum"), base=("capital_base", "first"))


def test_equity_policy_base_is_capital_plus_prior_net_pnl():
    led = run_ledger(**ledger_fixture(), capital=100_000.0, capital_policy="equity")
    day = daily_net(led)
    assert np.isclose(day.base.iloc[0], 100_000.0)
    assert np.allclose(day.base.iloc[1:].values, 100_000.0 + day.net.cumsum().iloc[:-1].values)
    assert day.base.nunique() > 1                                # it actually compounds


def test_equity_policy_resets_to_equity_not_to_the_initial_base():
    led = run_ledger(**ledger_fixture(), capital=100_000.0, capital_policy="equity")
    recon = led[led.event == "reconstitution"]
    assert np.isclose(recon.notional_after.sum(), recon.capital_base.iloc[0])
    assert not np.isclose(recon.notional_after.sum(), 100_000.0)  # A would reset to exactly the base


def test_policies_agree_on_the_opening_day_and_differ_afterwards():
    fixed = run_ledger(**ledger_fixture(), capital_policy="fixed")
    equity = run_ledger(**ledger_fixture(), capital_policy="equity")
    first = fixed.holding_period_start.min()
    assert np.allclose(fixed[fixed.holding_period_start == first].net_pnl.values,
                       equity[equity.holding_period_start == first].net_pnl.values)
    assert not np.isclose(fixed.net_pnl.sum(), equity.net_pnl.sum())


def test_both_policies_satisfy_the_identity_and_return_uses_its_own_base():
    for policy in ("fixed", "equity"):
        led = run_ledger(**ledger_fixture(), capital_policy=policy)
        residual = led.net_pnl - (led.price_pnl + led.funding_pnl - led.fee - led.slippage)
        assert residual.abs().max() < 1e-9
        assert (led.capital_base > 0).all()


def test_unknown_capital_policy_raises():
    try:
        run_ledger(**ledger_fixture(), capital_policy="compounded")
    except ValueError as error:
        assert "capital_policy" in str(error)
        return
    raise AssertionError("expected ValueError")


def test_default_policy_is_fixed():
    default = run_ledger(**ledger_fixture())
    fixed = run_ledger(**ledger_fixture(), capital_policy="fixed")
    assert default.equals(fixed)


SKIP = utc("2026-06-24 04:00")


def skip_table(stamps, interval=4.0):
    return pd.DataFrame({"stamp": pd.to_datetime(stamps, utc=True), "rate": 0.0001,
                         "mark": 60.0, "interval_hours": interval})


SKIP_MARKS = pd.DataFrame({"open": [61.0]}, index=pd.DatetimeIndex([SKIP]))


def test_registered_skip_becomes_an_explicit_flagged_zero_rate_row():
    rows = skip_table(["2026-06-24 00:00", "2026-06-24 08:00"])
    out = _add_exchange_skips("HYPEUSDT", rows, SKIP_MARKS)
    row = out[out.stamp == SKIP].iloc[0]
    assert row.exchange_skipped and row.rate == 0.0 and row.mark == 61.0 and row.interval_hours == 4.0
    assert out.exchange_skipped.sum() == 1 and len(funding_gaps(out)) == 0


def test_skip_is_not_applied_to_an_unregistered_contract():
    rows = skip_table(["2026-06-24 00:00", "2026-06-24 08:00"])
    out = _add_exchange_skips("BTCUSDT", rows, SKIP_MARKS)
    assert len(out) == 2 and len(funding_gaps(out)) == 1        # the hole still raises downstream


def test_skip_is_not_applied_when_the_neighbours_are_wrong():
    for stamps in (["2026-06-23 20:00", "2026-06-24 08:00"],    # 8h before, 4h after
                   ["2026-06-24 00:00", "2026-06-24 12:00"]):   # 4h before, 8h after
        out = _add_exchange_skips("HYPEUSDT", skip_table(stamps), SKIP_MARKS)
        assert len(out) == 2 and not out.exchange_skipped.any()


def test_skip_needs_the_mark_at_that_hour():
    rows = skip_table(["2026-06-24 00:00", "2026-06-24 08:00"])
    no_mark = pd.DataFrame({"open": [61.0]}, index=pd.DatetimeIndex([utc("2026-06-24 05:00")]))
    assert len(_add_exchange_skips("HYPEUSDT", rows, no_mark)) == 2


def test_skip_charges_nothing_in_funding_over():
    rows = _add_exchange_skips("HYPEUSDT", skip_table(["2026-06-24 00:00", "2026-06-24 08:00"]), SKIP_MARKS)
    paid = funding_over({"HYPEUSDT": rows}, "HYPEUSDT", utc("2026-06-24 00:00"), utc("2026-06-24 08:00"), 10.0)
    assert np.isclose(paid, -10.0 * 60.0 * 0.0001)             # only the 08:00 row charges


def test_real_nine_contracts_load_with_exactly_one_flagged_skip_each():
    names = sorted(EXCHANGE_SKIPPED[SKIP])
    out = load_settlements(names)
    for name in names:
        flagged = out[name][out[name].exchange_skipped]
        assert list(flagged.stamp) == [SKIP] and (flagged.rate == 0.0).all() and (flagged.mark > 0).all()


def test_real_other_contracts_are_untouched_and_ftt_still_raises():
    btc = load_settlements(["BTCUSDT"])["BTCUSDT"]
    assert not btc.exchange_skipped.any() and len(btc) == 7305
    try:
        load_settlements(["FTTUSDT"])
    except ValueError:
        return
    raise AssertionError("FTT's interval-change holes must still raise")


def test_skip_sensitivity_is_quantity_times_the_settlement_mark_times_the_rate():
    led = pd.DataFrame({"holding_period_start": [utc("2026-06-24")] * 3 + [utc("2026-06-25")],
                        "contract": ["HYPEUSDT", "TIAUSDT", "BTCUSDT", "HYPEUSDT"],
                        "quantity_after": [10.0, -20.0, 1.0, 99.0]})
    marks = {"HYPEUSDT": 60.0, "TIAUSDT": 2.0}
    st = {c: pd.DataFrame({"stamp": [SKIP], "mark": [m], "rate": [0.0]}) for c, m in marks.items()}
    out = skipped_funding_sensitivity(led, st, rates=(1e-3, -1e-3))
    expected = -(10.0 * 60.0) * 1e-3 - (-20.0 * 2.0) * 1e-3          # BTC unaffected; the next day is another period
    assert np.isclose(out[1e-3], expected) and np.isclose(out[-1e-3], -expected)
    assert set(skipped_funding_sensitivity(led, st)) == set(EXCHANGE_SKIP_SENSITIVITY_RATES)


def test_the_sensitivity_uses_the_mark_at_the_settlement_not_the_opening_notional():
    led = pd.DataFrame({"holding_period_start": [utc("2026-06-24")], "contract": ["HYPEUSDT"],
                        "quantity_after": [10.0], "notional_after": [10.0 * 50.0]})   # opened at 50
    st = {"HYPEUSDT": pd.DataFrame({"stamp": [SKIP], "mark": [80.0], "rate": [0.0]})}  # mark moved to 80 by 04:00
    assert np.isclose(skipped_funding_sensitivity(led, st, rates=(1e-3,))[1e-3], -10.0 * 80.0 * 1e-3)


# ---------------------------------------------------------------- the exposure hook
def notional_fixture(recon=False):
    """The fixture in notional mode, with the drifted reference weights the real pipeline supplies."""
    fixture = ledger_fixture()
    opening = pd.DataFrame({"contract": ["A", "B"], "weight": [0.5, 0.5]})
    reference_members = {LEDGER_DAYS[0]: opening}
    if recon:
        reference_members.update(fixture["members_by_date"])       # A and C from the reset
    else:
        fixture["members_by_date"] = {}
    _, weights = reference_basket(reference_members, fixture["marks"])
    fixture["reference_weights"] = weights.reindex(columns=["A", "B", "C"], fill_value=0.0)
    return fixture


def run_with(fn, recon=False, **kwargs):
    return run_ledger(**notional_fixture(recon), mode="notional", exposure_fn=fn, **kwargs)


def test_hook_default_reproduces_the_constant_exposure_ledger_exactly():
    plain = run_ledger(**ledger_fixture())
    hooked = run_ledger(**ledger_fixture(), exposure_fn=lambda day, held, base: 1.0)
    pd.testing.assert_frame_equal(plain, hooked, check_exact=True)


def test_hook_is_called_each_day_with_day_held_exposure_and_base():
    calls = []
    run_with(lambda day, held, base: calls.append((day, held, base)) or 1.0, capital=100_000.0)
    assert [c[0] for c in calls] == LEDGER_DAYS[:5]             # one call per scored day, in order
    assert calls[0][1] == 0.0 and all(c[2] == 100_000.0 for c in calls)
    assert calls[1][1] > 1.0                                    # A rose, so held exposure drifted above 1


def test_rows_record_held_and_target_exposure():
    led = run_with(lambda day, held, base: 0.7)
    assert (led.target_exposure == 0.7).all()
    assert led.groupby("holding_period_start").held_exposure.first().iloc[0] == 0.0


def test_answering_no_trade_with_the_held_exposure_does_not_trade():
    decide = lambda day, held, base: 1.0 if held == 0.0 else held
    led = run_with(decide)
    later = led[led.event == "hold"]
    assert len(later) > 0 and (later.trade.abs() < 1e-6).all()


def test_idle_weight_is_applied_once_and_zero_active_allocation_is_flat():
    book = Book(quantities={"A": 400.0, "B": 400.0}, idle_weight=0.2)
    marks = {"A": 100.0, "B": 100.0}                            # live 80,000 of a 100,000 base
    held = held_exposure(book, marks, 100_000.0, False)
    assert np.isclose(held, 1.0)                                # exposure of the ACTIVE allocation
    kept = target_notionals(book, WEIGHTS, marks, False, held, 100_000.0, "notional")
    assert np.isclose(sum(kept.values()), 80_000.0) and np.allclose(list(kept.values()), [40_000.0] * 2)
    assert np.isclose(held_exposure(book, marks, 100_000.0, True), 0.8)   # a reset makes all of it active
    book.idle_weight = 1.0
    assert held_exposure(book, marks, 100_000.0, False) == 0.0
    assert all(v == 0.0 for v in target_notionals(book, WEIGHTS, marks, False, 1.0, 100_000.0, "notional").values())


def test_a_sign_flip_trades_through_zero_with_the_right_accounting():
    led = run_with(lambda day, held, base: 1.0 if day <= LEDGER_DAYS[0] else -1.0)
    a = led[led.contract == "A"].set_index("holding_period_start")
    flip = a.iloc[1]
    marks = notional_fixture()["marks"]
    assert a.iloc[0].quantity_after > 0 > flip.quantity_after
    assert flip.notional_before > 0 > flip.notional_after
    assert np.isclose(flip.trade, flip.notional_after - flip.notional_before)
    assert abs(flip.trade) > abs(flip.notional_before)                          # closed the long AND opened the short
    assert np.isclose(flip.price_pnl, flip.quantity_after * (marks.loc[LEDGER_DAYS[2], "A"] - marks.loc[LEDGER_DAYS[1], "A"]))
    assert flip.price_pnl < 0                                                   # short while A rises
    residual = led.net_pnl - (led.price_pnl + led.funding_pnl - led.fee - led.slippage)
    assert residual.abs().max() < 1e-9


def test_flat_days_write_no_rows_and_a_flat_book_reopens_on_reference_weights():
    plan = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0}
    led = run_with(lambda day, held, base: plan[LEDGER_DAYS.index(day)])
    days = sorted(led.holding_period_start.unique())
    assert utc("2024-01-03") not in days                                        # flat and staying flat: no rows
    reopened = led[led.holding_period_start == utc("2024-01-04")]
    drifted = notional_fixture()["reference_weights"].loc[LEDGER_DAYS[3], ["A", "B"]]
    assert np.allclose(reopened.set_index("contract").notional_after.loc[["A", "B"]].values,
                       100_000.0 * drifted.values)                              # the day's reference weights


def test_reconstitution_rebuilds_the_new_members_even_when_exposure_is_unchanged():
    led = run_with(lambda day, held, base: 1.0, recon=True)
    day = led[led.event == "reconstitution"].set_index("contract")
    assert day.loc["B", "quantity_after"] == 0.0 and day.loc["C", "quantity_after"] > 0.0
    assert (day.fee + day.slippage).sum() > 0


def test_mixed_sign_holdings_raise():
    book = Book(quantities={"A": 1.0, "B": -1.0})
    for call in (lambda: held_exposure(book, {"A": 50.0, "B": 50.0}, 100.0, False),
                 lambda: target_notionals(book, WEIGHTS, {"A": 50.0, "B": 50.0}, False, 1.0, 100.0, "notional")):
        assert_raises(call, "mixed-sign")


def test_a_non_finite_exposure_from_the_hook_raises():
    assert_raises(lambda: run_with(lambda day, held, base: float("nan")), "exposure_fn")


def test_hook_receives_the_equity_base_under_the_equity_policy():
    bases = []
    run_with(lambda day, held, base: bases.append(base) or 1.0, capital_policy="equity")
    assert bases[0] == 100_000.0 and len(set(bases)) > 1


# ---------------------------------------------------------------- the per-side cost parameter
def test_cost_rates_split_the_all_in_cost_half_fee_half_slippage():
    assert cost_rates(10.0) == (0.0005, 0.0005) and cost_rates(20.0) == (0.001, 0.001)
    assert cost_rates(5.0) == (0.00025, 0.00025) and cost_rates(0.0) == (0.0, 0.0)
    for bad in (-1.0, float("nan"), float("inf")):
        assert_raises(lambda: cost_rates(bad), "cost_bp_per_side")


def test_the_default_cost_reproduces_the_fixed_constants_bit_for_bit():
    default = run_ledger(**ledger_fixture())
    explicit = run_ledger(**ledger_fixture(), cost_bp_per_side=10.0)
    pd.testing.assert_frame_equal(default, explicit, check_exact=True)
    assert np.isclose(default.fee.sum(), default.trade.abs().sum() * FEE_PER_SIDE)


def test_doubling_the_cost_doubles_fees_and_slippage_and_changes_nothing_else():
    low = run_ledger(**ledger_fixture(), cost_bp_per_side=10.0)
    high = run_ledger(**ledger_fixture(), cost_bp_per_side=20.0)
    assert np.allclose(high.fee, 2 * low.fee) and np.allclose(high.slippage, 2 * low.slippage)
    for column in ("trade", "quantity_after", "price_pnl", "funding_pnl", "notional_after"):
        assert np.array_equal(high[column], low[column]), column
    assert (high.net_pnl <= low.net_pnl + 1e-12).all()
    zero = run_ledger(**ledger_fixture(), cost_bp_per_side=0.0)
    assert (zero.fee == 0).all() and (zero.slippage == 0).all()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
