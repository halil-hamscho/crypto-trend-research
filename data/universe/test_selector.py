"""Run from the repo root: python -m data.universe.test_selector"""
from datetime import date

from data.universe.selector import PoolTooShallow, UniverseSelector

sel = UniverseSelector(min_history_days=0)


def test_contract_on():
    assert sel.contract_on(4172, date(2021, 6, 1)) == "LUNAUSDT"       # LUNA before the collapse
    assert sel.contract_on(4172, date(2023, 6, 1)) == "1000LUNCUSDT"   # same coin, renamed
    assert sel.contract_on(8916, date(2022, 7, 15)) is None            # ICP while halted
    assert sel.contract_on(11419, date(2026, 8, 1)) == "GRAMUSDT"      # TON after its rename
    assert sel.contract_on(1839, date(2020, 2, 2)) is None             # BNB before its listing


def test_acceptance():
    # 2020-02-09 is the first date the Binance dollar-volume screen can field a
    # full basket: it needs 20 daily observations, and most perps had only just
    # listed. ADA, XTZ and XMR are excluded here for thin Binance volume, not
    # for being untraded.
    members, exclusions = sel.select(date(2020, 2, 9))
    assert dict(zip(members.symbol, members.weight.mul(100).round(2))) == {
        "BTC": 75.10, "ETH": 10.23, "XRP": 5.07, "BCH": 3.39, "LTC": 2.05,
        "EOS": 1.90, "TRX": 0.61, "XLM": 0.59, "ETC": 0.57, "LINK": 0.50}
    assert dict(zip(exclusions.symbol, exclusions.reason)) == {
        "BSV": "not_trading", "USDT": "stable_or_wrapped", "BNB": "not_trading",
        "XTZ": "low_volume", "ADA": "low_volume", "XMR": "low_volume"}


def test_volume_screen_reads_binance_not_cmc():
    """The screen is the contract's own trailing median dollar volume."""
    day = date(2025, 2, 2)                     # the snapshot select() reads for 02-03
    contract, window = "SYSUSDT", sel.VOLUME_WINDOW_DAYS
    assert sel._dollar_volume[(contract, day)] < sel.min_volume_usd
    assert sel.VOLUME_MIN_OBSERVATIONS == 20 and window == 30
    btc = sel._dollar_volume[("BTCUSDT", day)]
    assert btc > 1e9                           # BTC clears the floor by orders of magnitude


def test_buffer_is_stateful():
    """An incumbent stays while it ranks inside 13; a newcomer needs rank 10."""
    dates = [date(2022, m, 1) for m in range(1, 13)]
    buffered = sel.select_buffered(dates)
    assert all(len(m) == 10 for m in buffered.values())
    plain = {d: set(sel.select(d)[0].contract) for d in dates}
    changes_plain = sum(len(plain[b] - plain[a]) for a, b in zip(dates, dates[1:]))
    changes_buffered = sum(
        len(set(buffered[b].contract) - set(buffered[a].contract))
        for a, b in zip(dates, dates[1:]))
    assert changes_buffered <= changes_plain   # buffering never adds churn
    for members in buffered.values():
        assert abs(members.weight.sum() - 1.0) < 1e-12


def test_junk_and_never_listed():
    _, exclusions = sel.select(date(2025, 2, 3))
    assert ("SYS", "low_volume") in zip(exclusions.symbol, exclusions.reason)
    _, exclusions = sel.select(date(2020, 3, 13))     # the crash: LEO briefly ranks high
    assert ("LEO", "no_contract") in zip(exclusions.symbol, exclusions.reason)


def test_refuses_to_guess():
    try:
        sel.select(date(2020, 1, 5))       # only 3 coins had traded yet
        assert False, "should have raised"
    except PoolTooShallow:
        pass


def test_every_date_has_ten():
    members = sel.select_all(start=date(2020, 2, 9))
    assert (members.groupby("trade_date").size() == 10).all()
    assert "SYS" not in set(members.symbol)


def test_warmup_moves_the_start():
    strict = UniverseSelector(min_history_days=64)
    strict.select_all(start=date(2020, 4, 14))
    try:
        strict.select_all(start=date(2020, 4, 13))
        assert False, "should have raised"
    except PoolTooShallow:
        pass


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
