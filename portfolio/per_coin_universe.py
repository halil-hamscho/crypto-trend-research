"""Which instruments may hold a per-coin position on each day.

Protocol: docs/october/per_coin_protocol_2026-10-07.md, Part 14 (Amendment 1) over Parts 2-3.
Membership is the sealed basket's own: `select_buffered` on the registered monthly schedule, so the
coins are exactly those of the basket study. A member is ELIGIBLE on day D when
  - it is a member that month (the latest reconstitution on or before D),
  - it traded on D (daily bar with volume > 0), D is not the first traded day of its span, and at
    least `min_hourly_marks` of D's 24 hourly marks exist (U5-U6), and
  - its current span has at least `min_traded_days` traded days up to and including D (S5).
A span ends when the contract goes more than `span_break_days` calendar days without a trade (U2);
the next trade starts a new instrument with its own history. Pure functions take the data as
arguments; the loaders at the bottom read it from disk.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.universe.coinmarketcap.new_fetcher import contract_map, normalize, panel
from data.universe.selector import UniverseSelector
from portfolio.basket import reconstitution_dates

REASONS = ("not_traded", "span_first_day", "marks_incomplete", "short_history")
ACTIVITY_LAG_DAYS = 1      # eligibility on day X reads trading activity from X - 1 (Clarification 1)


def traded_days(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Calendar dates (UTC midnight, tz-naive) on which the contract traded: volume > 0."""
    days = pd.to_datetime(bars.loc[bars.volume > 0, "timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    return pd.DatetimeIndex(days.drop_duplicates().sort_values())


SPAN_STARTS = "span_starts"     # bars.attrs key: registered forced span starts (a redenomination)


def forced_starts(bars: pd.DataFrame) -> tuple:
    """Days on which a registered span break starts a new instrument whatever the gap (e.g. BNX's
    1:100 token swap, docs/october/per_coin_support/span_break_register.csv). Carried on the bars as
    `attrs`, so every caller that splits spans (signal, eligibility, ledger) sees the same breaks."""
    return tuple(pd.Timestamp(d) for d in bars.attrs.get(SPAN_STARTS, ()))


def traded_spans(days: pd.DatetimeIndex, break_days: int, starts=()) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[first, last] traded day of each span. A gap of more than `break_days` calendar days with no
    trade ends a span: next - previous - 1 > break_days. A traded day in `starts` also begins one."""
    if len(days) == 0:
        return []
    idle = np.diff(days.to_numpy()).astype("timedelta64[D]").astype(int) - 1
    forced = np.isin(days.to_numpy()[1:], pd.DatetimeIndex(list(starts)).to_numpy()) if len(starts) else False
    breaks = np.flatnonzero((idle > break_days) | forced)
    starts = [0] + list(breaks + 1)
    ends = list(breaks) + [len(days) - 1]
    return [(days[s], days[e]) for s, e in zip(starts, ends)]


def span_prices(bars: pd.DataFrame, break_days: int) -> list[pd.Series]:
    """One price series per span: the daily open on traded days, indexed by date."""
    traded = bars[bars.volume > 0].copy()
    traded["day"] = pd.to_datetime(traded.timestamp, utc=True).dt.tz_localize(None).dt.normalize()
    traded = traded.drop_duplicates("day").set_index("day").sort_index()
    out = []
    for first, last in traded_spans(pd.DatetimeIndex(traded.index), break_days, forced_starts(bars)):
        s = traded.loc[first:last, "open"].astype(float)
        s.index = [d.date() for d in s.index]
        s.name = f"{first.date()}"
        out.append(s)
    return out


def hourly_mark_counts(marks: pd.DataFrame) -> pd.Series:
    """Hourly mark bars present per calendar day (index: date)."""
    stamps = marks["timestamp"] if "timestamp" in marks.columns else marks.index.to_series()
    hours = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True, format="ISO8601")).tz_localize(None)
    counts = pd.Series(1, index=hours.floor("D")).groupby(level=0).sum()
    counts.index = [d.date() for d in counts.index]
    return counts


def membership(end: date) -> dict:
    """The sealed basket's buffered top-10 membership on every reconstitution date through `end`."""
    return UniverseSelector(min_history_days=0).select_buffered(reconstitution_dates(end))


# Amendment 2: top-25 and top-50 sensitivities, buffer exit rank scaled like the sealed 10 -> 13.
DEEP_EXIT_RANK = {25: 33, 50: 65}


class DeepSelector(UniverseSelector):
    """The sealed selector's rules at a deeper pool (per-coin protocol Amendment 2).

    The screens are the sealed ones, inherited unchanged: stablecoin and wrapped exclusions, a Binance
    contract that traded that day, the 30-day median Binance dollar volume, history. Only two inputs
    differ: the coin-to-contract map (`cmc_contract_map_top100.csv`) and the pool depth (100). A
    contract that two coins shared (a token migration or duplicate listing) is used for a coin only
    inside that coin's claim range. Built with the sealed map and depth, it reproduces the sealed
    selector exactly (portfolio/test_per_coin.py). `data/universe/selector.py` is not edited.
    """

    def __init__(self, min_history_days: int, n: int, min_volume_usd: float = 10e6,
                 table=contract_map.DEEP_MAP_TABLE, depth: int = contract_map.DEEP_POOL_DEPTH):
        self.n = n
        self.min_volume_usd = min_volume_usd
        self.min_history_days = min_history_days
        universe = normalize.load().sort_values(["date", "market_cap"], ascending=[True, False])
        self._by_date = dict(tuple(universe.groupby("date")))
        self._pool = contract_map.pool_ids(universe, depth)
        self._history, self._dollar_volume, first_day = {}, {}, {}
        contracts = contract_map.load(table)
        for contract in contracts.contract.unique():
            bars = panel.load_ohlcv(contract)
            traded = bars[bars.volume > 0]
            days = traded.timestamp.dt.date
            first_day[contract] = days.iloc[0]
            for count, day in enumerate(days, start=1):
                self._history[(contract, day)] = count
            median = traded.quote_volume.rolling(
                self.VOLUME_WINDOW_DAYS, min_periods=self.VOLUME_MIN_OBSERVATIONS).median()
            for day, value in zip(days, median):
                self._dollar_volume[(contract, day)] = value
        self._contracts = {cmc_id: sorted(group.contract, key=first_day.get)
                           for cmc_id, group in contracts.groupby("cmc_id")}
        self._claims = {}
        if "claim_from" in contracts.columns:
            for row in contracts[contracts.claim_from.notna()].itertuples():
                self._claims.setdefault((row.cmc_id, row.contract), []).append(
                    (date.fromisoformat(str(row.claim_from)), date.fromisoformat(str(row.claim_to))))

    def contract_on(self, cmc_id: int, day: date) -> str | None:
        for contract in self._contracts.get(cmc_id, []):
            ranges = self._claims.get((cmc_id, contract))
            if ranges is not None and not any(a <= day <= b for a, b in ranges):
                continue
            if (contract, day) in self._history:
                return contract
        return None


def deep_membership(n: int, end: date, start: date | None = None) -> dict:
    """Buffered top-`n` membership (n = 25 or 50) on the registered monthly schedule."""
    dates = reconstitution_dates(end) if start is None else reconstitution_dates(end, start)
    return DeepSelector(min_history_days=0, n=n).select_buffered(dates, exit_rank=DEEP_EXIT_RANK[n])


def members_by_day(members_by_date: dict, days: list) -> pd.DataFrame:
    """Boolean [day x contract]: a member that month (the latest reconstitution on or before the day)."""
    resets = sorted(members_by_date)
    contracts = sorted({c for m in members_by_date.values() for c in m.contract})
    out = pd.DataFrame(False, index=days, columns=contracts)
    for i, reset in enumerate(resets):
        until = resets[i + 1] if i + 1 < len(resets) else None
        rows = [d for d in days if d >= reset and (until is None or d < until)]
        out.loc[rows, list(members_by_date[reset].contract)] = True
    return out


def eligibility(members_by_date: dict, days: list, bars: dict, mark_counts: dict, config) -> tuple:
    """(eligible, reason, instrument): three [day x contract] frames, as KNOWN AT 00:00 OF EACH DAY.

    Membership on day X is known at X 00:00 (it comes from the reset in force, decided on an earlier
    snapshot). A day's trading activity is not: whether X trades, and how many hourly marks it has,
    is known only when X ends. So eligibility on X reads activity from X - 1, the last completed day
    (per-coin protocol, Clarification 1): traded on X - 1, X - 1 not the first day of its span,
    complete marks on X - 1, and at least `min_traded_days` traded days in the span through X - 1.
    Whether the coin can actually trade on its fill day is execution, checked separately by the
    ledger (`LedgerInputs.tradable`).

    `eligible` is boolean; `reason` names why a member day is NOT eligible (one of REASONS, the first
    that applies) and is None otherwise; `instrument` labels the span a traded day belongs to
    (`<contract>#<first traded day>`), the key the signal is computed on. `bars` maps contract to its
    daily klines (timestamp, open, volume); `mark_counts` maps contract to hourly_mark_counts.
    """
    member = members_by_day(members_by_date, days)
    eligible = pd.DataFrame(False, index=days, columns=member.columns)
    reason = pd.DataFrame(None, index=days, columns=member.columns, dtype=object)
    instrument = pd.DataFrame(None, index=days, columns=member.columns, dtype=object)
    for contract in member.columns:
        tdays = traded_days(bars[contract])
        traded = {d.date() for d in tdays}
        count_in_span, span_of, first_days = {}, {}, set()
        for first, last in traded_spans(tdays, config.span_break_days, forced_starts(bars[contract])):
            first_days.add(first.date())
            for k, d in enumerate(tdays[(tdays >= first) & (tdays <= last)], start=1):
                count_in_span[d.date()] = k
                span_of[d.date()] = f"{contract}#{first.date()}"
        marks = mark_counts.get(contract, pd.Series(dtype=int))
        for day in days:
            instrument.at[day, contract] = span_of.get(day)
            if not member.at[day, contract]:
                continue
            seen = day - timedelta(days=ACTIVITY_LAG_DAYS)          # the last day completed by day 00:00
            if seen not in traded:
                why = "not_traded"
            elif seen in first_days:
                why = "span_first_day"
            elif int(marks.get(seen, 0)) < config.min_hourly_marks:
                why = "marks_incomplete"
            elif count_in_span[seen] < config.min_traded_days:
                why = "short_history"
            else:
                eligible.at[day, contract] = True
                continue
            reason.at[day, contract] = why
    return eligible, reason, instrument


# ---------------------------------------------------------------- the registry-wide universe (Amendment 1 sensitivity)
def liquidity_median(bars: pd.DataFrame, days: list, window_days: int, min_traded: int) -> pd.Series:
    """U4: for each day D, the median daily quote volume over the traded days in the `window_days`
    calendar days ending D (the caller asks for D - 1); NaN with fewer than `min_traded` traded days."""
    traded = bars[bars.volume > 0]
    stamps = pd.to_datetime(traded.timestamp, utc=True).dt.tz_localize(None).dt.normalize()
    series = pd.Series(traded.quote_volume.to_numpy(dtype=float), index=stamps.to_numpy())
    series = series[~series.index.duplicated()].sort_index()
    out = {}
    for d in days:
        end = pd.Timestamp(d)
        window = series[(series.index > end - pd.Timedelta(days=window_days)) & (series.index <= end)]
        out[d] = float(window.median()) if len(window) >= min_traded else np.nan
    return pd.Series(out)


def registry_membership(contracts, days: list, bars: dict, threshold_usd: float, config, excluded=()) -> dict:
    """U1, U3 and U4 as daily membership: {day X: contracts passing}, where X passes when the
    contract is in the registry, not excluded (Appendix A), and its median quote volume over the 30
    days ending X - 1 is at least `threshold_usd` with >= 20 traded days in that window. Tradability,
    marks and history (U5-U7, S5) are then applied by `eligibility`, which reads activity from X - 1
    (Clarification 1). There is no count cap (U8)."""
    gate = config.raw["universe"]["liquidity_gate"]
    window, minimum, lag = int(gate["window_days"]), int(gate["min_traded_days_in_window"]), int(gate["lag_days"])
    excluded = set(excluded)
    seen = [d - timedelta(days=lag) for d in days]
    medians = {c: liquidity_median(bars[c], seen, window, minimum) for c in contracts if c not in excluded}
    out = {}
    for d, s in zip(days, seen):
        names = [c for c, m in medians.items() if np.isfinite(m[s]) and m[s] >= threshold_usd]
        out[d] = pd.DataFrame({"contract": sorted(names)})
    return out


def registry_contracts() -> list:
    """The registry as frozen at the download (data/derived/binance/traded_ranges.csv, 900 contracts
    on 2026-10-07), so the universe cannot drift with a live listing."""
    return sorted(pd.read_csv(panel.TRADED_RANGES).contract)


def load_exclusions(path=None) -> set:
    from pathlib import Path
    path = path or Path(__file__).resolve().parent.parent / "docs/october/per_coin_support/exclusions.csv"
    return set(pd.read_csv(path).contract)


# ---------------------------------------------------------------- loaders (disk)
def load_bars(contracts) -> dict:
    return {c: panel.load_ohlcv(c) for c in contracts}


def load_mark_counts(contracts) -> dict:
    """Hourly mark counts per day. A contract for which Binance archives NO mark files at all (five
    on 2026-10-07, protocol Appendix B) gets an empty series, so U6 makes it never tradable. A
    contract that has archived marks but no built file raises: that is a pipeline failure."""
    out = {}
    for c in contracts:
        if not (panel.MARK_DIR / f"{c}.csv").exists():
            archived = [p for kind in ("monthly", "daily")
                        for p in (panel.RAW_DIR / f"futures/um/{kind}/markPriceKlines/{c}/1h").glob("*.zip")]
            if archived:
                raise FileNotFoundError(f"{c}: {len(archived)} mark zips archived but no built mark file")
            out[c] = pd.Series(dtype=int)
            continue
        out[c] = hourly_mark_counts(panel.load_mark_prices(c))
    return out
