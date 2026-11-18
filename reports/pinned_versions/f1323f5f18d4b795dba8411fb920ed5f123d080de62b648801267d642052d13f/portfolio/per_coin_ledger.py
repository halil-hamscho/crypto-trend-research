"""The per-coin ledger: books ten (or 25, 50, ~100) independent signed positions a day.

Design: docs/october/per_coin_ledger_design_2026-10-12.md (v3). Build plan:
docs/october/per_coin_ledger_implementation_2026-10-13.md (v2), which this module follows step by
step. The sealed basket ledger (`portfolio/basket.py`) is not edited; its per-row functions are
reused unchanged.

Built so far: step 1 (the inputs and their timing), step 2 (validate_targets), step 3 (the daily
loop on normal days: trade, hold and flat rows), step 4 (the daily table, reconciliations R1-R8 and
fingerprints). Step 1: Everything the daily loop will read is
loaded here, aligned by day and contract, and every decision price carries the instant it was
stamped, so the one-day lag can be tested by timestamp rather than by value.
"""
import hashlib
import numbers
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.universe.coinmarketcap.new_fetcher import panel
from portfolio.basket import (_describe, _read_settlements, _require_clean, cost_rates, funding_gaps, funding_over,
                              leg_row)
from portfolio.per_coin_signal import load_per_coin_config
from portfolio.per_coin_universe import load_bars, load_mark_counts, traded_days, traded_spans

HOLD = object()          # a target meaning "keep the held quantity; no trade"

EVENT_KINDS = ("settlement", "halt", "span_end")
EVENT_COLUMNS = ("contract", "kind", "first_day", "last_day", "instant", "evidence")
SUSPENSION_COLUMNS = ("contract", "start", "end", "evidence")


@dataclass(frozen=True)
class FillClock:
    """When fills happen and at which price: the 00:00 daily trade-price open (primary and
    same-bar), or the hourly mark open at 01:00 (the protocol's X2 sensitivity)."""
    hour: int
    price: str

    def __post_init__(self):
        if self.hour not in (0, 1):
            raise ValueError(f"FillClock hour must be 0 or 1, got {self.hour}")
        if self.price not in ("kline_open", "mark_open"):
            raise ValueError(f"FillClock price must be 'kline_open' or 'mark_open', got {self.price!r}")
        if (self.hour == 0) != (self.price == "kline_open"):
            raise ValueError("the 00:00 clock fills at the kline open; the 01:00 clock at the mark open")


PRIMARY = FillClock(0, "kline_open")
HOURLY = FillClock(1, "mark_open")


def fill_instant(day: date, clock: FillClock) -> pd.Timestamp:
    """The UTC instant a fill for holding period `day` happens."""
    return pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=clock.hour)


@dataclass(frozen=True)
class LedgerInputs:
    contracts: tuple
    days: list                        # start .. end + 1: the last row needs its exit price
    fill_prices: pd.DataFrame         # [day x contract] price at the fill instant; NaN = no trade that day
    decision_prices: pd.DataFrame     # [day x contract] the last price the decision for `day` may see
    decision_stamps: pd.Series        # day -> the UTC instant those decision prices were stamped
    tradable: pd.DataFrame            # [day x contract] traded bar, enough hourly marks, not a span's first day
    eligible: pd.DataFrame            # [holding day x contract] eligible AT THE DECISION: eligibility on D - lag
    settlements: dict                 # contract -> funding table, from load_settlements_registered
    events: pd.DataFrame              # event register (EVENT_COLUMNS)
    suspensions: pd.DataFrame         # suspension register (SUSPENSION_COLUMNS)
    clock: FillClock
    decision_lag_days: int
    capital: float
    cost_bp_per_side: float


# ------------------------------------------------------------------ funding, register-aware
def load_settlements_registered(contracts, suspensions, reader=_read_settlements) -> dict:
    """Funding settlements per contract, accepting only gaps a suspension register explains.

    The sealed `load_settlements` raises on any gap at load time, which leaves no room for an
    evidenced exchange suspension. This loader runs the same checks through the same sealed
    internals, and differs in one way: a gap is accepted when a register entry for that contract
    spans it, from the last recorded settlement before the gap to the first one after it. With an
    empty register (or None) it returns exactly what `load_settlements` returns (tested).
    """
    register = _validated_register(suspensions, SUSPENSION_COLUMNS, "suspension")
    out = {}
    for contract in contracts:
        rows = reader(contract)
        _require_clean(contract, rows)
        gaps = funding_gaps(rows)
        if len(gaps):
            mine = register[register.contract == contract]
            open_gaps = [g for g in gaps.itertuples()
                         if not ((mine.start <= g.previous) & (mine.end >= g.stamp)).any()]
            if open_gaps:
                raise ValueError(f"{contract}: {_describe(gaps.loc[[g.Index for g in open_gaps]])}")
        out[contract] = rows
    return out


# ------------------------------------------------------------------ prices and tradability
def _kline_open_frame(contracts, days, bars) -> pd.DataFrame:
    """The daily open (the 00:00 price) on days the contract traded; NaN otherwise."""
    out = pd.DataFrame(np.nan, index=days, columns=list(contracts))
    for c in contracts:
        traded = bars[c][bars[c].volume > 0]
        opens = pd.Series(traded.open.astype(float).to_numpy(),
                          index=pd.to_datetime(traded.timestamp, utc=True).dt.date.to_numpy())
        opens = opens[~opens.index.duplicated()]
        out[c] = opens.reindex(days).to_numpy()
    return out


def _mark_open_frame(contracts, days, marks, hour: int, traded_open: pd.DataFrame) -> pd.DataFrame:
    """The hourly mark open at `hour` UTC, on days the contract traded (a mark is not a trade, so
    a non-traded day still has no fill)."""
    out = pd.DataFrame(np.nan, index=days, columns=list(contracts))
    instants = [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=hour) for d in days]
    for c in contracts:
        m = marks[c]
        values = m["open"].astype(float).reindex(pd.DatetimeIndex(instants)).to_numpy()
        out[c] = np.where(traded_open[c].notna().to_numpy(), values, np.nan)
    return out


def _decision_price_frame(kline_open: pd.DataFrame, days, lag: int) -> tuple:
    """For each day D: the 00:00 open of day D - lag, and that instant. Same-bar (lag 0) and the
    01:00 clock decide on D 00:00; the primary one-day lag decides on D-1 00:00."""
    source = [d - timedelta(days=lag) for d in days]
    prices = kline_open.reindex(source)
    prices.index = days
    stamps = pd.Series([pd.Timestamp(s, tz="UTC") for s in source], index=days)
    return prices, stamps


def _tradable_frame(contracts, days, bars, mark_counts, min_marks: int, break_days: int) -> pd.DataFrame:
    """Traded bar that day (volume > 0), at least `min_marks` hourly marks, and not the first
    traded day of a span (protocol U5-U6)."""
    out = pd.DataFrame(False, index=days, columns=list(contracts))
    for c in contracts:
        tdays = traded_days(bars[c])
        traded = {d.date() for d in tdays}
        firsts = {first.date() for first, _ in traded_spans(tdays, break_days)}
        counts = mark_counts.get(c, pd.Series(dtype=int))
        out[c] = [d in traded and d not in firsts and int(counts.get(d, 0)) >= min_marks for d in days]
    return out


# ------------------------------------------------------------------ registers
def _validated_register(frame, columns, name) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=list(columns))
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} register lacks column(s) {missing}")
    blank = frame.evidence.isna() | (frame.evidence.astype(str).str.strip() == "")
    if blank.any():
        raise ValueError(f"{name} register: {int(blank.sum())} entr(y/ies) without evidence, "
                         f"e.g. {frame[blank].iloc[0].to_dict()}")
    if name == "event":
        bad = ~frame.kind.isin(EVENT_KINDS)
        if bad.any():
            raise ValueError(f"event register: unknown kind {sorted(set(frame.kind[bad]))}; "
                             f"allowed {EVENT_KINDS}")
        for contract, rows in frame.groupby("contract"):
            spans = sorted(zip(rows.first_day, rows.last_day))
            for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                if b0 <= a1:
                    raise ValueError(f"event register: entries for {contract} overlap ({a0}..{a1} and {b0}..{b1})")
    return frame.reset_index(drop=True)


def event_on(day: date, contract: str, events: pd.DataFrame):
    """The event-register entry covering (contract, day), or None. Entries never overlap (checked
    when the register is loaded)."""
    if events is None or len(events) == 0:
        return None
    hit = events[(events.contract == contract) & (events.first_day <= day) & (events.last_day >= day)]
    return None if hit.empty else next(hit.itertuples(index=False))


def _by_holding_day(eligible, decision_days, scored, contracts) -> pd.DataFrame:
    out = eligible.reindex(index=decision_days, columns=list(contracts), fill_value=False).astype(bool)
    out.index = scored
    return out


# ------------------------------------------------------------------ the package
def build_ledger_inputs(contracts, start: date, end: date, eligible: pd.DataFrame, clock: FillClock = PRIMARY,
                        decision_lag_days: int = 1, capital: float = 100_000.0, cost_bp_per_side: float = 10.0,
                        events=None, suspensions=None, bars=None, marks=None, mark_counts=None,
                        settlements=None, config=None) -> LedgerInputs:
    """Everything the daily loop reads, aligned by day and contract. Data are loaded from disk
    unless passed in (tests pass synthetic frames). `end` is the start of the last holding period,
    so the window runs to end + 1, the last row's exit price.

    `eligible` is per_coin_universe.eligibility by calendar day. It must cover every decision day,
    D - lag for D in start .. end (so start - 1 under the one-day lag); the package stores it
    re-indexed by holding day, so `inputs.eligible.at[D, c]` is the eligibility the decision for
    holding period D was allowed to see."""
    if decision_lag_days < 0:
        raise ValueError(f"decision_lag_days must be >= 0, got {decision_lag_days}")
    if not (np.isfinite(capital) and capital > 0):
        raise ValueError(f"capital must be finite and positive, got {capital}")
    contracts = tuple(contracts)
    config = config or load_per_coin_config()
    days = [start + timedelta(days=i) for i in range((end - start).days + 2)]
    scored = days[:-1]
    decision_days = [d - timedelta(days=decision_lag_days) for d in scored]
    missing = [d for d in decision_days if d not in eligible.index]
    if missing:
        raise ValueError(f"eligibility does not cover {len(missing)} decision day(s), first {missing[0]}")
    unknown = [c for c in eligible.columns if c not in contracts]
    if unknown:
        raise ValueError(f"eligibility names contract(s) the ledger was not given: {unknown}")

    bars = bars if bars is not None else load_bars(contracts)
    mark_counts = mark_counts if mark_counts is not None else load_mark_counts(contracts)
    absent = [c for c in contracts if c not in bars]
    if absent:
        raise ValueError(f"no daily bars for {absent}")

    history = [start - timedelta(days=i) for i in range(decision_lag_days, 0, -1)] + days   # D - lag reaches before start
    kline_open = _kline_open_frame(contracts, sorted(set(history)), bars)
    if clock.price == "kline_open":
        fill_prices = kline_open.reindex(days)
    else:
        marks = marks if marks is not None else {c: panel.load_mark_prices(c) for c in contracts}
        fill_prices = _mark_open_frame(contracts, days, marks, clock.hour, kline_open.reindex(days))
    decision_prices, decision_stamps = _decision_price_frame(kline_open, days, decision_lag_days)
    for d in days:
        if decision_stamps[d] > fill_instant(d, clock):
            raise ValueError(f"{d}: decision stamped {decision_stamps[d]} after the fill {fill_instant(d, clock)}")

    tradable = _tradable_frame(contracts, days, bars, mark_counts, config.min_hourly_marks, config.span_break_days)
    events = _validated_register(events, EVENT_COLUMNS, "event")
    suspensions = _validated_register(suspensions, SUSPENSION_COLUMNS, "suspension")
    if settlements is None:
        settlements = load_settlements_registered(contracts, suspensions)

    return LedgerInputs(
        contracts=contracts, days=days, fill_prices=fill_prices, decision_prices=decision_prices,
        decision_stamps=decision_stamps, tradable=tradable,
        eligible=_by_holding_day(eligible, decision_days, scored, contracts),
        settlements=settlements, events=events, suspensions=suspensions, clock=clock,
        decision_lag_days=decision_lag_days, capital=float(capital), cost_bp_per_side=float(cost_bp_per_side))


# ================================================================== step 2: the gatekeeper
def _check_value(day, contract, value):
    if value is HOLD:
        return value
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{day}: target for {contract} must be a finite number of dollars or HOLD, got {value!r}")
    return float(value)


def validate_targets(day: date, targets: dict, held_qty: dict, inputs: LedgerInputs, notes: list | None = None) -> dict:
    """Check and complete one day's targets before anything is booked (design D5-D6, plan v2 §4).

    Returns {contract: target dollars or HOLD} for every contract held or targeted. A held coin with
    no target gets 0 (close it); HOLD with nothing held is 0. Two stages, in this order:

    1. The DECISION must be valid, whatever happens at execution. Raises, naming coin and day: a
       value that is not finite dollars or HOLD; an unknown contract; a non-zero target, or HOLD on a
       position, for a coin not eligible at the decision (`inputs.eligible`, eligibility on D - lag),
       because an exiting coin may only go to 0.
    2. EXECUTION on the fill day. A coin that cannot trade needs an event-register entry, or the run
       stops. On a registered event day the order (decided the day before) is not executable: a held
       coin is held and a not-held coin stays flat, each change noted. Any trade on a tradable day
       needs a finite, positive fill price, or the run stops.
    """
    notes = notes if notes is not None else []
    out = {}
    for contract in sorted(set(held_qty) | set(targets)):
        if contract not in inputs.contracts:
            raise ValueError(f"{day}: target for unknown contract {contract}")
        target = _check_value(day, contract, targets.get(contract, 0.0))
        held = float(held_qty.get(contract, 0.0))
        eligible = bool(inputs.eligible.at[day, contract]) if contract in inputs.eligible.columns else False

        # 1. the decision
        if not eligible:
            if (target is HOLD and held != 0.0) or (target is not HOLD and target != 0.0):
                raise ValueError(f"{day}: {contract} is not eligible at the decision for this holding period; "
                                 f"its only valid target is 0, got {'HOLD' if target is HOLD else target}")
            decided = 0.0
        else:
            decided = 0.0 if (target is HOLD and held == 0.0) else target

        # 2. execution
        wants_trade = decided is not HOLD and not (held == 0.0 and decided == 0.0)
        if not bool(inputs.tradable.at[day, contract]):
            event = event_on(day, contract, inputs.events)
            if (held != 0.0 or wants_trade) and event is None:
                raise ValueError(f"{day}: {contract} cannot trade on its fill day "
                                 f"({'held' if held else 'target ' + repr(decided)}), and no registered event covers it")
            if held != 0.0:
                if decided is not HOLD:
                    notes.append(f"{day}: {contract} target {decided!r} not executable during registered "
                                 f"{event.kind}; position held")
                out[contract] = HOLD
            else:
                if decided != 0.0:
                    notes.append(f"{day}: {contract} target {decided!r} not executable during registered "
                                 f"{event.kind}; stays flat")
                out[contract] = 0.0
            continue
        if wants_trade:
            fill = inputs.fill_prices.at[day, contract]
            if not (np.isfinite(fill) and fill > 0):
                raise ValueError(f"{day}: {contract} has a trade requested but no finite, positive fill price ({fill!r})")
        out[contract] = decided
    return out


# ================================================================== step 3: the loop on normal days
STEP_7_8 = "halts and settlements are built in plan steps 7-8"


@dataclass(frozen=True)
class PerCoinLedger:
    rows: pd.DataFrame                # one row per coin per day: eligible, held or traded
    notes: tuple                      # execution notes from validate_targets (registered event days)
    daily: pd.DataFrame | None = None         # step 4
    fingerprint: str | None = None            # step 4
    inputs_fingerprint: str | None = None     # step 4


def _side(quantity: float) -> str:
    return "long" if quantity > 0 else "short" if quantity < 0 else "flat"


def _row(day, contract, inputs, *, event, quantity_before, notional_before, trade, quantity_held, fill, exit_,
         price_pnl, funding_pnl, fee, slippage, window) -> dict:
    """One ledger row. Ordinary rows (steps 3-6) carry no closing leg: close_trade = close_fee = 0 and
    quantity_after = quantity_held. Settlement and resume-close rows (steps 7-8) will set them."""
    return {
        "day": day, "contract": contract, "event": event,
        "eligible": bool(inputs.eligible.at[day, contract]) if contract in inputs.eligible.columns else False,
        "side": _side(quantity_held), "quantity_before": quantity_before, "notional_before": notional_before,
        "trade": trade, "quantity_held": quantity_held, "close_trade": 0.0, "quantity_after": quantity_held,
        "notional_after": quantity_held * fill if np.isfinite(fill) else 0.0,
        "fill_price": fill, "exit_price": exit_, "price_return": exit_ / fill - 1.0 if fill > 0 else np.nan,
        "price_pnl": price_pnl, "funding_pnl": funding_pnl, "fee": fee, "close_fee": 0.0, "slippage": slippage,
        "net_pnl": price_pnl + funding_pnl - fee - slippage, "capital_base": inputs.capital, "flags": "",
        "funding_start": window[0], "funding_end": window[1],
    }


def trade_row(day, next_day, contract, qty_before, target, inputs, stamps, rates) -> dict:
    """Trade from qty_before to the target dollars at the fill price, then hold to the next fill.
    The arithmetic is the sealed `leg_row`: trade, fee and slippage on |trade|, price P&L of the new
    position to the next fill price, funding on every settlement in (this fill, next fill]."""
    fill = float(inputs.fill_prices.at[day, contract])
    exit_ = float(inputs.fill_prices.at[next_day, contract])
    if not np.isfinite(exit_):
        if target != 0.0:
            raise NotImplementedError(f"{day}: {contract} held into {next_day}, which has no fill price; {STEP_7_8}")
        exit_ = fill                                     # a closing leg: the next price is irrelevant
    leg = leg_row(contract, inputs.settlements, qty_before, target, fill, exit_, *stamps,
                  fee_rate=rates[0], slippage_rate=rates[1])
    return _row(day, contract, inputs, event="trade", quantity_before=qty_before,
                notional_before=leg["notional_before"], trade=leg["trade"], quantity_held=leg["quantity_after"],
                fill=fill, exit_=exit_, price_pnl=leg["price_pnl"], funding_pnl=leg["funding_pnl"],
                fee=leg["fee"], slippage=leg["slippage"], window=stamps)


def hold_row(day, next_day, contract, qty, inputs, stamps) -> dict:
    """HOLD: the quantity is kept exactly (no trade, no cost); the notional drifts with price. Same
    price-P&L and funding formulas as `leg_row`, without a round trip through dollars, so the
    quantity carried forward is bit-for-bit the quantity held (R2)."""
    fill = float(inputs.fill_prices.at[day, contract])
    exit_ = float(inputs.fill_prices.at[next_day, contract])
    if not np.isfinite(exit_):
        raise NotImplementedError(f"{day}: {contract} held into {next_day}, which has no fill price; {STEP_7_8}")
    funding = funding_over(inputs.settlements, contract, *stamps, qty)
    return _row(day, contract, inputs, event="hold", quantity_before=qty, notional_before=qty * fill, trade=0.0,
                quantity_held=qty, fill=fill, exit_=exit_, price_pnl=qty * (exit_ - fill), funding_pnl=funding,
                fee=0.0, slippage=0.0, window=stamps)


def flat_row(day, next_day, contract, inputs, stamps) -> dict:
    """An eligible coin with no position before or after: zero P&L, but its price return is kept so the
    market / coin-specific split averages over every eligible coin (design D9).

    A missing fill or exit price would silently drop an eligible coin from that average, so it stops
    the run, unless the event register covers the day without a price (`day` or `next_day`). Then
    the row is kept with a blank return and flagged `excluded_no_price:<kind>`, an explicit
    treatment the attribution must count and report."""
    fill = float(inputs.fill_prices.at[day, contract])
    exit_ = float(inputs.fill_prices.at[next_day, contract])
    out = _row(day, contract, inputs, event="flat", quantity_before=0.0, notional_before=0.0, trade=0.0,
               quantity_held=0.0, fill=fill, exit_=exit_, price_pnl=0.0, funding_pnl=0.0, fee=0.0, slippage=0.0,
               window=stamps)
    if not (np.isfinite(fill) and fill > 0 and np.isfinite(exit_) and exit_ > 0):
        missing_on = day if not (np.isfinite(fill) and fill > 0) else next_day
        event = event_on(missing_on, contract, inputs.events)
        if event is None:
            raise ValueError(f"{day}: eligible coin {contract} has no price on {missing_on}, and no registered event "
                             f"covers it; its return would silently drop out of the attribution")
        out["price_return"], out["flags"] = np.nan, f"excluded_no_price:{event.kind}"
    return out


def run_per_coin_ledger(target_fn, inputs: LedgerInputs) -> PerCoinLedger:
    """The daily loop (plan §5). Each day: ask target_fn for targets with decision-time information
    only (held quantities and the decision prices stamped at `inputs.decision_stamps[day]`, never the
    fill price), validate them, then book one row for every coin that is held, targeted or eligible.
    Step 3 books trade, hold and flat rows; a held coin meeting a registered event, or held into a day
    with no fill price, stops with NotImplementedError until steps 7-8."""
    rates = cost_rates(inputs.cost_bp_per_side)
    held, rows, notes = {}, [], []
    for day, next_day in zip(inputs.days[:-1], inputs.days[1:]):
        stamps = (fill_instant(day, inputs.clock), fill_instant(next_day, inputs.clock))
        decision = {c: float(v) for c, v in inputs.decision_prices.loc[day].items()}
        raw = target_fn(day, dict(held), decision, inputs.capital)
        targets = validate_targets(day, raw, held, inputs, notes)
        eligible_today = {c for c in inputs.contracts if bool(inputs.eligible.at[day, c])}
        for contract in sorted(set(held) | set(targets) | eligible_today):
            qty = held.get(contract, 0.0)
            target = targets.get(contract, 0.0)
            if qty != 0.0 and event_on(day, contract, inputs.events) is not None:
                raise NotImplementedError(f"{day}: {contract} is held on a registered event day; {STEP_7_8}")
            if qty == 0.0 and (target is HOLD or target == 0.0):
                if contract not in eligible_today:
                    continue
                row = flat_row(day, next_day, contract, inputs, stamps)
            elif target is HOLD:
                row = hold_row(day, next_day, contract, qty, inputs, stamps)
            else:
                row = trade_row(day, next_day, contract, qty, target, inputs, stamps, rates)
            rows.append(row)
            if row["quantity_after"] == 0.0:
                held.pop(contract, None)
            else:
                held[contract] = row["quantity_after"]
    rows = pd.DataFrame(rows, columns=ROW_COLUMNS)
    daily = daily_table(rows, inputs)
    reconcile(rows, daily, inputs)
    return PerCoinLedger(rows=rows, notes=tuple(notes), daily=daily, fingerprint=fingerprint(rows),
                         inputs_fingerprint=inputs_fingerprint(inputs))


# ================================================================== step 4: daily table, reconciliations, fingerprints
ROW_COLUMNS = ("day", "contract", "event", "eligible", "side", "quantity_before", "notional_before", "trade",
               "quantity_held", "close_trade", "quantity_after", "notional_after", "fill_price", "exit_price",
               "price_return", "price_pnl", "funding_pnl", "fee", "close_fee", "slippage", "net_pnl",
               "capital_base", "flags", "funding_start", "funding_end")
TOL = 1e-9                     # dollars; every check is relative to the row's own size where that matters


def _turnover_causes(rows: pd.DataFrame, inputs: LedgerInputs) -> pd.DataFrame:
    """Per row: |trade| split into membership (a coin entering, or leaving because it is no longer
    eligible) and signal (everything else), plus |close_trade| (instrument ends, steps 7-8)."""
    prev = {d: p for p, d in zip(inputs.days[:-2], inputs.days[1:-1])}
    was_eligible = [bool(inputs.eligible.at[prev[d], c]) if d in prev and c in inputs.eligible.columns else False
                    for d, c in zip(rows.day, rows.contract)]
    traded = rows.trade.abs()
    exiting = ~rows.eligible.astype(bool)
    entering = (rows.quantity_before == 0.0) & ~pd.Series(was_eligible, index=rows.index) & rows.eligible.astype(bool)
    membership = traded.where(exiting | entering, 0.0)
    return pd.DataFrame({"membership": membership, "signal": traded - membership,
                         "instrument_end": rows.close_trade.abs()})


def daily_table(rows: pd.DataFrame, inputs: LedgerInputs) -> pd.DataFrame:
    """One row per scored day, flat days included (a day with no position has zero P&L and is never
    dropped: dropping quiet days would flatter the Sharpe). Returns are on the fixed capital base."""
    days = inputs.days[:-1]
    cap = inputs.capital
    r = rows.copy()
    held_notional = (r.quantity_held * r.fill_price.where(r.quantity_held != 0.0, 0.0)).fillna(0.0)
    r["gross_notional"] = held_notional.abs()
    r["net_notional"] = held_notional
    r["funding_long"] = r.funding_pnl.where(r.quantity_held > 0, 0.0)
    r["funding_short"] = r.funding_pnl.where(r.quantity_held < 0, 0.0)
    r["n_long"] = (r.quantity_held > 0).astype(int)
    r["n_short"] = (r.quantity_held < 0).astype(int)
    r["n_eligible_flat"] = (r.event == "flat").astype(int)
    causes = _turnover_causes(r, inputs) if len(r) else pd.DataFrame(columns=["membership", "signal", "instrument_end"])
    for k in causes.columns:
        r[f"turnover_{k}"] = causes[k] / cap
    sums = ["price_pnl", "funding_pnl", "funding_long", "funding_short", "fee", "slippage", "net_pnl",
            "gross_notional", "net_notional", "n_long", "n_short", "n_eligible_flat",
            "turnover_membership", "turnover_signal", "turnover_instrument_end"]
    daily = r.groupby("day")[sums].sum().reindex(days, fill_value=0.0)
    daily.index.name = "day"
    daily["turnover"] = daily[["turnover_membership", "turnover_signal", "turnover_instrument_end"]].sum(axis=1)
    daily["gross_exposure"] = daily.pop("gross_notional") / cap
    daily["net_exposure"] = daily.pop("net_notional") / cap
    daily["capital_base"] = cap
    daily["price_return"] = daily.price_pnl / cap
    daily["gross_return"] = (daily.price_pnl + daily.funding_pnl) / cap
    daily["net_return"] = daily.net_pnl / cap
    last = days[-1]
    terminal = r[(r.day == last) & (r.quantity_after != 0.0)]
    daily.attrs["terminal_positions"] = {c: float(q) for c, q in zip(terminal.contract, terminal.quantity_after)}
    return daily


def _fail(check, why, rows, mask):
    first = rows[mask].iloc[0]
    raise ValueError(f"reconcile {check}: {why}: {int(mask.sum())} row(s), first {first.contract} on {first.day}")


def _close(a, b, scale=None):
    scale = np.maximum(1.0, np.abs(scale if scale is not None else b))
    return np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) <= TOL * scale


def _utc_ns(values) -> np.ndarray:
    """Instants as int64 UTC nanoseconds (a naive stamp is taken as UTC)."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(values), utc=True))
    return idx.asi8


def _recomputed_funding(rows: pd.DataFrame, settlements: dict) -> np.ndarray:
    """Funding recomputed from the settlement tables alone: -quantity x sum(mark x rate) over every
    settlement in (funding_start, funding_end]. A cumulative sum and two binary searches per row,
    sharing no code with funding_over / settled_window."""
    out = np.zeros(len(rows))
    for contract, idx in rows.groupby("contract").indices.items():
        sub = rows.iloc[idx]
        live = sub.quantity_held.to_numpy() != 0.0
        if not live.any():
            continue
        s = settlements[contract].sort_values("stamp")
        stamps = _utc_ns(s.stamp)
        cum = np.concatenate([[0.0], np.cumsum((s.mark * s.rate).to_numpy(dtype=float))])
        a = np.searchsorted(stamps, _utc_ns(sub.funding_start)[live], side="right")
        b = np.searchsorted(stamps, _utc_ns(sub.funding_end)[live], side="right")
        values = np.zeros(len(sub))
        values[live] = -sub.quantity_held.to_numpy()[live] * (cum[b] - cum[a])
        out[idx] = values
    return out


def reconcile(rows: pd.DataFrame, daily: pd.DataFrame, inputs: LedgerInputs) -> None:
    """The eight mandatory checks (design §3, plan §7). Raises on the first breach, naming the check,
    the contract and the day. The row identity (R1) alone would pass if P&L were duplicated
    consistently; R7 recomputes price P&L and funding independently to catch that."""
    if list(daily.index) != list(inputs.days[:-1]):
        raise ValueError("reconcile: the daily table does not cover every scored day exactly")
    if rows.empty:
        return
    held = rows.quantity_held.to_numpy()
    # R1 identity
    ok = _close(rows.net_pnl, rows.price_pnl + rows.funding_pnl - rows.fee - rows.slippage)
    if not ok.all():
        _fail("R1", "net != price + funding - fee - slippage", rows, ~ok)
    # R2 quantity continuity, per contract, flat gaps included
    day_index = {d: i for i, d in enumerate(inputs.days)}
    for contract, sub in rows.groupby("contract", sort=False):
        sub = sub.sort_values("day")
        idx = sub.day.map(day_index).to_numpy()
        before, after = sub.quantity_before.to_numpy(), sub.quantity_after.to_numpy()
        consecutive = np.diff(idx) == 1
        bad = np.concatenate([[before[0] != 0.0],
                              np.where(consecutive, before[1:] != after[:-1], (before[1:] != 0.0) | (after[:-1] != 0.0))])
        if bad.any():
            _fail("R2", "quantity entering a day differs from the quantity left the day before", sub, bad)
    # R3 trade to position
    fill = rows.fill_price.to_numpy()
    moved = (held != 0.0) | (rows.trade.to_numpy() != 0.0) | (rows.quantity_before.to_numpy() != 0.0)
    ok = ~moved | (_close(held * fill, rows.notional_before + rows.trade, rows.notional_before.abs() + rows.trade.abs())
                   & _close(rows.notional_before, rows.quantity_before * fill, rows.notional_before))
    closing = rows.event.isin(["settlement", "resume_close"]).to_numpy()
    ok &= np.where(closing, (rows.quantity_after.to_numpy() == 0.0)
                   & _close(rows.close_trade, -held * rows.exit_price.to_numpy(), rows.close_trade),
                   rows.quantity_after.to_numpy() == held)
    if not ok.all():
        _fail("R3", "trade, quantity and notional disagree", rows, ~ok)
    # R4 daily totals
    sums = rows.groupby("day")[["price_pnl", "funding_pnl", "fee", "slippage", "net_pnl"]].sum()
    sums = sums.reindex(daily.index, fill_value=0.0)
    for col in sums.columns:
        ok = _close(daily[col], sums[col], sums[col])
        if not ok.all():
            raise ValueError(f"reconcile R4: daily {col} differs from its rows on {daily.index[~ok][0]}")
    # R5 turnover
    total = (rows.trade.abs().sum() + rows.close_trade.abs().sum()) / inputs.capital
    parts = daily[["turnover_membership", "turnover_signal", "turnover_instrument_end"]].sum(axis=1)
    if not (_close(daily.turnover.sum(), total, total) and _close(daily.turnover, parts, parts).all()):
        raise ValueError("reconcile R5: turnover does not equal total |trade| + |close_trade| over capital, "
                         "or its causes do not add up")
    # R6 funding by side, each side recomputed from signed holdings and the settlement tables
    funding = _recomputed_funding(rows, inputs.settlements)
    flat_paid = (held == 0.0) & (rows.funding_pnl.to_numpy() != 0.0)
    if flat_paid.any():
        _fail("R6", "funding booked on a row with no position held", rows, flat_paid)
    side_ok = rows.side.to_numpy() == np.where(held > 0, "long", np.where(held < 0, "short", "flat"))
    if not side_ok.all():
        _fail("R6", "a row's side disagrees with the sign of its holding", rows, ~side_ok)
    for side, mask in (("long", held > 0), ("short", held < 0)):
        expect = pd.Series(np.where(mask, funding, 0.0), index=rows.index).groupby(rows.day.to_numpy()).sum()
        expect = expect.reindex(daily.index, fill_value=0.0)
        ok = _close(daily[f"funding_{side}"], expect, expect)
        if not ok.all():
            raise ValueError(f"reconcile R6: daily {side} funding differs from its recomputation from signed "
                             f"holdings on {daily.index[~ok][0]}")
    # R7 independent recomputation, from the source tables, never from the row's own copies
    src = inputs.fill_prices.to_numpy(dtype=float)
    r_i = inputs.fill_prices.index.get_indexer(rows.day)
    c_i = inputs.fill_prices.columns.get_indexer(rows.contract)
    if (r_i < 0).any() or (c_i < 0).any() or (r_i + 1 >= len(inputs.fill_prices)).any():
        raise ValueError("reconcile R7: a row's day or contract is outside the input price table")
    fill_src, exit_src = src[r_i, c_i], src[r_i + 1, c_i]
    ordinary = rows.event.isin(["trade", "hold", "flat"]).to_numpy()
    if not ordinary.all():
        raise ValueError("reconcile R7: halt and settlement rows are not reconcilable until plan steps 7-8")
    same = lambda a, b: (a == b) | (np.isnan(a) & np.isnan(b))            # noqa: E731
    copied = same(rows.fill_price.to_numpy(dtype=float), fill_src) & (
        (held == 0.0) | same(rows.exit_price.to_numpy(dtype=float), exit_src))
    if not copied.all():
        _fail("R7", "a row's fill or exit price differs from the input price table", rows, ~copied)
    price = np.where(held != 0.0, held * (exit_src - fill_src), 0.0)
    ok = _close(rows.price_pnl, price, price)
    if not ok.all():
        _fail("R7", "price P&L differs from quantity_held x (exit - fill) recomputed from the input price table",
              rows, ~ok)
    ok = _close(rows.funding_pnl, funding, funding)
    if not ok.all():
        _fail("R7", "funding differs from its independent recomputation from the settlement tables", rows, ~ok)
    # R8 terminal positions
    last, exit_day = inputs.days[-2], inputs.days[-1]
    terminal = rows[(rows.day == last) & (rows.quantity_after != 0.0)]
    listed = daily.attrs.get("terminal_positions", {})
    if set(terminal.contract) != set(listed) or any(listed[c] != q for c, q in zip(terminal.contract, terminal.quantity_after)):
        raise ValueError("reconcile R8: terminal positions are not exactly those held after the last scored day")
    marked = terminal.exit_price.to_numpy() == inputs.fill_prices.loc[exit_day, list(terminal.contract)].to_numpy()
    if not marked.all():
        _fail("R8", f"a terminal position is not marked at the {exit_day} fill price", terminal, ~marked)


def _digest(*frames) -> str:
    h = hashlib.sha256()
    for f in frames:
        h.update(pd.util.hash_pandas_object(f, index=True).to_numpy().tobytes())
    return h.hexdigest()


def fingerprint(rows: pd.DataFrame) -> str:
    """sha256 of the row table, as strategy_runner fingerprints the sealed ledger."""
    return _digest(rows.astype({"day": str, "funding_start": str, "funding_end": str}))


def inputs_fingerprint(inputs: LedgerInputs) -> str:
    settings = pd.Series({"clock": f"{inputs.clock.hour}/{inputs.clock.price}", "lag": inputs.decision_lag_days,
                          "capital": inputs.capital, "cost": inputs.cost_bp_per_side,
                          "contracts": ",".join(inputs.contracts)}).astype(str)
    frames = [inputs.fill_prices, inputs.decision_prices, inputs.tradable.astype(int), inputs.eligible.astype(int),
              settings, inputs.events.astype(str), inputs.suspensions.astype(str)]
    frames += [inputs.settlements[c].astype({"stamp": str}) for c in sorted(inputs.settlements)]
    return _digest(*[f.reset_index(drop=False).astype(str) if isinstance(f, pd.DataFrame) else f for f in frames])
