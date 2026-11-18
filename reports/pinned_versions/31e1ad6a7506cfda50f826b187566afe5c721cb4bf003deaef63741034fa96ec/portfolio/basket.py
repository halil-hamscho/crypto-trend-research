from dataclasses import dataclass, field
from enum import Enum
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.universe.coinmarketcap.new_fetcher import panel


def load_marks(contracts: list[str]) -> pd.DataFrame:

    series = {}
    for contract in contracts:
        bars = panel.load_ohlcv(contract)
        bars = bars[bars.volume > 0]                 # zombie bars are not prices
        days = bars.timestamp.dt.date                # already a UTC datetime
        series[contract] = pd.Series(bars.open.values, index=days)
    return pd.DataFrame(series).sort_index()


SETTLEMENT_COLUMNS = ("stamp", "rate", "mark", "interval_hours")

# ASSUMED zero, cause unknown (exception recorded in the protocol change log, Part 10, Exception 1).
# On 2026-06-24 the 04:00 funding settlement is absent for nine 4h contracts: 441 settlements
# the day before, 18 that day. The archive, the live fundingRate API and Binance's website
# history all lack the row; trading and marks ran normally through 04:00; no announcement or
# changelog entry explains it. About 70% confident it is an exchange skip; an account-level
# payment that no feed records cannot be excluded. So the ledger charges ZERO for it, flagged,
# and `skipped_funding_sensitivity` reports what a nonzero rate would have meant. The frozen
# config says missing funding must raise: this is the one dated exception, and any other hole
# still raises. Only these nine contracts, only this stamp, and only when the neighbouring
# rows sit exactly one 4h interval either side.
EXCHANGE_SKIPPED = {
    pd.Timestamp("2026-06-24 04:00", tz="UTC"): frozenset({
        "BSVUSDT", "HYPEUSDT", "MUSDT", "POLUSDT", "RAVEUSDT",
        "TIAUSDT", "TRUMPUSDT", "USTCUSDT", "XTZUSDT"}),
}
# Sensitivity rates, not bounds: premium-index reconstructions of the missing 04:00 rate were
# about +5e-5 for most contracts, +3.8e-4 for RAVE and -1.4e-4 for TRUMP. Nothing establishes a maximum.
EXCHANGE_SKIP_SENSITIVITY_RATES = (5e-5, -5e-5, 4e-4, -4e-4)


def _add_exchange_skips(contract: str, settlements: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    """Insert each registered skipped settlement as an explicit, flagged zero-rate row.

    The row makes the chain contiguous and owes nothing (rate 0.0), and carries
    `exchange_skipped=True` so it is visible in every table and count. Nothing is
    inserted unless the contract and stamp are registered AND the rows either side
    are exactly one recorded interval away.
    """
    settlements = settlements.assign(exchange_skipped=False)
    for stamp, names in EXCHANGE_SKIPPED.items():
        if contract not in names or (settlements.stamp == stamp).any() or stamp not in marks.index:
            continue
        before, after = settlements[settlements.stamp < stamp], settlements[settlements.stamp > stamp]
        if before.empty or after.empty:
            continue
        b, a = before.iloc[-1], after.iloc[0]
        interval = pd.Timedelta(hours=b.interval_hours)
        if not (b.interval_hours == a.interval_hours and b.stamp + interval == stamp
                and a.stamp - interval == stamp):
            continue
        row = pd.DataFrame({"stamp": [stamp], "rate": [0.0], "mark": [marks.open.loc[stamp]],
                            "interval_hours": [b.interval_hours], "exchange_skipped": [True]})
        settlements = pd.concat([settlements, row]).sort_values("stamp")
    return settlements.reset_index(drop=True)


def skipped_funding_sensitivity(ledger: pd.DataFrame, settlements: dict,
                                rates=EXCHANGE_SKIP_SENSITIVITY_RATES) -> dict:
    """What the skipped settlements would have added to funding P&L IF the 04:00 rate had been r.

    A conditional sensitivity, not a bound: for every ledger row whose holding period contains a
    registered skipped stamp, the effect is -quantity_after x mark at that stamp x r (the ledger's
    own sign: positive r means longs pay). Returns {rate: dollars}, the ledger itself charging zero.
    """
    totals = {r: 0.0 for r in rates}
    for stamp, names in EXCHANGE_SKIPPED.items():
        start = pd.Timestamp(stamp.date(), tz="UTC")
        held = ledger[(ledger.holding_period_start == start) & ledger.contract.isin(names)]
        for row in held.itertuples():
            table = settlements[row.contract]
            mark = float(table.loc[table.stamp == stamp, "mark"].iloc[0])
            for r in rates:
                totals[r] += -row.quantity_after * mark * r
    return totals


def _read_settlements(contract: str) -> pd.DataFrame:
    """One contract's settlements joined to their marks, clipped to its mark range.

    NOT checked for gaps: load_settlements is the checked door, and this exists so
    a violation can be inspected instead of only raised.
    """
    rows = panel.load_funding(contract)
    if rows.empty:
        raise ValueError(f"{contract}: the funding file has no settlements")
    marks = panel.load_mark_prices(contract)
    stamp = rows.timestamp.dt.floor("h")
    settlements = pd.DataFrame({
        "stamp": stamp,
        "rate": rows.funding_rate.astype(float),
        "mark": marks.open.reindex(stamp).values,
        "interval_hours": rows.funding_interval_hours.astype(float),
    })
    settlements = settlements[(settlements.stamp >= marks.index.min())
                              & (settlements.stamp <= marks.index.max())]
    if settlements.empty:
        raise ValueError(f"{contract}: no funding settlement falls inside its mark range")
    if settlements.mark.isna().any():
        raise ValueError(f"{contract}: settlement without a mark price")
    return _add_exchange_skips(contract, settlements.reset_index(drop=True), marks)


def load_settlements(contracts: list[str]) -> dict[str, pd.DataFrame]:
    """Every funding settlement with its mark price, per contract, or ValueError.

    Columns: stamp (settlement floored to the hour), rate, mark, interval_hours
    (the cadence Binance recorded on that row). One row per settlement, never
    collapsed to a daily total. `rate` is positive when longs pay; the sign is
    applied once, in funding_over.

    A settlement outside [first mark bar, last mark bar] is dropped: before the
    first the contract was not yet tradable (APT 2022-10-19 00:00, ARB 2023-03-23
    08:00); after the last it is the archived tail of a dead contract (LUNA's
    2022-05-13 00:00 row, stamped after the final mark). No position could pay
    either. Inside that range nothing is dropped, filled or skipped: a missing
    mark, a non-finite rate or interval, a non-positive mark, unordered or
    duplicate stamps, or a spacing that matches neither row's recorded interval
    (see funding_gaps) raises. funding_over repeats the completeness check on
    every window a position actually holds.
    """
    out = {}
    for contract in contracts:
        settlements = _read_settlements(contract)
        _require_clean(contract, settlements)
        gaps = funding_gaps(settlements)
        if len(gaps):
            raise ValueError(f"{contract}: {_describe(gaps)}")
        out[contract] = settlements
    return out


def _require_columns(contract, rows):
    missing = [c for c in SETTLEMENT_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"{contract}: settlement table lacks column(s) {missing}")


def _require_clean(contract, rows):
    """Every rate and interval usable, every mark positive, stamps strictly increasing."""
    _require_columns(contract, rows)
    for name, positive in (("rate", False), ("mark", True), ("interval_hours", True)):
        values = rows[name].to_numpy(dtype=float)
        usable = np.isfinite(values) & ((values > 0) if positive else True)
        if not usable.all():
            first = rows["stamp"].iloc[int(np.flatnonzero(~usable)[0])]
            raise ValueError(f"{contract}: {int((~usable).sum())} settlement(s) with a missing or "
                             f"unusable {name}, first at {first}")
    if not (np.diff(rows["stamp"].values) > np.timedelta64(0)).all():
        raise ValueError(f"{contract}: settlement stamps are not strictly increasing")


def _off_interval(stamp, interval):
    """(mask, spacing in hours) per consecutive pair; the mask is True where the spacing
    equals neither row's recorded interval."""
    spacing = np.diff(stamp) / np.timedelta64(1, "h")
    return ~((spacing == interval[1:]) | (spacing == interval[:-1])), spacing


def funding_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    """Consecutive settlements spaced by something other than a recorded interval.

    Each row records the cadence in force at its settlement, and Binance changes
    cadence between rows (8h, 4h, 2h and 1h all occur, in both directions), so
    the spacing of a pair may equal EITHER row's interval. Any other spacing,
    a whole missing settlement included, is returned; nothing is assumed about
    8h. Rows must already be strictly increasing (see _require_clean).
    """
    bad, spacing = _off_interval(rows["stamp"].values, rows["interval_hours"].to_numpy(dtype=float))
    at = np.flatnonzero(bad) + 1                                   # the later row of each bad pair
    return pd.DataFrame({
        "previous": rows["stamp"].iloc[at - 1].to_numpy(), "stamp": rows["stamp"].iloc[at].to_numpy(),
        "spacing_hours": spacing[bad],
        "previous_interval_hours": rows["interval_hours"].iloc[at - 1].to_numpy(),
        "interval_hours": rows["interval_hours"].iloc[at].to_numpy(),
    })


def _describe(gaps: pd.DataFrame) -> str:
    shown = "; ".join(
        f"{pd.Timestamp(g.previous)} -> {pd.Timestamp(g.stamp)} ({g.spacing_hours:g}h after rows "
        f"recording {g.previous_interval_hours:g}h/{g.interval_hours:g}h)" for g in gaps.head(5).itertuples())
    more = f"; +{len(gaps) - 5} more" if len(gaps) > 5 else ""
    return f"{len(gaps)} settlement spacing(s) match neither recorded interval: {shown}{more}"


def settled_window(rows: pd.DataFrame, contract: str, start, end) -> pd.DataFrame:
    """The settlements in (start, end], certified complete, or ValueError.

    Complete means the settlement before `start` (if the table has one) and every
    row up to `end` form an unbroken chain at the recorded intervals, the first
    row is at most one interval after `start` when nothing precedes it, and the
    last row is followed by no settlement that was due by `end`. Holes, NaNs and
    a table that ends early all raise; holes beyond the window are not its concern.
    """
    _require_columns(contract, rows)
    if rows.empty:
        raise ValueError(f"{contract}: no funding settlements on record")
    stamp = rows["stamp"].values                                   # UTC instants, unsorted is allowed here
    t0, t1 = start.to_datetime64(), end.to_datetime64()
    earlier = stamp <= t0
    if earlier.any():
        chain = np.flatnonzero((stamp >= stamp[earlier].max()) & (stamp <= t1))
    else:
        chain = np.flatnonzero(stamp <= t1)
    if chain.size == 0:
        raise ValueError(f"{contract}: no funding settlement on record at or before {end}")
    chain = rows.iloc[chain]
    _require_clean(contract, chain)
    gaps = funding_gaps(chain)
    if len(gaps):
        raise ValueError(f"{contract}: window ({start}, {end}] is incomplete: {_describe(gaps)}")
    first, last = chain.iloc[0], chain.iloc[-1]
    if not earlier.any() and first.stamp - start > pd.Timedelta(hours=first.interval_hours):
        raise ValueError(f"{contract}: funding begins {first.stamp}, more than {first.interval_hours:g}h "
                         f"after the period start {start}")
    if last.stamp + pd.Timedelta(hours=last.interval_hours) <= end:
        raise ValueError(f"{contract}: funding ends {last.stamp}; the next settlement was due by {end}")
    return chain[chain["stamp"].values > t0]


SCORED_START = date(2021, 7, 5)    # the book opens here: first scored holding period
SCORED_END = date(2026, 8, 30)     # LAST holding period STARTS here and exits at open[08-31]
                                   # (Amendment 6: funding on disk ends 2026-08-31 16:00)

FEE_PER_SIDE = 0.0005         # registered 10 bp per side = 5 bp exchange fee (taker)...
SLIPPAGE_PER_SIDE = 0.0005    # ...+ 5 bp spread/impact, stored separately


class State(str, Enum):
    """Each contract-day is in exactly one state."""
    ACTIVE = "active"          # traded today: normal row
    HALTED = "halted"          # no volume, still listed: last valid mark, funding accrues, no trade
    SETTLING = "settling"      # delisting day: realise P&L to settlement price, quantity -> 0
    IDLE = "idle"              # freed allocation, unallocated until reconstitution
    REBRANDED = "rebranded"    # successor listed: migrate at its first bar, idle in the gap


@dataclass
class Book:
    """What we hold. Quantities persist; weights are derived, never stored.

    A dead leg's weight is not redistributed to survivors: it accumulates in
    idle_weight until the next reconstitution, which resets it to zero.
    """
    quantities: dict[str, float] = field(default_factory=dict)
    idle_weight: float = 0.0


def target_notionals(book, reference_weights, marks_today, is_reconstitution,
                     exposure, capital, mode):
    """Per-leg target notional. `mode` is 'units' (benchmark) or 'notional' (strategy).

    Weights come from the REFERENCE basket, never from the book's own contents,
    so the strategy can reopen after exposure has been zero.
    """
    if is_reconstitution:
        unpriced = [c for c in reference_weights if np.isnan(marks_today.get(c, np.nan))]
        if unpriced:                                          # classified or raise
            raise ValueError(f"reconstitution member without a mark today: {unpriced}")
        book.idle_weight = 0.0                                # reset, or it strands forever
        return {c: exposure * capital * w for c, w in reference_weights.items()}

    live = {c: q * marks_today[c] for c, q in book.quantities.items()
            if not np.isnan(marks_today.get(c, np.nan))}
    if mode == "units":
        return live                                           # hold; price drift is not a trade

    # notional: `exposure` is the exposure of the ACTIVE allocation, and the idle share
    # (a dead leg's freed weight) is applied here, once, and nowhere else.
    active = 1.0 - book.idle_weight
    if active <= 0.0:                                         # every leg dead: flat until reconstitution
        return {c: 0.0 for c in live}
    target = exposure * capital * active
    if not live:                                              # reopening from flat
        return {c: target * w for c, w in reference_weights.items()
                if not np.isnan(marks_today.get(c, np.nan))}
    _require_one_sign(live)
    weights = {c: reference_weights.get(c, 0.0) for c in live}
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"held legs {sorted(live)} carry no reference-basket weight")
    return {c: target * w / total for c, w in weights.items()}   # reference weights are the authority


def _require_one_sign(notionals):
    """All legs of a basket position share one direction; mixed signs are an unexplained holding."""
    values = [v for v in notionals.values() if v != 0.0]
    if values and min(values) < 0.0 < max(values):
        raise ValueError(f"mixed-sign holdings: {dict(notionals)}")


def held_exposure(book, marks_today, base, is_reconstitution) -> float:
    """Signed exposure of the ACTIVE allocation at today's marks, before trading.

    live notional / base / active fraction, where active = 1 - idle_weight. On a reconstitution
    day the allocation resets to full, so the active fraction is 1. With 20% idle and live
    exposure 0.8 this is 1.0, so a policy answering "no trade" with x = held leaves the
    survivors alone instead of shrinking them to 0.64. Zero when flat or when nothing is active.
    """
    live = {c: q * marks_today[c] for c, q in book.quantities.items()}
    _require_one_sign(live)
    active = 1.0 if is_reconstitution else 1.0 - book.idle_weight
    if active <= 0.0 or not live:
        return 0.0
    return sum(live.values()) / base / active


def funding_over(settlements, contract, start, end, quantity) -> float:
    """Funding P&L for one position over (start, end]. Negative = paid.

    `quantity` is signed (long > 0). `start` and `end` are UTC-aware timestamps.
    The half-open window puts a settlement at exactly `start` in the previous
    row, where the previous position lived. `stamp` is already floored to the
    hour by load_settlements. This is the one place the sign is applied.

    The window must be complete (settled_window) or this raises: a missing,
    NaN or unordered settlement can never become a smaller payment. A zero
    position owes nothing, so there is nothing to certify.
    """
    if quantity == 0:
        return 0.0
    window = settled_window(settlements[contract], contract, start, end)
    return -float((quantity * window["mark"] * window["rate"]).sum())


def cost_rates(cost_bp_per_side: float) -> tuple:
    """(fee, slippage) per side as fractions of traded notional, split half and half. 10 bp is the
    registered primary (5 + 5); the registered stress levels are 5 and 20 bp."""
    if not (np.isfinite(cost_bp_per_side) and cost_bp_per_side >= 0):
        raise ValueError(f"cost_bp_per_side must be finite and non-negative, got {cost_bp_per_side}")
    half = cost_bp_per_side / 2.0 / 10_000.0
    return half, half


def leg_row(contract, settlements, quantity_before, target, mark_today, mark_tomorrow,
            start, end, fee_rate=FEE_PER_SIDE, slippage_rate=SLIPPAGE_PER_SIDE) -> dict:
    """One contract over one holding period [start, end): trade at open, then hold.

    One position per row: the NEW position (quantity_after) earns the price move
    and pays every settlement in (start, end]. The settlement at `start` belongs
    to the previous row. Identity: net = price_pnl + funding_pnl - fee - slippage.
    """
    notional_before = quantity_before * mark_today
    trade = target - notional_before
    fee = abs(trade) * fee_rate
    slippage = abs(trade) * slippage_rate
    quantity_after = target / mark_today

    price_pnl = quantity_after * (mark_tomorrow - mark_today)
    funding_pnl = funding_over(settlements, contract, start, end, quantity_after)
    return {
        "holding_period_start": start, "contract": contract,
        "notional_before": notional_before, "trade": trade, "notional_after": target,
        "quantity_after": quantity_after,
        "price_pnl": price_pnl, "funding_pnl": funding_pnl, "fee": fee, "slippage": slippage,
        "net_pnl": price_pnl + funding_pnl - fee - slippage,
    }


def settle(book, contract, settlement_price, mark_start, settlements,
           period_start, delisting_stamp, weight) -> dict:
    """Delisting: a perp has no principal, so closing REALISES P&L and frees the allocation.

    `quantity` is the position held from `period_start`; any trade that set it at
    that open belongs to the caller's leg row. `weight` is the leg's allocation,
    which goes idle (not to the survivors) until the next reconstitution.
    Exchange fee only: the exchange settles at its own price, so there is no
    spread to cross and no slippage.
    """
    quantity = book.quantities.pop(contract)
    price_pnl = quantity * (settlement_price - mark_start)
    funding_pnl = funding_over(settlements, contract,       # every settlement up to the
                               period_start, delisting_stamp,  # actual delisting time
                               quantity)
    fee = abs(quantity * settlement_price) * FEE_PER_SIDE
    slippage = 0.0
    book.idle_weight += weight
    return {
        "holding_period_start": period_start, "contract": contract, "event": "settlement",
        "price_pnl": price_pnl, "funding_pnl": funding_pnl, "fee": fee, "slippage": slippage,
        "quantity_after": 0.0, "notional_after": 0.0,
        "net_pnl": price_pnl + funding_pnl - fee - slippage,
    }


def reference_basket(members_by_date, marks) -> tuple[pd.Series, pd.DataFrame]:
    """Costless basket: quantities from cap weights, drifting between reconstitutions.

    Returns (index level, weights per day). This is what the signal reads and what
    supplies the weights the books scale; it is not a portfolio and pays nothing.

    The index is chain-linked on the previous BASKET VALUE, not on capital. On a
    reconstitution day the old quantities are first valued at TODAY's marks, and
    the new quantities are built from that value, so the old basket's move since
    yesterday is counted rather than discarded. The starting value is arbitrary
    (only ratios enter the index).

    Every asset held on a day, including one about to be sold at a reconstitution,
    must have a mark that day, or this raises. Leaving an unpriced asset out and
    renormalising the survivors would turn a data hole into a crash and a rebound
    (two flat assets, one missing price: 100 -> 50 -> 100). There is no allowance
    for classified halts or settlements yet: a lifecycle event needs an explicit
    valuation rule, and until one is registered it must fail here, as in run_ledger.
    """
    quantities, levels, weights = {}, {}, {}
    index, previous_value = 100.0, None
    for day in marks.index:
        today = marks.loc[day]
        unpriced = [c for c in quantities if np.isnan(today[c])]
        if unpriced:
            raise ValueError(f"{day}: held contract without a mark (unclassified gap): {unpriced}")
        if day in members_by_date:
            members = members_by_date[day]
            unpriced = [c for c in members.contract if np.isnan(today[c])]
            if unpriced:
                raise ValueError(f"{day}: reconstitution member without a mark: {unpriced}")
            value = (sum(q * today[c] for c, q in quantities.items())
                     if quantities else 1_000_000.0)
            quantities = {row.contract: value * row.weight / today[row.contract]
                          for row in members.itertuples()}
        live = {c: q * today[c] for c, q in quantities.items()}
        value = sum(live.values())
        if previous_value:
            index *= value / previous_value
        levels[day], weights[day] = index, {c: n / value for c, n in live.items()}
        previous_value = value
    return pd.Series(levels), pd.DataFrame(weights).T.fillna(0.0)


def _require_window(marks, start, end):
    """Marks for every day in [start, end + 1], the last one being the final exit price."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    needed = [start + timedelta(days=i) for i in range((end - start).days + 2)]
    present = set(marks.index)
    missing = [d for d in needed if d not in present]
    if missing:
        raise ValueError(f"marks lack {len(missing)} of the {len(needed)} days from start {start} to "
                         f"the final exit {needed[-1]} (end {end}): first missing {[str(d) for d in missing[:5]]}")


CAPITAL_POLICIES = ("fixed", "equity")


def run_ledger(members_by_date, marks, settlements, reference_weights, start, end,
               capital=100_000.0, exposure=1.0, mode="units",
               capital_policy="fixed", exposure_fn=None, cost_bp_per_side=10.0) -> pd.DataFrame:
    """The table everything comes from: one row per contract per holding period.

    Each day D covers [D, D+1): trade at open[D], then the NEW position earns the
    move to open[D+1] and pays every settlement in (D 00:00, D+1 00:00].

    `capital_policy` sets the base that sizes targets AND is the return denominator
    (column `capital_base`; return = net_pnl / capital_base):
      "fixed"  (A, registered primary): `capital` every day. Each reconstitution
               resets exposure to that base, so the benchmark is a passive basket
               with monthly notional resets, not pure buy-and-hold.
      "equity" (B, sensitivity): net equity at the start of the day, i.e. `capital`
               plus every net P&L booked on earlier days (price + funding - costs).
               Perp exposure and equity are different quantities; only equity is
               the base.

    `exposure_fn(day, held_exposure, base) -> float` makes exposure a per-day decision (a
    strategy, in mode="notional"). It is called each day after the marks and gap checks,
    before targets are built, with the exposure of the active allocation currently held
    (see held_exposure; 0.0 on the opening day and when flat) and the day's capital base, and
    returns the target exposure x. Left None, `exposure` is used every day: identical to
    the constant-exposure ledger, which is how the benchmark runs. Nothing after D 00:00 is
    passed in. When no allocation is active (every leg dead) the book stays flat until
    reconstitution and the function is not called. Each row records `held_exposure` and
    `target_exposure`.

    `cost_bp_per_side` is the all-in cost of a trade as basis points of traded notional, split half fee
    and half slippage: 10 is the registered primary, 5 and 20 the registered stress levels. The default
    reproduces the earlier fixed constants bit for bit.

    `start` is the opening event: the book opens on the reference basket's
    drifted weights. Days in `members_by_date` are reconstitutions, which reset
    to fresh cap weights; a leg that is no longer a member is sold at full cost.
    Any held contract without a mark raises: halts, rebrands and delistings are
    classified by the caller or fail loud. So does funding that ends before a
    period does. `end` is the START of the last holding period (SCORED_END). `exposure` is a constant here.

    The window is exact: every day from `start` through `end + 1` (the last
    period's exit price) must be in `marks`, else ValueError. Nothing is
    shortened silently, and a missing opening day cannot turn into an empty book.
    Every leg that holds exposure through a period must have a complete funding
    window (funding_over); a leg sold at the period's open owes none.
    """
    if capital_policy not in CAPITAL_POLICIES:
        raise ValueError(f"capital_policy must be one of {CAPITAL_POLICIES}, got {capital_policy!r}")
    fee_rate, slippage_rate = cost_rates(cost_bp_per_side)
    _require_window(marks, start, end)
    days = list(marks.index)
    book, rows, equity = Book(), [], capital
    for i, day in enumerate(days[:-1]):
        if day < start:
            continue
        if day > end:
            break
        tomorrow = days[i + 1]
        if tomorrow != day + pd.Timedelta(days=1):
            raise ValueError(f"marks skip from {day} to {tomorrow}")
        today, next_marks = marks.loc[day], marks.loc[tomorrow]
        stamps = (pd.Timestamp(day, tz="UTC"), pd.Timestamp(tomorrow, tz="UTC"))

        event = "opening" if day == start else "reconstitution" if day in members_by_date else "hold"
        if event == "reconstitution":
            weights = members_by_date[day].set_index("contract").weight.to_dict()
        else:
            drifted = reference_weights.loc[day]
            weights = drifted[drifted > 0].to_dict()

        gaps = [c for c in book.quantities if np.isnan(today[c])]
        if gaps:
            raise ValueError(f"{day}: held contract without a mark (unclassified gap): {gaps}")

        base = capital if capital_policy == "fixed" else equity
        if base <= 0:
            raise ValueError(f"{day}: capital base {base} is not positive (equity wiped out)")
        held = held_exposure(book, today, base, event != "hold")
        if exposure_fn is None:
            x = exposure
        elif event == "hold" and book.idle_weight >= 1.0:
            x = 0.0                                              # nothing active: flat until reconstitution
        else:
            x = float(exposure_fn(day, held, base))
            if not np.isfinite(x):
                raise ValueError(f"{day}: exposure_fn returned {x}")
        targets = target_notionals(book, weights, today, event != "hold", x, base, mode)
        day_rows = []
        for contract in sorted(set(book.quantities) | set(targets)):
            target = targets.get(contract, 0.0)          # a leg with no target is sold
            quantity_before = book.quantities.get(contract, 0.0)
            if target == 0.0 and quantity_before == 0.0:
                continue
            mark_tomorrow = next_marks[contract]
            if np.isnan(mark_tomorrow):
                if target != 0.0:
                    raise ValueError(f"{day}: {contract} has no mark tomorrow (unclassified gap)")
                mark_tomorrow = today[contract]          # closing leg: price after is irrelevant
            row = leg_row(contract, settlements, quantity_before, target, today[contract],
                          mark_tomorrow, *stamps, fee_rate=fee_rate, slippage_rate=slippage_rate)
            row["event"] = event
            row["capital_base"] = base
            row["held_exposure"] = held
            row["target_exposure"] = x
            day_rows.append(row)
            if row["quantity_after"] == 0.0:
                book.quantities.pop(contract, None)
            else:
                book.quantities[contract] = row["quantity_after"]
        rows += day_rows
        equity += sum(r["net_pnl"] for r in day_rows)
    return pd.DataFrame(rows)


BASKET_START = date(2020, 2, 9)    # first date the volume screen fields ten names; the index starts here
ANNUALISATION = 365                # crypto trades every day


@dataclass
class BenchmarkResult:
    """Everything the benchmark produces: the ledger, the returns built from it, and the audit.

    ledger    one row per contract per holding period (see run_ledger)
    daily     one row per holding period: price/funding/fee/slippage/net P&L, the capital base,
              and each layer as a return on that base (layer / capital_base)
    ladder    price, gross (price + funding) and net: annualised mean, vol and Sharpe
    turnover  annualised, as a share of the capital base: opening, sweep, churn, total
    checks    audit numbers: funding drag, costs, BTC correlation and weight, skip bound
    reference the costless reference index (level) and its daily weights
    """
    ledger: pd.DataFrame
    daily: pd.DataFrame
    ladder: pd.DataFrame
    turnover: dict
    checks: dict
    reference: tuple
    capital_policy: str
    window: tuple


def reconstitution_dates(last: date, first: date = BASKET_START) -> list[date]:
    """The registered schedule: the basket start, then the first calendar day of every month
    through `last`. (Every calendar day is traded in crypto, so none moves.)"""
    months = pd.date_range(pd.Timestamp(first) + pd.offsets.MonthBegin(1), pd.Timestamp(last), freq="MS")
    return [first] + [d.date() for d in months]


INSTRUMENTS = ("basket", "btc")


def build_inputs(end: date = SCORED_END, instrument: str = "basket"):
    """(members_by_date, marks, settlements, reference_weights, reference_index), each checked on load.

    'basket': the registered monthly buffered top-10 membership. 'btc': the H2 comparator, one member
    (BTCUSDT) at weight 1 on the same dates, so the reference index is BTC's own price and everything
    downstream (signal, sizing, costs, funding, resets) runs with identical settings. Marks run one day
    past `end`, the last holding period's exit price.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"instrument must be one of {INSTRUMENTS}, got {instrument!r}")
    if instrument == "btc":
        only = pd.DataFrame({"contract": ["BTCUSDT"], "weight": [1.0]})
        members = {d: only for d in reconstitution_dates(end)}
    else:
        from data.universe.selector import UniverseSelector
        selector = UniverseSelector(min_history_days=0)       # warm-up is the signal's, not membership's
        members = selector.select_buffered(reconstitution_dates(end))
    contracts = sorted({c for m in members.values() for c in m.contract})
    marks = load_marks(contracts).loc[BASKET_START:end + timedelta(days=1)]
    index, weights = reference_basket(members, marks)
    return members, marks, load_settlements(contracts), weights, index


def sharpe(returns: pd.Series) -> float:
    """Registered: annualised arithmetic mean / annualised volatility (365-day year)."""
    return float(returns.mean() * ANNUALISATION / (returns.std() * np.sqrt(ANNUALISATION)))


def turnover_split(ledger: pd.DataFrame) -> dict:
    """Annualised turnover as a share of the capital base, split by cause.

    opening  the one-off first trade, averaged over the window
    sweep    |base - V| at each reconstitution: resizing the drifted basket back to the base
    churn    trading after scaling the old holdings to the base: entries, exits, weight resets
    total    every |trade|
    overlap  sweep + churn - reconstitution trades; the two partly offset (>= 0, triangle bound)

    Under 'fixed' the sweep is the P&L swept out or topped up each month; under 'equity' the base
    already contains the P&L, so it is small.
    """
    years = ledger.holding_period_start.nunique() / ANNUALISATION
    days = ledger.groupby("holding_period_start")
    parts = {"opening": 0.0, "sweep": 0.0, "churn": 0.0, "reconstitution": 0.0, "other": 0.0}
    for _, day in days:
        base, event = day.capital_base.iloc[0], day.event.iloc[0]
        traded = day.trade.abs().sum() / base
        if event == "opening":
            parts["opening"] += traded
        elif event == "reconstitution":
            value = day.notional_before.sum()
            scaled = day.notional_before * (base / value) if value > 0 else day.notional_before * 0.0
            parts["sweep"] += abs(base - value) / base
            parts["churn"] += (day.notional_after - scaled).abs().sum() / base
            parts["reconstitution"] += traded
        else:
            parts["other"] += traded                          # hold days: rounding only
    out = {k: v / years for k, v in parts.items()}
    out["total"] = out["opening"] + out["reconstitution"] + out["other"]
    out["overlap"] = out["sweep"] + out["churn"] - out["reconstitution"]
    return out


def daily_returns(ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per holding period: each layer as P&L and as a return on that day's capital base."""
    daily = ledger.groupby("holding_period_start").agg(
        price=("price_pnl", "sum"), funding=("funding_pnl", "sum"), fee=("fee", "sum"),
        slippage=("slippage", "sum"), net=("net_pnl", "sum"), base=("capital_base", "first"),
        exposure=("notional_after", "sum"))
    daily["cost"] = daily.fee + daily.slippage
    for layer in ("price", "funding", "cost", "net"):
        daily[f"{layer}_return"] = daily[layer] / daily.base
    daily["gross_return"] = daily.price_return + daily.funding_return
    daily.index = pd.DatetimeIndex(daily.index).tz_convert(None).date
    return daily


def ladder(daily: pd.DataFrame) -> pd.DataFrame:
    """Price, gross (price + funding) and net (gross - costs): annualised mean, vol and Sharpe."""
    rows = {}
    for name, column in (("price", "price_return"), ("gross", "gross_return"), ("net", "net_return")):
        r = daily[column]
        rows[name] = {"ann_mean": r.mean() * ANNUALISATION, "ann_vol": r.std() * np.sqrt(ANNUALISATION),
                      "sharpe": sharpe(r)}
    return pd.DataFrame(rows).T


def benchmark(capital_policy: str = "fixed", capital: float = 100_000.0,
              start: date = SCORED_START, end: date = SCORED_END, inputs=None,
              cost_bp_per_side: float = 10.0) -> BenchmarkResult:
    """Buy the same basket and hold it: same machine as the strategy, exposure 1.0, units mode.

    Same funding, same per-leg costs, same membership rule including the 10-in/13-out buffer, same
    treatment of every lifecycle event; a separately written benchmark would differ in a dozen small
    ways and every difference would land in alpha looking like skill. `capital_policy` is 'fixed'
    (A, registered: a passive basket with monthly notional resets) or 'equity' (B, sensitivity).
    `inputs` is build_inputs() output, to reuse one load across runs.
    """
    if capital_policy not in CAPITAL_POLICIES:                 # fail before any loading
        raise ValueError(f"capital_policy must be one of {CAPITAL_POLICIES}, got {capital_policy!r}")
    members, marks, settlements, weights, index = inputs if inputs is not None else build_inputs(end)
    ledger_ = run_ledger({d: m for d, m in members.items() if d > start}, marks, settlements, weights,
                         start, end, capital=capital, exposure=1.0, mode="units",
                         capital_policy=capital_policy, cost_bp_per_side=cost_bp_per_side)
    daily = daily_returns(ledger_)
    btc = (marks["BTCUSDT"].pct_change().reindex(pd.Index(index.index)) if "BTCUSDT" in marks
           else pd.Series(np.nan, index=index.index))                  # synthetic baskets have no BTC
    ref = index.pct_change()
    years = len(daily) / ANNUALISATION
    checks = {
        "days": len(daily), "years": years,
        "funding_drag_per_year": float(-daily.funding_return.sum() / years),
        "costs_per_year": float(daily.cost_return.sum() / years),
        "btc_corr_reference_index": float(ref.corr(btc)),
        "btc_weight_mean": float(weights["BTCUSDT"].mean()) if "BTCUSDT" in weights else float("nan"),
        "identity_max_residual": float((ledger_.net_pnl - (ledger_.price_pnl + ledger_.funding_pnl
                                                           - ledger_.fee - ledger_.slippage)).abs().max()),
        "skip_sensitivity_dollars": skipped_funding_sensitivity(ledger_, settlements),
        "exchange_skipped_rows_in_window": int(sum(s.exchange_skipped.sum() for s in settlements.values())),
    }
    return BenchmarkResult(ledger_, daily, ladder(daily), turnover_split(ledger_), checks,
                           (index, weights), capital_policy, (start, end))
