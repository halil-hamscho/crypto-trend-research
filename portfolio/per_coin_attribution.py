"""Where the per-coin book's P&L comes from: market versus coin-specific (protocol Part 7), and
whether the book is genuinely two-sided (Part 9 validity check).

For each day, with positions w (share of capital held through the day) and returns r over the coins
eligible at the decision (N of them), W = sum(w) and rbar = mean(r):

    sum(w r) = W x rbar  +  sum((w - W/N)(r - rbar))  +  sum over held-but-ineligible coins of w r
               market       coin-specific                ineligible holdings

an exact identity, applied separately to price (r = the coin's price return over the holding
period) and to funding (r = funding per unit of notional over the same window). Flat eligible coins
take part with w = 0; an eligible coin whose return is missing (flagged excluded_no_price by the
ledger) is excluded and counted. The split is checked against the ledger's own daily totals.
"""
import numpy as np
import pandas as pd

from portfolio.per_coin_ledger import _utc_ns
from portfolio.per_coin_signal import load_per_coin_config

_validity = load_per_coin_config().raw["inference"]["validity"]
TWO_SIDED = {"min_side_share": float(_validity["two_sided_min_side_share_of_gross"]),
             "min_share_of_days": float(_validity["min_share_of_two_sided_days"])}


def split(w: np.ndarray, r: np.ndarray, eligible: np.ndarray) -> tuple:
    """(market, coin_specific, ineligible) for one day and one layer. Exact: they sum to w . r."""
    w, r, eligible = np.asarray(w, float), np.asarray(r, float), np.asarray(eligible, bool)
    out_w, out_r = w[~eligible], r[~eligible]
    ineligible = float(np.where(out_w == 0.0, 0.0, out_w * out_r).sum())
    we, re = w[eligible], r[eligible]
    if len(we) == 0:
        return 0.0, 0.0, ineligible
    total_w, mean_r = float(we.sum()), float(re.mean())
    market = total_w * mean_r
    coin = float(((we - total_w / len(we)) * (re - mean_r)).sum())
    return market, coin, ineligible


def funding_per_unit(rows: pd.DataFrame, settlements: dict) -> np.ndarray:
    """Funding per unit of notional over each row's window: -sum(mark x rate) / fill price. Independent
    of the position, so flat eligible coins have one too. NaN where the fill price is missing."""
    out = np.full(len(rows), np.nan)
    for contract, idx in rows.groupby("contract").indices.items():
        sub = rows.iloc[idx]
        s = settlements[contract].sort_values("stamp")
        stamps = _utc_ns(s.stamp)
        cum = np.concatenate([[0.0], np.cumsum((s.mark * s.rate).to_numpy(dtype=float))])
        a = np.searchsorted(stamps, _utc_ns(sub.funding_start), side="right")
        b = np.searchsorted(stamps, _utc_ns(sub.funding_end), side="right")
        fill = sub.fill_price.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(np.isfinite(fill) & (fill > 0), -(cum[b] - cum[a]) / fill, np.nan)
    return out


def two_sided_days(position_vectors) -> list:
    """A day is two-sided if the book holds longs and shorts, each at least TWO_SIDED["min_side_share"]
    of gross. A flat day is not."""
    out = []
    for p in position_vectors:
        p = np.asarray(p, float)
        gross = float(np.abs(p).sum())
        longs, shorts = float(p[p > 0].sum()), float(-p[p < 0].sum())
        m = TWO_SIDED["min_side_share"] * gross
        out.append(bool(gross > 0 and longs >= m and shorts >= m))
    return out


def validity(two_sided) -> dict:
    """Part 9: below TWO_SIDED["min_share_of_days"] two-sided days, the result is labelled collapsed to
    market timing, whatever H1 says."""
    flags = list(two_sided)
    share = float(np.mean(flags)) if flags else 0.0
    return {"days": len(flags), "two_sided_days": int(sum(flags)), "two_sided_share": share,
            "collapsed_to_market_timing": share < TWO_SIDED["min_share_of_days"]}


def attribute(ledger, inputs) -> pd.DataFrame:
    """One row per scored day (as returns on the capital base): price and funding each split into
    market, coin-specific and ineligible; costs; the coin-specific price term net of ALL trading costs
    (conservative: some trades change W); counts; and the two-sided flag. Raises if the split does not
    reproduce the ledger's daily price or funding P&L."""
    rows, cap = ledger.rows, inputs.capital
    held = rows.quantity_held.to_numpy(dtype=float)
    w = np.where(held != 0.0, held * rows.fill_price.to_numpy(dtype=float) / cap, 0.0)
    r_price = rows.price_return.to_numpy(dtype=float)
    r_fund = funding_per_unit(rows, inputs.settlements)
    excluded = rows["flags"].astype(str).str.startswith("excluded_no_price").to_numpy()
    eligible = rows.eligible.to_numpy(dtype=bool) & ~excluded
    if (excluded & (w != 0.0)).any():
        raise ValueError("attribution: a held position is flagged as excluded for a missing price")
    days = list(ledger.daily.index)
    by_day = rows.groupby("day").indices
    records = []
    for d in days:
        i = by_day.get(d, np.array([], dtype=int))
        pm, pc, pi = split(w[i], r_price[i], eligible[i])
        fm, fc, fi = split(w[i], r_fund[i], eligible[i])
        (two,) = two_sided_days([w[i]])
        records.append({"day": d, "price_market": pm, "price_coin": pc, "price_ineligible": pi,
                        "funding_market": fm, "funding_coin": fc, "funding_ineligible": fi,
                        "n_eligible": int(eligible[i].sum()), "n_excluded": int(excluded[i].sum()),
                        "gross": float(np.abs(w[i]).sum()), "net": float(w[i].sum()), "two_sided": two})
    out = pd.DataFrame(records).set_index("day")
    out["costs"] = ((ledger.daily.fee + ledger.daily.slippage) / cap).to_numpy()
    out["coin_net_of_all_costs"] = out.price_coin - out.costs
    for layer in ("price", "funding"):
        got = out[f"{layer}_market"] + out[f"{layer}_coin"] + out[f"{layer}_ineligible"]
        want = ledger.daily[f"{layer}_pnl"].to_numpy() / cap
        bad = np.abs(got.to_numpy() - want) > 1e-12
        if bad.any():
            raise ValueError(f"attribution: the {layer} split does not reproduce the ledger on {out.index[bad][0]}")
    return out
