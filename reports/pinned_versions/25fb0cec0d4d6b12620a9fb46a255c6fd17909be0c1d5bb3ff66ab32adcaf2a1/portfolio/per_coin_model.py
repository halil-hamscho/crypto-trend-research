"""The per-coin portfolio model: decision tables and the daily target_fn the ledger calls.

Design: docs/october/per_coin_portfolio_model_design_2026-10-19.md. Sizing arithmetic:
portfolio/per_coin_sizing.py. The model reads only decision-time information: for holding day D
it uses the decision day X = D - lag, and every table row for X is built from data stamped at or
before X 00:00 (tested by rebuilding the tables from truncated data).

Tables, per decision day X and contract (each from the contract's current traded span only):
  forecast, sigma  the instrument forecast and floored annual vol of the span; on a day the coin did
                   not trade, the latest earlier row of the same span (design P3), carried to the day
                   before the next span begins, so a row never depends on later trading
  price            the daily open on traded days, NaN otherwise: what the ledger passes as the
                   decision price
  mark_price       the latest traded open at or before X of the span in force, for marking a held
                   coin whose decision price is missing. It runs on through a pause after the span's
                   last trade (a held position still needs a mark), up to the day before the next
                   span begins, and is never carried into a new span
  rho_bar          mean pairwise correlation of the coins eligible at X, daily open-to-open returns
                   within spans, over the 180 days ending X, pairs with >= 90 shared days; NaN if
                   undefined (the sizing then uses rho = 1, Clarification 3)
Variants (protocol Parts 8 and 14): "own" (the strategy), "k1" (+10), "k3" (historical-mean sign),
"k5" (the sealed basket forecast). K2 is the "own" sizing on a BTC-only eligibility frame with a
+10 forecast ("k1"), so N = 1 and IDM = 1.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from portfolio.per_coin_ledger import HOLD
from portfolio.per_coin_signal import instrument_forecast, load_per_coin_config
from portfolio.per_coin_sizing import dynamic_positions, primary_positions
from portfolio.per_coin_universe import span_prices

VARIANTS = ("own", "k1", "k3", "k5")
SIZINGS = ("primary", "dynamic")


@dataclass(frozen=True)
class DecisionTables:
    days: list
    forecast: pd.DataFrame
    sigma: pd.DataFrame
    price: pd.DataFrame
    mark_price: pd.DataFrame
    eligible: pd.DataFrame
    rho_bar: pd.Series
    variant: str


def _span_forecast(prices: pd.Series, config, variant: str, basket_forecast) -> tuple:
    fc = instrument_forecast(prices, config)
    available = fc.forecast.notna()                              # >= 64 traded days in the span (S5)
    if variant == "own":
        f = fc.forecast
    elif variant == "k1":
        f = pd.Series(np.where(available, 10.0, np.nan), index=fc.index)
    elif variant == "k3":
        mean = prices.pct_change().expanding().mean()
        f = pd.Series(np.where(available, 10.0 * np.sign(mean.to_numpy()), np.nan), index=fc.index)
    else:                                                         # k5: one basket forecast for every coin
        if basket_forecast is None:
            raise ValueError("variant k5 needs the sealed basket forecast series")
        f = pd.Series([float(basket_forecast.get(d, np.nan)) if a else np.nan
                       for d, a in zip(fc.index, available)], index=fc.index)
    return f, fc.vol_annual


def _calendar(series: pd.Series, first: date, last: date, fill: bool) -> pd.Series:
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    out = series.reindex(days)
    return out.ffill() if fill else out


def build_decision_tables(bars: dict, eligible: pd.DataFrame, config=None, variant: str = "own",
                          basket_forecast=None) -> DecisionTables:
    """Precompute every decision-day input. `eligible` is per_coin_universe.eligibility by calendar
    day (decision-time, Clarification 1); its index gives the decision days and its columns the
    contracts. `bars` maps contract to daily klines."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    config = config or load_per_coin_config()
    days = list(eligible.index)
    contracts = list(eligible.columns)
    frames = {k: pd.DataFrame(np.nan, index=days, columns=contracts)
              for k in ("forecast", "sigma", "price", "mark_price")}
    returns = {}
    for c in contracts:
        pieces = []
        spans = span_prices(bars[c], config.span_break_days)
        for k, prices in enumerate(spans):
            first, last = prices.index[0], prices.index[-1]
            until = spans[k + 1].index[0] - timedelta(days=1) if k + 1 < len(spans) else max(days[-1], last)
            f, s = _span_forecast(prices, config, variant, basket_forecast)
            # The fallback must not depend on whether the span trades again after X: carry the latest
            # row to the day before the next span (found by the truncation-rebuild test).
            columns = {"forecast": _calendar(f, first, until, True), "sigma": _calendar(s, first, until, True),
                       "price": _calendar(prices, first, last, False),
                       "mark_price": _calendar(prices, first, until, True)}
            for k, v in columns.items():
                hit = v.reindex(days)
                frames[k][c] = frames[k][c].where(hit.isna(), hit.to_numpy())
            pieces.append(_calendar(prices, first, last, False).pct_change(fill_method=None))
        returns[c] = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    returns = pd.DataFrame(returns).sort_index()
    raw = config.raw["risk"]["idm"]
    window, overlap = int(raw["window_days"]), int(raw["min_overlap_days"])
    rho = {}
    for x in days:
        names = [c for c in contracts if bool(eligible.at[x, c])]
        if len(names) < 2:
            rho[x] = np.nan
            continue
        block = returns.reindex(columns=names).loc[x - timedelta(days=window - 1):x]
        corr = block.corr(min_periods=overlap).to_numpy()
        pairs = corr[np.triu_indices(len(names), 1)]
        pairs = pairs[np.isfinite(pairs)]
        rho[x] = float(pairs.mean()) if len(pairs) else np.nan
    return DecisionTables(days=days, forecast=frames["forecast"], sigma=frames["sigma"], price=frames["price"],
                          mark_price=frames["mark_price"], eligible=eligible.astype(bool),
                          rho_bar=pd.Series(rho), variant=variant)


def make_target_fn(tables: DecisionTables, lag: int, sizing: str = "primary", notes: list | None = None,
                   band: float | None = None):
    """The ledger's target_fn (ledger design D5). For holding day D it decides on X = D - lag:
    eligible coins at X get the sizing's positions in dollars of `base`, or HOLD where the buffer keeps
    a held position; a held coin no longer eligible gets 0. `band` is the no-trade half-width: the
    frozen 0.10 when None, 0 for the full-rebalancing sensitivity and its twin.

    Missing prices, explicitly:
      - the ledger's decision prices must equal the model's own table at X (NaN where the coin did not
        trade), or the run stops;
      - a held coin with no price at X is marked at its last traded open within its span (noted), or
        the run stops if there is none;
      - an eligible coin without a finite forecast or a positive vol at X stops the run.
    """
    if sizing not in SIZINGS:
        raise ValueError(f"sizing must be one of {SIZINGS}, got {sizing!r}")
    notes = notes if notes is not None else []

    def positions(f, sigma, rho, h):
        if sizing == "primary":
            return primary_positions(f, sigma, rho, h, band)
        p, _, _, keep = dynamic_positions(f, sigma, rho, h, band)
        return p, keep

    def target_fn(day, held, prices, base):
        x = day - timedelta(days=lag)
        if x not in tables.eligible.index:
            raise ValueError(f"{day}: no decision tables for decision day {x}")
        _check_prices(x, prices)
        names = [c for c in tables.eligible.columns if bool(tables.eligible.at[x, c])]
        f = tables.forecast.loc[x, names].to_numpy(dtype=float)
        sigma = tables.sigma.loc[x, names].to_numpy(dtype=float)
        bad = [c for c, a, b in zip(names, f, sigma) if not (np.isfinite(a) and np.isfinite(b) and b > 0)]
        if bad:
            raise ValueError(f"{day}: eligible coin(s) {bad} without a forecast or volatility at decision day {x}")
        h = np.array([held.get(c, 0.0) * _mark(x, c, day) / base if held.get(c, 0.0) else 0.0 for c in names])
        p, keep = positions(f, sigma, float(tables.rho_bar.get(x, np.nan)), h)
        out = {}
        for c, value, kept, hc in zip(names, p, keep, h):
            out[c] = HOLD if (kept and hc != 0.0) else 0.0 if kept else float(value) * base
        for c in held:
            if c not in out:
                out[c] = 0.0                                          # no longer eligible: exit (design P6)
        return out

    def _check_prices(x, prices):
        mine = tables.price.loc[x]
        for c, given in prices.items():
            if c not in mine.index:
                continue
            ours = float(mine[c])
            if not ((np.isnan(given) and np.isnan(ours)) or given == ours):
                raise ValueError(f"decision prices disagree with the model's tables on {x} for {c}: "
                                 f"ledger {given!r}, model {ours!r}")

    def _mark(x, c, day):
        price = float(tables.price.at[x, c])
        if np.isfinite(price):
            return price
        last = float(tables.mark_price.at[x, c])
        if not np.isfinite(last):
            raise ValueError(f"{day}: held coin {c} has no traded price at or before decision day {x}")
        notes.append(f"{day}: held {c} marked at its last traded open ({last}) before decision day {x}")
        return last

    return target_fn
