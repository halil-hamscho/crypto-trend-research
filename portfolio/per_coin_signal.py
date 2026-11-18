"""The per-coin trend signal: the registered forecast, computed on each instrument's own price.

Protocol: docs/october/per_coin_protocol_2026-10-07.md, Part 3 (S1-S5), amended by Part 14.
Every number comes from docs/october/per_coin_support/frozen_config.yaml; a missing key or a
structure this module does not implement raises. The per-speed forecasts are produced by the same
`basket_forecast` that produced the sealed basket result, so the EWMA, normalisation, fixed scalars,
volatility floor and clipping are identical by construction. What is new here is only the
short-history rule (S5): a speed counts once the instrument has 2 x its slow span of traded days, and
the blend uses the diversification multiplier for the speeds available.
"""
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from portfolio.trend_signal import NORMALISATION, SignalConfig, basket_forecast

PER_COIN_CONFIG = Path(__file__).resolve().parent.parent / "docs/october/per_coin_support/frozen_config.yaml"
SPEED_AVAILABILITY_RULE = "2x_slow_span"


@dataclass(frozen=True)
class PerCoinConfig:
    signal: SignalConfig            # the registered EWMA / vol / scalar parameters, rebuilt from this file
    fdm_by_count: dict              # available speeds -> forecast diversification multiplier
    min_traded_days: int            # no forecast, and not eligible, below this
    span_break_days: int            # > this many calendar days without a trade starts a new instrument
    min_hourly_marks: int           # of 24, for a day to count as tradable
    raw: dict                       # the whole file, for the modules that read the rest
    config_hash: str


def _need(raw: dict, *path):
    node = raw
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"per-coin config is missing {'.'.join(path)}")
        node = node[key]
    return node


def _expect(raw: dict, expected, *path):
    found = _need(raw, *path)
    if found != expected:
        raise ValueError(f"per-coin config {'.'.join(path)} is {found!r}, this module implements {expected!r}")


def load_per_coin_config(path: Path = PER_COIN_CONFIG, raw: dict | None = None) -> PerCoinConfig:
    data = Path(path).read_bytes() if raw is None else b""
    raw = yaml.safe_load(data) if raw is None else raw
    _expect(raw, "sealed_basket_membership", "primary_universe", "source")
    _expect(raw, "ewmac_per_instrument", "signal", "rule")
    _expect(raw, "daily_open_on_traded_days", "signal", "price")
    _expect(raw, True, "signal", "ewma_adjust")
    _expect(raw, NORMALISATION, "signal", "normalisation")
    _expect(raw, "equal_over_available_speeds", "signal", "weights")
    _expect(raw, SPEED_AVAILABILITY_RULE, "signal", "speed_available_after_traded_days")
    _expect(raw, "ewma", "risk", "vol_estimator")
    speeds = tuple(tuple(int(v) for v in pair) for pair in _need(raw, "signal", "speeds"))
    scalars = {k: float(v) for k, v in _need(raw, "signal", "forecast_scalars").items()}
    missing = [f"{f}/{s}" for f, s in speeds if f"{f}/{s}" not in scalars]
    if missing:
        raise ValueError(f"per-coin config has no forecast scalar for speed(s) {missing}")
    per_speed_cap, blend_cap = float(_need(raw, "signal", "per_speed_cap")), float(_need(raw, "signal", "blend_cap"))
    if per_speed_cap != blend_cap:
        raise ValueError("per-speed and blend caps differ; the registered pipeline clips both at the same value")
    fdm_by_count = {int(k): float(v) for k, v in _need(raw, "signal", "fdm_by_available_count").items()}
    if sorted(fdm_by_count) != list(range(1, len(speeds) + 1)):
        raise ValueError(f"fdm_by_available_count must cover 1..{len(speeds)}, got {sorted(fdm_by_count)}")
    min_days = int(_need(raw, "signal", "min_traded_days_for_forecast"))
    if min_days != 2 * min(s for _, s in speeds):
        raise ValueError(f"min_traded_days_for_forecast {min_days} != 2 x the fastest slow span")
    signal = SignalConfig(
        speeds=speeds, scalars=scalars, fdm=fdm_by_count[len(speeds)], cap=blend_cap,
        vol_span=int(_need(raw, "risk", "vol_span_days")),
        floor_quantile=float(_need(raw, "risk", "vol_floor_quantile")),
        floor_window=int(_need(raw, "risk", "vol_floor_window_days")),
        floor_min_days=int(_need(raw, "risk", "vol_floor_min_days")),
        annualisation=int(_need(raw, "risk", "annualisation_days")),
        target_vol=float(_need(raw, "risk", "target_vol")),
        leverage_cap=float(_need(raw, "risk", "caps", "per_instrument_abs")),
        warmup_observations=2 * max(s for _, s in speeds),
        band_half_width=float(_need(raw, "execution", "band_half_width")),
        config_hash=hashlib.sha256(data).hexdigest() if data else "",
    )
    return PerCoinConfig(
        signal=signal, fdm_by_count=fdm_by_count, min_traded_days=min_days,
        span_break_days=int(_need(raw, "universe", "span_break_days_without_trade")),
        min_hourly_marks=int(_need(raw, "universe", "tradable_day", "min_hourly_marks_present")),
        raw=raw, config_hash=signal.config_hash)


def instrument_forecast(prices: pd.Series, config: PerCoinConfig) -> pd.DataFrame:
    """The forecast for one instrument (one contract over one traded span), at every traded day.

    `prices` is the instrument's daily open on its traded days only, oldest first: row i is its
    (i+1)-th traded day. Columns: one `f_<fast>/<slow>` per speed (scaled and clipped, NaN until that
    speed is available), `n_traded`, `n_speeds`, `forecast` (NaN below the minimum history),
    `vol_daily`, `vol_annual`, `standard` (target vol / vol, the risk scale before weights and IDM).
    Each row uses that day's price and earlier ones only.
    """
    c = config.signal
    base = basket_forecast(prices, replace(c, warmup_observations=max(len(prices), 1) + 1))  # S5 replaces the warm-up
    n = np.arange(1, len(prices) + 1)
    out = pd.DataFrame(index=prices.index)
    columns = []
    for fast, slow in c.speeds:
        name = f"f_{fast}/{slow}"
        available = n >= 2 * slow
        values = base[name].where(available)
        if not np.isfinite(values[available]).all():
            bad = values.index[available & ~np.isfinite(values.to_numpy())]
            raise ValueError(f"speed {name} not finite on {len(bad)} available day(s), first {bad[0]}")
        out[name] = values
        columns.append(name)
    out["n_traded"] = n
    out["n_speeds"] = out[columns].notna().sum(axis=1).astype(int)
    fdm = out.n_speeds.map(config.fdm_by_count)
    out["forecast"] = (out[columns].mean(axis=1, skipna=True) * fdm).clip(-c.cap, c.cap)
    out.loc[out.n_speeds == 0, "forecast"] = np.nan
    out["vol_daily"] = base.vol_daily
    out["vol_annual"] = base.vol_annual
    out["standard"] = base.standard
    return out

