"""The registered trend signal on the basket index, from the frozen configuration.

One signal on the reference index (never one per coin): four EWMA crossovers normalised by a
floored volatility, fixed Carver scalars, equal weights, a fixed diversification multiplier, and a
volatility-targeted ideal exposure. No parameter has a default: everything comes from
`prereg_support/frozen_config.yaml`, and a missing or changed structure raises.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.volatility import VolatilityEstimator
from strategies.ewma_crossover import EWMA_Trend_Following

FROZEN_CONFIG = Path(__file__).resolve().parent.parent / "docs/september/prereg_support/frozen_config.yaml"
FORECAST_UNIT = 10.0     # Carver's average absolute forecast; fixed in the registered pseudocode, not in the YAML
NORMALISATION = "(fast_ewma - slow_ewma) / (price * daily_return_vol)"
REGISTERED_CAP = 20.0    # BaseStrategy clips each speed at +-20 itself, so only the registered cap is implementable
PIPELINE = ["raw", "normalise", "scale_per_speed", "clip_pm20", "equal_weight_average", "apply_fdm", "clip_pm20"]


@dataclass(frozen=True)
class SignalConfig:
    speeds: tuple
    scalars: dict
    fdm: float
    cap: float
    vol_span: int
    floor_quantile: float
    floor_window: int
    floor_min_days: int
    annualisation: int
    target_vol: float
    leverage_cap: float
    warmup_observations: int
    band_half_width: float
    config_hash: str


def _need(raw: dict, *path):
    node = raw
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"frozen config is missing {'.'.join(path)}")
        node = node[key]
    return node


def _expect(raw: dict, expected, *path):
    found = _need(raw, *path)
    if found != expected:
        raise ValueError(f"frozen config {'.'.join(path)} is {found!r}, this runner implements {expected!r}")


def load_signal_config(path: Path = FROZEN_CONFIG, raw: dict | None = None) -> SignalConfig:
    """Read the registered signal and risk parameters. Missing keys raise; so does any structure
    this runner does not implement (non-equal weights, adjust=False, another pipeline order)."""
    data = Path(path).read_bytes() if raw is None else b""
    raw = yaml.safe_load(data) if raw is None else raw
    _expect(raw, "ewmac_on_basket_index_level", "signal", "rule")
    _expect(raw, "equal", "signal", "weights")
    _expect(raw, True, "signal", "ewma_adjust")
    _expect(raw, True, "risk", "vol_adjust")
    _expect(raw, "ewma", "risk", "vol_estimator")
    _expect(raw, "percentile_of_own_history", "risk", "vol_floor_rule")
    _expect(raw, PIPELINE, "signal", "pipeline_order")
    _expect(raw, NORMALISATION, "signal", "normalisation")
    _expect(raw, REGISTERED_CAP, "signal", "cap")
    _expect(raw, "ideal_exposure", "execution", "band_centre")
    _expect(raw, "trade_to_nearest_edge", "execution", "band_action")
    _expect(raw, True, "execution", "membership_trades_always_execute")
    _expect(raw, ["full_rebalance", "buffered"], "execution", "policies")
    _expect(raw, "buffered", "execution", "primary_policy")
    speeds = tuple(tuple(int(v) for v in pair) for pair in _need(raw, "signal", "speeds"))
    scalars = {k: float(v) for k, v in _need(raw, "signal", "forecast_scalars").items()}
    missing = [f"{f}/{s}" for f, s in speeds if f"{f}/{s}" not in scalars]
    if missing:
        raise ValueError(f"frozen config has no forecast scalar for speed(s) {missing}")
    config = SignalConfig(
        speeds=speeds, scalars=scalars, fdm=float(_need(raw, "signal", "fdm")),
        cap=float(_need(raw, "signal", "cap")), vol_span=int(_need(raw, "risk", "vol_span_days")),
        floor_quantile=float(_need(raw, "risk", "vol_floor_quantile")),
        floor_window=int(_need(raw, "risk", "vol_floor_window_days")),
        floor_min_days=int(_need(raw, "risk", "vol_floor_min_days")),
        annualisation=int(_need(raw, "risk", "annualisation_days")),
        target_vol=float(_need(raw, "risk", "target_vol")), leverage_cap=float(_need(raw, "risk", "leverage_cap")),
        warmup_observations=int(_need(raw, "sample", "warmup_observations")),
        band_half_width=float(_need(raw, "execution", "band_half_width")),
        config_hash=hashlib.sha256(data).hexdigest() if data else "",
    )
    _require_valid_domains(config)
    return config


def _require_valid_domains(c: SignalConfig) -> None:
    """Numerical domains the maths needs: nothing here is a choice, so a violation raises."""
    problems = []
    if len(set(c.speeds)) != len(c.speeds) or not c.speeds:
        problems.append("speeds must be non-empty and unique")
    if any(not (0 < fast < slow) for fast, slow in c.speeds):
        problems.append("every speed needs 0 < fast < slow")
    for name, value in (("fdm", c.fdm), ("cap", c.cap), ("target_vol", c.target_vol),
                        ("leverage_cap", c.leverage_cap)):
        if not (np.isfinite(value) and value > 0):
            problems.append(f"{name} must be finite and positive, got {value}")
    if not all(np.isfinite(v) and v > 0 for v in c.scalars.values()):
        problems.append("forecast scalars must be finite and positive")
    if not (0.0 < c.floor_quantile < 1.0):
        problems.append(f"vol floor quantile must be in (0, 1), got {c.floor_quantile}")
    if not (c.vol_span >= 2 and c.annualisation > 0 and 0 < c.floor_min_days <= c.floor_window):
        problems.append("vol span >= 2, annualisation > 0 and 0 < floor_min_days <= floor_window are required")
    if not (np.isfinite(c.band_half_width) and c.band_half_width >= 0):
        problems.append(f"band half-width must be finite and non-negative, got {c.band_half_width}")
    if c.warmup_observations < 1:
        problems.append("warm-up observations must be positive")
    if problems:
        raise ValueError("frozen config has invalid values: " + "; ".join(problems))


def basket_forecast(index: pd.Series, config: SignalConfig) -> pd.DataFrame:
    """The signal at every date, from the index through that date and nothing later.

    Columns: vol_daily, vol_annual (the one floored series used for both the forecast's
    normalisation and the sizing), one `f_<fast>/<slow>` per speed, `forecast`, `standard`
    (target vol / vol), `x_star` (ideal exposure, a signed multiple of capital), and the cap flags.
    `forecast` and `x_star` are NaN until the 512th observation. From then on every speed must be
    finite or this raises: a NaN is never read as zero.
    """
    values = index.to_numpy(dtype=float)
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError("index must be finite and positive on every date (no NaN, infinity, zero or negative)")
    if not index.index.is_monotonic_increasing or not index.index.is_unique:
        raise ValueError("index must be strictly increasing and unique by date")
    # BaseStrategy wants a DatetimeIndex; the reference index is keyed by date. Compute on the
    # datetime view and hand every column back on the caller's own index, positionally.
    close = pd.DataFrame({"close": index.to_numpy(dtype=float)}, index=pd.DatetimeIndex(pd.to_datetime(list(index.index))))
    estimator = VolatilityEstimator(span=config.vol_span, annualization_factor=config.annualisation,
                                    floor_quantile=config.floor_quantile, floor_window=config.floor_window,
                                    floor_min_periods=config.floor_min_days)
    out = pd.DataFrame(index=index.index)
    vol = estimator.daily(close["close"])
    out["vol_daily"] = vol.to_numpy()
    out["vol_annual"] = out.vol_daily * np.sqrt(config.annualisation)
    out["vol_floor_bound"] = out.vol_daily.to_numpy() > close["close"].pct_change().ewm(
        span=config.vol_span, adjust=True).std().to_numpy()                      # the floor lifted the estimate

    for fast, slow in config.speeds:
        strategy = EWMA_Trend_Following(
            vol_span=config.vol_span, annualization_factor=config.annualisation, fast=fast, slow=slow,
            forecast_scalar=config.scalars[f"{fast}/{slow}"], floor_quantile=config.floor_quantile,
            floor_window=config.floor_window, floor_min_periods=config.floor_min_days)
        pd.testing.assert_series_equal(strategy.volatility_estimator.daily(close["close"]), vol,
                                       check_names=False)       # one vol series, everywhere
        out[f"f_{fast}/{slow}"] = strategy.run(close.copy())["signal"].to_numpy()   # fixed scalar, clipped

    speed_columns = [f"f_{f}/{s}" for f, s in config.speeds]
    out["forecast"] = (out[speed_columns].mean(axis=1, skipna=False) * config.fdm).clip(-config.cap, config.cap)
    out["standard"] = config.target_vol / out.vol_annual
    out["x_star"] = (out.forecast / FORECAST_UNIT * out.standard).clip(-config.leverage_cap, config.leverage_cap)
    out["forecast_cap_hit"] = out.forecast.abs() >= config.cap
    out["x_star_cap_hit"] = (out.forecast / FORECAST_UNIT * out.standard).abs() >= config.leverage_cap

    warm = np.arange(len(out)) < config.warmup_observations - 1          # observations 1..511 are warm-up
    out.loc[warm, ["forecast", "x_star"]] = np.nan
    out.loc[warm, ["forecast_cap_hit", "x_star_cap_hit"]] = False
    scored = out.loc[~warm, speed_columns + ["forecast", "vol_daily", "standard", "x_star"]]
    bad = scored.index[~np.isfinite(scored.to_numpy(dtype=float)).all(axis=1)]
    if len(bad):
        raise ValueError(f"signal not finite on {len(bad)} scored day(s), first {bad[0]}")
    return out
