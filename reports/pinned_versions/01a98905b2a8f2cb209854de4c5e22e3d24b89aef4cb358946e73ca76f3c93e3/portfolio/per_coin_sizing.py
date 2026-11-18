"""The sizing arithmetic of the per-coin portfolio model, as pure functions over the eligible coins.

Design: docs/october/per_coin_portfolio_model_design_2026-10-19.md §2 (protocol R2-R8), and the
dynamic-risk-targeting sensitivity (Amendment 3, as fixed by Clarification 2). Positions are shares
of capital, signed; arrays are aligned over the coins eligible at the decision. Every parameter is
read from docs/october/per_coin_support/frozen_config.yaml. The daily target_fn (decision tables,
held positions, dollars, HOLD) is built on top of these in a later step.
"""
from dataclasses import dataclass

import numpy as np

from portfolio.per_coin_signal import load_per_coin_config


@dataclass(frozen=True)
class Sizing:
    target_vol: float
    band: float                  # half-width of the no-trade band, as a share of the standard position
    idm_cap: float
    overlay_multiple: float      # the overlay acts above this multiple of the target
    per_instrument_cap: float
    gross_cap: float
    rho_when_undefined: float    # Clarification 3


def _load() -> Sizing:
    raw = load_per_coin_config().raw
    risk, caps = raw["risk"], raw["risk"]["caps"]
    order = raw["dynamic_risk_targeting"]["order"]
    if order != ["unscaled_book", "ex_ante_vol", "multiplier", "ideal", "buffer_primary_band",
                 "per_instrument_cap", "gross_cap"]:
        raise ValueError(f"dynamic_risk_targeting order {order} is not the one this module implements")
    if raw["dynamic_risk_targeting"]["buffer_band"] != "primary":
        raise ValueError("this module implements the primary buffer band for dynamic risk targeting")
    return Sizing(target_vol=float(risk["target_vol"]), band=float(raw["execution"]["band_half_width"]),
                  idm_cap=float(risk["idm"]["cap"]),
                  overlay_multiple=float(risk["overlay"]["trigger_multiple_of_target"]),
                  per_instrument_cap=float(caps["per_instrument_abs"]), gross_cap=float(caps["gross_abs"]),
                  rho_when_undefined=float(risk["idm"]["rho_when_undefined"]))


SIZING = _load()


def _rho(rho_bar: float) -> float:
    """The correlation used everywhere: floored at 0 (R3), and 1 when undefined (Clarification 3),
    the conservative assumption that the coins move as one."""
    return SIZING.rho_when_undefined if not np.isfinite(rho_bar) else max(0.0, float(rho_bar))


def idm(n: int, rho_bar: float) -> float:
    """R3: min(cap, 1/sqrt(1/N + (1 - 1/N) rho)). One coin, or an undefined rho, gives 1, which is
    exactly the rho = 1 value (Clarification 3)."""
    if n <= 1:
        return 1.0
    rho = _rho(rho_bar)
    return float(min(SIZING.idm_cap, 1.0 / np.sqrt(1.0 / n + (1.0 - 1.0 / n) * rho)))


def standard_positions(sigma: np.ndarray, rho_bar: float) -> np.ndarray:
    """The position at a forecast of +10: w x IDM x target / sigma, with w = 1/N (R2, R4)."""
    n = len(sigma)
    if n == 0:
        return np.zeros(0)
    return (1.0 / n) * idm(n, rho_bar) * SIZING.target_vol / np.asarray(sigma, dtype=float)


def ex_ante_vol(p: np.ndarray, sigma: np.ndarray, rho_bar: float) -> float:
    """sqrt(p' S p) with S_ii = sigma_i^2 and S_ij = rho sigma_i sigma_j (R6)."""
    if len(p) == 0:
        return 0.0
    rho = _rho(rho_bar)
    risk = np.asarray(p, dtype=float) * np.asarray(sigma, dtype=float)
    variance = (1.0 - rho) * float(risk @ risk) + rho * float(risk.sum()) ** 2
    return float(np.sqrt(max(variance, 0.0)))


def _band(band) -> float:
    """The no-trade half-width: the frozen 0.10 unless a run spec sets it (0 for full rebalancing)."""
    band = SIZING.band if band is None else float(band)
    if not (np.isfinite(band) and band >= 0.0):
        raise ValueError(f"band must be finite and >= 0, got {band}")
    return band


def buffer_to_edge(held: np.ndarray, ideal: np.ndarray, standard: np.ndarray, band: float | None = None) -> tuple:
    """R5: hold inside ideal +/- band x standard; otherwise trade to the nearer edge. Band 0 is full
    rebalancing: a coin is held only if it already sits exactly at its ideal.
    Returns (positions, held_mask)."""
    half = _band(band) * np.asarray(standard, dtype=float)
    held, ideal = np.asarray(held, dtype=float), np.asarray(ideal, dtype=float)
    inside = np.abs(held - ideal) <= half
    out = np.where(inside, held, np.where(held > ideal + half, ideal + half, ideal - half))
    return out, inside


def _overlay(p, sigma, rho_bar):
    limit = SIZING.overlay_multiple * SIZING.target_vol
    v = ex_ante_vol(p, sigma, rho_bar)
    return p * (limit / v) if v > limit else p


def _caps(p):
    p = np.clip(p, -SIZING.per_instrument_cap, SIZING.per_instrument_cap)
    gross = float(np.abs(p).sum())
    return p * (SIZING.gross_cap / gross) if gross > SIZING.gross_cap else p


def primary_positions(forecast, sigma, rho_bar, held, band: float | None = None) -> tuple:
    """R8, in order: ideal -> buffer -> overlay -> per-coin cap -> gross cap.
    Returns (positions, hold_mask); a coin is HOLD only if the buffer kept it and nothing after
    the buffer moved it."""
    forecast, sigma = np.asarray(forecast, dtype=float), np.asarray(sigma, dtype=float)
    standard = standard_positions(sigma, rho_bar)
    ideal = forecast / 10.0 * standard
    after_buffer, kept = buffer_to_edge(held, ideal, standard, band)
    final = _caps(_overlay(after_buffer, sigma, rho_bar))
    return final, kept & (final == after_buffer)


def dynamic_positions(forecast, sigma, rho_bar, held, band: float | None = None) -> tuple:
    """Dynamic risk targeting (Amendment 3, Clarification 2), in order:
      1. unscaled book u = (f/10) x w x target / sigma;
      2. its ex-ante volatility V(u);
      3. M = target / V(u), unbounded; M = 0 when V(u) = 0 (no forecast, or nothing eligible).
         The target is the IDEAL book's risk; the final book can run below it or be capped;
      4. ideal x* = M u;
      5. buffer with the PRIMARY band (10% of w x IDM x target / sigma), not scaled by M;
      6. per-coin cap, then gross cap. No overlay.
    Returns (positions, ideal, M, hold_mask)."""
    forecast, sigma = np.asarray(forecast, dtype=float), np.asarray(sigma, dtype=float)
    n = len(forecast)
    if n == 0:
        return np.zeros(0), np.zeros(0), 0.0, np.zeros(0, dtype=bool)
    u = forecast / 10.0 * (1.0 / n) * SIZING.target_vol / sigma
    v = ex_ante_vol(u, sigma, rho_bar)
    m = SIZING.target_vol / v if v > 0.0 else 0.0
    ideal = m * u
    after_buffer, kept = buffer_to_edge(held, ideal, standard_positions(sigma, rho_bar), band)
    final = _caps(after_buffer)
    return final, ideal, m, kept & (final == after_buffer)
