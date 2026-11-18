"""Run from the repo root: python -m portfolio.test_per_coin_sizing

The sizing arithmetic of the per-coin portfolio model (design §2, protocol R2-R8) and of the
dynamic-risk-targeting sensitivity (Amendment 3, Clarification 2). Synthetic numbers only.
"""
import numpy as np

from portfolio.per_coin_sizing import (SIZING, buffer_to_edge, dynamic_positions, ex_ante_vol, idm,
                                       primary_positions, standard_positions)

N = 10
SIGMA = np.full(N, 0.80)
RHO = 0.64
FLAT = np.zeros(N)


def close(a, b, tol=1e-12):
    return np.allclose(a, b, rtol=0, atol=tol)


# ------------------------------------------------------------------ primary arithmetic
def test_parameters_come_from_the_frozen_config():
    assert (SIZING.target_vol, SIZING.band, SIZING.idm_cap, SIZING.overlay_multiple) == (0.30, 0.10, 2.5, 2.0)
    assert (SIZING.per_instrument_cap, SIZING.gross_cap) == (0.5, 3.0)


def test_idm_formula_cap_and_edges():
    assert abs(idm(N, RHO) - 1 / np.sqrt(0.1 + 0.9 * 0.64)) < 1e-15        # 1.2163
    assert idm(1, RHO) == 1.0 and idm(N, float("nan")) == 1.0
    assert idm(N, -0.2) == idm(N, 0.0) == min(2.5, np.sqrt(N))             # rho floored at 0, then capped
    assert idm(N, 0.0) == 2.5


def test_worked_example_from_the_design():
    s = standard_positions(SIGMA, RHO)
    assert close(s, 0.1 * idm(N, RHO) * 0.30 / 0.80)
    p, held = primary_positions(np.full(N, 10.0), SIGMA, RHO, FLAT)
    assert close(p, s - 0.10 * s) and not held.any()                      # from flat: to the inner edge
    assert abs(p[0] * 100_000 - 4_104.2) < 1.0                             # $4,560 ideal, minus the $456 band


def test_ideal_aligned_book_at_full_forecast_runs_exactly_at_target():
    s = standard_positions(SIGMA, RHO)
    assert abs(ex_ante_vol(s, SIGMA, RHO) - 0.30) < 1e-12                  # f = 10 on every coin


def test_buffer_holds_inside_the_band_and_trades_to_the_nearer_edge():
    s = np.array([0.04, 0.04])
    ideal = np.array([0.02, 0.02])
    new, hold = buffer_to_edge(np.array([0.0215, 0.05]), ideal, s)
    assert hold.tolist() == [True, False] and close(new, [0.0215, 0.024])
    new, hold = buffer_to_edge(np.array([-0.03, 0.0]), ideal, s)          # a flip, and an entry from flat
    assert close(new, [0.016, 0.016]) and not hold.any()


def test_the_overlay_cannot_bind_on_fresh_positions():
    """Design §2: ten maximum forecasts on one side give exactly 2 x target, the threshold."""
    ideal = 2.0 * standard_positions(SIGMA, RHO)
    assert abs(ex_ante_vol(ideal, SIGMA, RHO) - 0.60) < 1e-12
    p, _ = primary_positions(np.full(N, 20.0), SIGMA, RHO, FLAT)
    assert ex_ante_vol(p, SIGMA, RHO) <= 0.60 + 1e-12


# ------------------------------------------------------------------ dynamic risk targeting (Clarification 2)
WEAK = np.full(N, 0.5)                                                     # every forecast well under |f| = 1


def test_weak_forecasts_leave_the_primary_flat():
    """The primary by design: from flat, a coin trades only when |f| exceeds 1 (the band is 10% of
    a standard position, the ideal is f/10 of one). Weak forecasts mean a small or flat book."""
    p, hold = primary_positions(WEAK, SIGMA, RHO, FLAT)
    assert close(p, 0.0) and hold.all()


def test_weak_forecasts_are_scaled_up_and_traded_under_dynamic_risk_targeting():
    """The point of the sensitivity: always aim the IDEAL book at 30% risk. The same weak book's ideal is
    rescaled to 30% ex-ante risk; the traded book, from flat, sits at the band's inner edge (27%)."""
    p, ideal, m, hold = dynamic_positions(WEAK, SIGMA, RHO, FLAT)
    assert abs(ex_ante_vol(ideal, SIGMA, RHO) - 0.30) < 1e-12              # the ideal book is exactly on target
    assert m > 1 and not hold.any() and (p > 0).all()
    s = standard_positions(SIGMA, RHO)
    assert close(p, ideal - 0.10 * s)                                      # the primary band, not scaled by M
    assert abs(ex_ante_vol(p, SIGMA, RHO) - 0.27) < 1e-12                  # 90% of target from flat


def test_the_rejected_definition_would_have_stayed_flat():
    """Regression for the review finding: if the band scaled with M, weak forecasts would never trade."""
    p, ideal, m, _ = dynamic_positions(WEAK, SIGMA, RHO, FLAT)
    scaled_band_standard = m * standard_positions(SIGMA, RHO) / idm(N, RHO)    # w x M x 0.30 / sigma
    rejected, hold = buffer_to_edge(FLAT, ideal, scaled_band_standard)
    assert close(rejected, 0.0) and hold.all()


def test_an_undefined_correlation_is_treated_as_one_in_both_places():
    """Clarification 3: IDM 1 is the rho = 1 value; the covariance must agree, not assume independence."""
    p = standard_positions(SIGMA, 0.64)
    assert idm(N, float("nan")) == idm(N, 1.0) == 1.0
    assert ex_ante_vol(p, SIGMA, float("nan")) == ex_ante_vol(p, SIGMA, 1.0)
    assert ex_ante_vol(p, SIGMA, float("nan")) > ex_ante_vol(p, SIGMA, 0.0)
    aligned = np.full(N, 0.05)
    assert abs(ex_ante_vol(aligned, SIGMA, float("nan")) - 0.05 * 0.80 * N) < 1e-12   # moves as one


def test_zero_forecasts_cut_outside_the_band_and_leave_inside_it():
    p, ideal, m, hold = dynamic_positions(np.zeros(N), SIGMA, RHO, FLAT)
    assert m == 0.0 and close(ideal, 0.0) and close(p, 0.0)
    held = np.full(N, 0.03)
    s = standard_positions(SIGMA, RHO)
    p, ideal, m, hold = dynamic_positions(np.zeros(N), SIGMA, RHO, held)
    assert m == 0.0 and close(p, 0.10 * s) and not hold.any()             # down to the band's edge, not dumped
    inside = 0.05 * s                                                      # inside the zero-centred band
    p, _, _, hold = dynamic_positions(np.zeros(N), SIGMA, RHO, inside)
    assert close(p, inside) and hold.all()                                 # stays, possibly indefinitely
    p, _, m, _ = dynamic_positions(np.zeros(0), np.zeros(0), RHO, np.zeros(0))   # nothing eligible
    assert m == 0.0 and p.size == 0


def test_caps_bound_the_unbounded_multiplier():
    f = np.zeros(N)
    f[0] = 0.01                                                            # one tiny forecast, the rest zero
    sigma = SIGMA.copy()
    sigma[0] = 0.15                                                        # a calm coin
    p, ideal, m, _ = dynamic_positions(f, sigma, RHO, FLAT)
    assert m > 1e3 and abs(ideal[0] - 0.30 / 0.15) < 1e-9                  # alone at full risk: 2.0x capital
    assert abs(p[0] - 0.5) < 1e-12 and close(p[1:], 0.0)                   # the per-coin cap binds


def test_gross_cap_scales_the_whole_book():
    sigma = np.full(N, 0.05)                                               # absurdly calm, to force the gross cap
    p, _, _, _ = dynamic_positions(np.full(N, 3.0), sigma, 0.0, FLAT)
    assert abs(np.abs(p).sum() - 3.0) < 1e-12 and (np.abs(p) <= 0.5 + 1e-12).all()



def test_band_zero_is_full_rebalancing():
    """The full_rebalance spec and its twin pass band 0: every coin trades to its ideal unless it is
    already exactly there; the default is the frozen 0.10."""
    f, sigma, held = np.array([10.0, -5.0, 3.0]), np.array([0.6, 0.8, 1.0]), np.array([0.05, -0.02, 0.0])
    p, keep = primary_positions(f, sigma, 0.5, held, band=0.0)
    ideal = f / 10.0 * standard_positions(sigma, 0.5)
    assert np.allclose(p, ideal, rtol=0, atol=1e-15) and not keep.any()
    p, keep = primary_positions(f, sigma, 0.5, ideal, band=0.0)
    assert keep.all()
    assert (primary_positions(f, sigma, 0.5, held)[0] == primary_positions(f, sigma, 0.5, held, band=0.10)[0]).all()
    for bad in (-0.1, float("nan")):
        try:
            primary_positions(f, sigma, 0.5, held, band=bad)
        except ValueError:
            continue
        raise AssertionError("a negative or NaN band must raise")


# ------------------------------------------------------------------ stage C: the risk diagnostics (design 2026-10-27 §4)
def test_C1_the_ideal_book_runs_exactly_at_target_and_the_final_book_too_when_nothing_binds():
    """V(M u) = target for any forecasts, vols and correlation; with full rebalancing and no cap binding the
    held book is the ideal book, so its ex-ante risk is the target too, and the record says no cap bound."""
    rng = np.random.default_rng(11)
    unbound = 0
    for _ in range(200):
        n = int(rng.integers(1, 40))
        f, sigma = rng.uniform(-20, 20, n), rng.uniform(0.3, 2.0, n)
        rho = float(rng.choice([rng.uniform(-0.2, 0.95), np.nan]))
        d = {}
        p, ideal, m, _ = dynamic_positions(f, sigma, rho, np.zeros(n), band=0.0, detail=d)
        assert abs(d["v_ideal"] - SIZING.target_vol) < 1e-12 and abs(ex_ante_vol(ideal, sigma, rho) - SIZING.target_vol) < 1e-12
        assert abs(d["multiplier"] * d["v_unscaled"] - SIZING.target_vol) < 1e-12 and d["multiplier"] == m
        if d["per_instrument_cap_bound"] == 0 and not d["gross_cap_bound"]:
            assert close(p, ideal) and abs(d["v_final"] - SIZING.target_vol) < 1e-12
            assert abs(d["gross_final"] - np.abs(ideal).sum()) < 1e-12
            unbound += 1
    assert 20 < unbound < 200, unbound                                      # both branches exercised


def test_C2_a_binding_cap_lowers_the_realised_risk_and_is_recorded():
    # weak forecasts on low-vol coins: M is large and one coin's ideal exceeds 0.5
    f, sigma = np.array([0.5, 0.1, -0.1]), np.array([0.25, 1.0, 1.0])
    d = {}
    p, ideal, m, _ = dynamic_positions(f, sigma, 0.3, np.zeros(3), band=0.0, detail=d)
    assert np.abs(ideal).max() > SIZING.per_instrument_cap and d["per_instrument_cap_bound"] >= 1
    assert d["v_final"] < SIZING.target_vol - 1e-6 and np.abs(p).max() <= SIZING.per_instrument_cap
    # many diversified low-vol coins: the gross cap binds
    n = 40
    f, sigma = np.where(np.arange(n) % 2, 10.0, -10.0), np.full(n, 0.3)
    d = {}
    p, ideal, _, _ = dynamic_positions(f, sigma, 0.0, np.zeros(n), band=0.0, detail=d)
    assert np.abs(ideal).sum() > SIZING.gross_cap and d["gross_cap_bound"] and d["per_instrument_cap_bound"] == 0
    assert abs(d["gross_final"] - SIZING.gross_cap) < 1e-12 and d["v_final"] < SIZING.target_vol - 1e-6


def test_C3_the_twin_is_long_only_at_full_target_risk():
    """K1 under dynamic sizing (forecast +10 on every eligible coin) is the risk-targeted long-only control."""
    rng = np.random.default_rng(12)
    for _ in range(100):
        n = int(rng.integers(1, 40))
        sigma, rho = rng.uniform(0.3, 2.0, n), float(rng.uniform(0.0, 0.95))
        d = {}
        p, ideal, _, _ = dynamic_positions(np.full(n, 10.0), sigma, rho, np.zeros(n), detail=d)
        assert (ideal > 0).all() and (p >= 0).all() and abs(d["v_ideal"] - SIZING.target_vol) < 1e-12


def test_C4_diagnostics_never_change_a_decision():
    rng = np.random.default_rng(13)
    for _ in range(50):
        n = int(rng.integers(1, 30))
        f, sigma, held = rng.uniform(-20, 20, n), rng.uniform(0.3, 2.0, n), rng.normal(0, 0.1, n)
        for fn in (dynamic_positions, primary_positions):
            a, b = fn(f, sigma, 0.5, held), fn(f, sigma, 0.5, held, detail={})
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
    d = {}
    dynamic_positions(np.zeros(0), np.zeros(0), np.nan, np.zeros(0), detail=d)
    assert d["multiplier"] == 0.0 and d["v_final"] == 0.0 and d["per_instrument_cap_bound"] == 0


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
