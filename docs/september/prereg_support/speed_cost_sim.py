"""Speed-limit evidence for the EWMAC blend (amendment of 2026-09-23).

Supersedes turnover_sim.py and scalar_sim.py, which used `adjust=False`
(the protocol registers `adjust=True`), independent paths per speed, and a
single volatility assumption.

Uses SYNTHETIC prices only — no project data, no strategy returns. Turnover
depends on the filter AND on the price process, so the registered random-walk
basis is reported alongside a trending (AR(1)) sensitivity.

Outputs
-------
1. Annual forecast turnover per speed, on COMMON paths across speeds.
2. Cost in Sharpe units over a grid of annual volatilities, against Carver's
   ~0.13 SR/yr speed limit. A speed is admitted only if it passes across the
   WHOLE registered volatility range (50-80%), so no speed enters or leaves on
   a knife edge at one assumed volatility.
3. Implied forecast scalars and the forecast diversification multiplier for the
   admitted blend.

Run:  python docs/september/prereg_support/speed_cost_sim.py
"""

import numpy as np
import pandas as pd

SEED = 20260923
YEARS = 40
REPS = 40
SPEEDS = [2, 4, 8, 16, 32, 64]          # Carver's ladder, fast leg
VOL_GRID = [0.50, 0.55, 0.60, 0.64, 0.70, 0.80]
VOL_REGISTERED = 0.64                    # measured buy-and-hold basket vol
SPEED_LIMIT = 0.13                       # Carver, SR/yr of cost
COST_PER_SIDE = 0.0010                   # 10 bp, the registered primary
VOL_SPAN = 32                            # registered vol estimator
ADJUST = True                            # registered EWMA convention


def forecasts(prices: pd.Series) -> dict[str, pd.Series]:
    """Vol-normalised EWMAC forecasts, one per speed, on one price path."""
    vol = prices.pct_change().ewm(span=VOL_SPAN, adjust=ADJUST).std()
    price_vol = prices * vol
    out = {}
    for fast in SPEEDS:
        slow = 4 * fast
        raw = (prices.ewm(span=fast, adjust=ADJUST).mean()
               - prices.ewm(span=slow, adjust=ADJUST).mean())
        out[f"{fast}/{slow}"] = (raw / price_vol).iloc[4 * max(SPEEDS):]
    return out


def paths(rng, process: str, n: int) -> pd.Series:
    """One price path. `random_walk` is the registered basis; `trending` is the
    sensitivity the review asked for, because EWMAC turnover is NOT purely a
    property of the filter — it also depends on the price process. A trending
    series crosses its own averages less often, so turnover falls."""
    if process == "random_walk":
        rets = rng.normal(0, VOL_REGISTERED / np.sqrt(365), n)
    else:                                   # AR(1) in returns, positive autocorrelation
        shocks = rng.normal(0, VOL_REGISTERED / np.sqrt(365) * np.sqrt(1 - 0.05 ** 2), n)
        rets = np.zeros(n)
        for i in range(1, n):
            rets[i] = 0.05 * rets[i - 1] + shocks[i]
    return pd.Series(100 * np.exp(np.cumsum(rets)))


def turnover_table(rng, process: str) -> pd.Series:
    out = {k: [] for k in (f"{f}/{4*f}" for f in SPEEDS)}
    for _ in range(REPS):
        panel = pd.DataFrame(forecasts(paths(rng, process, 365 * YEARS)))
        scaled = (panel * (10 / panel.abs().mean())).clip(-20, 20)
        years = len(panel) / 365
        for name in panel:
            out[name].append(scaled[name].diff().abs().sum() / 10 / years)
    return pd.Series({k: float(np.mean(v)) for k, v in out.items()})


def main() -> None:
    rng = np.random.default_rng(SEED)
    turnover = {k: [] for k in (f"{f}/{4*f}" for f in SPEEDS)}
    panels = []

    for _ in range(REPS):
        # one path per replication, shared by every speed
        rets = rng.normal(0, VOL_REGISTERED / np.sqrt(365), 365 * YEARS)
        prices = pd.Series(100 * np.exp(np.cumsum(rets)))
        fc = forecasts(prices)
        panel = pd.DataFrame(fc)
        scalars = 10 / panel.abs().mean()
        scaled = (panel * scalars).clip(-20, 20)
        years = len(panel) / 365
        for name in panel:
            # turnover in units of the average position (|F| = 10)
            turnover[name].append(scaled[name].diff().abs().sum() / 10 / years)
        panels.append((scalars, scaled))

    turn = pd.Series({k: float(np.mean(v)) for k, v in turnover.items()})
    scalars = pd.concat([s for s, _ in panels], axis=1).mean(axis=1)

    print(f"Seed {SEED}, {REPS} paths x {YEARS}y, adjust={ADJUST}, "
          f"vol span {VOL_SPAN}, cost {COST_PER_SIDE*1e4:.0f} bp/side\n")

    print("Annual forecast turnover (units of average position), common paths:")
    print(turn.round(1).to_string(), "\n")

    print(f"Cost in SR/yr = turnover x cost / vol.  Speed limit {SPEED_LIMIT}.")
    grid = pd.DataFrame({f"{v:.0%}": turn * COST_PER_SIDE / v for v in VOL_GRID})
    grid["admitted"] = (grid.max(axis=1) <= SPEED_LIMIT)
    print(grid.round(3).to_string(), "\n")

    admitted = [k for k in turn.index if grid.loc[k, "admitted"]]
    print("Admitted across the whole 50-80% volatility range:", admitted)

    # FDM for the admitted blend, from its own correlations
    corr = pd.concat([sc[admitted] for _, sc in panels]).corr()
    w = np.full(len(admitted), 1 / len(admitted))
    fdm = 1 / np.sqrt(w @ corr.values @ w)
    print("\nImplied scalars (simulated, for comparison with Carver's fixed values):")
    print(scalars[admitted].round(2).to_string())
    print("\nCorrelation of scaled forecasts:")
    print(corr.round(2).to_string())
    print(f"\nFDM = 1/sqrt(w'Hw) = {fdm:.3f}")

    blend = pd.concat([sc[admitted].mean(axis=1) for _, sc in panels]) * fdm
    print(f"mean |blend| after FDM = {blend.abs().mean():.2f}")
    print(f"share of days at the +/-20 cap = {(blend.abs() >= 20).mean():.1%}")
    print("(FDM and the cap-hit rate above are random-walk figures, not claims "
          "about this market.)")

    # Sensitivity: the same rule on a trending process. Reported for disclosure;
    # the registered admission is the random-walk table above, fixed in advance.
    trend_turn = turnover_table(np.random.default_rng(SEED + 1), "trending")
    comparison = pd.DataFrame({
        "random_walk": turn,
        "trending_ar1": trend_turn,
        "cost@50%_rw": turn * COST_PER_SIDE / 0.50,
        "cost@50%_ar1": trend_turn * COST_PER_SIDE / 0.50,
    })
    comparison["same_verdict"] = ((comparison["cost@50%_rw"] <= SPEED_LIMIT)
                                  == (comparison["cost@50%_ar1"] <= SPEED_LIMIT))
    print("\nSensitivity — AR(1) returns, rho=0.05 (turnover is not filter-only):")
    print(comparison.round(3).to_string())


if __name__ == "__main__":
    main()
