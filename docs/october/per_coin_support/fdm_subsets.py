"""Forecast diversification multipliers for the nested speed subsets used while an
instrument's history is short (per-coin protocol, rule S5).

SYNTHETIC random walks only: no project data, no returns. Reuses the September
simulation (same seed, paths, adjust=True, 32-day vol span), so the four-speed
figure must reproduce the registered FDM of 1.15.

Run:  python docs/october/per_coin_support/fdm_subsets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "september" / "prereg_support"))
import speed_cost_sim as sim  # noqa: E402

SUBSETS = [["8/32"], ["8/32", "16/64"], ["8/32", "16/64", "32/128"],
           ["8/32", "16/64", "32/128", "64/256"]]


def main() -> None:
    rng = np.random.default_rng(sim.SEED)
    panels = []
    for _ in range(sim.REPS):
        rets = rng.normal(0, sim.VOL_REGISTERED / np.sqrt(365), 365 * sim.YEARS)
        panel = pd.DataFrame(sim.forecasts(pd.Series(100 * np.exp(np.cumsum(rets)))))
        panels.append((panel * (10 / panel.abs().mean())).clip(-20, 20))
    stacked = pd.concat(panels)
    print(f"Seed {sim.SEED}, {sim.REPS} paths x {sim.YEARS}y (the September simulation)\n")
    for subset in SUBSETS:
        corr = stacked[subset].corr().values
        w = np.full(len(subset), 1 / len(subset))
        print(f"{' + '.join(subset):34s} FDM = {1 / np.sqrt(w @ corr @ w):.3f}")


if __name__ == "__main__":
    main()
