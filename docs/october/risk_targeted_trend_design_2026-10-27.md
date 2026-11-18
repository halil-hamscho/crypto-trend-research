# Per-coin trend with explicit daily portfolio risk targeting: design for review

**Dated 2026-10-27. Approved by the researcher on 2026-10-29 as recommended (D1–D8 and the 20 bp stress
pair, §7): this is the stage-C specification, and the run driver's guard checks its SHA-256. Nothing has run.**
This is a new, separately registered construction. It does not reopen the completed per-coin study (sealed; tag
`per-coin-study-complete-2026-10-27`). Labels: **[V]** verified in this repository, **[R]** recommended, open
to review.

## 0. Why, stated honestly

- **The decision is design-led.** The researcher's objective is trend following *inside a deliberately
  controlled portfolio risk budget*. The frozen primary does not do that: it gives each coin a fixed risk
  budget and only cuts the book above 2× target, so the book's actual risk drifts with signal strength and
  correlation. Explicit daily risk targeting is a better fit to that objective.
- **It is not a passed performance gate.** The exploratory control (`octp_reproduction_findings_2026-10-26.md`
  §10) missed its pre-committed bar: α of risk-targeted trend over risk-targeted long-only +10.8%/yr, 5th
  percentile −1.7%. The registered stage-A dynamic-risk-targeting sensitivity (top 10) was also not
  significant: α vs twin +13.8%/yr, 5th percentile −10.7% [V]. A performance gain is a hypothesis to test,
  not a finding.
- **The sample is mined.** Every 2020–26 result here is in-sample, so this design must be fixed before any new
  run, counted as new trials, and judged finally on forward data.

## 1. What changes and what does not

**Unchanged** (the frozen per-coin protocol and config, every rule):
- universe rules;
- the signal: forecasts with fixed scalars, short-history speeds, ±20 caps;
- per-coin volatility and its floor;
- eligibility (X−1), the one-day lag and 00:00 fills;
- costs (10 bp, 5 bp settlement fee) and funding at every settlement;
- the registers, the ledger and its reconciliations, the attribution;
- inference: bootstrap, 10,000 replications, blocks 20/10/40/60, seed.

**Changes: only the sizing.** The book is rescaled every day to a fixed predicted risk. The engine already
implements this as `per_coin_sizing.dynamic_positions` (Amendment 3, ordered by Clarification 2) [V]. Each
day, on decision day X:

1. unscaled book uᵢ = (fᵢ / 10) × (1/N) × target / σᵢ over the eligible coins;
2. its predicted volatility V(u) from the risk estimator (D1);
3. multiplier M = target / V(u) (D3; M = 0 when V(u) = 0);
4. ideal book M·u;
5. buffer (D5);
6. per-coin cap, then gross cap (D3).

There is no overlay.

## 2. Decisions to fix before any run

| # | decision | options | recommendation |
|---|---|---|---|
| D1 | **Risk estimator** for V(u) | (a) constant-correlation covariance: σᵢσⱼρ̄, with ρ̄ the 180-day mean pairwise correlation of eligible coins through X, ≥ 90 shared days, ρ = 1 when undefined (Clarification 3); (b) a shrunk sample covariance (OCTP: 30% toward constant correlation, rebuilt weekly) | **(a) [R].** Already frozen, no new parameter. (b) was chosen by looking at OCTP's result and would add a free parameter (the shrinkage); a later trial at most |
| D2 | **Target** | 30% (frozen `target_vol`); 20% (OCTP) | **30% [R].** Sharpe barely depends on scale; only the caps and costs interact with it |
| D3 | **Scaling limits** | M unbounded with caps 0.5 per coin and 3.0 gross (Amendment 3); or a cap on M | **Unbounded M, frozen caps [R].** Bounding M would be a new tuned parameter. The caps bound leverage |
| D4 | **Weak signals** | as Clarification 2, weak forecasts are scaled up to full target risk; or shrink toward flat when forecasts are weak | **Keep Clarification 2 [R].** It is the mechanism under test. Disclosed risk: in signal-poor regimes the book runs full risk on weak, noisy forecasts, held back only by the caps |
| D5 | **Buffer** | the primary band, 10% of the standard position w·IDM·target/σ, not scaled by M (Clarification 2); or a band scaled by M (OCTP) | **Primary band [R].** Already frozen. Means more trading when M is large |
| D6 | **Execution order** | as §1 | **As §1 [R]** (Clarification 2, `frozen_config.yaml dynamic_risk_targeting.order`) |
| D7 | **Which books, which primary** | the registry $10m; the top 25 / top 50; the top 10 | **Primary: registry-wide $10m [R].** It's the broadest book, the one the OCTP control measured, and fully built and verified. Secondary: top 25 and top 50. The top-10 dynamic book already exists (sealed stage A) and is cited, not rerun. Disclosed: choosing the registry primary was informed by the OCTP work |
| D8 | **Cluster weights** | include, as in OCTP; or exclude | **Exclude [R].** One change at a time; clusters add an estimated structure and a tuned count (OCTP's own sweep: 0.65–1.20). Later trial at most |

## 3. Tests and reading

Each book runs with its **risk-targeted no-signal twin**: K1, forecast +10 on every eligible coin, under the
same dynamic sizing. That twin *is* the risk-targeted long-only control, so the existing identity check
(twins differ only in the forecast) enforces like-for-like [V: `per_coin_runs._pair`, `per_coin_inference`].

| result | comparison | rule |
|---|---|---|
| **H1′** (primary) | registry $10m risk-targeted trend vs its risk-targeted twin | Part 9 and Clarification 4 unchanged: headline FAIL / INCONCLUSIVE / H1 PASSES; warnings FRAGILE, NOT PER-COIN, COLLAPSED, LARGE CLAIM REJECTED |
| H1′ secondary | top 25 and top 50, each vs its twin | reported, not gating |
| vs the frozen primary | risk-targeted vs sealed primary-sizing book, same universe | descriptive: what the sizing change alone did |
| decomposition | long and short P&L, average M and its range, days at the gross cap, turnover | always reported |

The 20 bp stress for FRAGILE needs a 20 bp risk-targeted book and its twin for the primary, so two more
books.

**Trial count:** 58 + 3 new counted books (registry $10m, top 25, top 50, all risk-targeted) = **61** for
the deflated Sharpe. Twins and the 20 bp stress are not counted, as before.

**Expected outcome, stated now.** The minimum detectable net Sharpe is about 1.1. The exploratory figures
suggest a true difference near 0.5 if any. So the most likely registered reading is INCONCLUSIVE, and the
honest final test is forward.

## 4. Build and validation (step 3)

Existing [V]:
- `dynamic_positions` and its tests (weak-forecast behaviour, M = 0 at zero risk, the caps, the undefined-ρ
  fallback);
- the `dynamic_rt` run spec and twin (top 10, stage A).

To add:
1. Run specs: `registry_10m_dyn` and twin, `top_25_dyn` and twin, `top_50_dyn` and twin, plus the 20 bp
   primary pair, as a new stage ("C") with its own manifest.
2. **Synthetic tests:**
   - the ideal book's V(Mu) equals the target exactly when no cap binds;
   - a cap binding lowers the realised ex-ante risk and is recorded;
   - ρ̄ and σ at X use data through X only (truncation-rebuild, as model test M9);
   - the twin is long-only at full target risk.
3. **Reporting:** the daily M, V(u), gross exposure and the cap-binding days, per book.
4. **Freeze and run:** the blind real-data check, then freeze, then one registered attempt (the existing
   driver, stage C).

## 5. Forward evaluation (step 4)

- **A paper record of the frozen primary, from its freeze date.** Daily decisions at 00:00 UTC from live
  Binance data, with fills, funding and costs booked by the same ledger. Design reference:
  `docs/september/forward_record_design_2026-10-03.md`.
- **Needs:** a live daily data path. `fapi.binance.com` was reachable on 2026-10-26; the archive lags by a
  day for klines and by a month for funding.
- **Rule fixed now:** no claim before 12 months of forward data. The forward Sharpe is reported with its
  interval; nothing is tuned on it.

## 6. Questions for the researcher

1. Approve D1–D8 as recommended, or change any? D7 (the primary book) and D4 (weak signals) matter most.
2. Include the 20 bp stress pair (needed for FRAGILE), as recommended?
3. Start the forward-record build in parallel?

## 7. Approved specification: stage C (2026-10-29)

The researcher approved D1–D8 and the 20 bp stress pair as recommended. No rule in §1–3 may change; anything
else needs a protocol change-log row. Every setting not listed is the frozen primary's: one-day lag, 00:00
kline-open fill, buffer 0.10, window 2021-07-05 → 2026-08-30, $100k fixed base, seed 20261007. Universes,
registers, supplements and the dataset manifest are stage B's, unchanged.

| run | twin | universe | sizing | cost | role |
|---|---|---|---|---|---|
| `registry_10m_dyn` | `registry_10m_dyn_k1` | registry-wide, $10m | dynamic (Clarification 2) | 10 bp | **primary (H1′)**, counted |
| `registry_10m_dyn_20bp` | `registry_10m_dyn_20bp_k1` | registry-wide, $10m | dynamic | 20 bp | stress for FRAGILE, not counted |
| `top_25_dyn` | `top_25_dyn_k1` | top 25 (DeepSelector) | dynamic | 10 bp | secondary, counted |
| `top_50_dyn` | `top_50_dyn_k1` | top 50 (DeepSelector) | dynamic | 10 bp | secondary, counted |

- **Verdict (H1′):** `registry_10m_dyn` vs its twin by Part 9 and Clarification 4 in full, the stress from
  the 20 bp pair; deflated Sharpe at 61 trials (bracket 30–80).
- **Secondaries:** read descriptively by the same rules, without a stress pair (as in stage B).
- **vs the frozen primary:** each dynamic book against the sealed primary-sizing book of the same universe
  (`reports/per_coin_run/7cdd604ad20b/attempt_1/registry_10m`, `…/5498171a2afb/attempt_1/top_25`, `top_50`),
  verified against its `SHA256SUMS`; an identity check requires the two to differ in the sizing only.
  Descriptive.
- **Always reported per book:** daily M, V(u), the ideal and final ex-ante risk, gross exposure, days a
  per-coin or the gross cap binds, long and short P&L, plus everything stage B reports.
- **Order:** synthetic validation, then a blind real-data check, then the freeze (with the researcher's
  approval), then one registered attempt.
