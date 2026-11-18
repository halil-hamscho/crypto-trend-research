# Per-coin study, stage B: specification fixed before stage A runs

**Dated 2026-10-24. Fixed before any per-coin return on real data** (stage A has not run). This
document is the commitment the run driver's guard checks: its SHA-256 is written into the stage-A
run manifest, and stage A refuses to start if it is missing or changed. The staging is recorded in
the protocol's Part 13 change log (2026-10-24, *Staging*). Where this document and the protocol
disagree, the protocol governs.

## 1. Why there are two stages

Stage A (the top 10: the primary, controls K1–K3 and K5, and six counted sensitivities) needs no
contract-event handling: no top-10 member meets a halt, settlement or rebrand while it can be
held **[V: ledger implementation, real-inputs test]**. Stage B's universes do (protocol Part 15),
and the ledger rules for them (steps 7–8) and the event register are not built. Running stage A
first does not change any stage-A rule. Its cost is disclosed in §5.

## 2. The stage-B runs

Ten books: five counted sensitivities, each with its K1 twin (+10 forecast, identical settings).
Every setting not listed is the primary's: one-day lag, 00:00 kline-open fill, 10 bp per side,
buffer 0.10, primary sizing, window 2021-07-05 → 2026-08-30, $100k fixed base, seed 20261007.

| run | twin | universe | rule | counted as |
|---|---|---|---|---|
| `top_25` | `top_25_k1` | `deep_membership(25, …)`: enter 25 / exit 33, map `cmc_contract_map_top100.csv`, resets from 2021-07-01 | Part 15 (A2.1–A2.6) | `top_25` |
| `top_50` | `top_50_k1` | `deep_membership(50, …)`: enter 50 / exit 65, same map and schedule | Part 15 | `top_50` |
| `registry_25m` | `registry_25m_k1` | registry-wide U1–U8, Appendix A exclusions, $25m 30-day median liquidity gate | Parts 2–3, 14 | `registry_wide_25m` |
| `registry_10m` | `registry_10m_k1` | as above, $10m gate | Parts 2–3, 14 | `registry_wide_10m` |
| `registry_50m` | `registry_50m_k1` | as above, $50m gate | Parts 2–3, 14 | `registry_wide_50m` |

These are already the stage-B specs in `portfolio/per_coin_runs.py` (`RUNS`, stage "B"), checked
against the config by test D1.

## 3. Comparisons and reporting

- **Each variant against its own twin only** (Clarification 4a): α by the Part 9 paired bootstrap
  (mean block 20, 10,000 replications), and net Sharpe with its interval. The identity check
  requires the pair to differ only in the forecast.
- **Reported, never feeding the stage-A verdict.** The verdict (Clarification 4b) is computed from
  stage A alone and is not revisited after stage B.
- **Also reported per book:** the ladder, turnover with causes, funding by leg, the market /
  coin-specific / ineligible split, per-coin and per-year contributions, the two-sided share, IDM
  and ρ̄ paths, the eligible count, and every contract event met with its classification and
  settlement price.
- **LUNA measured** (protocol Part 9), in the books that hold it.
- **Deflated Sharpe:** unchanged at 58 trials; these five runs are already counted.

## 4. Implementation changes allowed before stage B runs

Only these. Anything else needs its own change-log row.

1. **Ledger steps 7–8** (`docs/october/per_coin_ledger_implementation_2026-10-13.md`): halts held at
   the last traded mark and exited at the first traded bar; settlements closed at the settlement
   price with the 5 bp settlement fee; span ends. The settlement price uses the **era-specific
   index average**: 60 minutes before 2024-11-11 08:00 UTC, 30 minutes after
   (`docs/october/binance_delisting_rules_2026-10-13.md`).
2. **The event and suspension registers:** every event the stage-B books can meet while held,
   classified on dated evidence (exchange settlement, halt, rebrand): at least LUNA, ICP, FTT,
   MATIC → POL, TON, BTT, RNDR → RENDER, FTM → S.
3. **Data completions:** ICPUSDT funding for 2021-05 → 2022-06 from the `ICPUSDT_SETTLED` archive,
   and the same check for every other `…_SETTLED` rename. Missing funding is never filled with zero.
4. **Universe code for the registry-wide books:** a registry-wide eligibility builder implementing
   U1–U8 and the liquidity gate as frozen; the deep universes use the existing `DeepSelector`.
5. **A stage-B dataset manifest** (run-driver design §4.1): the deep map, `traded_ranges.csv`, the
   registry list, `exclusions.csv`, both registers and the 1-minute index files, plus klines,
   funding and marks for every contract the stage-B selectors may examine. Checked at load exactly
   as in stage A.
6. **The driver's stage-B entry point**, reusing every stage-A check (dataset manifest, permitted
   subsets, identity, reconciliations, attribution, blind mode).

Every change above is tested on synthetic data first, and stage B gets its own blind real-data
check before its single run. **No rule, parameter or comparison in §2–3 may change.**

## 5. How stage-B results will be described

Stage-B results are produced **after stage A's results are known**, and every report says so.
They are sensitivities of the primary, so knowing stage A cannot change the verdict. It can bias
how stage B is built and read, which is why §2–4 are fixed here first.
