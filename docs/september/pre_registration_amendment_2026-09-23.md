# Amendments to the pre-registration

**Dated 2026-09-23. Pre-result: no strategy return has been computed.**

Amends `pre_registration_2026-09-22.md` in response to an independent review of
that document. Where the two conflict, **this amendment governs**. Machine-readable
form: `prereg_support/frozen_config.yaml`.

Labels: **[V]** verified here · **[L]** literature · **[O]** judgement.

---

## A1. Data terminology: the universe is reconstructed, not vintage

The protocol called the universe "point-in-time". The **selection rule** is
point-in-time (it uses only the D−1 snapshot), but the **underlying data is not**:
CoinMarketCap's history was fetched in September 2026 and contains retroactive
restatements — the 2020-11-17 row for LUNC reports a $1.83tn market cap, 5.4×
BTC, from post-collapse supply **[V]**. A one-day lag prevents ordinary
look-ahead; it cannot turn revised data into vintage data.

**Replacement wording, throughout:** *historically reconstructed, one-day-lagged
large-cap basket using date-labelled CMC data.* The limitation belongs in the
protocol, not only the summary, and any broad market claim requires replication
against a governed free-float source.

## A2. EWMA speeds: rule rebuilt, 64/256 restored, 4/16 removed

The previous rule excluded 64/256 on a warm-up argument that does not hold — 512
days still leaves 5.2 scored years — and that exclusion coincided with something
learned from this sample. The inclusion of 4/16 was knife-edge: its cost passed
only at an unstated 64% volatility assumption, and the supporting simulations
used `adjust=False` while the protocol registers `adjust=True`.

**New rule, outcome-independent.** A speed from Carver's ladder is admitted only
if its cost stays **at or below 0.13 SR/yr [L]** at the registered 10 bp per side
across the **entire registered volatility range of 50–80% annualised**. No speed
enters or leaves on one assumed volatility.

Evidence, rebuilt (`prereg_support/speed_cost_sim.py`: seed 20260923, 40 shared
paths × 40 years, `adjust=True`, 32-day vol span, common paths across speeds;
synthetic random walks only, no project data) **[V]**:

| speed | turnover/yr | 50% | 55% | 60% | 64% | 70% | 80% | admitted |
|---|---|---|---|---|---|---|---|---|
| 2/8 | 157.8 | 0.316 | 0.287 | 0.263 | 0.247 | 0.225 | 0.197 | no |
| 4/16 | 80.4 | 0.161 | 0.146 | 0.134 | 0.126 | 0.115 | 0.101 | **no** |
| 8/32 | 41.2 | 0.082 | 0.075 | 0.069 | 0.064 | 0.059 | 0.052 | yes |
| 16/64 | 22.5 | 0.045 | 0.041 | 0.037 | 0.035 | 0.032 | 0.028 | yes |
| 32/128 | 14.0 | 0.028 | 0.025 | 0.023 | 0.022 | 0.020 | 0.017 | yes |
| 64/256 | 10.4 | 0.021 | 0.019 | 0.017 | 0.016 | 0.015 | 0.013 | yes |

**Registered blend: 8/32, 16/64, 32/128, 64/256, equal-weighted.** 4/16 fails
below 63% volatility and is excluded by the rule, not by preference; 2/8 fails
everywhere. Nothing is now excluded for a reason learned from this sample.

Consequences:

| | before | after |
|---|---|---|
| forecast scalars **[V]** | 7.5 / 5.3 / 3.75 / 2.65 | **5.3 / 3.75 / 2.65 / 1.87** |
| FDM **[V]** | 1.14 | **1.15** (1/√(w′Hw), same simulation) |
| warm-up | 256 obs | **512 obs** (2 × 256) |
| first scored day | 2020-10-03 | **2021-06-16** (superseded by Amendment 2: 2021-07-05) |
| scored window | 2,159 days (5.92 y) | **1,903 days** (superseded by Amendment 2: 1,884 days, 5.16 y) |

Simulated mean |blend| after FDM is 8.97 with 7.0% of days at the ±20 cap, the
reference point for the realised diagnostics.

## A3. H2: hurdle, not equivalence; corrected power

**Claim narrowed to what the paired gap tests.** The headline is: *positive alpha
over the passive basket (H1) **and** a higher net mean return than an identically
implemented BTC-only trend strategy (H2).* It is **not** a claim about return
outside the joint span of BTC-trend and passive beta. The two-factor conditional-
alpha regression (S15) stays a **stricter secondary analysis** and never controls
the headline; an underpowered regression must not gate the conclusion **[O]**.

**δ is a basket-superiority hurdle, not an equivalence margin.** Schuirmann TOST
is withdrawn: an upper bound below +5%/yr shows no economically material basket
advantage; it does not show the two strategies lie within ±5% of each other.

> **Hurdle:** a multi-contract basket must add at least **5 percentage points of
> annualised net return** over matched BTC trend to justify its additional
> operational complexity — funding, liquidity, monitoring, reconstitution and
> delisting risk across ten contracts rather than one highly liquid contract. At
> the 30% volatility target that is ≈**0.17 incremental Sharpe** **[V]**, a
> reasonable institutional hurdle **[O]**. If the one-sided 95% bound rules out
> an advantage of that size, the evidence supports **BTC operational
> sufficiency**, not statistical equivalence.

**Corrected minimum detectable effects** (80% power, 5% one-sided, T = 5.21 y —
recomputed at T = 5.16 y in Amendment 2, which moves H1 to 30.6%/yr and the
ρ = 0.95 gap to 10.4%/yr;
MDE ≈ 2.485·σ/√T) **[V]**. The previous H2 figures used ρ = 0.97–0.977 while
labelling them 0.95–0.97:

| test | noise | MDE |
|---|---|---|
| H1, residual vol 28% | 28%/yr | **30.5%/yr** (appraisal ratio 1.09) |
| H2 gap, ρ = 0.95 | 9.5%/yr | **10.3%/yr** |
| H2 gap, ρ = 0.97 | 7.3%/yr | **8.0%/yr** |
| H2 gap, ρ = 0.977 | 6.4%/yr | **7.0%/yr** |

**Disclosed consequence:** the 5%/yr hurdle sits **below** the H2 MDE. With a gap
SE of ≈3%/yr (ρ = 0.97), the upper bound falls below 5% only when the estimated
gap is approximately zero or negative. So the hurdle test can conclude "no
material basket advantage" or "inconclusive", and can only conclude "basket
superior" for a gap far larger than the hurdle itself. This asymmetry is stated
in advance rather than discovered afterwards.

Sharpe SE at this sample length is **0.46** (was 0.40 at 5.92 years) **[V]**.

## A4. Implementation frozen, not only the concept

See `prereg_support/frozen_config.yaml` for the machine-readable version and the
appendix below for pseudocode. Points previously unspecified:

| item | registered value |
|---|---|
| reconstitution timestamp | first calendar day of the month, 00:00 UTC; if not a traded day, the next traded day. Weekly sensitivity: Mondays |
| membership buffer | **stateful**: an incumbent is retained while its cap rank ≤ 13; vacancies are filled from the top of the eligible list; applied identically to strategy and benchmark |
| dollar-volume screen | 30-day median of the contract's own Binance daily quote volume, ≥ $10m, lagged one day, **minimum 20 observations** in the window, else the coin is ineligible |
| index | simple (arithmetic) returns; basket return = Σ wᵢ·rᵢ; index chain-linked so the level is continuous across reconstitution while weights step |
| signal and execution timing | signal from the 00:00 UTC close of D; the resulting position is held over D+1; execution priced at that same close, with slippage carried entirely by the cost model |
| opening costs | **included** in the first scored return |
| terminal treatment | positions marked to the final mark, open[2026-08-31] (the last holding period starts 2026-08-30; Amendment 6); **no** terminal liquidation cost |
| capital base | fixed notional capital, **no compounding**, returns expressed on that constant base; turnover expressed as Σ per-leg traded notional ÷ capital |
| pipeline order | raw → normalise by price vol → per-speed scalar → clip ±20 → equal-weight average → × FDM → clip ±20 |
| missing price | the contract is untradable that day; position carried at the last traded price, exits per the halt rule |
| missing funding | **hard failure**. No zero-filling, no interpolation; the affected date is outside the funded sample |
| BTC-cap runs (S2, S3) | excess weight above the cap is redistributed **pro rata** across the remaining constituents |
| Newey–West | fixed bandwidth of **21 lags** (~1 month), set now, not chosen by result |

**Two corrections of substance:**

1. **Funding is not on a fixed 8-hour grid.** The data contain 296,534 rows at
   8h, **50,736 at 4h**, 1,805 at 1h and 147 at 2h **[V]**. The rule is: the
   **pre-trade** position pays the 00:00 UTC settlement, and the post-trade
   position pays **every subsequent recorded settlement**, whatever the interval.
   Interval must be read from the data, never assumed.
2. **Return clipping is forbidden.** `backtesting/enginee.py:174` clips daily net
   returns at ±50% **[V]**; that path must not be used for any registered run. A
   day that exceeds ±50% is a real result or a bug, and either way it is reported.

**Sharpe is defined** as annualised arithmetic mean excess return ÷ annualised
volatility of excess returns. `enginee.py:219` computes CAGR ÷ volatility
**[V]**, which is a different statistic and must not be used.

**Ladder consistency:** within a policy, all three accounting layers use **one
common position path and one common capital denominator**. Otherwise the
differences between rows do not isolate funding and costs.

**Acceptance tests** required before any registered run:

1. `net = price P&L + funding − fees − slippage`, per leg, per day, to 1e-9.
2. Σ leg notionals = basket notional after every trade, and weights sum to 1 at
   each reconstitution.
3. A fixture in which aggregate exposure is unchanged while constituents change
   must produce **non-zero** turnover and cost.
4. Benchmark run at exposure 1.0 reproduces the buy-and-hold basket series.
5. No NaN silently treated as zero in any funding or price column.

## A5. Inference gate and trial taxonomy

**The H1 gate is the registered primary test only:** the stationary block
bootstrap at **20-day mean blocks and 10 bp costs**. Failures at 40/60-day blocks
or at 20 bp change the **label** to "fragile"; they do not retroactively redefine
the formal test or close the H2 gate.

**Correction:** the earlier claim that no registered run can support the headline
was wrong. S1 (BTC-only) *supplies* H2, and S12 (20 bp) decides supported versus
fragile. The registry is therefore typed:

| type | runs | counted in the deflated Sharpe? |
|---|---|---|
| strategy configurations | S2, S3, S4–S7, S8, S9, S10, S16 | yes |
| cost scenarios | S11, S12, S13, S14 | no — one strategy under different cost assumptions |
| statistical analyses | S15, block-length sensitivities, Newey–West | no |
| descriptive diagnostics | coverage, LUNA split, forecast diagnostics, calendar years | no |
| comparator required by a declared test | S1 | no — part of H2, not a search |

**The deflated Sharpe is a partial, lower-bound correction**, because earlier
mining of this sample (the June 2026 IC variants, the voided 0.42) is not
enumerated. It is reported with sensitivity to assumed prior trial counts (for
example 10, 25 and 50 prior trials), so a reader can see how much of the
correction is missing.

## A6. Positioning and literature

**Registered positioning.** Not a new momentum signal. This is a
**pre-registered, implementation-aware replication and decomposition**: it
reconstructs how a conventional basket-level trend strategy would have performed
implemented through constituent perpetual futures — historical contract
availability, constituent-level funding, leg-level turnover and costs, renames
and delisting settlements, a funded passive benchmark — and tests whether
conclusions drawn from spot or index returns survive perpetual-futures economics,
and whether BTC alone is sufficient.

**Literature matrix** (to be completed in the paper; all four verified to exist
2026-09-23 **[V]**):

| study | universe | spot or perp | dynamic membership | funding | delistings | costs | BTC comparator | multiple-testing correction |
|---|---|---|---|---|---|---|---|---|
| Borgards (2021), *N. Am. J. Econ. Finance* | 20 coins | spot | no | n/a | no | limited | partial | no |
| Liu & Tsyvinski (2021), *RFS* | broad CMC | spot | yes | n/a | partial | limited | yes | partial |
| Le & Ruthbah (2023), SSRN 4551518 | crypto | spot | no | n/a | no | limited | yes | no |
| Rozario, Holt, West & Ng (2020), SSRN 3697981 / arXiv 2009.12155 | crypto | spot | no | n/a | no | partial | partial | no |
| **this study** | top-10 Binance perps, reconstructed lagged | **perp** | **yes, buffered** | **yes, per leg** | **yes, exchange settlement** | **per leg + measured spreads** | **yes, as a declared hypothesis** | **pre-registration, typed registry, partial DSR** |

**Reference corrections [V]:** Dmitrienko et al. is **2006**, *Biometrical
Journal* (doi:10.1002/bimj.200610274), not 2007. The Deflated Sharpe Ratio
journal DOI is **10.3905/jpm.2014.40.5.094**; the SSRN DOI previously paired with
the journal citation is the working-paper identifier.

## A7. Forward-test language

- The clean operational out-of-sample period begins when the **code** is complete,
  frozen and running automatically — not when the prose was written.
- The interval between this amendment and that date is labelled
  **"post-registration, pre-implementation holdout"**, separately from the live
  record.
- A git commit is a useful timestamp but **not** a durable academic registration.
  The amended protocol will be archived on OSF or Zenodo and its identifier
  recorded in `frozen_config.yaml` (`external_archive_doi`). **Author action,
  outstanding.**
- Forward failure conditions are separated: **position/accounting divergence from
  the recorded targets = implementation failure**; **realised volatility or costs
  far from expectation = model/assumption failure or regime change**, not
  automatically a coding error.

## A8. Repository consistency

- `codex.md` §3 said each coin is sized by its own trend — corrected to the
  aggregate basket signal.
- `CLAUDE.md` framed the research question as "out of sample" — corrected; the
  historical sample is explicitly not out-of-sample.
- The selector still applies the CMC 24h-volume floor and has no stateful 10/13
  buffer. Both are **build tasks** required by this protocol before any
  registered run, recorded in `CLAUDE.md`.
- The universe-turnover table now has a committed reproducer:
  `prereg_support/universe_churn.py` (reproduces monthly 41.7%/yr, weekly
  72.2%/yr exactly).

---

## Appendix: pseudocode for the registered run

```
# ---- universe -------------------------------------------------------------
for each reconstitution date R (first calendar day of month, 00:00 UTC):
    snapshot   = CMC rows dated R-1, sorted by market_cap desc
    eligible   = [row for row in snapshot if
                    row.cmc_id not in EXCLUDED and
                    contract_traded_on(row.cmc_id, R-1) and
                    median(binance_quote_volume[contract, R-31:R-1]) >= 10e6 and
                    count(binance_quote_volume[contract, R-31:R-1]) >= 20]
    held       = [c for c in held if cap_rank(c, eligible) <= 13][:10]   # buffer
    for c in eligible:                                                   # refill
        if len(held) == 10: break
        if c not in held: held.append(c)
    quantities = capital_weights(held, market_cap) / price[held, R]      # fixed until next R

# ---- index ----------------------------------------------------------------
basket_return[t] = sum(weight[i, t-1] * simple_return[i, t] for i in held)
index[t]         = index[t-1] * (1 + basket_return[t])                   # chain-linked

# ---- signal (close of D) --------------------------------------------------
vol[t]      = max(ewma_std(index_returns, span=32, adjust=True)[t], floor[t])
floor[t]    = quantile(vol[t-500:t], 0.05)                               # >=100 obs
for (fast, slow) in [(8,32), (16,64), (32,128), (64,256)]:
    raw     = ewma(index, fast, adjust=True) - ewma(index, slow, adjust=True)
    f[s, t] = clip(raw[t] / (index[t] * vol[t]) * scalar[s], -20, +20)
forecast[t] = clip(mean(f[:, t]) * 1.15, -20, +20)

# ---- sizing ---------------------------------------------------------------
if t < 512th observation: position = NaN; continue                       # warm-up
standard[t] = 0.30 / vol_annualised[t]
x_star[t]   = clip(forecast[t] / 10 * standard[t], -2.0, +2.0)

# ---- execution ------------------------------------------------------------
band        = 0.10 * standard[t]
if policy == "full":      target = x_star[t]
else:                     target = (held_exposure if |held_exposure - x_star[t]| <= band
                                    else nearest_edge(x_star[t], band))
target      = clip(target, -2.0, +2.0)                                   # cap again
leg_target[i] = target * weight[i, t] * capital                          # proportional
if t is a reconstitution date: leg_target reflects the new membership     # always trades

# ---- ledger (per leg, per day) -------------------------------------------
trade[i]    = leg_target[i] - leg_marked[i]
fee[i]      = |trade[i]| * cost_per_side
funding[i]  = sum(position_at(settlement) * rate(settlement)
                  for settlement in funding_rows[i, t])                  # any interval
price_pnl[i]= leg_marked[i] * simple_return[i, t]
funding_pnl[i] = -funding[i]                                             # Amendment 3 sign
net[i]      = price_pnl[i] + funding_pnl[i] - fee[i]
assert abs(net_total - (price_total - funding_total - fee_total)) < 1e-9

# ---- scoring --------------------------------------------------------------
scored window = 2021-07-05 .. 2026-08-30            # 1883 days (Amendment 6; was 08-31, 1884)
sharpe        = 365 * mean(net_return) / (sqrt(365) * std(net_return))
alpha, beta   = OLS(net_return_strategy ~ net_return_benchmark)
p_value       = stationary_bootstrap(paired daily returns, mean_block=20, reps=10000)
```


---

# Amendment 2 — sample start, forced by the registered screen

**Dated 2026-09-24. Pre-result: no strategy return has been computed.**

Implementing A4's investability screen exposed an arithmetic consequence rather
than a new decision. The screen needs **20 observations** of Binance dollar
volume, so a contract is ineligible for its first 20 traded days. In early 2020
most perps had only just listed, and the basket cannot field ten eligible names
until **2020-02-09** (thirteen, the buffer's exit rank, until 2020-02-26) **[V]**.

| | Amendment 1 | **Amendment 2** |
|---|---|---|
| basket start | 2020-01-21 | **2020-02-09** |
| warm-up (512 obs) ends | 2021-06-15 | **2021-07-04** |
| first scored day | 2021-06-16 | **2021-07-05** |
| scored window | 1,903 days (5.21 y) | **1,884 days (5.16 y)** |

No discretion is involved: the start is wherever the frozen screen first admits
ten names. Power is unchanged to the stated precision — Sharpe SE **0.46**;
MDE **30.6%/yr** for H1 and **8.0%/yr** for the H2 gap at ρ = 0.97 **[V]**.

Before the buffer's exit rank is available (2020-02-09 → 2020-02-25),
`select_buffered` falls back to the plain top-10 selection; with no ranks 11–13
to buffer against there is nothing for the rule to do. This is in the code and
stated here rather than left to the implementer.

**Verified side effects of the screen**, recorded so they are not mistaken for
drift later **[V]**: on 2020-02-09 ADA, XTZ and XMR are excluded for thin
*Binance* volume though they traded; SYS remains excluded, as under the old CMC
floor; with `min_history_days=64` the first workable start moves from 2020-03-24
to 2020-04-14.


---

# Amendment 3 — identification, attribution and housekeeping

**Dated 2026-09-24. Pre-result: no strategy return has been computed.**

Responds to a second independent review, of `design_rationale.md` and its own
adversarial review. Where earlier text conflicts, this governs.

## B1. Alpha attribution with a common beta (replaces the review's R1 fix)

**The identification problem is real.** The strategy runs at roughly 0.47×
average exposure against a benchmark permanently at 1.0, and funding is a large,
one-directional holding cost: over the registered window the mean annualised rate
is **11.52% on BTC, 13.71% on ETH, 14.00% on XRP** (realised 11.35 / 13.51 /
13.82%/yr) **[V: `prereg_support/instrument_facts.py`]**. Constant exposure
yields α ≈ 0, so nothing is mechanical, but α absorbs any covariance between
exposure and *total* return — including carrying less funding when it is
expensive.

**Rejected fix:** differencing two separately estimated alphas (price-layer minus
net-layer). Those regressions have different regressors and different betas, so
the difference is not an attribution.

**Registered fix — exact decomposition at a fixed β.** Estimate β once, in the
primary net regression. Hold it fixed and decompose the *same* α:

```
α_net = α_price + α_funding − α_cost
where  α_layer = mean(r_strategy,layer − β · r_benchmark,layer) × 365
```

with the identical β and the identical sample in every term. The three components
then sum to the headline α by construction. Layer-by-layer regressions with their
own betas may also be shown, **labelled diagnostics, never attribution**.

Also reported: funding P&L split by long days versus short days.

**Interpretation, fixed now.** If α is concentrated in the funding term, the
finding is *timing total perpetual returns*, not price timing — a different
mechanism, **not a lesser one**: reducing exposure when carry is expensive is
economically meaningful for anyone trading perps. The paper reports which
mechanism produced the result rather than asserting price timing by default.

## B2. H2 realised-risk diagnostics (H2 itself unchanged)

Both legs target 30% ex ante, so "not risk-matched" was too strong; they may
nonetheless differ **ex post**. Always reported alongside the gap: realised
volatility, mean absolute exposure and turnover for each leg, plus an **ex-post
volatility-matched gap** as a standing diagnostic. The registered H2 statistic
does not change, and the vol-matched version never replaces it.

## B3. Frictions as a primary *descriptive* estimand

Elevated from a supporting table, but **not** to a second confirmatory endpoint —
that would imply another hypothesis and another multiplicity problem. The study
has one confirmatory hypothesis (H1), one gated hypothesis (H2), and one
**primary descriptive objective**: the funding and execution-cost drag of
implementing crypto trend through perpetual futures, reported with intervals for
strategy, benchmark and BTC-trend. It is estimated precisely because the ladder
layers are paired same-day differences on identical positions — conditional, as
always, on the cost model.

## B4. Funding sign convention (ambiguity removed)

Rates are positive when longs pay. The ledger therefore records

```
funding_pnl = − position_notional × funding_rate      (negative when a long pays)
gross_pnl   = price_pnl + funding_pnl
net_pnl     = gross_pnl − fees − slippage
```

A short position with a positive rate *receives*, and the sign follows
automatically. Earlier text described "gross = price + funding" while the
pseudocode subtracted a positive payment; both now mean this.

## B5. Corrected sample figures everywhere

The Amendment 2 window is authoritative: basket start **2020-02-09**, warm-up
ends **2021-07-04**, scored **2021-07-05 → 2026-08-30**, **1,883 days (5.16 y)** (Amendment 6; the window ended 2026-08-31, 1,884 days, under Amendment 2).
Superseded figures (1,903 days, 2021-06-16) have been removed from the protocol
header and from Amendment 1's tables and pseudocode.

## B6. Registered sensitivities added

- **LUNA settlement counterfactual.** Alongside the reconstructed 30-minute index
  settlement, report the same result under the naive "last traded close"
  assumption. One number cannot show whether the conclusion depends on the
  reconstruction; two can.
- **Terminal liquidation.** The primary marks to the final close with no
  liquidation cost, on the stated rationale that the strategy continues past an
  arbitrary sample end. A hypothetical full liquidation is reported as a
  sensitivity, not folded into the primary.

## B7. Disclosures the reviews asked for

- **The 50–80% volatility range** in the speed rule brackets the basket's
  measured ~64% — a property of the instrument, not of any strategy return.
  Stated, not implied.
- **Turnover is not purely a property of the filter**; it depends on the price
  process. `speed_cost_sim.py` now reports an AR(1) trending sensitivity
  alongside the registered random-walk basis. Verdicts are unchanged for all six
  speeds (e.g. 4/16: 0.161 → 0.158 SR/yr at 50% vol; 2/8: 0.316 → 0.305) **[V]**.
  FDM 1.15 and the 7.0% cap-hit reference are labelled random-walk figures.
- **How often the vol floor and the leverage cap bind** is reported. At a 30%
  target against ~64% basket volatility the cap requires volatility below 30%
  (forecast 20) or 15% (forecast 10), so "never" is a plausible answer and a
  useful one.
- **Cap weighting** is described as the standard broad-market proxy, not the only
  defensible construction.
- **Effective breadth ≈1.4** is an illustration assuming equal weights and a
  constant average correlation; the basket is cap-weighted, so it is labelled a
  heuristic rather than a measurement.

## B8. Reproducibility and freezing

- `prereg_support/instrument_facts.py` reproduces every instrument number quoted
  in the documents, over the stated window. Figures quoted earlier without a
  window (11.8 / 14.0 / 14.7%/yr funding) came from whole files including
  pre-window settlements and are superseded by B1's values.
- `prereg_support/universe_churn.py` and `speed_cost_sim.py` reproduce the
  membership and speed tables.
- The protocol, both amendment files and the frozen config are timestamped with
  OpenTimestamps (`prereg_support/TIMESTAMPS.md`); `frozen_config.yaml` records
  `frozen_commit` on the commit that carries this amendment.
- `external_archive_doi` remains open: Zenodo's GitHub integration cannot be used
  on a private repository, so an embargoed OSF registration is the route, at
  write-up. **Author action, outstanding.**

---

# Amendment 4 — event timeline and state transitions

**Dated 2026-09-24. Pre-result: no strategy return has been computed.**

A technical review of `basket_walkthrough.md` found five defects that would have
changed P&L. Four are implementation errors; the fifth was a genuine ambiguity in
the protocol. All are settled here, before `basket.py` exists.

## C1. The daily timestamp is the bar's OPEN

**Verified [V]:** `close[t] == open[t+1]` exactly for BTCUSDT. `panel.build_ohlcv`
stores Binance's `open_time`, so the bar labelled D opens at D 00:00 UTC and
closes at D+1 00:00 UTC. Indexing the *close* by that label would trade on a
price that did not yet exist and misalign every funding settlement by a day.

**Registered convention.** Execution instants are **00:00 UTC**, and the price at
instant D is **`open[D]`** — which is by construction the price at D 00:00, and
equals `close[D−1]`. A position taken at D 00:00 earns bar D's return and pays
the settlements that fall inside bar D (D 00:00, 08:00, 16:00 for an 8-hour
contract). Using opens rather than shifted closes removes the off-by-one class of
error entirely.

## C2. Funding stays settlement-level, on the mark price

Collapsing a day's rates into one sum and multiplying by one notional is wrong:
the 08:00 and 16:00 notionals differ from the 00:00 one because the mark moves.

**Registered computation**, per contract per settlement:

```
funding_pnl(settlement) = − quantity_held(settlement)
                          × mark_price(settlement)
                          × funding_rate(settlement)
```

`mark_price` comes from **1-hour `markPriceKlines`** (`futures/um/monthly/
markPriceKlines/<contract>/1h`), whose open at hour H is the mark at H:00. Full
history 2020-01 → 2026-08 for every contract, ~0.01 MB/month **[V]**, so no
approximation is needed. The ledger keeps the settlement subledger and reports
`funding_open_pnl` (pre-trade position, 00:00) and `funding_post_pnl` (new
position, later settlements) separately, so a sign change at 00:00 stays
auditable.

**Hard failure, as registered:** a missing funding row, or a missing mark for a
settlement, raises. No `fill_value=0`, no `.get(day, 0.0)`.

## C3. Contract states are explicit

A skipped row is not a treatment. Each contract-day is in exactly one state:

| state | meaning | accounting |
|---|---|---|
| `active` | traded that day | normal ledger row |
| `halted` | no volume, contract still listed | carried at its last traded mark; **funding still accrues** if settlements are published; no trade |
| `settling` | the delisting day | realise P&L to the settlement price, charge the taker fee, quantity → 0 |
| `idle` | its allocation after settlement | held out of the basket until the next reconstitution; earns nothing |
| `rebranded` | successor listed while still a member | migrate at the successor's first traded bar; `idle` in the gap |

A contract that resumes after a halt marks from its **last valid price**, not from
a NaN, so no P&L is lost or invented.

## C4. Settlement realises P&L; there are no principal proceeds

A perpetual future has no principal. Closing a position realises price P&L and
releases exposure — it does not return `quantity × price` in cash. The earlier
sketch would have recorded a **negative** cash flow for a short LUNA position that
in fact made money.

**Registered treatment at settlement:** realise price P&L from the last mark to
the settlement price; apply that settlement's funding; charge the taker fee; set
quantity to zero; mark the freed allocation `idle`.

**And it is not redistributed.** Scaling the survivors back to the full target
would buy more of the remaining nine at the moment of a collapse — the opposite of
the registered rule. Until the next reconstitution the basket runs with the dead
leg's weight **unallocated**, so total exposure is below target by that weight.

## C5. Exposure semantics: units for the benchmark, notional for the strategy

The ambiguity the review exposed: "exposure 1.0" can mean *hold one basket*
(quantities fixed, weights drift) or *hold capital's worth of basket every day*
(notional restored daily, which trades on every price move). The protocol's prose
says the first for the benchmark; a naive implementation does the second.

**Registered:**

- **Benchmark — units.** Exposure 1.0 means one basket unit. Quantities are fixed
  between reconstitutions and weights drift. **The only benchmark trades are
  reconstitutions, settlements and rebrands.** Turnover is therefore membership
  turnover (≈40%/yr, `universe_churn.py`), which is what the sanity check
  compares against.
- **Strategy — notional.** The target is `x*ₜ × capital`, recomputed daily as
  forecast and volatility change, which is what volatility targeting means. Price
  drift therefore does create trades — and that is precisely what the 10% band
  exists to damp. The full-rebalancing policy (band = 0) is the unbuffered
  diagnostic and will show much higher turnover; that is expected, not a bug.

The two policies' turnover figures are therefore **not** comparable to the
benchmark's, and the report labels them separately.

## C6. Index construction versus the scored books

The index must exist across the 512-day warm-up so the slow EWMA can mature, but
no position is held then.

**Registered:** the index is built **costlessly** from 2020-02-09 — a reference
basket, quantities from cap weights, no fees, no funding, price return only. The
benchmark and strategy **books open on 2021-07-05**, the first scored day, and
their opening costs fall inside the scored window as registered. The warm-up
carries no book and contributes to no statistic.

## C7. Smaller corrections

- `fee` is split into **`fee`** (exchange) and **`slippage`**, since the
  registered 10 bp per side is ≈5 + ≈5. The ledger stores both; `net = price_pnl
  + funding_pnl − fee − slippage`.
- A missing or NaN **exposure** is a hard failure, never silently zero.
- Pre-flight validation before the loop: no duplicate dates, member uniqueness,
  weights summing to 1, and complete price/funding/mark coverage for every
  member-day. A gap raises before any number is produced.
- No NaN may enter P&L, quantities, exposure or funding silently.

## C8. Additional acceptance tests

Beyond the identity and the constituent-swap fixture:

1. A full 00:00 → 08:00 → 16:00 → next-00:00 timeline **including a position
   reversal**, checking each settlement is charged on the position actually held.
2. Missing funding, or a missing mark at a settlement, raises.
3. A temporary halt followed by resumed trading: no lost or invented P&L.
4. Contract death **long and short**: the short fixture must realise a gain.
5. A settled leg's allocation is **not** redistributed before the next
   reconstitution.
6. Opening costs appear on the first scored date, not before it.
7. Benchmark price returns match an independently constructed basket index.
8. Benchmark turnover contains only reconstitution, settlement and rebrand
   trades — no price-drift trading.

---

# Amendment 5 — holding-period returns and an independent reference basket

**Dated 2026-09-25. Pre-result: no strategy return has been computed.**

A second technical review found two P&L-critical defects remaining after
Amendment 4, plus eight smaller inconsistencies. All are settled here.

## D1. Rows are holding periods, not calendar events

Amendment 4 said a position entered at D 00:00 pays the D 00:00 settlement. It
cannot: the trade happens *after* that settlement. The Amendment 4 row also mixed
two positions — the pre-trade position's price P&L and 00:00 funding, and the
post-trade position's later funding — in one observation.

**Registered convention: one row = one holding period `[D, D+1)`, one position
throughout.**

| step | belongs to |
|---|---|
| D 00:00 settlement | the **previous** position — it closes the row `[D−1, D)` |
| trade at D 00:00, fee and slippage charged | the start of row `[D, D+1)` |
| price P&L `q_D × (open[D+1] − open[D])` | row `[D, D+1)` |
| every settlement in `(D 00:00, D+1 00:00]` | row `[D, D+1)` |

The row key is `holding_period_start = D`. Price, funding and the
signal-selected position now refer to the same position over the same interval,
which is what a "daily return" has to mean for the regressions to be
interpretable.

The separate `funding_open_pnl` / `funding_post_pnl` columns of Amendment 4 are
no longer needed for correctness — each row has one position — but funding is
still stored per settlement so a sign change at a trade stays auditable.

## D2. The reference index uses the previous basket value, not capital

**Verified [V]:** dividing price P&L by the original capital and compounding is
wrong once quantities are fixed and the basket value drifts. On a two-day fixture
of two constituents rising 10% a day, the correct index is 121.00 and the draft's
method gives 122.10 — an error that compounds, and that would have entered the
EWMA, the forecast and every position.

**Registered:**

```
basket_return[D] = Σ_i weight[D−1, i] × constituent_return[D, i]
index[D]         = index[D−1] × (1 + basket_return[D])
```

equivalently `value[D] / value[D−1] − 1` where `value = Σ qᵢ · priceᵢ`. The
denominator is the **previous basket value**. This is the chain-linked formula
already registered; the walkthrough's shortcut was not equivalent to it.

## D3. The reference basket is maintained independently

Targets must be derived from the **reference basket's** weights, not from the
strategy's own book. Otherwise, once exposure reaches zero the book empties, the
weights are unrecoverable, and the strategy can never reopen.

**Registered:** a reference basket (quantities from cap weights at each
reconstitution, drifting in between, no costs, no funding) is maintained for the
whole sample. It defines the index and supplies the weights that both the
benchmark and the strategy scale.

## D4. Opening is its own event

**2021-07-05 is a Monday; that month's reconstitution is 2021-07-01 [V].** The
books therefore open **mid-month**, on an ordinary day.

**Registered:** book opening is an explicit `opening` event on the first scored
day, using the membership in force and the reference basket's **drifted** weights
for that day. Opening costs are charged then, inside the scored window.

## D5. `idle_weight` resets at reconstitution

The Amendment 4 sketch multiplied new reconstitution targets by `(1 −
idle_weight)`, which would have left a dead leg's allocation idle **forever**.

**Registered:** `idle_weight` applies only between the settlement and the next
reconstitution. At reconstitution it resets to zero and the full target is
allocated across the new members.

## D6. Settlement takes every settlement up to the delisting timestamp

Delisting can happen intraday. The closing row includes **all** funding
settlements from the start of its holding period up to the actual delisting
timestamp — not only the 00:00 one.

## D7. Settlement charges the exchange fee, not modelled slippage

Automatic settlement is executed by the exchange at its own computed price; there
is no spread to cross. **Registered:** a settlement row charges the **taker fee
only**, with `slippage = 0`, and the two columns stay separate as in Amendment 4
C7. The sketch that merged them into a single `cost` field is superseded.

## D8. Pre-flight validation allows classified lifecycle events

"Complete price coverage for every member-day" contradicts the `halted` state.
**Registered:** every member-day must have either valid data **or** an explicitly
classified lifecycle event (`halted`, `settling`, `idle`, `rebranded`). An
unclassified gap raises; a classified one is accounted for by its state's rule.

## D9. Test corrections

- The benchmark trade-event test allows `opening` alongside `reconstitution`,
  `settlement` and `rebrand`.
- The idle-weight test applies only to the interval **between** the settlement
  and the next reconstitution; after it, allocation returns to full.

## D10. Mark prices are "timestamp-aligned", not "exact"

**Verified [V]:** 3,267 of BTCUSDT's 7,305 settlements are not exactly on the
hour (e.g. `2026-08-31 16:00:00.001`). Settlement times are floored to the hour
to join the 1-hour mark klines.

The mark used is therefore described as **timestamp-aligned** to the settlement,
not as the exact mark Binance applied. Before the first registered run, the join
is validated against a handful of settlements where the applied mark can be
checked independently, and any discrepancy is reported.
