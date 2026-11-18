# Pre-registration — Trend Following in Crypto Beta

**Dated 2026-09-22. Frozen on commit. No strategy return has been computed.**

> **Amended 2026-09-23, pre-result.** See
> `pre_registration_amendment_2026-09-23.md`, which governs where the two
> conflict: reconstructed-data terminology, the rebuilt speed rule (blend is now
> 8/32, 16/64, 32/128, 64/256; warm-up 512 days; scored window **2021-07-05 →
> 2026-08-31, 1,884 days** per Amendment 2), H2 as a superiority hurdle rather than an equivalence
> margin with corrected MDEs, the frozen implementation config and pseudocode,
> funding at every recorded settlement, the ban on return clipping, the H1 gate
> definition and the typed trial registry. **Amendment 3 (2026-09-24)** adds the
> common-β alpha attribution (price / funding / cost), H2 realised-risk
> diagnostics, frictions as a primary *descriptive* estimand, the funding sign
> convention, and the LUNA-settlement and terminal-liquidation sensitivities.
> **Amendment 4 (2026-09-24)** settles the event timeline after a technical
> review: bars are stamped with their OPEN time so `open[D]` is the price at
> D 00:00; funding is computed per settlement on that settlement's mark price
> (1h `markPriceKlines`), never as a daily sum; contract states are explicit
> (active / halted / settling / idle / rebranded); settlement realises P&L with
> no principal proceeds and the freed weight goes idle rather than to the
> survivors; the benchmark holds basket *units* while the strategy targets a
> *notional*; the index is built costlessly across the warm-up with books opening
> on the first scored day. **Amendment 5 (2026-09-25)** makes a ledger row a
> *holding period* `[D, D+1)` with one position throughout (the D 00:00
> settlement belongs to the previous row), fixes the reference index to divide by
> the **previous basket value** rather than capital, maintains the reference
> basket independently so the books scale its weights, makes book opening an
> explicit event, resets `idle_weight` at reconstitution, and charges settlement
> the exchange fee only. Machine-readable: `prereg_support/frozen_config.yaml`.

Labels: **[V]** verified in this repository or by direct check · **[L]** from the
cited literature · **[O]** judgement.

Supersedes the working draft `pre_registration_2026-09-21.md`. Supporting code:
`prereg_support/`.

---

## Part 1 — Why this document exists

### What pre-registration is

Fixing every choice that could be tuned, in writing, with a date, **before**
seeing the result. A result produced under fixed rules is a *test*. A result
produced after adjusting the rules is a *selection*.

The arithmetic is unforgiving here. The standard error of an annual Sharpe ratio
over this sample is ≈ 0.46 at the registered 5.16-year window **[V]**. A strategy with *no* skill, tried in 20
variants, produces a best-of-20 Sharpe around 0.75 by luck alone **[V:
arithmetic]** — better than the 0.42 this project previously reported and has
since voided.

### Where the idea comes from

- **Medicine and psychology.** Ioannidis (2005) argued most published findings
  are false, largely through flexibility in design and analysis **[L]**. Nosek,
  Ebersole, DeHaven & Mellor (2018) set out pre-registration as the remedy: it
  separates *generating* a hypothesis from *testing* one **[L]**.
- **Finance.** Harvey, Liu & Zhu (2016) showed that after decades of factor
  mining, conventional significance thresholds are far too lenient **[L]**;
  Harvey's 2017 AFA presidential address makes the same case about p-hacking in
  financial economics **[L]**. Bailey & López de Prado (2014) deflate the Sharpe
  ratio for the number of trials and non-normality; Bailey, Borwein, López de
  Prado & Zhu (2017) estimate the probability of backtest overfitting **[L]**.
  Both need an honest count of trials — hence the registry in Part 6.

### What it protects against *here*, concretely

1. Choosing the EWMA speed that happened to win on 2020–2026.
2. Choosing the volume floor, buffer width or cost level after seeing which made
   the result positive.
3. Starting or ending the sample where the curve looks best.
4. Reporting the buffered-net number and quietly dropping the unbuffered one.
5. Calling the per-coin variant "robustness" when it was really a second shot.

### What it cannot fix

**This sample has already been mined.** 2020–2026 has been inspected many times
in this project, so pre-registering today does **not** make it out-of-sample. It
limits *further* freedom and makes the remaining freedom countable. Only forward
data is genuinely out-of-sample (Part 8).

In particular, **"faster EWMAs win in crypto" and "drop 64/256" were both learned
from this sample**. Neither may be used to choose parameters. Where a decision
below happens to coincide with one of them, the coincidence is disclosed.

### Trial counting, disclosed

Earlier work on this sample (the June 2026 IC variants; the voided Sharpe of
0.42) is **not** enumerated in the registry. The deflated Sharpe therefore counts
only trials from this date forward, and this document states plainly that this is
not the first look at the sample **[O]**.

---

## Part 2 — The claim

**Working title:** *Trend Following in Crypto Beta: Evidence from Reconstructed
Constituent-Level Perpetual-Futures Portfolios.* (The earlier subtitle
"Point-in-Time" was withdrawn in Amendment A1: the selection rule is
point-in-time, the underlying data is a reconstruction.)

**Framing.** Aggregate crypto time-series momentum: **one** time-series trend
signal applied to a predetermined, **historically reconstructed, one-day-lagged**
basket (the *selection rule* is point-in-time; the underlying CMC data is a 2026
reconstruction with known retroactive restatements — see Amendment A1), asking whether aggregate
crypto beta can be timed. This is **not** cross-sectional momentum — no ranking
of coins, no winners-versus-losers. The comparable literature is BTC and
crypto-index trend following, not the cross-sectional crypto factor literature.

**Claim.** After funding, turnover, fees, slippage and delisting settlements,
does trend-following a reconstructed, one-day-lagged large-cap crypto basket add
value beyond **both** passive basket exposure **and** an otherwise identical
BTC-only trend strategy? Scope (Amendment A3): "adds value" means positive α over
the passive basket **and** a higher net mean return than matched BTC trend — not
a claim about return outside their joint span.

**Economic hypothesis.** Aggregate crypto beta shows enough time-series
persistence that conventional trend following — Carver's standard EWMAC speeds,
filtered only by his cost and data rules — improves its return distribution,
especially by cutting exposure or going short during sustained declines.

**Null.** After realistic perpetual-futures economics, the strategy has no
positive alpha over the funded same-basket benchmark.

**Not claimed:** that momentum selects winning coins; that ten coins provide ten
independent trends; that the data is a true vintage point-in-time feed; that the strategy beats buy-and-hold on absolute return; that
2020–2026 is out-of-sample; that the instrument represents all crypto rather than
Binance-tradable large caps.

**Positioning (Amendment A6).** Not a new momentum signal: a pre-registered,
implementation-aware **replication and decomposition** of conventional crypto
trend conclusions under constituent-level perpetual-futures economics.

**Contribution either way.** Existing evidence for crypto trend following rests
largely on spot prices, BTC alone, survivor universes and incomplete cost models.
If the result is positive, aggregate crypto trend is exploitable after
implementation frictions. If negative, apparent crypto trend profits do not
survive perpetual-futures economics. Both are publishable; that symmetry is the
safeguard.

**Precision of language.** The basket is *not* "95% BTC by weight". BTC averages
**66%** of basket weight (range 53–80%); **95.4%** is the share of daily basket
return *variance* explained by BTC (R² = 0.954, daily correlation 0.977) **[V]**.

---

## Part 3 — The frozen protocol

### 3.1 Instrument and universe

| # | decision | value |
|---|---|---|
| U1 | instrument | cap-weighted top-10 basket of Binance USDT-M perps, **historically reconstructed and one-day-lagged** (not vintage point-in-time; Amendment A1), from `data/universe/selector.py` |
| U2 | ranking | by `market_cap`, never CMC's `rank` field |
| U3 | information lag | membership uses the D−1 snapshot; signal at close D; position held over D+1 |
| U4 | reconstitution | **monthly** (primary). Quantities fixed between reconstitutions, so weights drift with prices |
| U5 | membership buffer | **enter at rank 10, exit at rank 13**, applied identically to strategy and benchmark (it defines the instrument) |
| U6 | investability screen | **30-day median of the contract's own Binance daily dollar volume ≥ $10m**, lagged, past data only. Replaces the CMC 24h-volume floor (multi-venue and wash-traded; Cong, Li, Tang & Yang 2023 estimate >70% wash trading on unregulated venues **[L]**). Open interest is *not* a screen: `metrics` archives begin 2021-12 for alts **[V]** |
| U7 | exclusions | stablecoins, wrapped and gold-pegged tokens by `cmc_id` (`selector.py`) |
| U8 | asset key | `cmc_id`, never the symbol |
| U9 | tradability | the contract must have traded *that day* (volume > 0), read from its bars |

### 3.2 Signal

| # | decision | value |
|---|---|---|
| S1 | rule | EWMA crossover on the **basket index level**, one signal for the basket |
| S2 | speeds | **8/32, 16/64, 32/128, 64/256**, equal-weighted blend (Amendment A2) |
| S3 | speed selection rule | **(Amendment A2)** a speed is admitted only if its cost stays ≤ **0.13 SR/yr [L]** at 10 bp per side across the **whole registered 50–80% volatility range**, so no speed turns on one assumed volatility. 2/8 fails everywhere (0.20–0.32); 4/16 fails below 63% vol (0.10–0.16); the other four pass throughout. Evidence: `prereg_support/speed_cost_sim.py` |
| S4 | normalisation | (fast EWMA − slow EWMA) ÷ (price × daily return vol) — Carver's definition **[V]** |
| S5 | EWMA initialisation | `adjust=True` (no phantom pre-sample history). Current `ewma_crossover.py` uses `adjust=False` and must change **[V]** |
| S6 | forecast scalars | **fixed**: 8/32 **5.3**, 16/64 **3.75**, 32/128 **2.65**, 64/256 **1.87** (Carver, pysystemtrade `futures_chapter15`) **[V]** |
| S7 | blend multiplier (FDM) | **fixed 1.15**, from 1/√(w′Hw) with equal weights and correlations from synthetic random walks (`prereg_support/speed_cost_sim.py`; no project data) **[V]** |
| S8 | barred | rolling scalar estimation in `strategies/base.py` and rolling FDM in `strategies/combination/forecast_combiner.py` — both fit to this sample |
| S9 | cap | forecast clipped to **±20** |

### 3.3 Risk and sizing

| # | decision | value |
|---|---|---|
| R1 | volatility estimator | 32-day EWMA of past returns (Carver uses 35; difference disclosed) |
| R2 | volatility floor | **5th percentile of the estimate's own past 500 days** (expanding from 100 days; past data only) — Carver's `robust_vol_calc` rule **[V]**. Applied to both forecast normalisation and sizing |
| R3 | annualisation | **365** days |
| R4 | target volatility | **30%** |
| R5 | leverage cap | **2.0**, applied to x\* *and again* to the final position after buffer logic |
| R6 | ideal exposure | x\*ₜ = (forecastₜ / 10) × (target vol ÷ estimated volₜ) |
| R7 | cash rate | **0**. Perps are unfunded: P&L is already an excess return, and collateral interest accrues equally to strategy and benchmark, cancelling in excess returns. Affects reported absolute returns only, never α |

### 3.4 Execution

| # | decision | value |
|---|---|---|
| X1 | policies | **full rebalancing** (trade to x\* daily, buffer = 0) and **buffered** (primary) |
| X2 | band | half-width = **0.10 × (target vol ÷ estimated vol)** — 10% of *standard* exposure, centred on x\*. Inside → no trade; outside → trade **to the nearest edge** (Carver) |
| X3 | membership trades | monthly constituent changes always execute, whatever the band says |
| X4 | leg trading | when aggregate exposure changes, all legs scale **proportionally at current drifted quantities**. Weights reset only at reconstitution |
| X5 | accounting | per leg: target notional and quantity, turnover from each leg's marked pre-trade position, per-leg fee/spread/slippage, funding on the position held at each settlement. **Never** costs from the change in aggregate exposure |
| X6 | funding timestamp | trades execute at the 00:00 UTC close **after** that settlement: the **pre-trade** position pays the 00:00 settlement, the new position pays **every subsequent recorded settlement, whatever its interval**. The data contain 8h, 4h, 2h and 1h intervals **[V]**; the interval is read from the data, never assumed (Amendment A4) |
| X8 | return clipping | **forbidden.** `backtesting/enginee.py:174` clips daily net returns at ±50% **[V]** and must not be used for any registered run |
| X9 | capital and compounding | fixed notional capital, no compounding; turnover = Σ per-leg traded notional ÷ capital. Opening costs enter the first scored return; positions are marked to the final close with no terminal liquidation cost |
| X7 | funding treatment | a **holding** cost, not a transaction cost; sits inside gross return. Never filled with zero. Sign convention (Amendment 3): rates are positive when longs pay, so `funding_pnl = − position × rate`, `gross = price_pnl + funding_pnl`, `net = gross − fees − slippage`. A short with a positive rate receives |

### 3.5 Costs

| # | decision | value |
|---|---|---|
| C1 | **primary** | **10 bp per side**, every leg, every date: ≈5 bp Binance VIP-0 USDⓈ-M taker fee **[L]** + ≈5 bp spread/impact allowance **[O]**. Taker assumed throughout |
| C2 | stress | **5 bp and 20 bp** |
| C3 | measured (registered) | 5 bp fee + half the **Abdi & Ranaldo (2017)** spread, rolling 60-day past-only window; negative or missing → 10 bp fallback **[L]** |
| C4 | measured (registered) | 5 bp fee + half the **effective spread from Binance `aggTrades`**: in each 1-minute window with trades on both sides, mean(buyer-initiated price) − mean(seller-initiated price); daily = median across windows; contract-quarter = median across sampled days |
| C5 | aggTrades sampling | deterministic: the **first seven days of the middle month of each quarter** (Feb/May/Aug/Nov 1–7) for each member contract. Separately and labelled apart: the **final 30 days before settlement** of every delisted contract (stress sample, never averaged into the base estimate) |
| C6 | validation | true quotes from Binance `bookTicker`, which exists **only 2023-05-16 → 2024-04** for every symbol **[V]**; three-way comparison in that window. It is a relatively calm period and likely understates 2022-era spreads |

No authoritative 2020–2026 Binance fee history was located; the stress range
carries that uncertainty. **The validation cannot change C1.** If the estimators
understate true spreads, that is reported as a limitation.

### 3.6 Sample

| # | decision | value |
|---|---|---|
| T1 | basket start | **2020-02-09** (Amendment 2: the registered volume screen needs 20 observations, and the basket cannot field ten eligible names before then) |
| T2 | warm-up | **512 observations** (2 × slowest span, 256). Signals computed from the start; **no position before the close of 2021-07-04** (Amendment 2). Warm-up days excluded from all statistics (NaN, not zero) |
| T3 | scored window | **2021-07-05 → 2026-08-31**, 1,884 days (~5.16 years), identical for strategy and benchmark (Amendment 2) |
| T4 | end date | funding coverage ends 2026-08-31. Prices run to 2026-09-17 but a net number past 08-31 would be missing its largest cost |

### 3.7 Contract death and halts

Binance settles a delisted perp at the **average index price every second over
the final 30 minutes** (1,800 observations) and charges the **taker fee**; the
last 30 minutes are reduce-only and open positions are settled automatically
**[V: Binance delisting FAQ]**. "Exit at the last daily close" is wrong in both
price and fee.

Measured scope **[V]**: of 10 mapped contracts that stopped trading before
2026-08-31, only four were ever basket members (MATIC 15 months, EOS 11, TON 11,
LUNA 8), and **only LUNAUSDT was still a member when its contract died**
(2022-05-12) *[superseded for the registered membership: see Part 10, Correction 1]*. MATIC, EOS and TON left the top 10 between 10 months and 4 years
before retirement, and all three retirements were rebrands (MATIC→POL,
EOS→Vaulta/A, TON→GRAM) with ~9-day gaps before the successor listed. **FTT was
never a member**, so the FTX case does not touch this study. Among the 28 member
contracts only ICP ever halted (103 zero-volume days, 2022-06-11 → 2022-09-21),
and ICP was a member only in 2021-06 — **no halt ever affects a held position**.

Rules, written even where nothing currently triggers them:

1. Dies while held → exit at the reconstructed settlement price: mean of
   1-minute `indexPriceKlines` over the final 30 minutes before the announced
   delisting time, taker fee charged. (30 one-minute closes approximate 1,800
   one-second observations; stated as such.)
2. Proceeds sit in **cash** until the next reconstitution; nothing is
   redistributed to other legs.
3. Halts while held → marked at the last traded price; exits to cash at the first
   subsequent traded bar or the next reconstitution, whichever comes first.
4. Rebrand while held → migrate to the successor at its first traded bar; cash in
   the gap.
5. **LUNA is reported separately**: its P&L contribution, and the headline result
   with and without it.

### 3.8 Benchmark

The same engine with exposure fixed at **1.0**: no forecast, no vol target, no
band; identical membership rule, per-leg costs, funding and delisting treatment
(`N-bench`).

---

## Part 4 — The primary test

### 4.1 Series produced

Six strategy series and three benchmark series, all always reported:

| accounting layer | full rebalancing | buffered |
|---|---|---|
| price only (what a spot/index study measures) | P-full | P-buf |
| gross = price + funding | G-full | G-buf |
| net = gross − fees − slippage | N-full | **N-buf (primary)** |

Benchmark: P-bench, G-bench, **N-bench**. The gaps between rows are findings in
their own right — funding drag, cost drag, and the value of execution discipline
— and are paired same-day differences, so they are estimated far more precisely
than any Sharpe.

### 4.1b Alpha attribution (Amendment 3)

β is estimated once in the primary net regression and held fixed while the same α
is decomposed across the ladder layers:
`α_net = α_price + α_funding − α_cost`, each term `mean(r_strategy,layer −
β·r_benchmark,layer) × 365` on the identical sample. The terms sum to the
headline by construction. Layer-by-layer regressions with their own betas are
diagnostics, never attribution. Funding P&L is also split by long versus short
days. If α is concentrated in the funding term, the finding is *timing total
perpetual returns*, which is a different mechanism from price timing and is
reported as such rather than assumed away.

### 4.2 H1

r(N-buf)ₜ − r_cash = α + β·(r(N-bench)ₜ − r_cash) + εₜ  (daily; α annualised ×365)

**H0: α ≤ 0 versus H1: α > 0**, one-sided at 5%.

Interpretation: α > 0 means the strategy improves the investment opportunity set
beyond passive exposure to the basket. It does **not** mean every investor should
add it — residual volatility and correlation matter, which is what the appraisal
ratio (α ÷ residual vol) quantifies.

### 4.3 Inference

Stationary block bootstrap (Politis & Romano 1994 **[L]**) on **paired daily
return pairs** (strategy, benchmark) — never on raw prices, since rebuilding
EWMAs across joined blocks would create artificial signal paths. 10,000
replications; mean block length **20 days**; α and β re-estimated in every
replication. **Pass = the 5th percentile of the bootstrapped α exceeds zero.**
Sensitivity at mean block lengths 10, 40 and 60. Newey & West (1987) standard
errors **[L]** reported alongside as an analytical cross-check; if the two
disagree materially, that is reported, not resolved by preference.

### 4.4 Result classification, fixed in advance

| # | condition | wording |
|---|---|---|
| 1 | α > 0, 5th percentile > 0 at 20-day blocks, **and** still > 0 at 40 and 60 blocks **and** at 20 bp costs | **statistically supported historical alpha** (not out-of-sample proof) |
| 2 | α > 0 but the interval includes 0 | **economically positive, statistically inconclusive** |
| 3 | α ≤ 0 | **the primary hypothesis fails** |
| 4 | passes at 20-day blocks but fails at 40/60 or at 20 bp | **fragile** |

Given SE ≈ 0.4 in Sharpe units, **case 2 is the most likely outcome for a modest
true effect**, and is stated as such rather than spun.

### 4.5 Full versus buffered

| N-full | N-buf | reading |
|---|---|---|
| α > 0 | α > 0 | survives even naive trading |
| α ≤ 0 | α > 0 | exists only with execution discipline — reported as exactly that |
| α ≤ 0 | α ≤ 0 | fails |
| α > 0 | α ≤ 0 | suspicious — audit the buffer before reporting |

**G-buf tripwire.** Buffering trades later, so gross should fall slightly
(G-buf < G-full) while net rises (N-buf > N-full). If **G-buf exceeds G-full**,
the buffer may be seeing information it should not: audit the code before
reporting. In a single sample this can happen by chance, so it triggers an audit,
not an automatic bug verdict.

### 4.6 Minimum detectable effects

IID design approximations, 80% power, 5% one-sided, **1,884 days (5.16 y)**;
MDE ≈ 2.485·σ/√T **[V]** (corrected in Amendment A3):

| test | assumed noise | detectable effect |
|---|---|---|
| H1 | ~28%/yr residual vol | ≈ **30.6%/yr** α (appraisal ratio ≈ 1.09) |
| H2 gap, ρ = 0.95 | 9.5%/yr | ≈ **10.4%/yr** |
| H2 gap, ρ = 0.97 | 7.3%/yr | ≈ **8.0%/yr** |
| H2 gap, ρ = 0.977 | 6.4%/yr | ≈ **7.0%/yr** |
| two-factor (S15) | its own residuals | computed at run time |

0.977 is the *basket/BTC* correlation, not the correlation between the two net
trend strategies; the realised strategy correlation is reported and the MDE
recomputed from it. **H1 can only confirm a large effect.**

---

## Part 5 — H2: beyond BTC-only trend

The strongest sceptical objection — "this is Bitcoin trend following with extra
instruments and costs" — is tested as a declared hypothesis, not an appendix.

**Headline scope (Amendment A3).** The claim is: positive α over the passive
basket **and** a higher net mean return than matched BTC trend. It is *not* a
claim about return outside the joint span of BTC-trend and passive beta; S15
remains a stricter secondary analysis and never gates the headline.

**H2.** Paired daily difference Δₜ = N(basket-trend)ₜ − N(BTC-trend)ₜ;
**H0: E[Δ] ≤ 0**, one-sided 5%, same paired stationary block bootstrap. The
BTC-only strategy uses **identical** dates, target vol, leverage cap, buffer,
execution assumptions, funding, fees and slippage.

**The H1 gate is the registered primary test only** — the bootstrap at 20-day
mean blocks and 10 bp costs. Failures at 40/60-day blocks or at 20 bp change the
*label* to "fragile"; they do not redefine the formal test or close the H2 gate
(Amendment A5).

**Fixed-sequence gatekeeping.** H1 is tested at 5%; **H2 only if H1 passes**.
This controls family-wise error at 5% without splitting the threshold (FDA
multiple-endpoints guidance; Dmitrienko et al. 2007 **[L]**). If H1 fails, H2 is
still estimated and reported, but it cannot support the combined headline claim.

| H1 | H2 | defensible conclusion |
|---|---|---|
| pass | pass | basket trend adds value beyond passive beta and BTC trend |
| pass | fail | trend works; incremental basket value unproven |
| fail | pass | basket may improve on BTC trend, but active trend does not beat passive beta |
| fail | fail | headline claim unsupported |

**Realised-risk diagnostics (Amendment 3).** Both legs target 30% ex ante, so
they are matched by construction but may differ ex post. Always reported:
realised volatility, mean absolute exposure and turnover for each leg, and an
ex-post volatility-matched gap as a standing diagnostic. The registered H2
statistic is unchanged.

**Basket-superiority hurdle (Amendment A3; this is *not* an equivalence test).**
An insignificant H2 does **not** prove BTC-only is sufficient. δ = **5 percentage
points of annualised net return**: the amount a ten-contract basket must add over
matched BTC trend to justify its extra funding, liquidity, monitoring,
reconstitution and delisting complexity — ≈**0.17 incremental Sharpe** at the 30%
target **[V]**, an institutional hurdle **[O]**. If the one-sided 95% bound rules
out an advantage of that size, the evidence supports **BTC operational
sufficiency**, not statistical equivalence. Note in advance: δ sits *below* the
H2 MDE, so this test can return "no material advantage" or "inconclusive", and
can only return "basket superior" for a gap far larger than δ.

**Accounting bridge** (exact, not inference):

> basket trend − BTC trend = basket alt-leg P&L + (basket BTC-leg P&L −
> standalone BTC-trend P&L)

split into price, funding, fees and slippage. Attribution, not causal proof.

**Secondary (S15), two-factor conditional alpha:**
N(basket-trend) = α + b₁·N(BTC-trend) + b₂·N(benchmark) + ε. This is *not* formal
mean–variance spanning, which imposes further restrictions (Huberman & Kandel
1987 **[L]**).

**Alternative-index robustness**, not dose-response: 30% and 35% BTC-capped
baskets. Capping changes the instrument, signal, vol scaling, funding, turnover
and benchmark simultaneously, so a rising α would be suggestive, never proof
that alts caused it.

---

## Part 6 — Registered runs and the trial registry

**Primary:** N-buf versus N-bench (H1). **Declared hypothesis test:** H2 (fixed
sequence). All runs below are executed and reported whatever they show, and each
takes a registry row. **Correction (Amendment A5):** it is *not* true that none
can bear on the headline — S1 supplies H2's comparator and S12 decides supported
versus fragile. The registry is typed (configurations / cost scenarios /
statistical analyses / diagnostics / declared comparators); only strategy
configurations count as trials in the deflated Sharpe, which is reported as a
**partial lower-bound** correction with sensitivity to assumed prior trial
counts.

| # | run | purpose |
|---|---|---|
| S1 | BTC-only trend | attribution; the H2 comparator |
| S2 | 30% BTC-capped basket | alternative index |
| S3 | 35% BTC-capped basket | alternative index |
| S4–S7 | each speed alone (8/32, 16/64, 32/128, 64/256) | breakdown of the blend |
| S8 | sign-only forecast (±1, no sizing by strength) | how much work the scaling does |
| S9 | weekly reconstitution | schedule sensitivity |
| S10 | daily reconstitution | schedule sensitivity (a different, higher-turnover instrument) |
| S11 / S12 | cost stress 5 bp / 20 bp | cost sensitivity |
| S13 / S14 | measured costs: Abdi–Ranaldo / aggTrades effective spread | realism checks |
| S15 | two-factor conditional alpha | secondary to H2 |
| S16 | **per-coin TSMOM** — one signal per constituent, each sized by its own vol, summed | **a different hypothesis** (do individual coins trend?). Run **last**, after the primary is locked. It cannot rescue a failed H1 |

**Reconstitution frequency evidence** (universe only, no strategy returns, 10/13
buffer; reproducer `prereg_support/universe_churn.py`) **[V]**:

| schedule | rebalances | entries/yr | turnover |
|---|---|---|---|
| monthly | 79 | 4.5 | 41.7%/yr |
| weekly | 344 | 4.7 | 72.2%/yr |

Weekly barely changes membership — the rank buffer absorbs churn — but adds
~30 pp/yr of turnover through weight resets (≈3 bp/yr of capital at 10 bp).
Monthly is standard index practice and stays primary; weekly is S9. Frequency
does not change the number of scored observations and so does not affect power.

### Trial registry (starts empty)

Every run from this date forward gets a row, including failures. Anything not in
the table above is additionally flagged as a post-registration trial.

| date | run id | what changed vs the primary | result | returns stored at |
|---|---|---|---|---|
| | | | | |

---

## Part 7 — Reported regardless of outcome

**Primary descriptive objective (Amendment 3).** Besides the confirmatory H1 and
the gated H2, the study has one primary *descriptive* estimand: the funding and
execution-cost drag of implementing crypto trend through perpetual futures,
reported with intervals for strategy, benchmark and BTC-trend. It is not a
second hypothesis test and creates no multiplicity; it is the measurement this
sample can make precisely, because the ladder layers are paired same-day
differences on identical positions — conditional on the cost model.

1. The **nine-series ladder** (P/G/N × full/buffered, plus three benchmark
   series), each with turnover, trade count and average exposure. Within a
   policy, all three layers use one common position path and one common capital
   denominator, or the differences do not isolate funding and costs.
2. **Sharpe with bootstrap confidence intervals** for every run. Sharpe is
   defined as annualised arithmetic mean excess return ÷ annualised volatility of
   excess returns. `enginee.py:219` computes CAGR ÷ volatility **[V]**, a
   different statistic, and must not be used.
3. **α, β, appraisal ratio** (α ÷ residual vol), plus Newey–West as a
   cross-check.
4. **Convexity**: N-buf = α_c + β·r_ex + γ·[r_ex² − mean(r_ex²)] + ε, with the
   squared term centred so the intercept stays interpretable; estimated on daily
   **and** non-overlapping monthly returns, since trend convexity is expected at
   longer horizons (Treynor & Mazuy 1966 **[L]**). Explanatory, never a second
   definition of success.
5. **Calendar-year** results with confidence intervals, labelled **descriptive,
   not tests**. (Regime sub-periods such as "2022" are *not* used: those
   boundaries are hindsight from this sample.)
6. **Coverage**: share of true top-10 market cap without a Binance perp
   (mean 0.41%, ≈0 after 2020) **[V]**.
7. **LUNA's contribution**, with the headline shown with and without it, and
   (Amendment 3) the same result under the naive "last traded close" settlement
   as a counterfactual, so the reader can see whether the conclusion depends on
   the reconstructed settlement price.
8. **Forecast diagnostics**: realised mean |forecast| per speed and for the
   blend, share of days at the ±20 cap.
9. **H2 and the δ = 5% equivalence result**, including when inconclusive.
10. **Cost comparison**: 10 bp assumed vs Abdi–Ranaldo vs trade-based, with the
    bookTicker validation.
11. **The result classification** (supported / inconclusive / fails / fragile)
    stated explicitly.
12. **How often the volatility floor and the leverage cap bind** (Amendment 3),
    and a **terminal-liquidation sensitivity** beside the primary's
    mark-to-final-close treatment.
13. **Deflated Sharpe** computed over the registry, with the disclosure in
    Part 1 that earlier trials are not enumerated.

---

## Part 8 — Forward paper trading

**Two dates.** *OOS data start* = the day this document is committed: every price
after it was unseen when the rules were fixed. *Live record start* = the first
day the daily job runs and commits on its own. The gap is backfilled once
`basket.py` exists and is **labelled as backfilled** — weaker evidence, since the
code was written during that window; the defence is that the spec is frozen today
and every code change is logged.

**Frozen:** all parameters plus the git commit hash of the code.

**Daily record, append-only, committed daily:** basket index, each speed's
forecast and the blend, vol estimate, target exposure, per-leg target quantities,
prices used. Commit timestamps are the evidence.

**Allowed without breaking the registration:** documented bug fixes,
exchange-driven events (rename, delisting), logged data outages. **Not allowed:**
any parameter change, adding or removing a speed, re-scaling forecasts.

**Power, stated in advance.** Confirming a Sharpe near 0.5 at conventional
significance would take ~25+ years of forward data; one year has a Sharpe SE near
1.0 **[V: arithmetic]**. The forward record **cannot** reach statistical
confirmation within this study's horizon. What it can do: verify the
implementation (realised vol vs the 30% target, turnover vs backtest, cost drag
vs 10 bp), catch a broken result quickly, and accumulate.

**Cadence:** 6 and 12 months, then annually; no changes between reviews; the same
ladder and diagnostics as the backtest.

**Lag:** funding publishes monthly, so net numbers close ~1 month behind; gross
and positions are daily.

**Forward failure conditions** (implementation wrong, not unlucky): realised vol
> 1.5× target over any 3-month window; realised cost drag > 2× the assumed 10 bp;
realised positions diverging from the recorded targets.

**Future work:** real fills are the only way to verify the 10 bp assumption.

---

## Part 9 — Falsification

The hypothesis is abandoned if:

1. **H1 lands in case 3** (α ≤ 0 on N-buf versus N-bench), or
2. **H1 is case 4 (fragile)** *and* the measured-cost runs (S13/S14) put costs
   above the 10 bp assumption, or
3. the **accounting identity fails**: net ≠ price P&L + funding − fees −
   slippage, per leg, per day. Then no result is reported at all until it
   reconciles.

A positive result does **not** establish the effect: it establishes a magnitude
on a mined sample, to be tested forward.

---

## Part 10 — Change log

Amendments after this date are permitted; silent ones are not. Each gets a row.

| date | what changed | why | before any result? |
|---|---|---|---|
| 2026-09-22 | document frozen | — | yes |
| 2026-09-30 | **Correction 1** (same file; documentation only, no design change): the statement that only LUNAUSDT was still a member when its contract died was measured under an earlier membership rule. Under the registered monthly buffered membership (enter at 10, exit at 13, carried from 2020-02-09) LUNAUSDT is never held: 79 selections, 24 distinct contracts. It is held only in the weekly (Monday) sensitivity, the week of 2022-05-09; the daily sensitivity drops it before it dies. See `basket_event_inventory.md`. The LUNA reporting rules (with/without, settlement counterfactual) stay registered and apply to the variants that hold it; the primary run exercises no contract death. The original wording is kept above and in the archived, timestamped version (`prereg_support/ots_2026-09-24_amendment3/`) | regenerated event inventory from the actual registered membership (`basket_event_inventory.md`) | n/a: corrects a description, changes no rule or result |
| 2026-09-30 | **Amendment 6** (`pre_registration_amendment_2026-09-23.md`): last scored holding period starts 2026-08-30 and ends 2026-08-31 00:00, exiting at open[2026-08-31]; scored window 2021-07-05 → **2026-08-30**, **1,883 days** (was 2021-07-05 → 2026-08-31, 1,884). Terminal rule unchanged: mark to the final mark, no liquidation cost. Only the date moves | the last row of the old window, [2026-08-31, 2026-09-01), needs the 2026-09-01 00:00 funding settlement and funding on disk ends 2026-08-31 16:00; hard-fail on missing funding would raise. Verified that all 19 contracts held in the window have funding through at least 2026-08-31 16:00 when live on 08-31. The 09-01 settlement was not fetched from another source | **no** — an accounting sanity run of the monthly benchmark ledger through 2026-08-31 had been executed before this change (its net Sharpe, 0.34, was seen). The change is forced by data coverage, not chosen from that result |
| 2026-09-25 | **Amendment 5** (same file): rows are holding periods with one position; index denominator is the previous basket value, not capital; reference basket maintained independently and supplies weights; explicit opening event; idle_weight resets at reconstitution; settlement takes every settlement to the delisting timestamp and charges fee only; pre-flight allows classified lifecycle events; mark join described as timestamp-aligned pending validation | second technical review of `basket_walkthrough.md` | yes |
| 2026-09-24 | **Amendment 4** (same file): bar-open execution convention; per-settlement funding on mark prices with hard failure on gaps; explicit contract state machine; settlement realises P&L with the freed weight idle; units-vs-notional exposure semantics; costless warm-up index with books opening on the first scored day; fee and slippage stored separately; eight added acceptance tests | technical review of `basket_walkthrough.md` | yes |
| 2026-09-24 | **Amendment 3** (same file): common-β alpha attribution across price/funding/cost layers; H2 realised-risk diagnostics (H2 unchanged); frictions as a primary *descriptive* estimand; funding sign convention; superseded sample figures removed; LUNA settlement and terminal-liquidation sensitivities; AR(1) turnover sensitivity; reproducible instrument-fact script | second independent review, of the design rationale | yes |
| 2026-09-24 | **Amendment 2** (same file): basket start 2020-02-09, warm-up ends 2021-07-04, scored window 2021-07-05 → 2026-08-31 (1,884 days) | forced arithmetic: the registered 20-observation volume screen cannot field ten names earlier | yes |
| 2026-09-23 | **Amendment 1** (`pre_registration_amendment_2026-09-23.md`): reconstructed-data terminology; speed rule rebuilt (blend 8/32…64/256, warm-up 512, scored window then 1,903 days, superseded by Amendment 2); H2 recast as a superiority hurdle with corrected MDEs; implementation config, pseudocode and acceptance tests; funding at every recorded settlement; return clipping forbidden; Sharpe defined; H1 gate and typed registry; literature matrix; reference corrections | independent review of the frozen document | yes |

Recorded pre-freeze history, for completeness: a three-speed blend
(8/32, 16/64, 32/128) was recorded first on 2026-09-21; the speed-selection rule
(S3) added 4/16 the same day, before any strategy result. The phrase "slow trend
following" in the hypothesis was replaced with the rule-based wording for the
same reason.

---

## References

Verified 2026-09-21/22. Author, year, title and venue checked for each; anything
that could not be confirmed was dropped.

1. Abdi, F., and A. Ranaldo. 2017. "A Simple Estimation of Bid-Ask Spreads from
   Daily Close, High, and Low Prices." *Review of Financial Studies* 30(12):
   4437–4480.
2. Bailey, D. H., J. M. Borwein, M. López de Prado, and Q. J. Zhu. 2017. "The
   Probability of Backtest Overfitting." *Journal of Computational Finance*
   20(4): 39–69. doi:10.21314/jcf.2016.322
3. Bailey, D. H., and M. López de Prado. 2014. "The Deflated Sharpe Ratio."
   *Journal of Portfolio Management* 40(5): 94–107. doi:10.3905/jpm.2014.40.5.094
   (SSRN working paper: 10.2139/ssrn.2460551)
4. Binance. "Delisting of Futures Contracts." Binance Futures Support FAQ
   (settlement = 1,800 one-second index prices over the final 30 minutes; taker
   fee charged).
5. Binance. "Binance Futures Fee Structure and Fee Calculations." Current version
   (VIP-0 USDⓈ-M example: 2 bp maker / 5 bp taker).
6. Binance. "Introduction to Binance Futures Funding Rates." Current version.
7. binance/binance-public-data, issue #372, "bookTicker Data Missing Since
   March" — open, no official explanation; dataset published only 2023-05 →
   2024-04 and only for futures.
8. Carver, R. 2015. *Systematic Trading*. Harriman House. Reference
   implementation: `pysystemtrade` (`futures_chapter15` forecast scalars;
   `robust_vol_calc` volatility floor; `forecast_scalar` pooled estimation).
9. Carver, R. "How fast should we trade?" *Investment Idiocy* (qoppac), 2020 —
   the ≈0.13 SR/yr speed limit.
10. Cong, L. W., X. Li, K. Tang, and Y. Yang. 2023. "Crypto Wash Trading."
    *Management Science* 69(11): 6427–6454. doi:10.1287/mnsc.2021.02709
11. Corwin, S. A., and P. Schultz. 2012. "A Simple Way to Estimate Bid-Ask
    Spreads from Daily High and Low Prices." *Journal of Finance* 67(2): 719–760.
    (Considered; Abdi–Ranaldo preferred for less liquid assets.)
12. Dmitrienko, A., et al. **2006**. *Biometrical Journal*.
    doi:10.1002/bimj.200610274 — gatekeeping procedures. With FDA, "Multiple
    Endpoints in Clinical Trials" guidance.
13. Harvey, C. R. 2017. "Presidential Address: The Scientific Outlook in
    Financial Economics." *Journal of Finance* 72(4): 1399–1440.
    doi:10.1111/jofi.12530
14. Harvey, C. R., Y. Liu, and H. Zhu. 2016. "…and the Cross-Section of Expected
    Returns." *Review of Financial Studies* 29(1): 5–68. doi:10.1093/rfs/hhv059
15. Huberman, G., and S. Kandel. 1987. "Mean-Variance Spanning." *Journal of
    Finance* 42(4). doi:10.1111/j.1540-6261.1987.tb03917.x
16. Ioannidis, J. P. A. 2005. "Why Most Published Research Findings Are False."
    *PLoS Medicine* 2(8): e124. doi:10.1371/journal.pmed.0020124
17. Lo, A. W. 2002. "The Statistics of Sharpe Ratios." *Financial Analysts
    Journal* 58(4): 36–52. doi:10.2469/faj.v58.n4.2453
18. Moskowitz, T. J., Y. H. Ooi, and L. H. Pedersen. 2012. "Time Series
    Momentum." *Journal of Financial Economics* 104(2): 228–250.
    doi:10.1016/j.jfineco.2011.11.003
19. Newey, W. K., and K. D. West. 1987. "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    *Econometrica* 55(3): 703–708.
20. Nosek, B. A., C. R. Ebersole, A. C. DeHaven, and D. T. Mellor. 2018. "The
    Preregistration Revolution." *PNAS* 115(11): 2600–2606.
    doi:10.1073/pnas.1708274114
21. Politis, D. N., and J. P. Romano. 1994. "The Stationary Bootstrap." *Journal
    of the American Statistical Association* 89(428): 1303–1313.
22. Schuirmann, D. J. 1987. (Cited for context only; the TOST equivalence
    framing was **withdrawn** in Amendment A3 in favour of a one-sided
    superiority hurdle.) Schuirmann's TOST: *Journal of Pharmacokinetics and
    Biopharmaceutics* 15(6): 657–680.
24. Borgards, O. 2021. "Dynamic Time Series Momentum of Cryptocurrencies."
    *North American Journal of Economics and Finance* 57: 101428.
25. Le, T., and U. Ruthbah. 2023. "Trend-following Strategies for Crypto
    Investors." SSRN 4551518 (Monash Centre for Financial Studies).
26. Liu, Y., and A. Tsyvinski. 2021. "Risks and Returns of Cryptocurrency."
    *Review of Financial Studies* 34(6): 2689–2727. doi:10.1093/rfs/hhaa113
27. Rozario, E., S. Holt, J. West, and S. Ng. 2020. "A Decade of Evidence of
    Trend Following Investing in Cryptocurrencies." SSRN 3697981 /
    arXiv:2009.12155.
23. Treynor, J. L., and K. Mazuy. 1966. "Can Mutual Funds Outguess the Market?"
    *Harvard Business Review* 44(4): 131–136.

Index methodology sources (monthly reconstitution, concentration caps,
liquidity screens) are catalogued in `backtest_review.md`.
