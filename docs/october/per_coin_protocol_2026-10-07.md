# Per-coin trend following: frozen protocol

**Dated 2026-10-07. Frozen on commit (tag `per-coin-protocol-2026-10-07`). Amended 2026-10-12 (Part 14, tag `per-coin-amendment-1-2026-10-12`): the primary universe is now the sealed basket's top-10 membership. Amended 2026-10-13 (Part 15, tag `per-coin-amendment-2-2026-10-13`): top-25 and top-50 sensitivities. No per-coin
strategy return, forecast or position has been computed.** The data was downloaded and
checked (coverage, checksums, instrument ends, weekend price behaviour) but no return of
any strategy or control was calculated on it.

Machine-readable companion: `docs/october/per_coin_support/frozen_config.yaml`. Where
the two disagree, this document governs and the disagreement is a bug to log in Part 13.
Nothing here changes without a dated change-log row.

**How the open decisions were settled.** The draft of the same date left 14 decisions
open. The researcher delegated them on 2026-10-07 ("approve any decision that you must").
They were settled by the assistant, on the reasons given in each row, before any engine
existed that could compute a return. Part 12 lists each decision and the alternative
rejected.

Labels: **[V]** verified in this repository · **[L]** from the cited literature ·
**[O]** judgement.

---

## Part 0 — What the basket study did and did not show

The basket study asked whether one trend signal, fixed in advance and applied to a
cap-weighted top-10 basket of Binance USDT-M perpetuals, beat holding the same basket.
It measured this after assumed trading costs and measured funding, from 2021-07-05 to
2026-08-30. Bitcoin explains 95% of the basket's daily variance, so this was in effect a
test of timing Bitcoin.

It found no detectable edge:
- **Alpha:** +1.2%/yr with same-bar fills and −1.0%/yr with a one-day lag. The lagged
  figure is the honest one.
- **Sharpe:** 0.04, with a 90% range for alpha of −21% to +24% a year.
- **Compounded:** $100k became about $82k, against about $117k for the passive basket.
- **Where the gross gain went:** price timing earned about 5%/yr gross. Funding took
  about 3 points and costs about 1, because a book that is always one sign is long
  exactly when longs pay.
- **Concentration:** two months (June 2022 and February 2024) account for the whole
  positive path.

The study rules out a large edge (one-sided 95% upper bound on Sharpe about 0.76), not a
modest one. Its protocol was fixed in advance and timestamped, but it was applied to a
sample that had already been inspected, so it is an in-sample replication with a fixed
protocol, not a confirmatory test.

It says nothing about per-coin trend following, where each coin has its own signal and
the book can be long one coin and short another. The June 2026 IC work ran on a biased
universe, so it is not evidence either way.

---

## Part 1 — The claim and what this study can do

**Hypothesis.** A separate trend signal and position on each liquid Binance USDT-M
perpetual, combined through a risk model, earns a positive net return over a no-signal
long book of the same universe, sized by the same risk model. The edge comes from the
**coin-specific** component of P&L (Part 7), not from timing the market as a whole.

**Prior evidence, both ways.**
- **For:**
  - Moskowitz, Ooi & Pedersen (2012): diversified gross Sharpe above 1 **[L]**.
  - Man Group (Dec 2024): a 10–15-coin portfolio is optimal, with the level probably
    inflated by the short history **[L]**.
  - A public per-coin Binance backtest (`OctopusTakopi/crypto-trend-following`): net
    Sharpe 1.04 at a $5m liquidity floor and 0.50 at $25m. These are the author's own
    figures, not reproduced here.
- **Against:**
  - Hurst, Ooi & Pedersen (2017): single-market trend Sharpe near 0.4 **[L]**.
  - Huang, Li, Wang & Zhou (2020): 47 of 55 assets have t < 1.65, and a
    historical-mean-sign rule does about as well **[L]**.
  - Kim, Tse & Wald (2016): much of TSMOM's alpha is volatility scaling **[L]**.
  - Coins are about 0.7 correlated **[V, old universe]**.

**Status of the sample.** There is no unused 2020–26 Binance perpetual data. The
historical run is an in-sample replication with a fixed protocol: it can reject a large
claim, not confirm a modest one. Only forward paper trading (Part 11) can confirm.

---

## Part 2 — Universe

| # | rule |
|---|---|
| U1 | **Instruments:** every Binance USDT-M perpetual in the registry, delisted ones included (`registry.list_contracts()`, 900 contracts on 2026-10-07) **[V]**. |
| U2 | **Instrument key:** a contract symbol plus one *continuous traded span*. A span ends when the contract goes **more than 30 calendar days** without a traded bar; trading after that starts a new instrument with its own history and warm-up. The download found 9 such relisting gaps, plus 161 contracts that stopped trading **[V]**. CoinMarketCap ids cover only 62 coins, so they cannot key a registry-wide universe. |
| U3 | **Exclusions** (Appendix A, frozen): stablecoin and fiat-pegged perps; gold-backed tokens; composite-index perps; and every non-crypto underlying (equities, ETFs and leveraged ETFs, stock indices, commodities, FX, pre-IPO company exposure). |
| U4 | **Liquidity gate:** median of the contract's own daily quote volume over the 30 days ending D−1, at least 20 traded days in the window, **≥ $25m**. Past data only. Sensitivities: $10m and $50m. |
| U5 | **Tradable on day D:** the daily bar exists with volume > 0 **[V: constraint 2]**, D is not the instrument's first traded day (a partial bar), and U6 holds. |
| U6 | **Completeness:** at least 23 of the 24 hourly mark bars for D exist. |
| U7 | **Eligible on D** (may hold a new or changed position): not excluded (U3), passes U4 on D−1 data, tradable on D (U5), and has the history S5 requires. |
| U8 | **No cap on the number of instruments.** The risk model handles correlation; a count cap would add a free parameter. |

**Disclosure on U4.** Part of the case for $25m came from the public repo's own result:
its Sharpe halves from $5m to $25m. Choosing a floor after seeing that is a choice informed
by results on the same years. It is conservative (it lowers the comparable claim) and is
disclosed. The ex-ante reason is cost credibility: 10 bp per side is plausible only for
deep books.

---

## Part 3 — Signal (the registered rule, applied per instrument)

| # | rule |
|---|---|
| S1 | One EWMA-crossover forecast per instrument, on its own price. **Price series:** the daily open on the instrument's traded days, so `open[D]` is the price at D 00:00. A non-traded day inside a span contributes no observation. |
| S2 | **Speeds:** 8/32, 16/64, 32/128, 64/256, as registered **[V]**. The September speed-cost rule is **not** re-run on coin volatilities: above 63% vol it would admit 4/16, a faster speed admitted through a volatility range measured on this data. |
| S3 | Fixed scalars 5.3 / 3.75 / 2.65 / 1.87; each speed clipped to ±20; equal weights; FDM; blend clipped to ±20; `adjust=True` **[V]**. |
| S4 | Normalisation: (fast − slow) ÷ (price × daily return vol), the instrument's own floored vol (R1). |
| S5 | **Short history.** A speed is *available* once the instrument has 2 × its slow span of traded days (64, 128, 256, 512 days). The forecast is the equal-weighted mean of the available speeds, times the FDM for that set: **1.00, 1.04, 1.09 and 1.15** for one to four speeds. Those values come from the September synthetic simulation (no project data), which reproduces the registered 1.15 (`per_coin_support/fdm_subsets.py`) **[V]**. An instrument with fewer than 64 traded days has no forecast and is not eligible. |

---

## Part 4 — Risk model and sizing

| # | rule |
|---|---|
| R1 | **Instrument vol:** 32-day EWMA of past daily returns, floored at the 5th percentile of its own past 500 days (expanding from 100), annualised ×√365 **[V: registered R1–R2]**. |
| R2 | **Instrument weight:** wᵢ = 1/Nₜ over the instruments eligible on D. |
| R3 | **Diversification multiplier:** IDMₜ = min(2.5, 1/√(1/Nₜ + (1 − 1/Nₜ)·ρ̄ₜ)). ρ̄ₜ is the mean pairwise Pearson correlation of daily open-to-open returns among the eligible instruments over the 180 days ending D, using pairs with ≥ 90 overlapping days, floored at 0. If no pair qualifies, IDM = 1. With equal weights only the average correlation matters, so ρ̄ is a complete input, not a simplification **[O]**. Using instrument rather than sub-system correlations is conservative (a smaller IDM). Cap 2.5 (Carver) **[L]**. |
| R4 | **Ideal position** (signed notional ÷ capital): x\*ᵢ = (fᵢ/10) × wᵢ × IDMₜ × (0.30 ÷ σᵢ). Target vol 30% **[V]**. |
| R5 | **Buffer:** per instrument, ±10% of its standard position (wᵢ·IDMₜ·0.30/σᵢ) around x\*ᵢ. Inside the band, hold; outside, trade to the nearest edge **[V: registered X2]**. |
| R6 | **Risk overlay:** ex-ante book vol V = √(p′Σp), with Σᵢᵢ = σᵢ², Σᵢⱼ = ρ̄ₜσᵢσⱼ. If V > 0.60 (2 × target), every position is scaled by 0.60/V **[O]**. |
| R7 | **Caps:** \|pᵢ\| ≤ 0.5 (per instrument), then Σ\|pᵢ\| ≤ 3.0 (gross, scaled proportionally) **[O]**. |
| R8 | Order each day: x\* → buffer → overlay → per-instrument cap → gross cap. |
| R9 | 365-day annualisation; cash rate 0 **[V]**. |

Disclosure: ρ̄ and the IDM are estimated from past data, so they are not look-ahead. But
they are estimated quantities. The September protocol barred rolling *forecast* scalars;
the IDM cannot be fixed in advance when the universe changes size.

---

## Part 5 — Execution, costs, funding, contract events

| # | rule |
|---|---|
| X1 | **Primary fill: one-day lag.** Targets are computed on D from data through D 00:00 (prices `open[≤D]`, eligibility from D−1). They are traded at **D+1 00:00** (`open[D+1]`). "Next-open" would be the same instant as the same-bar fill and is not used. |
| X2 | **Sensitivities:** same-bar fill (trade at `open[D]`; the optimistic bound only), and a 1-hour delay: fill at D 01:00 at the hourly **mark** open, a proxy for the trade price, disclosed as such. |
| X3 | **No trade on a non-tradable day.** The position is held and the decision waits for the next tradable day. |
| X4 | An instrument that leaves the eligible set is traded to zero at the next fill, whatever the band says. |
| X5 | Policies: **buffered (primary)** and full rebalancing (reported). |
| C1 | **Costs:** 10 bp per side on every traded notional (5 fee + 5 slippage) **[L]/[O]**. Stress: **20 bp and 30 bp** per side. No liquidity tiering. |
| F1 | **Funding:** at every recorded settlement, on that settlement's hourly mark, for the position held. Positive rates mean longs pay. **Never filled with zero.** The 2026-06-24 04:00 exception (nine named contracts, sealed protocol Exception 1) carries over unchanged; any other missing settlement on a held position raises **[V]**. |
| F2 | Funding is reported **separately for long and short positions**, by year, with its correlation with net exposure. |
| E1 | **Instrument end while held** (a contract that stops trading, or the end of a span under U2): exit at the mean of the 1-minute **index-price** closes over the 30 minutes before the close of the last hourly mark bar, with the 5 bp taker fee only **[V: sealed protocol §3.7 and Amendment D7]**. Data: 1-minute index klines for the last traded day and the next, downloaded for all 170 ends (Part 10). If they are absent, exit at the last hourly mark close, flagged. |
| E2 | **Halt inside a span** (zero-volume days ≤ 30): hold at the last traded price; funding accrues on any recorded settlement; exit or adjust at the first traded bar **[V: registered rule 3]**. |
| E3 | **Ledger:** one row per instrument per holding period [D, D+1), with `net = price P&L + funding − fees − slippage` exact on every row **[V: basket ledger identity]**. |
| E4 | **Capital:** fixed $100k base, no compounding. Compounded figures are reported next to the arithmetic ones. |

---

## Part 6 — Sample

| # | rule |
|---|---|
| T1 | Data from 2020-01-01, the start of Binance's archive **[V]**. Signals and correlations are computed from each instrument's first traded day. |
| T2 | **Scored window: 2021-07-05 → 2026-08-30** (1,883 days, 5.16 years), the sealed basket window. It is chosen for exact comparability with the basket result (K4) and was fixed before any per-coin return existed. September 2026 is not scored. |
| T3 | Books open on 2021-07-05 with the instruments eligible that day. Before that, nothing is held. |

---

## Part 7 — The mechanism split

With positions wᵢ over the Nₜ eligible instruments, W = Σwᵢ, and r̄ the equal-weighted
mean return of the eligible set:

    Σ wᵢ rᵢ  =  W · r̄   +   Σ (wᵢ − W/Nₜ)(rᵢ − r̄)
                market       coin-specific

This is an accounting identity, exact on every day **[O]**. It is applied separately to
the price and funding layers, using the same day's eligible set. Positions in instruments
that are held but not eligible (X3, E2) count in the identity. Their return enters r̄ only
if they are in the eligible set; otherwise they are reported as a third, "ineligible
holdings" term, so the split remains exact.

**Prediction.** The coin-specific *price* component is positive. It is also reported net
of all trading costs, which is conservative because some trades change W.

---

## Part 8 — Controls (same engine, universe, risk model, costs, funding)

| # | control | definition |
|---|---|---|
| K1 | **No-signal long book** (primary comparator) | every eligible instrument at a fixed forecast of +10; same vol scaling, IDM, buffer, overlay, caps and fills. This is both the handoff's "vol-targeted long basket" and its "vol targeting with no trend signal". Built through one engine, they are the same object. |
| K2 | **BTC** | BTCUSDT alone at forecast +10, wᵢ = 1, IDM 1, same 30% target, costs, funding and fills. |
| K3 | **Historical-mean sign** | forecast = +10 × sign(expanding mean of the instrument's daily returns over its span); a zero mean gives 0. Same availability (64 days) and sizing (Huang et al. 2020) **[L]**. |
| K4 | **Sealed basket results** | referenced, not rerun (`reports/registered_run/`). |

---

## Part 9 — Hypotheses, inference, classification

**H1 (primary).** Daily `r(N-buf) = α + β·r(K1 net) + ε`. H0: α ≤ 0, one-sided 5%.
**Pass:** the 5th percentile of the stationary-block-bootstrap α is above zero. The
bootstrap uses paired daily returns, mean block 20 days, 10,000 replications, seed
**20261007**, and re-estimates α and β in every replication (Politis & Romano 1994)
**[L]**. Newey–West with 21 lags is a cross-check; a material disagreement is reported,
not resolved by preference **[L]**.

**H2 (mechanism).** The coin-specific price component is above zero, by the same bootstrap.

**H3 (secondary, not gating).** Alpha against K2 and against K3, by the same method.

**Validity check.** A day is *two-sided* if the book holds at least one long and one short
position, each side at least 10% of gross. If fewer than 20% of scored days are two-sided,
the result is labelled **collapsed to market timing**, whatever H1 says.

**Classification, fixed in advance.** All conditions are evaluated at the primary
settings: one-day lag, buffered, 10 bp, $25m floor.

| outcome | condition | wording |
|---|---|---|
| **pass** | H1 passes, still passes at 20 bp and at 40- and 60-day blocks, H2 passes, and the validity check holds | "in-sample replication consistent with a per-coin trend edge; not confirmation" |
| **fragile** | H1 passes at the primary settings but fails at 20 bp or at 40/60-day blocks | "fragile" |
| **timing, not per-coin** | H1 passes but H2 fails, or the validity check fails | "the return is market timing or volatility scaling, not a per-coin edge" |
| **inconclusive** | α > 0 but the interval includes 0 | "inconclusive" |
| **large claim rejected** | one-sided 95% upper bound on net Sharpe < 1.0 (observed net Sharpe below about 0.28) | "Sharpe ≈ 1 is rejected on clean data". It can accompany "inconclusive" or "fail" |
| **fail** | α ≤ 0 | "the per-coin hypothesis fails on this sample" |

**Power (IID planning figures, 80% power, one-sided 5%, 5.16 years).**
- The minimum detectable net Sharpe (or appraisal ratio against K1) is about **1.10**.
- **Chance of passing H1:** 0.76 at a true 1.04, 0.73 at 1.00, 0.52 at 0.75, **0.31 at
  0.50** and 0.23 at 0.40.
- **What that implies:** at the $25m floor the public figure is about 0.5, and this test
  would pass only about a third of the time even if that figure were true. The most
  likely outcome for a real but modest edge is therefore *inconclusive*, and that is
  stated before the run.

**Always reported:**
- the ladder (price / gross / net × full / buffered) and turnover;
- funding by leg;
- the market / coin-specific / ineligible split;
- per-instrument and per-year contributions;
- LUNA separately;
- the realised paths of IDM and ρ̄;
- the eligible count over time;
- compounded and arithmetic returns;
- the deflated Sharpe ratio (Bailey & López de Prado 2014) **[L]**.

**Registered sensitivities**, each counted as a trial:
- same-bar fill;
- 01:00 fill;
- 20 bp and 30 bp costs;
- $10m and $50m floors;
- full rebalancing.

Further checks, not counted as trials (inference settings or subsamples): 10-, 40- and
60-day blocks, and the result without LUNA.

**Trial count for the deflated Sharpe:** N = 35 (prior inventory) + 11 (the basket
primary and its 10 registered configurations) + 8 (this primary and its 7
sensitivities) = **54**, with 30 and 80 as a bracket.

---

## Part 10 — Data, as downloaded (2026-10-07)

| item | status |
|---|---|
| daily klines + funding, all 900 contracts | done, checksum-verified; prices to 2026-10-06, funding to 2026-09-30 23:00 **[V]** |
| hourly mark prices over each traded range | done for 895 of 900; 5 have no archived marks (Appendix B) |
| 1-minute index klines, last traded day and the next, for all 170 instrument ends | done; 338 of 340 files, the 2 absent belong to DOTECOUSDT (excluded, Appendix A) **[V]** |
| the 62 previously mapped contracts | identical to the sealed run's inputs through 2026-08-31, OHLCV and funding; the pre-download copy is `data/derived/snapshot_2026-10-07_pre_registry/` **[V]** |
| known gaps | six contracts lack the 2026-10-05 daily file (outside the window); ICP 2022-09-22..26 (delisted and relisted) |

---

## Part 11 — Forward record

Whatever the outcome, the frozen rule is paper-traded forward unchanged
(`docs/september/forward_record_design_2026-10-03.md`). A pass is not a reason to trade
money; a fail is not a reason to tune.

---

## Part 12 — Decisions settled on 2026-10-07 (from the draft)

| draft # | decision | chosen | rejected, and why |
|---|---|---|---|
| U2 | span break | 30 days without trading | shorter would split ordinary halts; longer would join unrelated relistings |
| U4 | liquidity floor | $25m (sensitivities $10m, $50m) | $5m: 10 bp costs are not credible for thin books |
| U6 | completeness | ≥ 23 of 24 hourly marks | none: a partial day would be priced as a full one |
| S5 | short history | per-speed availability with subset FDMs | a full 512-day warm-up: the median contract has 465 traded days **[V]**, so most of the universe would never trade |
| R3 | correlation input | 180-day mean pairwise correlation, ≥ 90 overlapping days | a full shrunk matrix: with equal weights the IDM needs only the mean, and a full matrix adds estimation noise and a shrinkage parameter |
| R5–R6 | overlay | scale down above 2× target vol | none: a crowded one-sign book could run far above target |
| R6–R7 | caps | 0.5 per instrument, 3.0 gross | the basket's single 2.0 cap: it does not apply to a book of many legs |
| C1 | costs | flat 10 bp; stress 20 and 30 bp | liquidity tiers: a new fitted schedule |
| K1 | controls | one no-signal book covers both handoff controls | a cap-weighted vol-targeted basket: a different universe, already on record as K4 |
| validity | two-sided days | < 20% two-sided ⇒ collapsed | none: a one-sign book would answer the basket's question again |
| T2 | window | the sealed window | extend to 2026-09-29: one more month buys little power and breaks comparability with K4 |
| E1 | instrument ends | the sealed settlement rule, index-price based | last daily close: wrong in price and fee (sealed protocol §3.7) |
| X2 | hourly fill | mark-price proxy at 01:00 | downloading hourly trade klines: a sensitivity does not justify it |
| U3 | exclusions | name list checked by weekend mark behaviour (Appendix A) | names alone: several tickers are ambiguous |

---

## Part 13 — Change log

| date | change | reason | before any return? |
|---|---|---|---|
| 2026-10-07 | protocol frozen | — | yes |
| 2026-10-12 | **Amendment 1**: primary universe = the sealed basket's top-10 membership; registry-wide universe becomes a counted sensitivity; control K5 and test H4 added | see Part 14 | yes |
| 2026-10-13 | **Amendment 2**: per-coin top-25 and top-50 sensitivities, with their own coin-to-contract map and schedule | see Part 15 | yes |
| 2026-10-13 | **Clarification 1**: eligibility on day X uses only information known at X 00:00. Membership is read on X; trading activity (traded, complete hourly marks, not a span's first day, history count) is read from X − 1, the last completed day. U5–U7 read "tradable on D" as same-day activity, which is not known when the decision is stamped. Whether the coin can trade on its fill day remains a separate execution check (X3). Convention stated, not changed: a day's open counts as a price only if the day traded, as in the sealed run (Binance's open is the first trade's price). Effect on the top 10: 34 short-history member-days instead of 32; nothing else | a reviewer found that eligibility for day X used X's full-day volume and mark count, unavailable at X 00:00 | yes |
| 2026-10-19 | **Amendment 3**: one counted sensitivity, *dynamic risk targeting* (the public per-coin repo's sizing). Each day the whole book is rescaled so its ex-ante volatility (the R6 constant-correlation covariance at ρ̄) equals the 30% target, with no upper bound on the multiplier; then the gross cap of 3× and the per-coin cap of 0.5× apply. It replaces the IDM and the overlay; everything else is as in the primary. Trials 58 | it tests whether always running at full target risk, even on weak forecasts, drives the public repo's reported Sharpe | yes |
| 2026-10-22 | **Clarification 2** (to Amendment 3): the dynamic-risk-targeting sensitivity, in order. (1) Unscaled book uᵢ = (fᵢ/10) × wᵢ × 0.30/σᵢ. (2) Its ex-ante volatility V(u) under the R6 constant-correlation covariance at ρ̄. (3) M = 0.30/V(u), no upper bound; if V(u) = 0 (every eligible forecast exactly 0, or no eligible coin), M = 0. (4) Ideal positions x\* = M·u. (5) The buffer uses the **primary's** band, 0.10 × wᵢ × IDM × 0.30/σᵢ, which does not scale with M. (6) Per-coin cap 0.5, then gross cap 3.0. No overlay. The 30% is the risk of the **ideal** book, before the buffer and caps; the final book can run below it (from flat, at the band's inner edge) or be cut by the caps. At M = 0 every ideal is 0: a held position outside the zero-centred band is cut to the band's edge, and one inside it (\|h\| ≤ 10% of a standard position) **stays, possibly indefinitely**, as in the primary | as first written, the band scaled with M too, so from flat a coin traded only if \|f\| > 1 whatever M was: weak forecasts left the book flat, the opposite of the experiment's intent (review, 2026-10-22). A synthetic test now proves the chosen behaviour | yes |
| 2026-10-22 | **Clarification 3**: the correlation fallback. When ρ̄ is undefined (no pair of eligible coins with ≥ 90 shared days, or a single coin), **both** the IDM and the ex-ante covariance of R6 and Amendment 3 use ρ = 1: the coins are treated as moving as one. IDM = 1, the R3 fallback, is exactly the ρ = 1 value, so the two now agree. A negative ρ̄ is floored at 0 in both, as R3 states | the code fell back to ρ = 0 (independence) in the covariance while the IDM fell back to 1; independence understates risk exactly when correlation cannot be measured, silencing the overlay and inflating the dynamic multiplier (review, 2026-10-22) | yes |
| 2026-10-24 | **Clarification 4**: how results are compared and read. (a) **Twins:** every strategy variant (each counted sensitivity, and the 20 bp stress test) is compared with a K1 no-signal book on identical settings (universe, timing, costs, sizing, buffer); the two differ only in the forecast, and a sensitivity's α is measured against its twin. Twins are comparators, not trials. (b) **Verdict:** a headline from H1 at the primary settings (FAIL: α ≤ 0; INCONCLUSIVE: 5th percentile ≤ 0; H1 PASSES otherwise), plus warnings evaluated independently and **all** reported whenever they hold: FRAGILE (fails at 20 bp or at 40/60-day blocks), NOT PER-COIN (H2 fails), COLLAPSED TO MARKET TIMING (< 20% two-sided days, whatever H1 says), LARGE CLAIM REJECTED (95th percentile of bootstrapped net Sharpe < 1.0). "Pass" requires H1 PASSES with no FRAGILE, NOT PER-COIN or COLLAPSED warning; LARGE CLAIM REJECTED is always shown and does not prevent a pass. The Part 9 table's six wordings are read through this | the Part 9 table left overlapping outcomes unordered, so one warning could hide another; and a sensitivity compared with the primary's K1 would measure the setting, not the signal (review of the run-driver design, 2026-10-24) | yes |
| 2026-10-24 | **Staging**: the registered runs execute in two stages. **Stage A** (17 books): the primary, K1, K2, K3, K5, and the counted sensitivities cost 20 bp, cost 30 bp, same-bar, 01:00 fill, full rebalancing and dynamic risk targeting, each with its K1 twin. **Stage B** (10 books): top 25, top 50 and the three registry-wide floors, each with its twin. Stage B's runs, comparisons, reporting and the only implementation changes it is allowed are fixed before stage A in `docs/october/per_coin_stage_b_specification.md`, whose hash the stage-A run manifest records. The verdict is computed from stage A alone; stage-B results are reported as produced after stage A's results were known. No rule, parameter, trial count or comparison changes | stage B needs ledger steps 7–8 and an event register (halts, settlements, rebrands) that no top-10 book meets; waiting for them would hold up a primary that does not depend on them | yes |
| 2026-10-26 | **Clarification 5** (stage B, after stage A's result was known; no rule, parameter or comparison changes). (a) **ICP funding:** Binance's archive lacks the old ICPUSDT contract's funding (2021-05-11 → 2022-08-31; no `ICPUSDT_SETTLED` funding folder). It is taken from Binance's own `fapi/v1/fundingRate`, accepted only because the API reproduces the archive exactly on all 1,095 overlapping settlements (rates and millisecond timestamps); stored as `data/derived/binance/funding_supplement/ICPUSDT.csv`, the archive file untouched (`per_coin_support/build_icp_funding_supplement.py`). F2 (never zero-fill) is kept. (b) **E1, the settlement instant:** Binance keeps archiving hourly marks after a settlement (FTT settled 04:00 on 2022-11-14; its marks run to 23:00), so "the close of the last hourly mark bar" is read as the settlement instant T: the last 1-minute trade bar, on Binance's half-hour grid, matching every dated announcement on file (LUNA 15:30, FTT 04:00, ICP 09:00). S is E1's mean of the 30 one-minute index closes before T. The stage-B specification's §4.1 line on an era-specific 60/30-minute window conflicts with E1; the protocol governs, so 30 minutes is booked and the 60-minute price is reported as a diagnostic. Register: `per_coin_support/event_register.csv` (8 settlements), covering T's day and the two after. (c) **Scope now:** stage B runs top 25 and top 50 with their twins; the three registry-wide books stay specified and unbuilt, and will be reported as run later if they are. (d) **Halts inside a span** (ledger step 7) are not built: none occurs while a top-25 or top-50 coin can be held (scanned), and one would stop the run. (e) **Funding schedule gaps:** the sealed spacing check flags 51 places in 36 stage-B contracts that are Binance's schedule, not lost payments (13 cadence switches such as FTT 8h→2h on 2022-11-10; the 2026-06-24 04:00 skip on 35 more 4h contracts, Part 10 Exception 1 carried as the config states, charged zero; FTT's post-settlement row; two gaps with no trading inside, LIT and PUMP). Each is accepted only because Binance's API returns exactly the archive's rows around it (or the klines show no trade inside it): `per_coin_support/funding_gap_register.csv`, built by `build_funding_gap_register.py`. Any other gap still raises. (f) One top-50 member (币安人生USDT) has no archived marks; U6 makes it never tradable, so it is never held and its empty funding table is never read | the stage-B build met a missing data source and an E1 wording that the archive's behaviour after settlement makes ambiguous | **no** (after stage A; before any stage-B return) |
| 2026-10-26 | **Clarification 6** (the registry-wide $10m book, built at the researcher's request after the stage-A and top-25/50 results were known; no rule, parameter or comparison changes; the $25m and $50m registry books stay specified and unbuilt). (a) **Registry:** U1 reads the registry as frozen at the download (`data/derived/binance/traded_ranges.csv`, 900 contracts on 2026-10-07), not a live listing. (b) **U4:** the median is taken over the traded days (volume > 0) in the 30 calendar days ending X − 1, with at least 20 of them. (c) **BNX token swap:** BNXUSDT went from 119.93 to 1.411 across a 10-day halt (2023-02-11 → 02-22), a 1:100 redenomination. Holding it through as an E2 halt would book a fake ±99% move, so the old token is settled at E1 on 2023-02-11 and the resumed contract is a new instrument (U2) by a registered span break (`per_coin_support/span_break_register.csv`). A scan of all 660 contracts for open-to-open moves beyond ×2.5 found no other redenomination among eligible coins (the rest are multi-hour crashes or pumps). (d) **Events:** the register is now built from a scan of every stage-B universe: 54 settlements, the 8 earlier ones unchanged; two unscheduled ends (AIA, OM) take T as the close of the last traded minute, flagged. No halt inside a span is met. (e) **Funding:** the gap register covers all 390 affected contracts (489 gaps), each verified against Binance's API or the klines; the 2026-06-24 skip on 1-hour contracts (01:00–04:00) is treated as the same exchange event, charged zero. Five settlements at hours with no mark bar (AERGO, CTK, CVC, CVX, SLP, all in 2025 and outside every eligible stretch) are registered as `mark_hole`: dropped, and the ledger refuses any held window that touches one. (f) **Late funding archives:** besides ICP, two registry contracts' fundingRate archives start after they were first eligible (BNX from 2023-02, eligible from 2022-06; TLM from 2023-03, eligible from 2021-09). Their earlier funding comes from Binance's API on the Clarification-5 terms (identical to the archive on every one of the first twelve months' overlapping settlements: 1,095 and 1,098). Older API rows carry no interval, so each is inferred from the spacing to the next settlement; a spacing that differs from both neighbours while they agree (a hole in a steady cadence) stops the build, and a cadence switch is registered as such (`per_coin_support/build_funding_supplements.py`, which also rebuilds ICP's file byte-identically) | the registry book meets a redenomination, unscheduled delistings, and mark gaps that the top-10/25/50 books did not | **no** (after stages A and B; before any registry-book return) |
| 2026-10-27 | **Errata** (documentation only; no rule, code, data or result changes). (1) `portfolio/per_coin_universe.py` module docstring says a member is eligible if "it traded on D"; the code reads activity on X − 1, as Clarification 1 requires (`ACTIVITY_LAG_DAYS = 1`, `eligibility`). (2) `data/universe/coinmarketcap/new_fetcher/registry.py` (line 15) says the live API is not usable (HTTP 451); `fapi.binance.com` was reachable on 2026-10-26 and used for the Clarification 5 and 6 funding work. (3) Clarification 5(a) cites `per_coin_support/build_icp_funding_supplement.py`; it was renamed and generalised to `build_funding_supplements.py` (Clarification 6f), which rebuilds ICP's supplement byte-identically. The files themselves are left unedited so their hashes still match the sealed run manifests | stale text found when CLAUDE.md was rederived from the implementation | n/a (no result affected) |
| 2026-10-29 | **Stage C: a separately registered construction** (after every per-coin result was known; no earlier rule, parameter, comparison or result changes). Per-coin trend with explicit daily portfolio risk targeting: the Amendment 3 / Clarification 2 sizing, unchanged, on the registry-wide $10m book (primary, H1′ against its risk-targeted K1 twin, which is the risk-targeted long-only control), with top 25 and top 50 as secondaries and a 20 bp stress pair. Specification: `docs/october/risk_targeted_trend_design_2026-10-27.md` §7, whose hash the run driver checks. Three new counted trials: the deflated Sharpe uses **61** for stage C (stages A and B keep 58; the frozen config is not edited). Recorded as a **design-led research decision**: the objective is trend inside a controlled portfolio risk budget. It is not a passed performance gate: the exploratory control (`octp_reproduction_findings_2026-10-26.md` §10) missed its bar, and the stage-A dynamic sensitivity was not significant | the researcher's objective is trend at a controlled portfolio risk, which the primary's fixed per-coin budget does not deliver | **no** (after stages A and B) |

---

## Part 14 — Amendment 1 (2026-10-12): the top-10 per-coin design

**Status: before any per-coin return.** No engine code exists yet. The researcher decided
this on 2026-10-12. Where this Part conflicts with Parts 2–10, **this Part governs**.

### Why

Under U1–U8 the book would hold about 60–110 coins a day. That figure is an approximate
count from volume and listing age only, with no returns. Three problems follow:
- **Costs.** A flat 10 bp per side is not credible for the thinnest of them.
- **Evidence.** Published evidence puts the benefit of breadth at its peak near 10–15 coins
  (Man Group, Dec 2024) **[L]**.
- **Clarity.** The researcher's question is narrower: on the **same coins** as the sealed
  basket study, do ten separate signals beat one basket signal?

Holding the membership fixed isolates that question. Coins, costs and funding stay the
same; only the signal structure changes.

### What changes

| # | was | now |
|---|---|---|
| U1–U4, U7–U8 | every liquid registry perp, $25m gate, no count cap | **the sealed basket's membership**: `UniverseSelector(min_history_days=0).select_buffered(reconstitution_dates(...))`. That is the top 10 by market cap from the D−1 snapshot, the registered $10m 30-day Binance volume screen, the 10-in/13-out buffer and monthly reconstitution, exactly as in `portfolio/basket.py: build_inputs` **[V]** |
| U2 | contract + traded span | the member's contract from the sealed `contract_map`, keyed by `cmc_id` (constraint 3). The span rule still applies to signal history |
| U3 | name list + weekend rule | the selector's own exclusions by `cmc_id` (stablecoins, wrapped and gold tokens). Appendix A applies to the registry-wide sensitivity only |
| U5–U6, S1–S5, R1–R9, X1–X5, C1, F1–F2, E1–E4, T1–T3 | — | **unchanged.** A member is *eligible* on D if it is a member that month, tradable on D (U5–U6) and has ≥ 64 traded days (S5) |
| R2 | 1/Nₜ over eligible | 1/Nₜ over **eligible members** (normally 10, equal risk; not cap weights) |
| R3 | ρ̄ over eligible instruments | ρ̄ over eligible members |

### New control and test

- **K5 — one basket signal, per-coin sizing.** Every eligible member gets the **same
  forecast**: the sealed basket's registered forecast for that day
  (`portfolio/trend_signal.basket_forecast` on the sealed reference index). Weights, IDM,
  buffer, overlay, caps, fills, costs and funding are the same as the strategy's. The only
  difference between the strategy and K5 is ten signals versus one.
- **H4 (secondary, not gating):** the daily net-return gap, strategy − K5, is above zero,
  by the Part 9 bootstrap. This answers the researcher's question directly. The sealed
  basket result (K4) differs from K5 in weighting (cap weights versus equal risk) and is
  reported alongside, not tested.

### Sensitivity and trial count

The original registry-wide design (U1–U8 with Appendix A exclusions, at the $25m floor and
at $10m and $50m) becomes a **counted sensitivity**, three trials. The $10m/$50m floors no
longer apply to the primary. Counted trials here are the primary, same-bar, 01:00, 20 bp,
30 bp, full rebalancing and the three registry-wide runs, nine in all. Trials for the
deflated Sharpe: 35 + 11 + 9 = **55**; bracket 30–80.

### Power

Unchanged in Sharpe units: the minimum detectable Sharpe is about 1.10 over 5.16 years.
Ten coins at ρ̄ ≈ 0.7 are about 1.4 independent bets, so a modest per-coin edge is
likely to read as *inconclusive*. This is stated before the run.

---

## Part 15 — Amendment 2 (2026-10-13): top-25 and top-50 per-coin sensitivities

**Status: before any per-coin return.** The researcher asked for the per-coin design at 25 and
50 coins as well as 10 (primary) and the registry-wide ~100 (Amendment 1 sensitivity). This
Part adds two **counted sensitivities**. It changes nothing in the primary.

### Rules

Everything is as in the primary (Part 14) except the following:

| # | rule |
|---|---|
| A2.1 | **Membership:** the top 25, or the top 50, by market cap from the D−1 snapshot. The sealed screens are inherited unchanged: stablecoin and wrapped exclusions, a Binance contract that traded that day, a 30-day median Binance dollar volume of at least $10m, and history. Monthly reconstitution. The **buffer is scaled like the sealed 10 → 13: enter at 25 / exit at 33; enter at 50 / exit at 65.** Code: `portfolio/per_coin_universe.DeepSelector`, a subclass. The sealed `data/universe/selector.py` is not edited |
| A2.2 | **Coin-to-contract map:** `data/universe/cmc_contract_map_top100.csv`, 263 rows, 259 coins, 260 contracts. CoinMarketCap's daily table holds the top 100, so 100 is the deepest pool available. It was built by the sealed builder at depth 100 (`docs/october/per_coin_support/build_deep_map.py`). At depth 20 the same builder reproduces the sealed 62-row map exactly **[V]** |
| A2.3 | **Shared contracts** (token migrations and duplicate listings: 1000SATS, BNX, BTT, DYDX, EGLD, KNC) are resolved by price. Each coin's claim is limited to the days the contract's close is within 5% of that coin's price × scale. Where claims overlap, the coin with more matching days keeps its whole matching range. A claim that never matches is dropped. The day-level ranges are in the map **[V]** |
| A2.4 | **Price validation:** 258 of 263 rows pass the sealed one-month check. Three more (FTT, RAY, BNX) pass on every day they traded (median ratio within 0.1%). The last two (NU, ZKJ) never traded on a day the coin was in the top 100, so they can never be selected **[V]** |
| A2.5 | **Schedule:** the monthly resets start **2021-07-01**, the reset in force on the first scored day. The sealed schedule start (2020-02-09) cannot be used: the top 25 can first be filled on 2020-09-01 and the top 50 on 2021-02-01, because too few coins had liquid Binance perps before then **[V]**. The top-10 primary keeps the sealed schedule |
| A2.6 | Equal risk per eligible member, IDM from the members' average correlation, and every other Part 3–5 and Part 14 rule, unchanged |

### What changes for the ledger: contract events are real here

Unlike the top 10, these books meet contract events while a coin may be held under the
one-day lag **[V]**:
- **Top 25:** LUNA (May 2022), ICP (halt, June 2022), FTT (November 2022), MATIC → POL
  (September 2024), TON (June 2026).
- **Top 50:** the same, plus BTT (January 2022), RNDR → RENDER (July 2024), FTM → S
  (January 2025).

The ledger's halt, settlement and span-end rules (ledger design v3) apply. **Before either
sensitivity runs**, each event must be classified on evidence (exchange settlement, halt or
rebrand) and its settlement rule confirmed for its date. A sourced history of Binance's
delisting rules is in progress.

### Trial count

Two counted trials: 55 + 2 = **57**; bracket 30–80.

---

## Appendix A — Exclusions (frozen)

**The list is `per_coin_support/exclusions.csv`: 209 of 900 contracts**, produced by
`per_coin_support/classify_underlyings.py`. That script uses prices only, never a strategy
or control return.

**Rule.** A contract is excluded if either of these holds:
1. **Name list.** It is on the name list of stablecoin/FX perps, gold tokens,
   composite-index perps and known non-crypto underlyings.
2. **Weekend rule.** It was first listed on or after **2025-12-01**, and its **weekend
   ratio is below 0.55**. The weekend ratio is mean |hourly mark change| on Saturday and
   Sunday ÷ the same on weekdays. A market-hours underlying barely moves at weekends.
   Binance's first non-crypto perps (XAU, XAG, TSLA) listed on 2025-12-11 **[V]**, and
   the median unnamed contract scores 0.87.

| category | count |
|---|---|
| equity | 121 |
| equity ETF / leveraged ETF | 27 |
| market-hours underlying (weekend rule only: e.g. V, VRT, NOW, NET, TEAM, BX, WEN, CAT) | 37 |
| commodity (XAU, XAG, XPT, XPD, COPPER, NATGAS, CL, BZ) | 8 |
| pre-IPO (OPENAI, ANTHROPIC, SPCX, ZHIPU, MINIMAX, UNITREE, CXMT) | 7 |
| composite index (BTCDOM, DEFI, FOOTBALL, BLUEBIRD, DOTECO) | 5 |
| gold token (PAXG, XAUT) | 2 |
| stablecoin / FX (USDC, USDBRL) | 2 |

Agreement: 137 contracts are caught by both signals, 35 by name only (mostly listed too
recently for a ratio, or above the threshold), and 37 by the weekend rule only.

Corrections the cross-check forced, before freezing:
- **CVXUSDT stays in.** It is Convex Finance (listed 2022-09, ratio 0.82), not Chevron.
- **ANCUSDT stays in.** It is Anchor Protocol (2022). Its low ratio (0.41) reflects its
  collapse, and the date floor excludes it from the weekend rule.

Contracts listed **after** the scored window that are too new for a ratio (ACN, AGPU, BWET,
CYPH, HUT, MOONSHOT, OURA, PATH, RUM, SECZ, TWST, 哈基米) are unclassified. They cannot
affect the historical run and must be classified, with a change-log row, before the
forward record starts.

## Appendix B — Data counts (2026-10-07)

| item | count |
|---|---|
| registry contracts | 900 (listed by year: 2020 81 · 2021 59 · 2022 26 · 2023 99 · 2024 131 · 2025 242 · 2026 262) |
| still trading at the last bar / stopped | 739 / 161 |
| median traded days per contract | 465 |
| daily kline + funding files | 48,548 + 227 gap-filled daily files, all checksum-verified |
| hourly marks | 895 contracts. **5 have none in Binance's archive** (哈基米, 币安人生, 我踏马来了, 牛来, 龙虾): they fail U6 and are never tradable. 8 daily mark files are genuinely absent (early-2021 days); U6 handles them |
| instrument ends (E1) | 170: 161 stops + 9 relisting gaps; 1-minute index files 338 of 340 |
| prices / funding coverage | 2020-01-01 → 2026-10-06 / → 2026-09-30 23:00 |

## References

Bailey & López de Prado (2014), The deflated Sharpe ratio · Carver (2015), *Systematic
Trading* · Huang, Li, Wang & Zhou (2020), Time-series momentum: is it there? · Hurst, Ooi
& Pedersen (2017), A century of evidence on trend-following investing · Kim, Tse & Wald
(2016), Time series momentum and volatility scaling · Lo (2002), The statistics of Sharpe ratios · Man
Group (Dec 2024), In crypto we trend · Moskowitz, Ooi & Pedersen (2012), Time series
momentum · Newey & West (1987) · Politis & Romano (1994), The stationary bootstrap.
Unverified and **not cited**: the "independent Binance ledger" for BTC funding, Summitward
(Aug 2026), the 2026 post-ETF TSMOM preprint.
