# Design rationale — Trend Following in Crypto Beta

Written 2026-09-24, for review. The protocol
(`pre_registration_2026-09-22.md` + `pre_registration_amendment_2026-09-23.md`)
says **what** is fixed. This says **why**, what was traded away, and where the
design could still be wrong. Nothing here overrides the protocol; if the two
disagree, the protocol wins and this file is the bug.

Labels: **[V]** verified in this repo · **[L]** literature · **[O]** judgement.

---

## 1. The question, and why it is this one

> Does trend-following the crypto market *as a whole* make money, net of the real
> costs of trading perpetual futures?

An earlier version of this project asked a different question — can momentum pick
which coins to hold — and the answer turned out to be "no, not in a tradeable
way". The June 2026 work decomposed a momentum information coefficient of ≈0.058
(t ≈ 2.2) into **winner-picking +0.55** and **timing −0.006**, with a within-coin
hit rate of **49.7%** — a coin flip **[V]**. The apparent skill was mostly
survivorship: coins that did well were still in the sample.

Two facts then push towards one aggregate bet rather than ten:

- **Breadth is an illusion here.** Average pairwise correlation across the crypto
  top-50 is ≈0.7 **[V]**, which is about **1.4 independent bets**, not 50
  (heuristic: effective breadth ≈ N / (1 + (N−1)ρ̄), 50 / (1 + 49×0.7) ≈ 1.4 —
  an illustration assuming equal weights and a constant average correlation; the
  basket is cap-weighted, so treat it as an order of magnitude, not a
  measurement) **[O]**. Expanding the
  universe made Sharpe *worse* (top-10 0.43 → top-25 0.18 → top-50 0.00) **[V]**.
- **The information that exists is about the market, not the cross-section.**

So: one signal, one instrument. The intuition is that crypto is closer to a
single asset with ten tickers than to a diversified portfolio, and a design that
pretends otherwise spends its degrees of freedom on noise.

**Tradeoff accepted:** this cannot discover that some subset of coins trends
better than others. That question is registered separately (S16, per-coin TSMOM)
and explicitly cannot rescue the main result.

## 2. The instrument

### 2.1 Why a cap-weighted top-10 basket

A "crypto market" proxy has to answer: which coins, what weights, how often
re-picked.

| choice | why | what it costs |
|---|---|---|
| **top 10** | holds **88.3% of top-100 market cap on average** (80.0% min, 93.0% at the sample end) **[V]**; beyond it coins are thin and wash-traded | omits the tail, where any diversification would come from |
| **cap-weighted** | the standard broad-market proxy — it represents what the market holds and smuggles in no rebalancing premium. Not the only defensible construction: capped variants are registered as S2/S3 **[O]** | BTC dominates: 66% average weight, 95.4% of return variance **[V]** |
| **Binance USDT-M perps** | the thing we could actually trade: shortable, leverable, funding observable | it is *Binance-tradable large-cap crypto*, not crypto. Exchange listing policy enters the universe |
| **monthly reconstitution** | standard index practice (Bloomberg Galaxy, Nasdaq); weekly adds 30 pp/yr turnover for 0.2 extra entries/yr **[V]** | membership can be up to a month stale |

The uncomfortable consequence is stated rather than hidden: **the basket is
mostly BTC**. That is why "does the basket beat BTC-trend?" is a declared
hypothesis (H2) and not a footnote. A reader's first objection should be the
paper's second test.

### 2.2 The buffer: hysteresis, not a filter

Ranks 9–12 shuffle constantly. Re-picking strictly by rank means buying a coin at
rank 10 and selling it at rank 11 a month later, paying twice for nothing.

**Rule:** a coin enters at rank 10 and only leaves when it falls past rank 13.
Intuition: a thermostat with a dead band — heat on at 19°, off at 21°, so it
doesn't chatter at 20°.

The measured effect is an ~24% cut in entries and exits with gross returns
essentially unchanged (daily return correlation 0.99996) **[V]**. It is applied
to the benchmark too, because it changes *what the basket is*, not how it is
traded — if only one side buffered, the comparison would contain a basket
difference as well as a strategy difference.

### 2.3 Drifting weights

Between reconstitutions, quantities are fixed and weights drift with prices. A
cap-weighted index does not trade when prices move; it trades when the index is
re-picked. Resetting weights daily would be a different instrument — one that
systematically sells winners — and would import a rebalancing effect into a study
about trend.

### 2.4 What "point-in-time" honestly means

The selection *rule* uses only the previous day's snapshot. The *data* is a 2026
reconstruction of CoinMarketCap history, and CMC restates: the 2020-11-17 row for
LUNC reports a $1.83tn market cap — 5.4× BTC — from post-collapse supply **[V]**.

A one-day lag stops ordinary look-ahead; it cannot turn revised data into vintage
data. So the instrument is described as **historically reconstructed and
one-day-lagged**, and a broad market claim would need replication against a
governed free-float source. **This is the single weakest input in the study.**

### 2.5 The liquidity screen

CMC volume is a poor screen: Cong et al. (2023) estimate >70% wash trading on
unregulated venues **[L]**, and it aggregates venues we do not trade. The screen
is therefore **the contract's own Binance dollar volume**, 30-day median, ≥$10m,
≥20 observations.

**Tradeoff discovered in implementation:** requiring 20 observations means a
contract is ineligible for its first 20 traded days, which pushed the basket
start from 2020-01-21 to **2020-02-09** **[V]**. That is a real cost (19 days)
for a real protection (one printed day of volume cannot admit a coin).

Open interest would be a better investability measure, but the archives start
2021-12 for alts, so it cannot cover the sample **[V]**.

## 3. The signal

### 3.1 EWMAC, in one paragraph

Take a fast moving average and a slow one. When fast is above slow, prices have
been rising lately relative to their longer history — an uptrend. The *distance*
between them measures strength, but in price units, so a $3,000 gap means
something different at $5,000 than at $100,000. Divide by the typical daily move:
now it reads "the averages are 2 daily moves apart", which means the same thing
in any market at any price. Multiply by a fixed scalar so an average day reads
10, and cap at ±20.

### 3.2 Why a blend of speeds

We do not know the horizon at which crypto trends, if it does. Picking one speed
is a bet on that horizon; picking the speed that worked on this sample is
snooping. Averaging several speeds is the admission of ignorance:

- each speed is a noisy estimate of the same underlying "is it trending?"
- they are correlated (0.87 adjacent, 0.30 across the range) **[V]** but not
  identical, so averaging cancels some noise
- the combined forecast is smaller than its parts, so a fixed multiplier (FDM
  1.15) restores the intended scale

**Tradeoff:** a blend cannot beat the best single speed in hindsight. That is the
point. In exchange, it cannot be accused of having chosen it.

### 3.3 Selecting speeds by a cost rule

Carver's "speed limit": a rule should not spend more than ≈**0.13 Sharpe per
year** on trading costs **[L]**. Cost in Sharpe units is

```
cost_SR = annual_turnover × cost_per_side ÷ annual_volatility
```

The intuition for the denominator: costs are a fixed toll per trade, but the
prize scales with volatility. A 10 bp toll is trivial in a 64%-vol market and
expensive in a 15%-vol one. Crypto's volatility is why fast rules are even
arguable here.

Turnover depends on the filter **and** on the price process — it is not a
property of the filter alone — but it needs no project data, so it is measured on
synthetic paths, with a trending (AR(1)) sensitivity beside the random-walk basis
(all six verdicts unchanged: 4/16 moves 0.161 → 0.158 SR/yr at 50% vol) (`prereg_support/speed_cost_sim.py`) **[V]**.
The 50–80% range brackets the basket's measured ~64% annualised volatility — a
property of the instrument, not of any strategy return, and stated here so the
range is not mistaken for an arbitrary choice **[O]**:

| speed | turnover/yr | cost SR @50% vol | @80% vol | admitted |
|---|---|---|---|---|
| 2/8 | 158 | 0.316 | 0.197 | no |
| 4/16 | 80 | 0.161 | 0.101 | no |
| 8/32 | 41 | 0.082 | 0.052 | yes |
| 16/64 | 23 | 0.045 | 0.028 | yes |
| 32/128 | 14 | 0.028 | 0.017 | yes |
| 64/256 | 10 | 0.021 | 0.013 | yes |

**Why the volatility *range* matters.** An earlier version of this rule evaluated
cost at a single assumed 64% volatility, where 4/16 scored 0.126 against a 0.13
limit — admitted by 0.004. A reviewer pointed out that this is not a rule but a
coincidence: at 60% vol it fails. Requiring a speed to pass across 50–80% makes
the admission robust to an assumption we cannot pin down **[O]**.

**The honesty problem this fixed.** The first version also excluded 64/256 on a
"warm-up doesn't fit" argument — but 512 days still leaves 5.2 scored years, and
"drop 64/256" was a conclusion previously drawn *from this sample*. Reverse-
engineering a rule to reach a remembered answer is exactly the failure mode
pre-registration exists to prevent. 64/256 is now in.

**Cost paid:** including 64/256 doubles the warm-up (256 → 512 days) and removes
~8 months of scored sample.

### 3.4 Fixed scalars, not estimated ones

Carver's scalars come from pooling dozens of futures markets over decades
**[V: pysystemtrade]**. For this project they are constants fixed *outside* the
sample under test. The alternative in the existing code rescales by a rolling
250-day median of the forecast's own magnitude — not look-ahead, but it
re-normalises the signal to its own recent behaviour, so "strong" means "strong
relative to last year". That changes behaviour, not just scale, and it is fitted
to the sample under test. Barred.

**Risk accepted:** futures-derived scalars may not fit crypto, so the realised
mean |forecast| may not be 10. That is measurable and is reported (with the
cap-hit rate and a sign-only variant), and it affects *size*, not the sign of the
bet.

### 3.5 `adjust=True`

`adjust=False` seeds an exponential average with the first observation, which is
equivalent to assuming the price sat flat at that value forever beforehand. Early
in the sample the slow average is then anchored to invented history, producing a
trend signal that is partly an artefact. For a 128-day span, 13.5% of the weight
still sits on that invented history after 128 days **[V: arithmetic]**. The
512-day warm-up already covers it; `adjust=True` removes it entirely.

## 4. Risk and sizing

### 4.1 Volatility targeting

```
position = capital × (forecast / 10) × (target_vol ÷ current_vol)
```

Two ideas stacked. **Forecast/10** scales the bet with conviction: 10 is a normal
bet, 20 the maximum, negative is short. **target ÷ current vol** makes a "normal
bet" mean the same amount of *risk* regardless of regime — in a calm market you
hold more coins for the same risk, in a wild one fewer.

**Why it matters more in crypto than anywhere else:** realised volatility ranges
roughly 30–150% across regimes. Without targeting, position risk would be
dominated by whatever volatility happened to be, not by the signal.

### 4.2 Why a floor and a cap

`1/vol` is unbounded as vol → 0, and low-vol periods are precisely when a shock is
most damaging (volatility clusters). Also a halted or frozen contract has
measured vol of exactly zero, which reads as a riskless asset **[V: CLAUDE.md
constraint 2]**.

- **Floor** (5th percentile of the estimate's own past 500 days, Carver's rule):
  caps the *estimate* before it produces nonsense. Adaptive, past-only, nothing
  for us to tune.
- **Leverage cap 2.0:** a hard stop on the *position* if something slips through.

They overlap deliberately; the failure they guard against is silent.

### 4.3 30% target and the invariance argument

Sharpe is invariant to leverage, so the target sets return and drawdown scale but
not the ratio. It is fixed anyway, because "we ran it at 40% and it looked better"
is a knob, and the drawdown number that readers see must not be chosen after the
fact.

## 5. Execution

### 5.1 Buffering, and why it is not free

Two policies are run: trade to the exact target daily, or trade only when the
position drifts outside a band of 10% of the standard exposure, and then only to
the band's edge.

The intuition is the same dead band as the membership buffer, one level down.
Chasing a forecast from 10.3 to 10.4 across ten contracts costs real money and
changes the risk position negligibly.

**The catch, stated in advance:** buffering is *also* a smoother. Trading later
means following the signal slightly less closely. So gross should fall a little
and net should rise. If gross *rises*, the buffer is seeing something it
shouldn't — a tripwire, not proof of a bug, since one sample can produce it by
chance.

**Both policies are reported side by side.** Reporting only the buffered number
is the named trap: it is the flattering one, since it pays less cost.

### 5.2 Leg-level accounting, and the bug it prevents

The tempting cost formula is `cost × |position_t − position_{t−1}|` on the basket
scalar. It is wrong, and wrong in the direction that flatters: on a reconstitution
day the basket exposure can be unchanged while one constituent is sold and
another bought. The scalar formula charges **zero** for two real trades.

So the ledger is per contract per day, and turnover is
`Σᵢ |Nᵢ,target − Nᵢ,marked|`. The acceptance test is a fixture where aggregate
exposure is constant and constituents swap: it must produce non-zero cost.

### 5.3 Funding is a holding cost, not a transaction cost

**Sign convention (Amendment 3):** rates are positive when longs pay, so
`funding_pnl = − position × rate`, `gross = price_pnl + funding_pnl`, and
`net = gross − fees − slippage`. A short with a positive rate receives.

Perpetual futures have no expiry; instead longs and shorts exchange a funding
payment several times a day to keep the contract near spot. Longs have paid, on average
over the registered window, **11.5%/yr on BTC, 13.7% on ETH, 14.0% on XRP**
(mean annualised; realised 11.4 / 13.5 / 13.8%/yr)
**[V: `prereg_support/instrument_facts.py`]**. You cannot avoid it by trading less,
so it belongs in *gross* return alongside price P&L — it is the cost of the
exposure, not of the execution.

**The trap:** assuming 8-hour settlements. This data contains 296,534 rows at 8h,
**50,736 at 4h**, 1,805 at 1h and 147 at 2h **[V]**. Hard-coding three payments a
day would misstate funding for every 4-hour contract. Intervals are read from the
data.

### 5.4 Contract death

Binance settles a delisted perp at the **average index price over the final 30
minutes** (1,800 one-second observations) and charges the taker fee **[V]**.
"Last daily close" is wrong in both price and fee.

Scope, measured rather than assumed **[V]**: of ten mapped contracts that stopped
trading, four were ever basket members, and **only LUNA was still held when its
contract died** (2022-05-12) *[superseded for the registered membership: see the 2026-09-30 correction at the end]*. The other three (MATIC, EOS, TON) were rebrands
that happened 10 months to 4 years after those coins left the top ten. FTT was
never a member. Of 28 member contracts only ICP ever halted, a year after it left.

So the whole delisting apparatus exists for **one event** — but it is the one
where a position of unknown sign was held into a near-total collapse, so the
settlement price could matter enormously. LUNA's contribution is reported separately, with
the headline shown with and without it, decided *before* seeing which way it cuts.

## 6. Costs

Nothing about crypto execution costs is authoritative, so the design layers
assumptions rather than pretending to precision:

| layer | value | what it is for |
|---|---|---|
| primary | 10 bp/side (≈5 fee + ≈5 spread) | one number that cannot be tuned |
| stress | 5 and 20 bp | brackets the uncertainty |
| measured A | Abdi & Ranaldo (2017) spread from daily high/low/close | full-sample realism check **[L]** |
| measured B | effective spread from Binance `aggTrades` | same venue, signed trades, covers dead coins |
| validation | true quotes from `bookTicker` | ground truth where it exists |

**The data finding that shaped this:** `bookTicker` (real bid/ask) exists only
**2023-05 → 2024-04**, for every symbol, and Binance never explained why it
stopped **[V]**. So true spreads for the 2022 stress period are unobtainable at
any price short of a commercial vendor. `aggTrades`, however, covers 2020→now
including LUNA and FTT through their collapse months, and its rows are signed, so
an *effective* spread can be estimated from actual executions.

**Why the primary stays a flat 10 bp:** the measured estimators contain choices
(which estimator, which window) and noise. The headline must rest on a number
nobody can adjust; the measured variants say whether that number was generous or
stingy, and are reported either way.

## 7. What is actually being tested

### 7.1 H1: alpha against the same basket

```
r_strategy − r_cash = α + β (r_benchmark − r_cash) + ε
```

The benchmark is buy-and-hold of the **same** basket, paying the same funding and
costs. Using the same universe on both sides cancels the listing and survivorship
effects that would otherwise masquerade as skill.

**Why alpha and not the Sharpe difference:** the claim is "adds value beyond
passive crypto beta". Positive α means the strategy earned something its exposure
to the basket does not explain — which is exactly that claim. Sharpe difference
answers "which is better alone?", a different question, and is reported but not
tested.

**Why linear and not Treynor–Mazuy.** In `r = α + β·r_m + γ·r_m²`, successful
timing loads on **γ**, and α can come out *negative* for a strategy that times
well. Using TM's α as the test statistic could reject a working strategy. So the
linear α is the test and γ is the mechanism evidence.

### 7.2 Inference

Daily crypto returns are fat-tailed, volatility-clustered and serially dependent;
positions persist for weeks. Textbook standard errors would be far too
optimistic. So: **stationary block bootstrap** (Politis & Romano 1994) **[L]** on
**paired** strategy/benchmark daily returns, 20-day mean blocks, 10,000
replications, with Newey–West as an analytical cross-check.

Two details that matter:
- **Resample returns, not prices.** Rebuilding EWMAs across randomly joined price
  blocks would manufacture signal paths that never existed.
- **Pair the two series.** Their contemporaneous relationship is the thing being
  estimated.

### 7.3 What this sample can and cannot do

Sharpe SE ≈ **0.46** over 5.16 years. Minimum detectable effects **[V]**:

| test | MDE (80% power, 5% one-sided) |
|---|---|
| H1 α | **≈30.6%/yr** (appraisal ratio ≈1.09) |
| H2 gap, ρ=0.97 | **≈8.0%/yr** |

**H1 can only confirm a large effect.** The most likely honest outcome for a
modest true edge is "economically positive, statistically inconclusive" — which
is why that outcome has pre-written wording, so it is not spun in either
direction.

**The precise part.** Cost and funding drags are *paired same-day differences on
identical positions*, so they are estimated far more tightly than any Sharpe.
"Perp frictions cost X%/yr" is a defensible finding whichever way the headline
falls — conditional on the cost model, which is itself uncertain.

### 7.4 H2 and the hurdle

The basket is mostly BTC, so the strongest objection is "this is BTC trend with
extra steps". H2 tests the **paired daily gap** against an otherwise identical
BTC-only trend strategy. Gating: H2 is confirmatory only if H1 passes
(fixed-sequence, which controls family-wise error without splitting α) **[L]**.

The **superiority hurdle** δ = 5 pp/yr asks an economic question, not a
statistical one: a ten-contract basket must add at least that much over one
liquid contract to be worth the funding, monitoring, reconstitution and delisting
complexity — ≈0.17 incremental Sharpe at a 30% target.

**Stated asymmetry:** δ sits *below* the gap's MDE, so this test can say "no
material advantage" or "inconclusive", but can only say "basket superior" for a
gap far larger than δ. Declaring that in advance stops it being discovered as a
convenient surprise.

### 7.5 Trials

Sixteen secondary runs are registered, typed by kind — strategy configurations
count as trials for the deflated Sharpe; cost scenarios, statistical analyses,
diagnostics and declared comparators do not. Earlier mining of this sample is
*not* enumerated, so the deflated Sharpe is reported as a **partial lower-bound**
correction with sensitivity to assumed prior trial counts.

## 8. Pseudocode

```
# ---------- universe (monthly, D-1 information) --------------------------
for R in reconstitution_dates:                       # 1st of month, 00:00 UTC
    snapshot = cmc_rows(date=R-1) sorted by market_cap desc
    eligible = [row for row in snapshot if
                  row.cmc_id not in EXCLUDED                       # stables, wrapped
                  and contract_traded_on(row.cmc_id, R-1)          # volume > 0 that day
                  and median(quote_volume[contract, R-31:R-1]) >= 10e6
                  and count(quote_volume[contract, R-31:R-1]) >= 20]
    held = [c for c in held if rank(c, eligible) <= 13][:10]       # buffer: stay to 13
    held += [c for c in eligible if c not in held][:10 - len(held)] # refill from the top
    quantity[i] = capital * cap_weight(i) / price[i, R]            # fixed until next R

# ---------- index --------------------------------------------------------
basket_return[t] = Σ weight[i, t-1] * simple_return[i, t]
index[t]         = index[t-1] * (1 + basket_return[t])             # chain-linked

# ---------- signal (close of D) -----------------------------------------
vol[t]   = max( ewma_std(index_returns, span=32, adjust=True)[t],
                quantile(vol[t-500:t], 0.05) )                     # Carver floor
for (fast, slow), scalar in [((8,32),5.3), ((16,64),3.75),
                             ((32,128),2.65), ((64,256),1.87)]:
    raw     = ewma(index, fast, adjust=True) - ewma(index, slow, adjust=True)
    f[s, t] = clip(raw[t] / (index[t] * vol[t]) * scalar, -20, +20)
forecast[t] = clip(mean over s of f[s, t] * 1.15, -20, +20)        # FDM

# ---------- sizing -------------------------------------------------------
if t < 512th observation: position[t] = NaN; continue              # warm-up excluded
standard[t] = 0.30 / annualised(vol[t])
x_star[t]   = clip(forecast[t] / 10 * standard[t], -2.0, +2.0)

# ---------- execution ----------------------------------------------------
band = 0.10 * standard[t]
if policy == "full":
    target = x_star[t]
else:                                                              # buffered
    target = held_exposure if abs(held_exposure - x_star[t]) <= band
             else nearest_edge(x_star[t], band)                    # to the edge, not the centre
target = clip(target, -2.0, +2.0)                                  # cap again, post-band
leg_target[i] = target * weight[i, t] * capital                    # proportional, drifted weights
if t in reconstitution_dates: leg_target reflects new membership   # always trades

# ---------- ledger (per contract, per day) ------------------------------
trade[i]     = leg_target[i] - leg_marked[i]
fee[i]       = abs(trade[i]) * cost_per_side
funding[i]   = Σ over settlements actually recorded on day t:
                   position_at(settlement) * rate(settlement)      # 8h, 4h, 2h or 1h
price_pnl[i]   = leg_marked[i] * simple_return[i, t]
funding_pnl[i] = -funding[i]                                       # rate>0 => a long pays
net[i]         = price_pnl[i] + funding_pnl[i] - fee[i]
assert abs(Σ net - (Σ price_pnl - Σ funding - Σ fee)) < 1e-9       # the identity

# ---------- events -------------------------------------------------------
if contract delists while held:
    exit_price = mean(index_price_1m[last 30 minutes before delisting])
    charge taker fee; proceeds -> cash until next reconstitution
    tag rows (LUNA 2022-05-12: held only in the weekly sensitivity; none in the registered run, see 2026-09-30 correction)

# ---------- scoring ------------------------------------------------------
window  = 2021-07-05 .. 2026-08-30                                 # 1,883 days (Amendment 6)
α_net   = α_price + α_funding − α_cost                             # common β, Amendment 3
sharpe  = 365 * mean(net_return) / (sqrt(365) * std(net_return))   # mean/vol, not CAGR/vol
α, β    = OLS(strategy_excess ~ benchmark_excess)
p_value = stationary_bootstrap(paired daily returns, mean_block=20, reps=10000)
gap     = mean(strategy_net - btc_trend_net)                       # H2, if H1 passes
```

## 9. Pre-mortem: how this could still be wrong

Ranked by how much damage it would do **[O]**.

1. **The CMC reconstruction is not vintage.** Restated caps could put the wrong
   coins in the basket on some dates. Mitigation: a restatement tripwire (not yet
   implemented) and, for any broad claim, replication on a governed source.
2. **The ledger has a silent accounting bug.** Costs or funding understated makes
   everything downstream flattering. Mitigation: the identity test, the swap
   fixture, and an independent recomputation of at least one month by hand.
3. **The cost model is optimistic.** 10 bp assumes a liquid market and small
   size; 2022 spreads are unobservable. Mitigation: the 20 bp stress decides
   "supported" vs "fragile", and measured variants are reported.
4. **The result is one event.** With 1,883 days (Amendment 6) and, in the weekly sensitivity only, one catastrophic delisting,
   LUNA could dominate that variant. Mitigation: pre-committed separate reporting.
5. **The sample is mined regardless.** Everything above limits *further* freedom;
   none of it makes 2020–2026 out-of-sample. Only the forward record does.
6. **Carver's parameters may not transfer to crypto.** Mitigation: diagnostics
   (mean |forecast|, cap-hit rate, sign-only variant) that make a mismatch
   visible instead of silent.

## 10. Reviewed

**Second review (2026-09-24), incorporated as Amendment 3.** The attribution fix
was revised: differencing two separately estimated alphas is not an attribution,
because the regressions have different betas — so β is fixed from the primary net
regression and the same α is decomposed across layers. H2 is matched *ex ante* by
construction, so "not risk-matched" was too strong; realised-risk diagnostics are
added instead and H2 is unchanged. Frictions become a primary **descriptive**
estimand rather than a co-primary endpoint, which would have implied a second
hypothesis. Funding sign conventions, superseded sample dates, the LUNA
settlement counterfactual, the terminal-liquidation sensitivity and reproducible
fact scripts are all in that amendment. Funding avoidance is **not** a lesser
result: it would mean the system times total perpetual returns rather than price
returns alone.


`design_rationale_review.md` (2026-09-24) raises eleven findings against this
document. Three are substantive and require a dated amendment before the first
registered run: α should also be estimated on the **price-only** layer, so
"timing prices" is separated from "paying less funding"; H2's comparator is not
risk-matched and must not be described as a clean instrument comparison; and the
frictions decomposition should be elevated to a co-primary *reported* result,
since it is the part this sample can measure precisely. R9–R11 have been fixed in
this file.

## 11. Open questions for reviewers

1. Is the **cap-weighted, uncapped** basket the right primary estimand, given
   that it is ~95% BTC by variance, or should the 30%-capped version be primary
   with uncapped as robustness?
2. Is the **superiority hurdle** (δ = 5 pp/yr) economically right, or should the
   threshold be set on incremental Sharpe rather than return?
3. Is the **taker assumption** too conservative? A patient implementation would
   post rather than cross, reducing cost materially.
4. Is **excluding funding from execution cost** (treating it as a holding cost)
   the right accounting for a strategy whose sign flips? A short *receives*
   positive funding, so the strategy's funding P&L is signal-dependent.
5. Does the **512-day warm-up** buy enough to justify losing 8 months, or should
   the slowest speed be dropped after all — accepting that this reopens the
   snooping question?

---

## Correction, 2026-09-30

Under the registered monthly buffered membership (enter at 10, exit at 13, carried from 2020-02-09) LUNAUSDT is never held: 79 selections, 24 distinct contracts. It is held only in the weekly (Monday) sensitivity, the week of 2022-05-09; the daily sensitivity drops it before it dies. See `basket_event_inventory.md`. The statements above that LUNA was "still held when its contract died" and is "the only case in this sample" were measured under an earlier membership rule and do not describe the registered monthly basket. The delisting apparatus (`settle`) is exercised by the weekly sensitivity only; the primary run holds no contract through a death, halt or rebrand. Also corrected: the scored window is 1,883 days (Amendment 6). The previously timestamped version of this document is preserved, with its proof, in `prereg_support/ots_2026-09-24_amendment3/`.
