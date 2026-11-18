# Trial 62: risk-targeted per-coin trend with a fixed risk budget by coin size — design for review

**Dated 2026-10-30. Approved by the researcher 2026-11-05 (§2 as recommended; T6 widened to all books, §5).
Nothing is frozen or run.** A new, separately registered
construction and new counted trials (62–64, §3). It changes one thing in the sealed stage-C construction
(`risk_targeted_trend_design_2026-10-27.md` §7; result `reports/per_coin_run/1df6ed18299b/attempt_1`): how the
risk budget is split across coins. Labels: **[V]** verified in this repository, **[R]** recommended.

## 0. Why, stated honestly

- **Where the idea came from.** The exploratory long-side decomposition
  (`long_side_decomposition_2026-10-30.md`), run after stage C was known, found that equal risk per coin over
  ~130 coins puts 63% of long risk in coins ranked 51+ or unranked. The book is net short on 80% of days, and
  its longs lose through which coins they hold [V]. The idea is therefore **learned from this mined sample**.
  That is why it is a counted trial, fixed here before any run, and judged finally on forward data.
- **What it tests.** Whether giving each size tier of the market a fixed share of risk, instead of each coin an
  equal share, makes risk-targeted trend detectably better than its own risk-targeted long-only twin.
- **What it may cost.** Stage C's profit came mostly from shorting small coins: the smallest bucket earned
  +20.7%/yr short [V]. A tier budget cuts that bucket's share of risk from about two thirds to one third, so the
  short side will likely earn less. **A lower Sharpe than stage C is a plausible outcome**, and the design
  does not assume otherwise. The case for it is a better-balanced book: less dependence on shorting small,
  hard-to-trade coins, where a flat 10 bp cost is least credible.
- **Researcher's choice (2026-10-30):** a fixed risk budget by coin size alone. Not scaling down weak
  forecasts and restricting the universe are excluded (one change at a time).

## 1. What changes and what does not

**Unchanged:** everything in stage C. That covers:
- the universe (registry-wide, $10m liquidity gate, Clarification 6) and eligibility (X−1);
- the signal (fixed Carver scalars, short-history speeds, ±20 caps) and per-coin volatility with its floor;
- the one-day lag and 00:00 fills; costs (10 bp, 5 bp settlement fee) and funding;
- the registers, ledger, reconciliations and attribution;
- dynamic risk targeting in the Clarification 2 order: 30% target, unbounded M, caps 0.5 / 3.0, no overlay,
  weak forecasts scaled up as before;
- inference: bootstrap, 10,000 replications, blocks 20/10/40/60, seed 20261007.

**Changes: only the instrument weight wᵢ** in the unscaled book, uᵢ = (fᵢ/10) × **wᵢ** × target / σᵢ. In stage C
wᵢ = 1/N. Here:

> wᵢ = (budget of coin i's tier) / (number of eligible coins in that tier on decision day X)

The same wᵢ enters the primary band (0.10 × wᵢ × IDM × target / σᵢ), exactly as 1/N does in stage C. M then
rescales the whole book to the target, so the tier budget sets the **mix** of risk, not its level.

## 2. Decisions to fix before any build

| # | decision | options | recommendation |
|---|---|---|---|
| T1 | **Tiers** | (a) top 10 / 11–50 / the rest; (b) BTC+ETH / the rest; (c) four or more tiers | **(a) [R].** The breakpoints are the universes this project has used since 2026-06 (top 10, top 25/50), not chosen from the decomposition. Disclosed: the decomposition's buckets coincide with them. (b) makes the book a BTC/ETH timer again, the basket study's null |
| T2 | **Budget across tiers** | (a) equal thirds; (b) market-cap weighted; (c) square root of tier market cap | **(a) equal thirds [R].** Carver's handcrafting (*Systematic Trading*, 2015, ch. 4): equal weight across groups, then equal within. It has no fitted parameter. (b) puts ~90% in the top 10 and recreates the cap-weighted basket. (c) adds an arbitrary exponent |
| T3 | **Within a tier** | equal; risk-parity on σ | **Equal [R].** The 1/σᵢ in uᵢ already equalises risk per coin within a tier, as in stage C |
| T4 | **Ranking source** | CMC `market_cap` (Constraint 1, never `rank`); Binance quote volume | **CMC market cap of day X−1 [R]** (Constraint 4: CMC's day D is usable on D+1). Coins are mapped through `cmc_contract_map_top100.csv` inside each claim range. Ranks are taken among the coins **eligible at X**, ties broken by `cmc_id`. An eligible contract with no mapped coin that day goes to the third tier |
| T5 | **Empty or thin tier** | redistribute; leave the budget unused | **Redistribute equally [R].** If a tier has no eligible coin at X, its third is split equally across the non-empty tiers. Logged daily. Under dynamic risk targeting the two options give the same positions: M rescales the book to the target either way, and with equal budgets the remaining tiers keep equal shares |
| T6 | **Books** | primary only; plus top 25 / top 50 secondaries | **Approved: all books.** Registry $10m (primary) with its K1 twin under the same tier weights, a 20 bp stress pair for FRAGILE, and top 25 / top 50 as secondaries. Inside a top-25 or top-50 book the third tier is almost always empty, so by T5 risk splits half to the top 10 and half to the rest |
| T7 | **Primary comparison** | vs its tier-weighted twin; vs stage C | **vs its tier-weighted, risk-targeted K1 twin [R]** (H1′, Part 9 and Clarification 4 in full). vs stage C's sealed book: descriptive only, after an identity check that the two differ only in the weights |

## 3. Tests and reading

| result | comparison | rule |
|---|---|---|
| **H1″** (primary) | `registry_10m_tier` vs `registry_10m_tier_k1` | Part 9 / Clarification 4: FAIL / INCONCLUSIVE / H1 PASSES, warnings FRAGILE, NOT PER-COIN, COLLAPSED, LARGE CLAIM REJECTED; stress from `registry_10m_tier_20bp` vs its twin |
| secondaries | `top_25_tier`, `top_50_tier` vs their twins | read descriptively with the same rules, as stage C's secondaries |
| vs stage C | each tier book vs its sealed stage-C counterpart (`registry_10m_dyn`, `top_25_dyn`, `top_50_dyn`) | descriptive: what the tier budget alone did, after an identity check that the two differ only in the weights |
| decomposition | long and short P&L by tier; risk share by tier vs the budget; net exposure; M; cap days; turnover | always reported |

- **Trial count:** 61 + 3 = **64** for the deflated Sharpe (bracket 30–80): the three tier books
  (`registry_10m_tier`, `top_25_tier`, `top_50_tier`), counted as stage C counted its secondaries. Twins and the
  stress pair are not counted.
- **Expected outcome, stated now:** INCONCLUSIVE is the most likely reading. The minimum detectable net
  Sharpe is ≈1.1, and stage C itself was INCONCLUSIVE. A Sharpe below stage C's 0.50 would not contradict the
  design's purpose (§0).

## 4. Build and validation

- **To add:**
  - a tier table by decision day: rank, tier, wᵢ, from the lagged CMC table and the deep map, read through the
    checked loader;
  - a `weights` input in `dynamic_positions` (default 1/N, so every sealed book is unchanged);
  - run specs as stage "D" with their own manifest.
- **Synthetic tests:**
  - with every coin in one tier, the weights reduce to 1/N and reproduce stage C's positions exactly;
  - at forecasts of ±10, each tier's share of the ideal book's stand-alone risk (Σ |position| × σ) equals its
    budget. (Its share of portfolio variance does not in general: that depends on the correlation and on how
    many coins share the tier, so it is reported, not imposed);
  - a tier's rank uses CMC day X−1 only (truncation-rebuild);
  - redistribution when a tier is empty;
  - the twin is long-only at the target.
- **Then, each with the researcher's approval:** the blind real-data check, the freeze, and one registered
  attempt.

## 5. Fixed at approval (2026-11-05)

- **Tiers:** CMC market-cap rank 1–10 / 11–50 / the rest, among the contracts eligible at X, on CMC day X−1. The
  researcher's first proposal, dollar tiers ($500m+ / $100–500m / $10–100m), was set aside before any build: the
  CMC table holds the top 100 coins a day only (the 100th coin's median cap over the window is about $540m), so
  the two lower dollar tiers could not be told apart.
- **Coin of a contract:** the deep map (`cmc_contract_map_top100.csv`); a row with a claim range applies only
  inside it, on the CMC day X−1. A contract claimed by two coins on one day stops the run. Ties in market cap are
  broken by `cmc_id`, then contract name.
- **Budgets:** one third per non-empty tier, equal within a tier: wᵢ = (1 / number of non-empty tiers) / (coins in
  i's tier). With one non-empty tier this is 1/N exactly, and the book is stage C's.
- **Scope:** dynamic sizing only (the stage-C construction); the tier weights replace 1/N in the unscaled book
  and in the primary band, and nothing else.

## 6. Questions for the researcher

1. Approve T1–T7 as recommended, or change any? **Approved 2026-11-05: rank tiers, T2–T7 as recommended, T6
   all books.**
2. (Open) Start forward paper trading of stage C in parallel? It needs 12 months, and trial 62 can join the forward
   record if it is frozen.
