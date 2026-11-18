"""Exclusion list for the per-coin universe (protocol U3, Appendix A).

Two independent signals, combined; no strategy return is used:
1. a name list of known non-crypto, stablecoin, gold-token and composite-index perps;
2. weekend behaviour of the hourly MARK price. An underlying whose home market shuts at the
   weekend moves much less on Saturday/Sunday than on weekdays. Statistic: mean |hourly log
   change| on weekend hours / the same on weekday hours, over the contract's traded range.
   Measured on 2026-10-07 before this script: AAPLUSDT 0.34, ARMUSDT 0.26, BTCUSDT 0.60.

Output: exclusions.csv (contract, category, reason, weekend_ratio) and a review table of every
contract on which the two signals disagree. Run from the repo root after the marks are built:
    python docs/october/per_coin_support/classify_underlyings.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.universe.coinmarketcap.new_fetcher import panel, registry  # noqa: E402

OUT = Path(__file__).resolve().parent
THRESHOLD = 0.55          # weekend/weekday ratio below this => market-hours underlying
LISTED_FROM = "2025-12-01"  # Binance's first non-crypto perps (XAU, XAG, TSLA) listed 2025-12-11;
                            # older contracts are crypto (ANC, 2022, has 0.41 from its collapse)
MIN_HOURS = 24 * 28       # fewer marks than four weeks: the ratio is not computed

NAMES = {
    "stablecoin_or_fx": "USDC USDBRL",
    "gold_token": "PAXG XAUT",
    "composite_index": "BTCDOM DEFI FOOTBALL BLUEBIRD DOTECO",
    "commodity": "XAU XAG XPT XPD COPPER NATGAS CL BZ",
    "equity_etf": ("BITO EWJ EWT EWY EWZ GDX IWM NVDL QQQ SMH SOXL SOXS SPY SQQQ TQQQ TSLL TZA "
                   "URNM UVXY XBI XLE TMF TBT KORU CSOPSAMSUNG2L CSOPSKHYNIX2L KODEX200"),
    "equity": ("AAOI AAPL ADBE ALAB AMAT AMC AMD AMZN ANET APLD APP ARM ASML ASTS AVGO AXTI BABA "
               "BMNR BRKB CIEN COHR COIN COST CRCL CRDO CRM CRML CRWD CRWV CSCO CVNA DDOG DELL "   # not CVX: Convex Finance (2022)
               "DIS DJT DKNG EBAY FLNC GEV GLW GME GOOGL GS HD HIMS HOOD HPE IBM INTC IONQ IREN JPM "
               "KLAC KO LITE LLY LRCX MARA MDB META MRK MRNA MRVL MSFT MSTR MU NBIS NFLX NKE NOK "
               "NVDA NVO OKLO ORCL PANW PDD PLTR PYPL QCOM RDDT RIVN RKLB SHOP SMCI SNDK SNOW SOFI "
               "SONY TEM TER TSLA TSM TTWO TXN UBER UNH WDC WMT XOM ZM ZS MP ONDS USAR "
               "SAMSUNG SAMSUNGEM SKHYNIX HYUNDAI HANMI NAVER LGELECTRONICS "
               "HK0625 HK0700 HK0992 HK1810 TENCENT MEITUAN KUAISHOU POPMART BYD ZHONGJI"),
    "pre_ipo": "OPENAI ANTHROPIC SPCX ZHIPU MINIMAX UNITREE CXMT",
}


def weekend_ratio(contract: str) -> tuple[float, int]:
    path = panel.MARK_DIR / f"{contract}.csv"
    if not path.exists():
        return np.nan, 0
    m = pd.read_csv(path, usecols=["timestamp", "close"])
    t = pd.to_datetime(m.timestamp, utc=True, format="ISO8601")
    r = pd.Series(np.log(m.close.astype(float).to_numpy()), index=t).sort_index().diff()
    r = r[r.index.to_series().diff() == pd.Timedelta(hours=1)].dropna()   # consecutive hours only
    if len(r) < MIN_HOURS:
        return np.nan, len(r)
    wk = r.index.dayofweek >= 5
    weekday = r[~wk].abs().mean()
    return (float(r[wk].abs().mean() / weekday) if weekday > 0 else np.nan), len(r)


def main() -> None:
    contracts = registry.list_contracts() if len(sys.argv) < 2 else json.load(open(sys.argv[1]))
    by_name = {f"{base}USDT": cat for cat, names in NAMES.items() for base in names.split()}
    ranges = panel.load_traded_ranges().set_index("contract")
    rows = []
    for c in contracts:
        ratio, hours = weekend_ratio(c)
        named = by_name.get(c)
        first = ranges.first_traded.get(c)
        flagged = bool(np.isfinite(ratio) and ratio < THRESHOLD and str(first) >= LISTED_FROM)
        rows.append({"contract": c, "first_traded": ranges.first_traded.get(c), "name_category": named,
                     "weekend_ratio": ratio, "hours": hours, "weekend_flag": flagged})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "underlying_review.csv", index=False)
    disagree = df[(df.name_category.notna()) != df.weekend_flag]
    print(f"{len(df)} contracts; named {df.name_category.notna().sum()}; weekend-flagged {df.weekend_flag.sum()}; "
          f"disagree {len(disagree)}")
    print(disagree.sort_values("weekend_ratio").to_string(index=False))
    excluded = df[df.name_category.notna() | df.weekend_flag].copy()
    excluded["basis"] = np.where(excluded.name_category.notna() & excluded.weekend_flag, "name+weekend",
                                 np.where(excluded.name_category.notna(), "name", "weekend"))
    excluded["category"] = excluded.name_category.fillna("market_hours_underlying")
    excluded[["contract", "category", "basis", "first_traded", "weekend_ratio"]].sort_values("contract").to_csv(
        OUT / "exclusions.csv", index=False)
    print(f"\nEXCLUDED {len(excluded)}:", excluded.category.value_counts().to_dict())
    print("by basis:", excluded.basis.value_counts().to_dict())
    print("\nweekend ratio quantiles (unnamed):",
          df[df.name_category.isna()].weekend_ratio.quantile([.01, .05, .1, .5, .9]).round(2).to_dict())


if __name__ == "__main__":
    main()
