"""
Raw CMC bodies -> universe_daily.csv

NO exclusions, NO screens, NO top-N here. This is a faithful transcription; every
judgement call lives in the selector so it can be changed without a re-download.
The old scraper filtered stablecoins BEFORE writing to disk (scraper.py:190),
which destroyed rank information at rest in 81 of 86 snapshots. Never again.

CSV, not parquet: neither pyarrow nor fastparquet is installed, and the rest of
the repo is CSV throughout. 245k rows is nothing to read.
"""

import json
import pandas as pd

from config import DATA_DIR
from data.universe.coinmarketcap.new_fetcher import raw_store

UNIVERSE_TABLE = DATA_DIR / "derived" / "universe_daily.csv"

COLUMNS = [
    "date", "cmc_id", "symbol", "name", "rank", "market_cap", "price",
    "volume_24h", "circulating_supply", "num_market_pairs", "last_updated",
]

def rows_for_date(date: str) -> list[dict]:
    payload = json.loads(raw_store.read(date))
    out = []
    for row in payload["data"]:
        quote = row["quotes"][0]
        out.append({
            "date": date,
            "cmc_id": int(row["id"]),
            "symbol": row["symbol"],
            "name": row["name"],
            "rank": int(row["cmcRank"]),
            "market_cap": _f(quote.get("marketCap")),
            "price": _f(quote.get("price")),
            "volume_24h": _f(quote.get("volume24h")),
            "circulating_supply": _f(row.get("circulatingSupply")),
            "num_market_pairs": row.get("numMarketPairs"),
            # kept ONLY as a restated-vintage flag (§2.5). Never used as a clock.
            "last_updated": row.get("lastUpdated"),
        })
    return out

def build(dates: list[str] | None = None) -> pd.DataFrame:
    """Normalize every stored snapshot into one long table and write it to disk.

    Pass `dates` to rebuild a subset; the default is everything in raw/.
    """
    dates = sorted(dates if dates is not None else _available_dates())

    all_rows: list[dict] = []

    for i, date in enumerate(dates, start=1):    # 2,451 iterations: "2020-01-01", ...
        try:
            coins_on_that_date = rows_for_date(date)   # 100 dicts, one per coin
        except Exception as exc:
            # Name the file. Without this a single bad snapshot gives a
            # traceback with no clue which of 2,451 dates produced it.
            raise ValueError(f"{date}: could not normalize — {exc}") from exc

        for coin in coins_on_that_date:      # {"date": "2020-01-01", "cmc_id": 1,
            all_rows.append(coin)            #  "symbol": "BTC", "rank": 1, ...}

        if i % 500 == 0:
            print(f"  normalized {i}/{len(dates)} dates", flush=True)

    frame = pd.DataFrame(all_rows, columns=COLUMNS)   # 2,451 x 100 = 245,100 rows

    # Typed once, here, so no consumer has to re-parse strings. CSV loses dtypes
    # on the round-trip, so load() below re-applies exactly these two.
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["last_updated"] = pd.to_datetime(
        frame["last_updated"], utc=True, format="ISO8601"
    )

    for row in implausible_caps(frame).itertuples():
        print(f"  WARNING implausible cap: {row.date} {row.symbol} "
              f"${row.market_cap / 1e9:,.0f}bn vs BTC ${row.btc_market_cap / 1e9:,.0f}bn "
              f"({row.market_cap / row.btc_market_cap:.1f}x), supply {row.circulating_supply:.3g}, "
              f"volume ${row.volume_24h / 1e6:,.1f}m", flush=True)

    UNIVERSE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(UNIVERSE_TABLE, index=False)
    return frame


def implausible_caps(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose market cap exceeds BTC's on the same date.

    Nothing has ever been larger than BTC, so when CMC serves one it is a
    retroactive supply restatement rather than a market event. On 2020-11-17 it
    reports LUNC (cmc_id 4172) with 5.88e12 tokens and a $1.83tn cap, 5.4x BTC:
    the post-2022-collapse supply applied to a 2020 price.

    Reported, not filtered. With one known instance across 245,077 rows, being
    told when something is impossible beats silently dropping rows under a rule
    we do not fully understand. The selector's $10m volume floor happens to
    remove this row ($6.55m traded), but that is luck, not a guard — and it
    stops being true if the floor is ever lowered below $6.55m.
    """
    btc = frame.loc[frame.cmc_id == 1].set_index("date").market_cap
    others = frame.loc[frame.cmc_id != 1].copy()
    others["btc_market_cap"] = others.date.map(btc)
    return others[others.market_cap > others.btc_market_cap]


def load() -> pd.DataFrame:
    """Read the table back with the dtypes build() wrote."""
    frame = pd.read_csv(UNIVERSE_TABLE, parse_dates=["last_updated"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def _f(value) -> float | None:
    """CMC nulls -> None rather than a float('nan') that hides in arithmetic."""
    return None if value is None else float(value)


def _available_dates() -> list[str]:
    """Every date present in raw/, from the filenames."""
    return [
        path.name[: -len(".json.gz")]
        for path in raw_store.RAW_CMC_DIR.rglob("*.json.gz")
    ]
