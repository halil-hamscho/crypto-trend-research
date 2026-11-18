"""The record type the raw store writes: one CoinMarketCap historical-listings day, stored verbatim.

The fetcher that produced these records is not published. To rebuild them, store each date's
CoinMarketCap historical-listings response as received (REPLICATION.md)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RawSnapshot:
    date: str            # "YYYY-MM-DD"
    url: str
    body: bytes          # verbatim response bytes: what gets stored
    fetched_at: str      # ISO8601 UTC
    n_rows: int
