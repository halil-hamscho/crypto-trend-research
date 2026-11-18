# Replication

The code, frozen designs and sealed results are here; the market data is not (it is several gigabytes and not mine to redistribute).

## Without data

```bash
pip install -r requirements.txt     # Python 3.12, exact versions used for every run
python verify_manifests.py          # sealed runs' recorded hashes match this code and config
python run_tests.py                 # synthetic tests; data-dependent ones report SKIP
```

## Rebuilding the data

Data goes in `data/raw/` (as downloaded) and `data/derived/` (rebuilt from raw), both gitignored.

- **Binance** (prices, funding, hourly marks, settlement files): downloaded and checksum-verified by the code in `data/universe/coinmarketcap/new_fetcher/` (the folder name is historical; most of it is Binance code). Run `registry.build()`, then `panel.download()`, `panel.build_all()`, `panel.fill_gaps()`, `panel.build_all()`, then the same for marks. Always pass the full contract list.
- **CoinMarketCap** (daily market-cap ranks, 2020-01-01 to 2026-09-16): the download script is not published. Save each date's historical listings, as received, to `data/raw/cmc/<year>/<date>.json.gz`, then run `normalize.build()`. Caps are restated on some dates, so a rebuild may not match exactly.

The event, funding-gap and exclusion registers are committed. The coin-to-contract maps (`data/universe/cmc_contract_map*.csv`) contain CoinMarketCap identifiers and are not published; rebuild them from the listings with `contract_map.build()` and `docs/october/per_coin_support/build_deep_map.py`.

## Running a registered stage

The run driver checks every input file against the hashes frozen with the data and stops if any differ. To experiment on rebuilt data, call the engine (`portfolio/per_coin_*.py`) directly and treat the results as new, unregistered runs.

Some documents cite internal design notes that are not published; every design, pre-registration and protocol a run was checked against is in `docs/`.
