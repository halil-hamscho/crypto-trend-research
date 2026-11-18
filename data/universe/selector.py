"""Universe selector: filter-then-rank, one day lagged.

The selection rule is point-in-time (it reads only the D-1 snapshot), but the
underlying CMC history is a 2026 reconstruction, not a vintage feed. See
`docs/september/pre_registration_amendment_2026-09-23.md` A1.

The universe is "the top N coins by market cap among those whose Binance USDT-M
perp traded that day" — not "the top N cryptocurrencies". Coins that never had a
Binance perp (LEO, HT, CRO, BGB) are left out by design, so build the benchmark
from this same selection.
"""

from datetime import date, timedelta

import pandas as pd

from data.universe.coinmarketcap.new_fetcher import contract_map, normalize, panel

# Stablecoins, wrapped and gold-pegged tokens, by cmc_id (symbols get reused).
EXCLUDED_CMC_IDS = frozenset({
    825,    # USDT
    3408,   # USDC
    4943,   # DAI
    2563,   # TUSD
    4687,   # BUSD
    3330,   # USDP (formerly PAX)
    3306,   # GUSD
    26081,  # FDUSD
    27772,  # PYUSD
    29470,  # USDe
    36148,  # USD1
    7129,   # UST / USTC
    3717,   # WBTC
    4705,   # PAXG
    5176,   # XAUT
})


class PoolTooShallow(Exception):
    """The selection reached a coin outside the contract map, so its perp was never checked."""
class UniverseSelector:
    # Investability screen, frozen 2026-09-22: the contract's own Binance dollar
    # volume, not CMC's multi-venue figure, which is largely wash-traded
    # (Cong et al. 2023). Median over the trailing window, so one printed day
    # cannot admit a coin.
    VOLUME_WINDOW_DAYS = 30
    VOLUME_MIN_OBSERVATIONS = 20

    def __init__(self, min_history_days: int, n: int = 10, min_volume_usd: float = 10e6):
        self.n = n
        self.min_volume_usd = min_volume_usd
        self.min_history_days = min_history_days

        universe = normalize.load().sort_values(["date", "market_cap"], ascending=[True, False])
        self._by_date = dict(tuple(universe.groupby("date")))
        self._pool = contract_map.pool_ids(universe)

        # (contract, day it traded) -> days traded so far, which is the warmup clock
        self._history = {}
        # (contract, day) -> trailing median Binance dollar volume, the screen
        self._dollar_volume = {}
        first_day = {}
        contracts = contract_map.load()
        for contract in contracts.contract.unique():
            bars = panel.load_ohlcv(contract)
            traded = bars[bars.volume > 0]
            days = traded.timestamp.dt.date
            first_day[contract] = days.iloc[0]
            for count, day in enumerate(days, start=1):
                self._history[(contract, day)] = count

            # The window ends on the day itself: every bar in it closed before
            # the trade date, because select() reads the D-1 snapshot.
            median = traded.quote_volume.rolling(
                self.VOLUME_WINDOW_DAYS, min_periods=self.VOLUME_MIN_OBSERVATIONS
            ).median()
            for day, value in zip(days, median):
                self._dollar_volume[(contract, day)] = value

        # cmc_id -> its contracts, oldest first
        self._contracts = {
            cmc_id: sorted(group.contract, key=first_day.get)
            for cmc_id, group in contracts.groupby("cmc_id")
        }
    def contract_on(self, cmc_id: int, day: date) -> str | None:
        """The contract that traded this coin on `day`, or None. On a rename, the older one."""
        for contract in self._contracts.get(cmc_id, []):
            if (contract, day) in self._history:
                return contract
        return None

    def screen(self, row, day: date) -> tuple[str | None, str | None]:
        """(reason, None) if the coin is excluded, else (None, contract)."""
        if row.cmc_id in EXCLUDED_CMC_IDS:
            return "stable_or_wrapped", None
        if row.cmc_id not in self._pool:
            return "outside_pool", None
        if row.cmc_id not in self._contracts:
            return "no_contract", None
        contract = self.contract_on(row.cmc_id, day)
        if contract is None:
            return "not_trading", None
        dollar_volume = self._dollar_volume.get((contract, day))
        if dollar_volume is None or not dollar_volume >= self.min_volume_usd:
            return "low_volume", None
        if self._history[(contract, day)] < self.min_history_days:
            return "short_history", None
        return None, contract
    def select(self, trade_date: date, n: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(members, exclusions) for `trade_date`, from the previous day's snapshot.

        `n` overrides the configured basket size, which is how `select_buffered`
        reads the eligible list out to the exit rank.
        """
        n = self.n if n is None else n
        day = trade_date - timedelta(days=1)
        members, exclusions = [], []

        for cap_rank, row in enumerate(self._by_date[day].itertuples(), start=1):
            reason, contract = self.screen(row, day)
            if reason == "outside_pool":
                raise PoolTooShallow(f"{trade_date}: reached rank {cap_rank} ({row.symbol}) "
                                     f"with {len(members)} members")
            if reason:
                exclusions.append((row.symbol, cap_rank, reason))
                continue
            members.append((row.cmc_id, row.symbol, contract, cap_rank, row.market_cap))
            if len(members) == n:
                break

        # A short basket whose weights still sum to 1 looks like a valid answer,
        # which is the failure this whole selector exists to remove. PoolTooShallow
        # should already have fired; this is the backstop.
        if len(members) != n:
            raise PoolTooShallow(
                f"{trade_date}: only {len(members)} of {n} members after scanning "
                f"all {len(self._by_date[day])} ranked coins"
            )

        members = pd.DataFrame(members, columns=["cmc_id", "symbol", "contract", "cap_rank", "market_cap"])
        members["weight"] = members.market_cap / members.market_cap.sum()
        exclusions = pd.DataFrame(exclusions, columns=["symbol", "cap_rank", "reason"])
        return members, exclusions

    def select_buffered(self, trade_dates, exit_rank: int = 13) -> dict:
        """Membership with the registered rank buffer: enter at `n`, leave at `exit_rank`.

        Stateful by construction — an incumbent is kept while it still ranks
        inside `exit_rank`, and vacancies are filled from the top of the eligible
        list. Reconstitution dates are passed in, so the caller decides monthly,
        weekly or daily; the rule itself does not change. Both the strategy and
        the benchmark must use this, because the buffer defines the instrument.
        """
        held: list[str] = []
        out = {}
        for trade_date in trade_dates:
            try:
                eligible, _ = self.select(trade_date, n=exit_rank)
            except PoolTooShallow:
                # Early dates cannot field `exit_rank` names at all; with no
                # ranks to buffer against, the plain selection is the answer.
                eligible, _ = self.select(trade_date)

            order = list(eligible.contract)
            keep = [c for c in held if c in order][:self.n]
            for contract in order:
                if len(keep) == self.n:
                    break
                if contract not in keep:
                    keep.append(contract)
            if len(keep) != self.n:
                raise PoolTooShallow(f"{trade_date}: buffered selection held {len(keep)} of {self.n}")

            members = eligible[eligible.contract.isin(keep)].copy()
            members["weight"] = members.market_cap / members.market_cap.sum()
            out[trade_date] = members.reset_index(drop=True)
            held = keep
        return out

    def select_all(self, start: date) -> pd.DataFrame:
        """Members for every trade date from `start`, as one long table."""
        trade_dates = [day + timedelta(days=1) for day in self._by_date]
        return pd.concat(
            [self.select(d)[0].assign(trade_date=d) for d in trade_dates if d >= start],
            ignore_index=True,
        )
