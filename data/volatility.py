"""
Centralized Volatility Calculation that produces an annualized series, which all components consume

"""
import pandas as pd

class VolatilityEstimator:
    def __init__(
        self,
        span: int = 20,
        annualization_factor: int = 365,
        floor_quantile: float | None = None,
        floor_window: int = 500,
        floor_min_periods: int = 100,
    ):
        """
        span: EWMA span for volatility calculation (default 20 ~= 1 month)
        annualization_factor = Factor to annualize daily volatility (252 for stocks, 365 for crypto)

        floor_quantile: Carver's volatility floor (pysystemtrade `robust_vol_calc`).
            When set, the estimate cannot fall below that quantile of its own past
            `floor_window` days. Positions scale with 1/vol, so an estimate drifting
            towards zero in a quiet stretch would size an unbounded bet just before
            the stretch ends. The registered protocol uses 0.05 over 500 days.
            Left None, no floor is applied, which is the pre-2026-09 behaviour.
        """
        self.span = span
        self.annualization_factor = annualization_factor
        self.floor_quantile = floor_quantile
        self.floor_window = floor_window
        self.floor_min_periods = floor_min_periods

    def daily(self, price: pd.Series) -> pd.Series:
        """
        Calculate daily volatility using EWMA. 
        
        Used for forecast normalization (converting raw signals to dimensionless ratios)
        """

        returns = price.pct_change()

        # adjust=True: a true weighted average of the observations that exist,
        # with no phantom pre-sample history standing in for the missing past.
        vol = returns.ewm(span=self.span, adjust=True).std()

        if self.floor_quantile is None:
            return vol

        floor = vol.rolling(
            self.floor_window, min_periods=self.floor_min_periods
        ).quantile(self.floor_quantile)
        return vol.where(vol >= floor, floor).fillna(vol)

    def annualized(self, price: pd.Series) -> pd.Series:
        """
        Calculate annualized volatility = annualization factor x sqrt(variance)

        - Produces the "V" (instrument volatility) used in Carvers position sizing formula:
        (1/N) x C x (T/V) x (F/10)

        """
 
        return self.daily(price) * (self.annualization_factor ** 0.5)
    


