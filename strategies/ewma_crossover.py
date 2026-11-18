"""
EWMA Crossover Strategy
Ratio of 4
Fast 16 days and slow 64 days
Fast 32, Slow 128
Fast 8, Slow 32
"""
import pandas as pd
from strategies.base import BaseStrategy
from data.volatility import VolatilityEstimator

import logging

logger = logging.getLogger(__name__)

class EWMA_Trend_Following(BaseStrategy):
    def __init__(self, vol_span, annualization_factor, fast=16, slow=64,
                 forecast_scalar=None, floor_quantile=None, floor_window=500, floor_min_periods=100):
        """forecast_scalar and floor_quantile are opt-in so legacy callers are unchanged. A
        registered run passes both: the scalar fixes the forecast (no rolling median) and the
        floor applies to the volatility the forecast is normalised by."""
        super().__init__(
            name = "ewma_trend_following",
            params = {"fast": fast, "slow": slow},
            required_columns = ("close",),
            forecast_scalar = forecast_scalar,
        )
        self.fast = fast
        self.slow = slow
        
        # Use centralized volatilty estimator for consistency across the system
        self.volatility_estimator = VolatilityEstimator(
            span=vol_span,
            annualization_factor=annualization_factor,
            floor_quantile=floor_quantile,
            floor_window=floor_window,
            floor_min_periods=floor_min_periods,
        )

    def generate_normalized_forecast(self, df: pd.DataFrame) -> pd.Series:
        """
        1. Compute EWMAs crossover
        2. Normalize trend signal by volatility
        3. Return normalized forecast
        """
        close = df["close"] # get column of close prices
        # applying .mean() is how we tell pandas to nroamlize the weights so they sum to 1
        # adjust=True, as registered: adjust=False seeds the average with the
        # first price, which reads as a trend that is really invented history.
        fast_ewma = close.ewm(span=self.fast, adjust=True).mean()

        slow_ewma = close.ewm(span=self.slow, adjust=True).mean()

        raw_forecast = fast_ewma - slow_ewma          # cross over strength

        normalized_forecast = self._normalize_trend_signal_by_volatility(raw_forecast, df.copy())

        if logger.isEnabledFor(logging.DEBUG):
            tail_len = 5
            logger.debug(
                "EWMA tails | fast=%s | slow=%s | raw=%s | normalized=%s",
                fast_ewma.tail(tail_len).tolist(),
                slow_ewma.tail(tail_len).tolist(),
                raw_forecast.tail(tail_len).tolist(),
                normalized_forecast.tail(tail_len).tolist(),
            )

        return normalized_forecast

    def _normalize_trend_signal_by_volatility(self, raw_forecast: pd.Series, df: pd.DataFrame) -> float:
        """
        1. Compute daily vol using centralized estimator
        2. Convert volatility to price units
        3. Normalize trend signal by volatility
        4. Return normalized forecast
        """
        # Returns Vol (%)
        daily_vol = self.volatility_estimator.daily(df["close"])

        # Convert to price units ($/day volatility)
        price_vol = df["close"] * daily_vol

        # Dimensionless: ($/day difference) / ($/day volatility) = dimensionless ratio
        normalized_forecast = raw_forecast / price_vol 
        
        return normalized_forecast

