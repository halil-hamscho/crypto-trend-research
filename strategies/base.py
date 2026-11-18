# Base Class for All Strategies

# Input: DataFrame
# Output: Scaled Forecast(-20 to + 20)
import pandas as pd
import numpy as np
from typing import Iterable, Dict
import logging

logger = logging.getLogger(__name__)

# Normalized forecasts are clipped to this range BEFORE computing the rolling
# median used for scaling.  Post-crash coins (e.g. LUNA → $0.00005) produce
# normalized values in the millions because price_vol → 0.  Letting those
# contaminate the 250-period median window permanently miscalibrates the scalar.
# The original un-clipped forecast is still passed through to the final ±20 cap.
MEDIAN_CLIP_THRESHOLD = 500.0
class BaseStrategy:
    def __init__(
        self, 
        name: str = "BaseStrategy",
        params: Dict | None = None,
        required_columns: Iterable[str] = ("close",),
        forecast_scalar: float | None = None,
    ):
        """
        forecast_scalar: a fixed scalar (Carver's published value for the rule).
            Registered runs must pass one. The rolling-median alternative below
            is estimated on the sample under test, which re-normalises the
            forecast to the last year of its own behaviour and is barred by the
            pre-registration.
        """
        self.name = name
        self.params = params or {}
        self.required_columns = required_columns
        self.forecast_scalar = forecast_scalar
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point for strategy execution.
        
        This method orchestrates the full forecast generation pipeline:
        1. Generate normalized (dimensionless) forecast (implemented by subclasses)
        2. Scale forecast to target average absolute value (~10) (Raw Forecast x Forecast Scalar)
        3. Capped forecast (-20 to +20)
        
        """
        self._validate(df)
        
        # Step 1: Generate normalized (dimensionless) forecast (subclass implements this)
        normalized_forecast = self.generate_normalized_forecast(df.copy())
        
        # Step 2: Scale forecast to target average absolute value
        if self.forecast_scalar is not None:
            scaled_forecast = normalized_forecast * self.forecast_scalar
        else:
            scaled_forecast = self._scale_forecast(normalized_forecast)

        # Step 3: Cap forecast at +/- 20
        # This ensures that forecasts are comparable across strategies
        capped_forecast = scaled_forecast.clip(lower=-20.0, upper=20.0)

        n_capped = (scaled_forecast.abs() > 20).sum()
        if n_capped > 0:
            pct_capped = (n_capped / len(scaled_forecast)) * 100
            if pct_capped > 10:  # Only warn if >10% capped (suggests scaling issue)
                logger.warning(
                    f"Excessive capping: {pct_capped:.1f}% of forecasts hit ±20 limit "
                    f"(range before cap: [{scaled_forecast.min():.1f}, {scaled_forecast.max():.1f}])"
                )
        
        return pd.DataFrame({"signal": capped_forecast})
    
    def generate_normalized_forecast(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate normalized (dimensionless) forecast values. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_normalized_forecast method")
    
    def _scale_forecast(self, raw_forecast: pd.Series, target_avg_abs: float = 10.0, 
                       lookback_periods: int = 250) -> pd.Series:
        """
        Scale normalized (dimensionless) forecast to target average absolute value (Carver's method).
        
        Uses rolling median of absolute values to calculate scalar.
        
        Args:
            raw_forecast: Raw forecast values
            target_avg_abs: Target average absolute value (default 10, Carver's standard)
            lookback_periods: Rolling window size (default 250 = ~1 year of daily data)
            @TODO: might have to change the window size to 365 since crypto 24/7

        """
        # Clip extreme values ONLY for computing the rolling median.
        # See MEDIAN_CLIP_THRESHOLD at the top of this file for the rationale.
        n_blown = (raw_forecast.abs() > MEDIAN_CLIP_THRESHOLD).sum()
        if n_blown > 0:
            logger.debug(
                "%s | _scale_forecast: %d values exceed ±%.0f median-clip threshold "
                "(min=%.1f, max=%.1f) — clipped for median only, ±20 cap handles the rest",
                self.name, n_blown, MEDIAN_CLIP_THRESHOLD,
                raw_forecast.min(), raw_forecast.max(),
            )
        raw_for_median = raw_forecast.clip(-MEDIAN_CLIP_THRESHOLD, MEDIAN_CLIP_THRESHOLD)
        # min_periods=20: establish a real scalar after ~20 bars instead of leaving
        # the first `lookback_periods` (250) bars NaN -> defaulting the scalar to 10
        # (i.e. raw*10), which massively over-caps the warmup (~53% of early bars hit
        # ±20). The window still grows to the full lookback. See
        # diagnostic/diagnostic_forecast_capping.py.
        rolling_median_abs = raw_for_median.abs().rolling(
            window=lookback_periods, min_periods=min(20, lookback_periods)
        ).median()

        # Calculate scalar: target / current median
        scalar = target_avg_abs / rolling_median_abs.replace(0, np.nan).fillna(1.0)

        # Apply scalar to the original (un-clipped) forecast
        scaled = raw_forecast * scalar
        
        if logger.isEnabledFor(logging.DEBUG):
            tail_len = 5
            logger.debug(
                "Scaling | scalar_tail=%s | scaled_tail=%s",
                scalar.tail(tail_len).tolist(),
                scaled.tail(tail_len).tolist()
            )

        return scaled
    
    def _validate(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required structure."""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be sorted in ascending order")
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise TypeError("DataFrame index must be datetime")

    # TODO: Add a method to plot the scaled forecast 