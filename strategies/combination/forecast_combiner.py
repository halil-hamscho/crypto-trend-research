import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union
from strategies.base import BaseStrategy

class ForecastCombiner:
    """
    Combines multiple forecasts usign Carver's methodology.

    Can be used for both:
    - Level 2: Combining variations into strategy forcast
    - Level 3: Combining strategies into instrument forecast

    Key Responsibilities:
    1. Estimate weights from RETURN series correlations
    2. Calculate FDM from FORECAST series correlations
    3. Combine weighted forecasts and apply FDM
    3. Cap at [-20, +20]
    """

    def __init__(self, correlation_lookback: int = 365, fdm: float | None = None):
        """
        fdm: a fixed forecast diversification multiplier. Registered runs must
            pass one (1.15 for the four registered speeds); the rolling estimate
            below is fitted to the sample under test and is barred by the
            pre-registration.
        """
        self.correlation_lookback = correlation_lookback
        self.fdm = fdm

    def combine(
        self,
        forecasts: Dict[str, pd.Series],
        return_series: Dict[str, pd.Series],
        weight_method: str = "equal",
        return_details: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, Dict[str, float], float]]:
        """
        Combine multiple forecasts into a single forecast.

        Args:
            forecasts: Dict of forecast name -> forecast series
            return_series: Dict of forecast name -> return series
            weight_method: Method to estimate weights ("equal", "correlation", "performance")
            return_details: If True, also return weights and average FDM

        Returns:
            If return_details=False: Combined forecast series
            If return_details=True: Tuple of (combined_forecast, weights_dict, avg_fdm)
        """
        forecast_df = pd.DataFrame(forecasts)
        return_df = pd.DataFrame(return_series)

        # Step 1: Estimate weights from RETURN series
        weights = self._estimate_weights_from_returns(return_df, weight_method)

        # Step 2: FDM — fixed when registered, rolling only for exploratory work
        if self.fdm is not None:
            fdm = pd.Series(self.fdm, index=forecast_df.index)
        else:
            fdm = self._calculate_rolling_fdm(forecast_df, weights)

        # Step 3: Combine: sum(w_i * F_i) * FDM
        weight_series = pd.Series(weights)
        # (time, n_forecasts) * weights --> row wise sum (horizontally across)
        combined = (forecast_df * weight_series).sum(axis=1) * fdm

        # Step 4: Cap at [-20, + 20]
        capped = combined.clip(lower=-20, upper=20)

        if return_details:
            # Return average FDM for reporting (exclude warmup period)
            valid_fdm = fdm[fdm != 1.0]
            avg_fdm = valid_fdm.mean() if len(valid_fdm) > 0 else 1.0
            return capped, weights, avg_fdm

        return capped

    def _estimate_weights_from_returns(
        self, 
        return_df: pd.DataFrame, 
        method: str
    ) -> Dict[str, float]:
        """
            Estimate weights using forecast RETURN series.

            - Returns capture actual profitability
            - Returns account for timing and volatility regimes
            - Two forecasts may have correlated signals but different return profiles
        """
        n = len(return_df.columns)

        if method == "equal":
            return {col: 1.0 / n for col in return_df.columns}

        elif method == "correlation":
            # lower correlation with others -> higher weight
            corr_matrix = return_df.corr()

            # Calculate average correlation for each forecast
            avg_corr = {}
            for col in corr_matrix.columns:
                other_corrs = corr_matrix[col].drop(col)
                avg_corr[col] = other_corrs.mean()

            # Inverse correlation weighting
            inv_corr = {k: 1.0 / (1.0 + v) for k, v in avg_corr.items()}
            total = sum(inv_corr.values())
            weights = {k: v / total for k, v in inv_corr.items()}
            return weights

        elif method == "performance":
            # Higher sharpe ratio -> higher weight
            # TODO: Implement with bootstrapping to avoid overfitting
            raise NotImplementedError(
                "Performance-based weighting requires bootstrap implementation"
            )

        else:
            raise ValueError(f"Unknown weight method: {method}")

    def _calculate_rolling_fdm(
        self,                                                       
        forecast_df: pd.DataFrame,                                  
        weights: Dict[str, float]                                   
    ) -> pd.Series:                                                 
        """                                                         
            Calculate rolling Forecast Diversification Multiplier (FDM).
                                                                        
            Why rolling instead of static?                              
            ─────────────────────────────                               
            Forecast correlations are NOT stationary:                   
            - Strong trends: trend-following strategies correlate highly -> FDM drops                                                   
            - Choppy markets: strategies diverge -> FDM rises           
            Rolling captures these regime changes dynamically.          
                                                                        
            CRITICAL: Uses correlation of FORECAST SERIES, not return series!                                                         
            ────────────────────────────────────────────────────────────                                                        
            Formula: FDM = 1 / sqrt(W' × H × W)                         
                                                                        
            Where:                                                      
                W = Weight vector (must sum to 1.0)                     
                H = Correlation matrix of scaled FORECASTS              
                                                                        
            Why forecast correlations (not return correlations)?        
            - FDM measures diversification of the SIGNALS themselves    
            - Low forecast correlation = forecasts provide different information                                                     
            - Independent of whether their returns correlate            
                                                                        
            FDM capped at 2.5 per Carver to prevent excessive scaling.  
                                                                        
            Edge cases handled:                                         
            ───────────────────                                         
            - Single forecast -> FDM = 1.0 (no diversification possible)
            - Warmup period (< lookback rows) -> FDM = 1.0              
            - Zero variance in window (constant forecast) -> FDM = 1.0  
            - NaN in correlation matrix -> FDM = 1.0                    
            - Weights not summing to 1.0 -> normalized automatically    
                                                                        
            Weight vector alignment:                                    
            ────────────────────────                                    
            We iterate over forecast_df.columns (NOT weights.keys()) to guarantee                                                       
            w is in the same column order as correlation matrix H.      
                                                                        
            Why numpy over pandas for the loop:                         
            ───────────────────────────────────                         
            - Pandas slicing: creates new DataFrame, copies index, validates alignment                                             
            - Numpy slicing: returns a VIEW (no copy), just pointer arithmetic                                                      
                                                                        
            Window indexing math:                                       
            ─────────────────────                                       
            i = L-1: values[0:L]     -> first valid (L rows available)  
            i = L:   values[1:L+1]   -> rolling forward                 
            Rows 0 through L-2 keep default FDM = 1.0                   
                                                                        
            Variance reduction interpretation:                          
            ──────────────────────────────────                          
            vr = W' × H × W                                             
            - vr = 1.0:  no diversification (identical forecasts)       
            - vr = 0.5:  FDM = 1.41 (moderate diversification)          
            - vr = 0.25: FDM = 2.0 (strong diversification)             
        """                                                         
        L = self.correlation_lookback                               
        n_cols = len(forecast_df.columns)                           
        n_rows = len(forecast_df)                                   
                                                                    
        if n_cols == 1:                                             
            return pd.Series(np.ones(n_rows), index=forecast_df.index)                                        
                                                                    
        w = np.array([weights[col] for col in forecast_df.columns]) 
                                                                    
        w_sum = w.sum()                                             
        if not np.isclose(w_sum, 1.0):                              
            w = w / w_sum                                           
                                                                    
        values = forecast_df.values # convert to 2d numpy array                          
        out = np.ones(n_rows)                                       
                                                                    
        for i in range(L - 1, n_rows):                              
            window = values[i + 1 - L : i + 1]                      
                                                                    
            stds = np.std(window, axis=0)                           
            if np.any(stds == 0): # if any forecast constant over a window                
                out[i] = 1.0                                        
                continue                                            
                                                                    
            H = np.corrcoef(window, rowvar=False)                   
                                                                    
            if np.any(np.isnan(H)):                                 
                out[i] = 1.0                                        
                continue                                            
                                                                    
            vr = w @ H @ w                                          
                                                                    
            if vr <= 0 or np.isnan(vr):                             
                fdm = 1.0                                           
            else:                                                   
                fdm = 1.0 / np.sqrt(vr)                             
                                                                    
            out[i] = min(fdm, 2.5)                                  
                                                                    
        return pd.Series(out, index=forecast_df.index)              
                                                                