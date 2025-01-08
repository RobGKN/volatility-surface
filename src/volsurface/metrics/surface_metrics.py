from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SurfaceMetrics:
    """
    Handles basic surface metric calculations for volatility surfaces
    """
    def __init__(self, surface: np.ndarray, strikes: np.ndarray, 
                 maturities: np.ndarray, spot: float):
        self.surface = surface
        self.strikes = strikes
        self.maturities = maturities
        self.spot = spot
        
    def compute_basic_metrics(self) -> Dict[str, float]:
        """
        Compute essential surface metrics needed for visualization and analysis
        """
        try:
            # Find ATM index (closest strike to spot)
            atm_idx = np.abs(self.strikes - self.spot).argmin()
            atm_vols = self.surface[atm_idx, :]
            
            metrics = {
                # ATM metrics
                "atm_vol": float(atm_vols[0]),  # Current ATM vol
                "average_atm_vol": float(np.mean(atm_vols)),
                
                # Overall surface metrics
                "min_vol": float(np.min(self.surface)),
                "max_vol": float(np.max(self.surface)),
                "average_vol": float(np.mean(self.surface)),
                
                # Basic shape metrics
                "skew_1m": float(self._compute_skew(0)),  # 1-month skew if available
                "term_structure_slope": float(self._compute_term_slope())
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing basic metrics: {str(e)}")
            raise RuntimeError(f"Failed to compute basic metrics: {str(e)}")
    
    def _compute_skew(self, maturity_idx: int = 0) -> float:
        """
        Compute simple skew measure for a given maturity slice
        Returns difference between 25D put and call vols
        """
        # Simplified approach using strike differentials
        mid_idx = len(self.strikes) // 2
        lower_idx = mid_idx // 2
        upper_idx = mid_idx + mid_idx // 2
        
        return (self.surface[lower_idx, maturity_idx] - 
                self.surface[upper_idx, maturity_idx])
    
    def _compute_term_slope(self) -> float:
        """
        Compute basic term structure slope using ATM vols
        """
        atm_idx = np.abs(self.strikes - self.spot).argmin()
        atm_vols = self.surface[atm_idx, :]
        
        if len(self.maturities) > 1:
            return np.polyfit(self.maturities, atm_vols, 1)[0]
        return 0.0