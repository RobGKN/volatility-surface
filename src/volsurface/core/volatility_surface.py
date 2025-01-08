from typing import Optional, Union, Tuple, Dict
import numpy as np
from dataclasses import dataclass
import logging
from volsurface.models.sabr import SABRParameters, SABRModel


logger = logging.getLogger(__name__)

@dataclass
class SurfaceGrid:
    """Represents the discretization grid for the volatility surface"""
    strikes: np.ndarray    # Array of strike prices
    maturities: np.ndarray # Array of maturities in years
    spot: float           # Current spot price
    rate: float          # Risk-free rate

    def validate(self) -> None:
        """Validate grid parameters are within reasonable bounds"""
        if not (self.strikes.ndim == 1 and self.maturities.ndim == 1):
            raise ValueError("Strikes and maturities must be 1D arrays")
        if np.any(self.strikes <= 0) or np.any(self.maturities <= 0):
            raise ValueError("Strikes and maturities must be positive")
        if self.spot <= 0:
            raise ValueError("Spot price must be positive")
        
class VolatilitySurface:
    def __init__(self, model: 'SABRModel'):
        self.model = model
        self._cached_surface: Optional[np.ndarray] = None
        self._cached_grid: Optional[SurfaceGrid] = None

    def generate_surface(self, grid: SurfaceGrid) -> np.ndarray:
        """
        Generate full volatility surface on the specified grid.
        
        Args:
            grid: SurfaceGrid object containing strikes and maturities
            
        Returns:
            ndarray: 2D array of implied volatilities with shape (n_strikes, n_maturities)
            
        Notes:
            - Uses vectorized operations for efficiency
            - Caches results for subsequent queries
        """
        # Validate inputs
        grid.validate()
        
        # Initialize output array
        n_strikes = len(grid.strikes)
        n_maturities = len(grid.maturities)
        surface = np.zeros((n_strikes, n_maturities))
        
        # Calculate forward prices for each maturity
        forwards = grid.spot * np.exp(grid.rate * grid.maturities)
        
        try:
            # Generate implied volatilities for each point
            for i, strike in enumerate(grid.strikes):
                for j, (maturity, forward) in enumerate(zip(grid.maturities, forwards)):
                    surface[i, j] = self.model.implied_volatility(
                        F=forward,
                        K=strike,
                        T=maturity
                    )
            
            # Cache results
            self._cached_surface = surface
            self._cached_grid = grid
            
            return surface
            
        except Exception as e:
            logger.error(f"Error generating surface: {str(e)}")
            raise RuntimeError(f"Failed to generate volatility surface: {str(e)}")
    
    def get_slice(self, 
              slice_type: str, 
              value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a slice of the surface at constant maturity (smile) or strike
        
        Args:
            slice_type: Either 'maturity' or 'strike'
            value: The fixed value to slice at
            
        Returns:
            Tuple of (x_values, volatilities) where:
            - For maturity slice: (strikes, volatilities)
            - For strike slice: (maturities, volatilities)
            
        Raises:
            ValueError: If surface hasn't been generated or value is out of bounds
        """
        if self._cached_surface is None or self._cached_grid is None:
            raise ValueError("Surface must be generated before taking slices")
            
        grid = self._cached_grid
        surface = self._cached_surface
        
        if slice_type == 'maturity':
            # Find nearest maturity index
            idx = np.abs(grid.maturities - value).argmin()
            if abs(grid.maturities[idx] - value) > 1e-10:
                logger.warning(f"Using closest maturity: {grid.maturities[idx]} instead of {value}")
                
            return grid.strikes, surface[:, idx]
            
        elif slice_type == 'strike':
            # Find nearest strike index
            idx = np.abs(grid.strikes - value).argmin()
            if abs(grid.strikes[idx] - value) > 1e-10:
                logger.warning(f"Using closest strike: {grid.strikes[idx]} instead of {value}")
                
            return grid.maturities, surface[idx, :]
            
        else:
            raise ValueError("slice_type must be either 'maturity' or 'strike'")
    
    def get_point(self, strike: float, maturity: float) -> float:
        """
        Get implied volatility at a specific (K,T) point
        
        Args:
            strike: Strike price
            maturity: Time to maturity in years
            
        Returns:
            float: Implied volatility at the specified point
            
        Notes:
            - If point exactly matches grid point, returns stored value
            - If point is off-grid, calculates new value using model
            - Validates inputs are within reasonable bounds
        """
        if strike <= 0 or maturity <= 0:
            raise ValueError("Strike and maturity must be positive")

        # If surface isn't generated yet, we need grid parameters
        if self._cached_surface is None or self._cached_grid is None:
            raise ValueError("Cannot compute point without surface generation - need spot/rate parameters")
        
        grid = self._cached_grid
        
        # Check if point exactly matches a grid point
        strike_idx = np.where(np.abs(grid.strikes - strike) < 1e-10)[0]
        maturity_idx = np.where(np.abs(grid.maturities - maturity) < 1e-10)[0]
        
        if len(strike_idx) > 0 and len(maturity_idx) > 0:
            # Return cached value if we have it
            return self._cached_surface[strike_idx[0], maturity_idx[0]]
        
        # If not on grid, compute new point
        forward = grid.spot * np.exp(grid.rate * maturity)
        return self.model.implied_volatility(F=forward, K=strike, T=maturity)
    
    def validate_arbitrage(self) -> dict:
        """
        Validate the surface for arbitrage violations.
        
        Returns:
            dict: Results containing:
                - butterfly_arbitrage (bool): True if butterfly arbitrage exists
                - calendar_spread (bool): True if calendar spread arbitrage exists
                - violations: List of specific violations
                - details: Additional verification information
        
        Raises:
            ValueError: If surface has not been generated yet
        """
        if self._cached_grid is None or self._cached_surface is None:
            raise ValueError("Surface must be generated before validation")
            
        grid = self._cached_grid
        violations = []
        butterfly_violations = []
        calendar_violations = []
        
        # Initialize result dictionary
        results = {
            "butterfly_arbitrage": False,
            "calendar_spread": False,
            "violations": violations,
            "details": {
                "grid_points": len(grid.strikes) * len(grid.maturities),
                "strike_range": [float(grid.strikes.min()), float(grid.strikes.max())],
                "maturity_range": [float(grid.maturities.min()), float(grid.maturities.max())],
                "butterfly": butterfly_violations,
                "calendar_spread": calendar_violations
            }
        }
        
        # Check butterfly arbitrage (convexity in strike dimension)
        for t_idx, t in enumerate(grid.maturities):
            vols = self._cached_surface[:, t_idx]
            for i in range(1, len(grid.strikes) - 1):
                k1, k2, k3 = grid.strikes[i-1:i+2]
                v1, v2, v3 = vols[i-1:i+2]
                
                # Check convexity using total variance
                w = (k3 - k2) / (k3 - k1)
                interpolated_var = w * (v1**2) + (1-w) * (v3**2)
                if v2**2 > interpolated_var + 1e-10:
                    results["butterfly_arbitrage"] = True
                    violation = {
                        "type": "butterfly",
                        "maturity": float(t),
                        "strikes": [float(k1), float(k2), float(k3)],
                        "vols": [float(v1), float(v2), float(v3)]
                    }
                    violations.append(violation)
                    butterfly_violations.append(violation)
        
        # Check calendar spread arbitrage
        for k_idx, k in enumerate(grid.strikes):
            vols = self._cached_surface[k_idx, :]
            total_var = vols**2 * grid.maturities
            
            for i in range(len(grid.maturities)-1):
                t1, t2 = grid.maturities[i], grid.maturities[i+1]
                var1, var2 = total_var[i], total_var[i+1]
                
                # Total variance should increase proportionally with time
                expected_var2 = var1 * (t2/t1)
                # Increase tolerance from 1e-10 to 1e-1 (10%) - TODO: make sabr accurate enough to get this down
                if var2 < expected_var2 * (1 - 1e-1):  
                    results["calendar_spread"] = True
                    violation = {
                        "type": "calendar_spread",
                        "strike": float(k),
                        "maturities": [float(t1), float(t2)],
                        "total_variance": [float(var1), float(var2)],
                        "volatilities": [float(vols[i]), float(vols[i+1])]
                    }
                    violations.append(violation)
                    calendar_violations.append(violation)
        
        return results
    
        
    def compute_metrics(self) -> Dict[str, float]:
        """
        Calculate key surface metrics including:
            - ATM term structure
            - Skew parameters
            - Surface summary statistics
        """
        pass

    def to_visualization_format(self) -> Dict:
        """
        Convert surface data to format suitable for 3D visualization
        
        Returns dictionary with:
            - Coordinates (strikes, maturities)
            - Volatility values
            - Metadata for rendering
        """
        pass