from typing import Optional, Union, Tuple, Dict
import numpy as np
from dataclasses import dataclass
import logging

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
            Tuple of (x_values, volatilities) where x_values are strikes or maturities
        """
        pass
    
    def get_point(self, strike: float, maturity: float) -> float:
        """
        Get implied volatility at a specific (K,T) point
        
        Uses interpolation if point is not on the grid
        """
        pass
    
    def validate_arbitrage(self) -> Dict[str, bool]:
        """
        Check if surface satisfies no-arbitrage conditions
        
        Returns dict of test results including:
            - Butterfly arbitrage
            - Calendar spread arbitrage
        """
        pass
    
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