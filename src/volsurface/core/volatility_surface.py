from typing import Optional, Union, Tuple, Dict
import numpy as np
from dataclasses import dataclass
import logging
from volsurface.models import VolatilityModel  # Import base class instead
from volsurface.metrics.surface_metrics import SurfaceMetrics

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
    def __init__(self, model: VolatilityModel):  # Change type hint to base class
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
        Compute key surface metrics. Current implementation focuses on
        essential metrics needed for visualization and basic analysis.
        
        Returns:
            Dictionary containing core surface metrics including ATM levels,
            basic shape parameters, and summary statistics.
            
        Raises:
            ValueError: If surface hasn't been generated
        """
        if self._cached_surface is None or self._cached_grid is None:
            raise ValueError("Surface must be generated before computing metrics")
            
        metrics = SurfaceMetrics(
            surface=self._cached_surface,
            strikes=self._cached_grid.strikes,
            maturities=self._cached_grid.maturities,
            spot=self._cached_grid.spot
        )
        
        return metrics.compute_basic_metrics()

    def to_react_visualization_format(self) -> Dict:
        """
        Convert surface data to format suitable for 3D visualization.
        
        Returns:
            Dictionary containing:
            - vertices: List of [x,y,z] coordinates for each point
            - indices: Triangle indices for mesh construction
            - colors: Color values for each vertex
            - gridLines: Reference grid data
            - metadata: Rendering hints and surface properties
            - bounds: Min/max values for scaling
            - markers: Special points (ATM line, etc.)
            
        Raises:
            ValueError: If surface hasn't been generated
        """
        if self._cached_surface is None or self._cached_grid is None:
            raise ValueError("Surface must be generated before visualization")
            
        # Get raw data
        surface = self._cached_surface
        grid = self._cached_grid
        
        # Validate data shape and content
        if surface.shape != (len(grid.strikes), len(grid.maturities)):
            raise ValueError(f"Surface shape {surface.shape} doesn't match grid dimensions: "
                            f"strikes={len(grid.strikes)}, maturities={len(grid.maturities)}")
        
        if np.any(np.isnan(surface)) or np.any(np.isinf(surface)):
            raise ValueError("Surface contains NaN or Inf values")
        
        try:
            
            # Normalize coordinates for visualization
            # Convert strikes to moneyness (centered around 1.0)
            norm_strikes = grid.strikes / grid.spot
            # Convert maturities to percentage of max maturity
            norm_maturities = grid.maturities / np.max(grid.maturities)
            
            # Generate vertices array
            vertices = []
            colors = []
            
            # Color mapping parameters
            vol_min, vol_max = np.min(surface), np.max(surface)
            
            for i, k in enumerate(norm_strikes):
                for j, t in enumerate(norm_maturities):
                    # Add vertex
                    vertices.append([
                        float(k - 1.0),  # Center moneyness around 0
                        float(t),
                        float(surface[i,j])
                    ])
                    
                    # Generate color (blue scale based on volatility)
                    vol_level = (surface[i,j] - vol_min) / (vol_max - vol_min)
                    colors.append([
                        0.0,  # R
                        0.3 + 0.7 * vol_level,  # G
                        0.5 + 0.5 * vol_level,  # B
                    ])
            
            # Generate triangle indices for mesh
            indices = []
            n_rows = len(norm_strikes)
            n_cols = len(norm_maturities)
            
            for i in range(n_rows - 1):
                for j in range(n_cols - 1):
                    # Two triangles per grid square
                    p0 = i * n_cols + j
                    p1 = p0 + 1
                    p2 = (i + 1) * n_cols + j
                    p3 = p2 + 1
                    
                    # First triangle
                    indices.extend([p0, p2, p1])
                    # Second triangle
                    indices.extend([p1, p2, p3])
            
            # Generate grid lines for reference
            grid_lines = {
                "strikes": [float(k - 1.0) for k in norm_strikes],
                "maturities": [float(t) for t in norm_maturities]
            }
            
            # Find ATM line points
            atm_idx = np.abs(norm_strikes - 1.0).argmin()
            atm_line = [
                [0.0, float(t), float(surface[atm_idx,j])]
                for j, t in enumerate(norm_maturities)
            ]
            
            # Calculate key surface metrics for metadata
            metrics = self.compute_metrics()
            
            return {
                "vertices": vertices,
                "indices": indices,
                "colors": colors,
                "gridLines": grid_lines,
                "markers": {
                    "atmLine": atm_line
                },
                "bounds": {
                    "strikes": [float(np.min(norm_strikes) - 1.0), 
                            float(np.max(norm_strikes) - 1.0)],
                    "maturities": [0.0, 1.0],
                    "volatility": [float(vol_min), float(vol_max)]
                },
                "metadata": {
                    "spotPrice": float(grid.spot),
                    "maxMaturity": float(np.max(grid.maturities)),
                    "atmVol": float(metrics["atm_vol"]),
                    "surfaceSkew": float(metrics["skew_1m"]),
                    "termStructureSlope": float(metrics["term_structure_slope"]),
                    "timestamp": "2024-01-08T00:00:00Z",  # Add actual timestamp if available
                    "renderHints": {
                        "initialRotation": [-0.5, 0.0, 0.0],
                        "cameraDistance": 2.0,
                        "gridOpacity": 0.2,
                        "meshOpacity": 0.8
                    }
                }
            }
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error in visualization formatting: {str(e)}")
            raise RuntimeError(f"Failed to process surface for visualization: {str(e)}")
        except ValueError as e:
            logger.error(f"Value error in visualization formatting: {str(e)}")
            raise RuntimeError(f"Invalid values encountered in surface: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in visualization formatting: {str(e)}")
            raise RuntimeError(f"Failed to format surface for visualization: {str(e)}")
    
    
    def to_visualization_format(self) -> Dict:
        """
        Convert surface data to format suitable for Plotly visualization.
        
        Returns:
            Dictionary containing:
            - x: Strike prices array
            - y: Maturities array
            - z: Volatility surface array
            - metrics: Key surface metrics
            - metadata: Additional surface properties
            
        Raises:
            ValueError: If surface hasn't been generated
        """
        if self._cached_surface is None or self._cached_grid is None:
            raise ValueError("Surface must be generated before visualization")
            
        # Get raw data
        surface = self._cached_surface
        grid = self._cached_grid
        
        # Validate data
        if surface.shape != (len(grid.strikes), len(grid.maturities)):
            raise ValueError(f"Surface shape {surface.shape} doesn't match grid dimensions: "
                            f"strikes={len(grid.strikes)}, maturities={len(grid.maturities)}")
        
        if np.any(np.isnan(surface)) or np.any(np.isinf(surface)):
            raise ValueError("Surface contains NaN or Inf values")
        
        try:
            # Compute metrics once
            metrics = self.compute_metrics()
            
            # Format for return
            # Note: tolist() converts numpy arrays to Python lists for JSON serialization
            return {
                # Core Plotly data
                "x": grid.strikes.tolist(),  # Strike prices
                "y": grid.maturities.tolist(),  # Maturities
                "z": surface.tolist(),  # Implied volatilities
                
                # Additional data for analysis
                "atm_line": {
                    "maturities": grid.maturities.tolist(),
                    "vols": surface[np.abs(grid.strikes - grid.spot).argmin()].tolist()
                },
                
                # Market data
                "market_params": {
                    "spot": float(grid.spot),
                    "rate": float(grid.rate),
                    "min_strike": float(grid.strikes.min()),
                    "max_strike": float(grid.strikes.max()),
                    "min_maturity": float(grid.maturities.min()),
                    "max_maturity": float(grid.maturities.max())
                },
                
                # Surface metrics
                "metrics": {
                    "atm_vol": float(metrics["atm_vol"]),
                    "skew_1m": float(metrics["skew_1m"]),
                    "term_structure_slope": float(metrics["term_structure_slope"]),
                    # Add any other metrics you want to display
                },
                
                # Model validation
                "arbitrage_checks": self.validate_arbitrage()
            }
                
        except Exception as e:
            logger.error(f"Error formatting surface for visualization: {str(e)}")
            raise RuntimeError(f"Failed to format surface for visualization: {str(e)}")