import pytest
import numpy as np
from copy import deepcopy

from volsurface.core.volatility_surface import VolatilitySurface, SurfaceGrid
from volsurface.models.sabr import SABRModel, SABRParameters

@pytest.fixture
def basic_grid():
    """Create a simple test grid"""
    return SurfaceGrid(
        strikes=np.array([90, 100, 110]),
        maturities=np.array([0.5, 1.0, 1.5]),
        spot=100.0,
        rate=0.02
    )

@pytest.fixture
def sabr_model():
    """Create a SABR model with typical parameters"""
    params = SABRParameters(
        alpha=0.4,  # ATM vol level
        beta=0.5,   # CEV parameter
        rho=-0.2,   # Correlation
        nu=0.4      # Vol of vol
    )
    return SABRModel(params)

def test_surface_generation_shape(basic_grid, sabr_model):
    """Test if surface dimensions match input grid"""
    surface = VolatilitySurface(sabr_model)
    result = surface.generate_surface(basic_grid)
    
    assert result.shape == (3, 3)  # Should match grid dimensions
    assert not np.any(np.isnan(result))  # No NaN values
    assert np.all(result > 0)  # All vols should be positive

def test_surface_grid_validation():
    """Test grid validation catches invalid inputs"""
    with pytest.raises(ValueError):
        SurfaceGrid(
            strikes=np.array([-90, 100, 110]),  # Negative strike
            maturities=np.array([0.5, 1.0, 1.5]),
            spot=100.0,
            rate=0.02
        ).validate()

def test_surface_caching(basic_grid, sabr_model):
    """Test that surface caching works"""
    surface = VolatilitySurface(sabr_model)
    
    # Generate surface first time
    result1 = surface.generate_surface(basic_grid)
    
    # Should use cached version
    assert surface._cached_surface is not None
    assert np.array_equal(surface._cached_surface, result1)

def test_realistic_values(basic_grid, sabr_model):
    """Test if generated volatilities are in realistic range"""
    surface = VolatilitySurface(sabr_model)
    result = surface.generate_surface(basic_grid)
    
    # Check if vols are in reasonable range (e.g., 10% to 100%)
    assert np.all(result > 0.1)
    assert np.all(result < 1.0)