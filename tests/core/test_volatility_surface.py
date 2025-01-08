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
    model = SABRModel(params)
    print(f"Type of model in fixture: {type(model)}")
    return model

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
    
    # Check if vols are in reasonable range (e.g., 1% to 100%)
    assert np.all(result > 0.01)
    assert np.all(result < 1.0)

def test_surface_slices(basic_grid, sabr_model):
   """Test both maturity and strike slices of the surface"""
   surface = VolatilitySurface(sabr_model)
   surface.generate_surface(basic_grid)

   # Test maturity slice (smile)
   x_vals, vols = surface.get_slice('maturity', 1.0)
   assert len(x_vals) == len(basic_grid.strikes)
   assert len(vols) == len(basic_grid.strikes)
   assert np.array_equal(x_vals, basic_grid.strikes)

   # Test strike slice (term structure)
   x_vals, vols = surface.get_slice('strike', 100.0)
   assert len(x_vals) == len(basic_grid.maturities)
   assert len(vols) == len(basic_grid.maturities)
   assert np.array_equal(x_vals, basic_grid.maturities)

   # Test invalid slice type
   with pytest.raises(ValueError):
       surface.get_slice('invalid', 1.0)

   # Test slice before generation
   new_surface = VolatilitySurface(sabr_model)
   with pytest.raises(ValueError):
       new_surface.get_slice('maturity', 1.0)

def test_get_point(basic_grid, sabr_model):
    """Test getting specific points from the surface"""
    surface = VolatilitySurface(sabr_model)
    surface.generate_surface(basic_grid)

    # Test exact grid point
    vol = surface.get_point(strike=100.0, maturity=1.0)
    assert isinstance(vol, float)
    assert vol > 0

    # Test off-grid point
    vol_off_grid = surface.get_point(strike=95.0, maturity=0.75)
    assert isinstance(vol_off_grid, float)
    assert vol_off_grid > 0

    # Test error handling for ungenerated surface
    new_surface = VolatilitySurface(sabr_model)
    with pytest.raises(ValueError, match="Cannot compute point without surface generation"):
        new_surface.get_point(strike=100.0, maturity=1.0)

    # Test invalid inputs
    with pytest.raises(ValueError, match="Strike and maturity must be positive"):
        surface.get_point(strike=-100.0, maturity=1.0)
    with pytest.raises(ValueError, match="Strike and maturity must be positive"):
        surface.get_point(strike=100.0, maturity=-1.0)
   
def test_validate_arbitrage(basic_grid, sabr_model):
    """Test arbitrage validation on a typical surface"""
    surface = VolatilitySurface(sabr_model)
    surface.generate_surface(basic_grid)
    
    # Debug surface values
    print("\nDebug: Surface Shape:", surface._cached_surface.shape)
    print("Debug: Surface Values:\n", surface._cached_surface)
    
    results = surface.validate_arbitrage()

    # Debug violations if any exist
    if results["calendar_spread"]:
        print("\nDebug: Calendar Spread Violations:", results["details"]["calendar_spread"])
    if results["butterfly_arbitrage"]:
        print("\nDebug: Butterfly Violations:", results["details"]["butterfly"])

    # Basic structure checks
    assert isinstance(results, dict)
    assert "butterfly_arbitrage" in results
    assert "calendar_spread" in results
    assert "details" in results
    assert isinstance(results["details"], dict)
    
    # SABR should not have arbitrage violations
    assert not results["butterfly_arbitrage"]
    assert not results["calendar_spread"]
    assert len(results["violations"]) == 0

def test_validate_arbitrage_with_violations(basic_grid, sabr_model):
    """Test arbitrage validation with manually introduced violations"""
    # Create a smaller grid for testing violations
    grid = SurfaceGrid(
        strikes=np.array([90, 100, 110]),
        maturities=np.array([0.5, 1.0]),
        spot=100.0,
        rate=0.02
    )
    
    surface = VolatilitySurface(sabr_model)
    surface.generate_surface(grid)
    
    # Store original surface for debugging
    original_surface = surface._cached_surface.copy()
    print("\nDebug: Original Surface:\n", original_surface)
    
    # Introduce violations
    surface._cached_surface[1, 0] *= 1.5  # Butterfly arbitrage: middle vol too high
    surface._cached_surface[:, 1] = surface._cached_surface[:, 0] * 0.8  # Calendar spread: later vol too low
    
    # Debug modified surface
    print("\nDebug: Modified Surface:\n", surface._cached_surface)
    
    results = surface.validate_arbitrage()
    
    # Debug results
    print("\nDebug: Validation Results:", results)
    
    # Verify violations were detected
    assert results["butterfly_arbitrage"], "Should detect butterfly arbitrage"
    assert results["calendar_spread"], "Should detect calendar spread arbitrage"
    assert len(results["violations"]) > 0, "Should have recorded violations"
    assert any(v["type"] == "butterfly" for v in results["violations"]), "Should include butterfly violation"
    assert any(v["type"] == "calendar_spread" for v in results["violations"]), "Should include calendar spread violation"
    
def test_validate_arbitrage_error_handling():
    """Test error handling for ungenerated surface"""
    surface = VolatilitySurface(sabr_model)
    
    with pytest.raises(ValueError):
        surface.validate_arbitrage()