import pytest
import numpy as np
from copy import deepcopy

from volsurface.core.volatility_surface import VolatilitySurface, SurfaceGrid
from volsurface.models.sabr import SABRModel, SABRParameters

# Reuse existing fixtures for consistency
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
        alpha=0.4,
        beta=0.5,
        rho=-0.2,
        nu=0.4
    )
    return SABRModel(params)

@pytest.fixture
def basic_surface(basic_grid, sabr_model):
    """Create a basic surface with generated data"""
    surface = VolatilitySurface(sabr_model)
    surface.generate_surface(basic_grid)
    return surface

def test_visualization_format_basic_structure(basic_surface):
    """Test if visualization format output has all required keys and correct structure"""
    result = basic_surface.to_react_visualization_format()
    
    # Check all required top-level keys
    required_keys = {
        "vertices", "indices", "colors", "gridLines",
        "markers", "bounds", "metadata"
    }
    assert set(result.keys()) == required_keys
    
    # Check data types and shapes
    assert isinstance(result["vertices"], list)
    assert isinstance(result["indices"], list)
    assert isinstance(result["colors"], list)
    assert len(result["vertices"]) == len(result["colors"])
    
    # Verify all vertices are 3D points
    assert all(len(v) == 3 for v in result["vertices"])
    # Verify all colors are RGB values
    assert all(len(c) == 3 for c in result["colors"])

def test_visualization_format_null_surface():
    """Test handling of ungenerated surface"""
    surface = VolatilitySurface(sabr_model)
    with pytest.raises(ValueError, match="Surface must be generated before visualization"):
        surface.to_react_visualization_format()

def test_vertex_coordinates(basic_surface, basic_grid):
    """Test if vertex coordinates are correctly normalized and transformed"""
    result = basic_surface.to_react_visualization_format()
    vertices = result["vertices"]
    
    # Check moneyness normalization (strikes/spot - 1)
    unique_x = sorted(set(v[0] for v in vertices))
    expected_x = sorted(k/basic_grid.spot - 1.0 for k in basic_grid.strikes)
    np.testing.assert_array_almost_equal(unique_x, expected_x)
    
    # Check maturity normalization (t/max_t)
    unique_y = sorted(set(v[1] for v in vertices))
    expected_y = sorted(t/max(basic_grid.maturities) for t in basic_grid.maturities)
    np.testing.assert_array_almost_equal(unique_y, expected_y)

def test_mesh_indices(basic_surface):
    """Test if mesh indices are valid and properly connected"""
    result = basic_surface.to_react_visualization_format()
    indices = result["indices"]
    vertices = result["vertices"]
    
    # Check index bounds
    assert max(indices) < len(vertices)
    assert min(indices) >= 0
    
    # Verify triangle structure (indices should come in groups of 3)
    assert len(indices) % 3 == 0
    
    # Each vertex should be referenced at least once
    used_vertices = set(indices)
    assert len(used_vertices) == len(vertices)

def test_color_mapping(basic_surface):
    """Test if color values are valid and properly mapped to volatility levels"""
    result = basic_surface.to_react_visualization_format()
    colors = result["colors"]
    
    # Check color value ranges
    assert all(0 <= c <= 1 for color in colors for c in color)
    
    # Verify blue gradient implementation
    for color in colors:
        assert color[0] == 0.0  # R should be 0
        assert 0.3 <= color[1] <= 1.0  # G between 0.3 and 1.0
        assert 0.5 <= color[2] <= 1.0  # B between 0.5 and 1.0

def test_grid_lines(basic_surface):
    """Test grid line generation and ATM marker"""
    result = basic_surface.to_react_visualization_format()
    
    # Check grid lines structure
    assert "strikes" in result["gridLines"]
    assert "maturities" in result["gridLines"]
    assert isinstance(result["gridLines"]["strikes"], list)
    assert isinstance(result["gridLines"]["maturities"], list)
    
    # Verify ATM line
    assert "atmLine" in result["markers"]
    atm_line = result["markers"]["atmLine"]
    assert len(atm_line) == len(basic_surface._cached_grid.maturities)
    assert all(point[0] == 0.0 for point in atm_line)  # ATM should be at moneyness = 0

def test_bounds_calculation(basic_surface):
    """Test if bounds are correctly calculated and formatted"""
    result = basic_surface.to_react_visualization_format()
    bounds = result["bounds"]
    
    # Check bounds structure
    assert all(key in bounds for key in ["strikes", "maturities", "volatility"])
    assert all(isinstance(bound, list) and len(bound) == 2 
              for bound in bounds.values())
    
    # Verify bound ranges
    assert bounds["maturities"] == [0.0, 1.0]  # Normalized maturities
    assert bounds["strikes"][0] < bounds["strikes"][1]
    assert bounds["volatility"][0] < bounds["volatility"][1]

def test_metadata_completeness(basic_surface):
    """Test if metadata contains all required fields with valid values"""
    result = basic_surface.to_react_visualization_format()
    metadata = result["metadata"]
    
    # Check required metadata fields
    required_fields = {
        "spotPrice", "maxMaturity", "atmVol", "surfaceSkew",
        "termStructureSlope", "timestamp", "renderHints"
    }
    assert set(metadata.keys()) == required_fields
    
    # Check render hints
    render_hints = metadata["renderHints"]
    assert "initialRotation" in render_hints
    assert "cameraDistance" in render_hints
    assert "gridOpacity" in render_hints
    assert "meshOpacity" in render_hints

def test_error_handling_invalid_data(basic_surface):
    """Test handling of invalid data in surface"""
    # Inject NaN/Inf values
    surface = deepcopy(basic_surface)
    surface._cached_surface[0, 0] = np.nan
    
    with pytest.raises(ValueError, match="Surface contains NaN or Inf values"):
        surface.to_react_visualization_format()

def test_format_with_minimal_grid(sabr_model):
    """Test visualization format with minimal 2x2 grid"""
    minimal_grid = SurfaceGrid(
        strikes=np.array([90, 110]),
        maturities=np.array([0.5, 1.0]),
        spot=100.0,
        rate=0.02
    )
    
    surface = VolatilitySurface(sabr_model)
    surface.generate_surface(minimal_grid)
    result = surface.to_react_visualization_format()
    
    # Should still produce valid visualization data
    assert len(result["vertices"]) == 4  # 2x2 grid = 4 points
    assert len(result["indices"]) == 6   # 2 triangles = 6 indices
    assert len(result["colors"]) == 4    # One color per vertex

def test_format_with_large_grid(sabr_model):
    """Test visualization format with a larger grid"""
    large_grid = SurfaceGrid(
        strikes=np.linspace(50, 150, 20),
        maturities=np.linspace(0.1, 2.0, 15),
        spot=100.0,
        rate=0.02
    )
    
    surface = VolatilitySurface(sabr_model)
    surface.generate_surface(large_grid)
    result = surface.to_react_visualization_format()
    
    # Verify correct number of vertices and indices
    expected_vertices = 20 * 15  # grid points
    expected_triangles = 2 * (19 * 14)  # 2 triangles per grid square
    
    assert len(result["vertices"]) == expected_vertices
    assert len(result["indices"]) == expected_triangles * 3