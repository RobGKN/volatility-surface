from fastapi.testclient import TestClient
from volsurface.api.main import app

# Simple initialization, no keyword args needed
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_surface_generation():
    test_params = {
        "strikes": [90, 100, 110],
        "maturities": [0.5, 1.0],
        "spot": 100,
        "rate": 0.02,
        "sabr_params": {
            "alpha": 0.4,
            "beta": 1.0,
            "rho": -0.4,
            "nu": 0.6
        }
    }
    
    response = client.post("/api/surface", json=test_params)
    if response.status_code != 200:
        print(f"Error response: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data