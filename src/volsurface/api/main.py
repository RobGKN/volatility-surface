from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Optional

from ..models.factory import ModelFactory
from pydantic import BaseModel
from ..core.volatility_surface import VolatilitySurface, SurfaceGrid

app = FastAPI(title="Volatility Surface API")

# Define the SABR parameters model
class SABRParamsModel(BaseModel):
    alpha: float
    beta: float
    rho: float
    nu: float

class SurfaceParams(BaseModel):
    strikes: List[float]
    maturities: List[float]
    spot: float
    rate: float = 0.0
    sabr_params: SABRParamsModel
    model_type: str = "quantlib_sabr"


# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://volatility-surface-nine.vercel.app",
        "https://volatility-surface-xru6.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_surface_data(surface_data):
        """For debugging to_vizualization"""
        strikes = surface_data['x']
        maturities = surface_data['y']
        surface = surface_data['z']
        
        # Check dimensions
        if len(surface) != len(strikes):
            raise ValueError(f"Surface first dimension ({len(surface)}) doesn't match strikes ({len(strikes)})")
        
        if any(len(row) != len(maturities) for row in surface):
            raise ValueError("Surface second dimension doesn't match maturities")
        
        # Check for reasonable values
        if any(v <= 0 or v > 2 for row in surface for v in row):
            raise ValueError("Found unreasonable volatility values (<=0 or >200%)")
        
        # Print sample values
        print(f"Surface shape: {len(surface)}x{len(surface[0])}")
        print(f"Sample values:")
        print(f"- Top left (shortest maturity, lowest strike): {surface[0][0]:.4f}")
        print(f"- Top right (shortest maturity, highest strike): {surface[-1][0]:.4f}")
        print(f"- Bottom left (longest maturity, lowest strike): {surface[0][-1]:.4f}")
        print(f"- Bottom right (longest maturity, highest strike): {surface[-1][-1]:.4f}")

@app.post("/api/surface")
async def generate_surface(params: SurfaceParams):
    try:
        # Create model using factory
        model = ModelFactory.create_model(
            params.model_type,
            {
                'alpha': params.sabr_params.alpha,
                'beta': params.sabr_params.beta,
                'rho': params.sabr_params.rho,
                'nu': params.sabr_params.nu
            }
        )
        
        # Rest of the implementation remains the same
        surface = VolatilitySurface(model=model)
        grid = SurfaceGrid(
            strikes=np.array(params.strikes),
            maturities=np.array(params.maturities),
            spot=params.spot,
            rate=params.rate
        )
        
        # Generate surface
        surface.generate_surface(grid)
        surface_data = surface.to_visualization_format()
        
        # for debugging
        validate_surface_data(surface_data)
        
        return {
            "success": True,
            "data": surface_data,
            "metadata": {
                "spot": params.spot,
                "rate": params.rate,
                "model_type": params.model_type,
                "sabr_params": {
                    "alpha": params.sabr_params.alpha,
                    "beta": params.sabr_params.beta,
                    "rho": params.sabr_params.rho,
                    "nu": params.sabr_params.nu
                }
            }
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}