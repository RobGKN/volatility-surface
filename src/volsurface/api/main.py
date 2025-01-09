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