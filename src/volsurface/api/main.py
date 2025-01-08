from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Optional

from ..models.sabr import SABRModel, SABRParameters
from ..core.volatility_surface import VolatilitySurface, SurfaceGrid

# Define the SABR parameters model
class SABRParamsModel(BaseModel):
    alpha: float
    beta: float
    rho: float
    nu: float

# Define the surface parameters model - renamed field to match request
class SurfaceParams(BaseModel):
    strikes: List[float]
    maturities: List[float]
    spot: float
    rate: float = 0.0
    sabr_params: SABRParamsModel

app = FastAPI(title="Volatility Surface API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/surface")
async def generate_surface(params: SurfaceParams):
    try:
        print(f"Full parameters object: {params}")
        print(f"SABR parameters: {params.sabr_params}")
        
        # Create SABR parameters
        sabr_params = SABRParameters(
            alpha=params.sabr_params.alpha,
            beta=params.sabr_params.beta,
            rho=params.sabr_params.rho,
            nu=params.sabr_params.nu
        )
        
        # Create SABR model
        model = SABRModel(params=sabr_params)
        
        # Create volatility surface object
        surface = VolatilitySurface(model=model)
        
        # Create surface grid from input parameters
        grid = SurfaceGrid(
            strikes=np.array(params.strikes),
            maturities=np.array(params.maturities),
            spot=params.spot,
            rate=params.rate
        )
        
        # Generate the surface on the grid
        surface.generate_surface(grid)
        
        # Now we can get the visualization data
        surface_data = surface.to_visualization_format()
        
        return {
            "success": True,
            "data": surface_data,
            "metadata": {
                "spot": params.spot,
                "rate": params.rate,
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