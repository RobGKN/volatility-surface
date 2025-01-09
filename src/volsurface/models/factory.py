# src/volsurface/models/factory.py

from enum import Enum
from typing import Dict, Any

from .base import VolatilityModel
from .custom_sabr import CustomSABRModel, SABRParameters
from .quantlib_sabr import QuantLibSABRModel, QuantLibSABRParameters

class ModelType(Enum):
    CUSTOM_SABR = "custom_sabr"
    QUANTLIB_SABR = "quantlib_sabr"

class ModelFactory:
    """Factory for creating volatility models"""
    
    @staticmethod
    def create_model(
        model_type: str,
        params: Dict[str, float]
    ) -> VolatilityModel:
        """
        Create a volatility model instance
        
        Args:
            model_type: Type of model to create ("custom_sabr" or "quantlib_sabr")
            params: Dictionary with model parameters (alpha, beta, rho, nu)
            
        Returns:
            VolatilityModel instance
        """
        try:
            model_type = ModelType(model_type)
        except ValueError:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if model_type == ModelType.CUSTOM_SABR:
            params = SABRParameters(
                alpha=params['alpha'],
                beta=params['beta'],
                rho=params['rho'],
                nu=params['nu']
            )
            return CustomSABRModel(params)
            
        elif model_type == ModelType.QUANTLIB_SABR:
            params = QuantLibSABRParameters(
                alpha=params['alpha'],
                beta=params['beta'],
                rho=params['rho'],
                nu=params['nu']
            )
            return QuantLibSABRModel(params)
            
        raise ValueError(f"Unhandled model type: {model_type}")