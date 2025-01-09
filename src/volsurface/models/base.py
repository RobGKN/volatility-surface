# src/volsurface/models/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class ModelParameters:
    """Base class for model parameters"""
    def validate(self) -> None:
        """Validate parameters are within acceptable ranges"""
        pass

class VolatilityModel(ABC):
    """Abstract base class for volatility models"""
    
    @abstractmethod
    def implied_volatility(self, F: float, K: float, T: float) -> float:
        """
        Calculate implied volatility for given inputs
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity (in years)
            
        Returns:
            float: Implied volatility value
        """
        pass
    
    @abstractmethod
    def calibrate(self, 
                 market_vols: np.ndarray,
                 strikes: np.ndarray,
                 forwards: np.ndarray,
                 times: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate model to market data
        
        Args:
            market_vols: Observed market volatilities
            strikes: Strike prices
            forwards: Forward prices
            times: Time to maturity for each point
            
        Returns:
            Dict containing calibration results and diagnostics
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> ModelParameters:
        """Get current model parameters"""
        pass