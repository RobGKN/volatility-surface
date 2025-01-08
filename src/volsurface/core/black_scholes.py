from enum import Enum
from typing import Union, Tuple
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

@dataclass
class BSMInputs:
    """Container for Black-Scholes-Merton model inputs"""
    S: float  # Spot price
    K: float  # Strike price
    T: float  # Time to maturity (in years)
    r: float  # Risk-free rate (annualized)
    sigma: float  # Volatility (annualized)
    
    def validate(self) -> None:
        """Validate inputs are within reasonable bounds"""
        if self.S <= 0 or self.K <= 0:
            raise ValueError("Spot and strike prices must be positive")
        if self.T <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        
class BlackScholes:
    """Black-Scholes option pricing and related calculations."""
    
    @staticmethod
    def d1(inputs: BSMInputs) -> float:
        """
        Calculate d1 component of Black-Scholes formula
        
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        """
        return (np.log(inputs.S / inputs.K) + 
                (inputs.r + inputs.sigma**2 / 2) * inputs.T) / \
                (inputs.sigma * np.sqrt(inputs.T))
    
    @staticmethod
    def d2(d1: float, sigma: float, T: float) -> float:
        """
        Calculate d2 component of Black-Scholes formula
        
        d2 = d1 - σ√T
        """
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def price(inputs: BSMInputs, option_type: OptionType) -> float:
        """
        Calculate Black-Scholes option price
        
        Call = S⋅N(d1) - K⋅e^(-rT)⋅N(d2)
        Put = K⋅e^(-rT)⋅N(-d2) - S⋅N(-d1)
        """
        inputs.validate()
        
        d1 = BlackScholes.d1(inputs)
        d2 = BlackScholes.d2(d1, inputs.sigma, inputs.T)
        
        disc_factor = np.exp(-inputs.r * inputs.T)
        
        if option_type == OptionType.CALL:
            return inputs.S * norm.cdf(d1) - inputs.K * disc_factor * norm.cdf(d2)
        else:
            return inputs.K * disc_factor * norm.cdf(-d2) - inputs.S * norm.cdf(-d1)

    @staticmethod
    def vega(inputs: BSMInputs) -> float:
        """
        Calculate option vega (∂V/∂σ)
        
        Vega = S⋅√T⋅N'(d1)
        """
        d1 = BlackScholes.d1(inputs)
        return inputs.S * np.sqrt(inputs.T) * norm.pdf(d1)

class ImpliedVolatility:
    """Calculate implied volatility using Newton-Raphson method"""
    
    def __init__(self, tolerance: float = 1e-5, max_iterations: int = 100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def calculate(self, 
                 market_price: float,
                 inputs: BSMInputs,
                 option_type: OptionType) -> Tuple[float, int]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Returns:
            Tuple[float, int]: (implied_vol, num_iterations)
        """
        if market_price <= 0:
            raise ValueError("Market price must be positive")
            
        # Initial guess using simplified Brenner-Subrahmanyam approximation
        if option_type == OptionType.CALL:
            vol_init = np.sqrt(2 * np.pi / inputs.T) * (market_price / inputs.S)
        else:
            vol_init = np.sqrt(2 * np.pi / inputs.T) * (market_price / inputs.K)
            
        vol = max(0.001, min(vol_init, 5.0))  # Bound initial guess
        
        for i in range(self.max_iterations):
            inputs.sigma = vol
            price = BlackScholes.price(inputs, option_type)
            vega = BlackScholes.vega(inputs)
            
            if abs(price - market_price) < self.tolerance:
                return vol, i + 1
                
            if abs(vega) < 1e-10:  # Avoid division by zero
                raise ValueError("Vega too close to zero")
                
            vol = vol - (price - market_price) / vega
            vol = max(0.001, min(vol, 5.0))  # Bound the estimate
            
        raise RuntimeError(f"Failed to converge after {self.max_iterations} iterations")