# src/volsurface/models/quantlib_sabr.py

import QuantLib as ql
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any

from .base import ModelParameters, VolatilityModel

class QuantLibSABRParameters(ModelParameters):
    """SABR parameters wrapper for QuantLib implementation"""
    def __init__(self, alpha: float, beta: float, rho: float, nu: float):
        self.alpha = alpha  # Initial volatility
        self.beta = beta    # CEV parameter
        self.rho = rho      # Correlation
        self.nu = nu        # Vol of vol
        
    def validate(self) -> None:
        """Validate parameters for QuantLib implementation"""
        if not 0 <= self.beta <= 1:
            raise ValueError("Beta must be between 0 and 1")
        if not -1 <= self.rho <= 1:
            raise ValueError("Rho must be between -1 and 1")
        if self.alpha <= 0 or self.nu <= 0:
            raise ValueError("Alpha and nu must be positive")

class QuantLibSABRModel(VolatilityModel):
    """
    QuantLib-based SABR implementation that matches our interface
    """
    def __init__(self, params: QuantLibSABRParameters):
        self._params = params
        self._params.validate()
        
        # QuantLib setup
        self._today = ql.Date().todaysDate()
        self._calendar = ql.TARGET()
        self._dayCounter = ql.Actual365Fixed()
        
    @property
    def parameters(self) -> ModelParameters:
        return self._params
    
    def implied_volatility(self, F: float, K: float, T: float) -> float:
        """
        Calculate implied volatility using QuantLib's SABR implementation
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity in years
            
        Returns:
            float: Implied volatility
        """
        try:
            # Create SABR smile section
            sabr_section = ql.SabrSmileSection(
                T,  # Time to expiry
                F,  # Forward rate
                [self._params.alpha, 
                 self._params.beta,
                 self._params.nu,
                 self._params.rho]
            )
            
            return sabr_section.volatility(K)
            
        except RuntimeError as e:
            # Handle QuantLib exceptions gracefully
            raise ValueError(f"QuantLib SABR calculation failed: {str(e)}")
            
    def calibrate(self,
                market_vols: np.ndarray,
                strikes: np.ndarray,
                forwards: np.ndarray,
                times: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate SABR parameters to market data using QuantLib
        
        Note: This implementation uses QuantLib's optimization routines
        """
        def objective(x):
            alpha, beta, rho, nu = x
            
            if not (0 < beta < 1 and -1 < rho < 1 and alpha > 0 and nu > 0):
                return 1e6
                
            try:
                # Update parameters temporarily
                self._params = QuantLibSABRParameters(alpha, beta, rho, nu)
                
                # Calculate model vols
                model_vols = np.array([
                    self.implied_volatility(f, k, t)
                    for f, k, t in zip(forwards, strikes, times)
                ])
                
                # Weight ATM options more heavily
                weights = np.exp(-0.5 * ((strikes - forwards) / forwards) ** 2)
                mse = np.mean(weights * (market_vols - model_vols) ** 2)
                
                # Add regularization
                reg_strength = 0.1
                reg_beta = reg_strength * (beta - 0.5) ** 2
                reg_nu = reg_strength * nu ** 2
                
                return float(mse + reg_beta + reg_nu)
                
            except (ValueError, RuntimeError):
                return 1e6
        
        # Initial parameters
        x0 = [
            self._params.alpha,
            self._params.beta,
            self._params.rho,
            self._params.nu
        ]
        
        # Setup QuantLib optimization
        optimizationMethod = ql.LevenbergMarquardt()
        endCriteria = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)
        
        try:
            # Create QuantLib array for optimization
            ql_params = ql.Array(x0)
            problem = ql.Problem(
                lambda x: objective(x),
                ql.NoConstraint(),
                ql_params
            )
            
            optimizationMethod.minimize(
                problem,
                endCriteria
            )
            
            # Get optimized parameters
            opt_params = problem.currentValue()
            
            # Update model parameters
            self._params = QuantLibSABRParameters(
                alpha=float(opt_params[0]),
                beta=float(opt_params[1]),
                rho=float(opt_params[2]),
                nu=float(opt_params[3])
            )
            
            return {
                "success": True,
                "final_error": float(problem.functionValue()),
                "iterations": problem.functionEvaluation(),
                "message": "Optimization succeeded",
                "parameters": {
                    "alpha": self._params.alpha,
                    "beta": self._params.beta,
                    "rho": self._params.rho,
                    "nu": self._params.nu
                }
            }
            
        except RuntimeError as e:
            return {
                "success": False,
                "message": f"QuantLib optimization failed: {str(e)}",
                "parameters": {
                    "alpha": self._params.alpha,
                    "beta": self._params.beta,
                    "rho": self._params.rho,
                    "nu": self._params.nu
                }
            }