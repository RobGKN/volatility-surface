from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from ..core.black_scholes import BSMInputs, BlackScholes, OptionType, ImpliedVolatility
from .base import ModelParameters, VolatilityModel
from ..core.black_scholes import ImpliedVolatility

class SABRParameters(ModelParameters):
    """SABR model parameters"""
    def __init__(self, alpha: float, beta: float, rho: float, nu: float):
        self.alpha = alpha  # Initial volatility (similar to Black-Scholes sigma)
        self.beta = beta    # CEV parameter (0 <= beta <= 1)
        self.rho = rho      # Correlation between spot and vol (-1 <= rho <= 1)
        self.nu = nu        # Volatility of volatility (volvol)
    
    def validate(self) -> None:
        """Validate SABR parameters are within valid ranges"""
        if not 0 <= self.beta <= 1:
            raise ValueError("Beta must be between 0 and 1")
        if not -1 <= self.rho <= 1:
            raise ValueError("Rho must be between -1 and 1")
        if self.alpha <= 0 or self.nu <= 0:
            raise ValueError("Alpha and nu must be positive")

class CustomSABRModel(VolatilityModel):
    """
    Custom implementation of the SABR model
    """
    def __init__(self, 
                params: SABRParameters,
                tolerance: float = 1e-5,
                max_iterations: int = 100,
                moneyness_threshold: float = 0.1):
        self._params = params
        self._params.validate()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.moneyness_threshold = moneyness_threshold
        
    @property
    def parameters(self) -> ModelParameters:
        return self._params

    def implied_volatility(self, F: float, K: float, T: float) -> float:
        """Calculate implied volatility using existing implementation"""
        return self._hagan_implied_vol(F, K, T)
        
    def calibrate(self,
                market_vols: np.ndarray,
                strikes: np.ndarray,
                forwards: np.ndarray,
                times: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate SABR parameters to market data
        """
        def objective(x):
            alpha, beta, rho, nu = x
            
            # Add penalty for parameters near bounds
            if not (0 < beta < 1 and -1 < rho < 1 and alpha > 0 and nu > 0):
                return 1e6
            
            try:
                # Update parameters temporarily
                temp_params = SABRParameters(alpha, beta, rho, nu)
                self._params = temp_params
                
                # Calculate model vols
                model_vols = np.array([
                    self.implied_volatility(f, k, t)
                    for f, k, t in zip(forwards, strikes, times)
                ])
                
                # Weighted MSE with emphasis on ATM options
                weights = np.exp(-0.5 * ((strikes - forwards) / forwards) ** 2)
                mse = np.mean(weights * (market_vols - model_vols) ** 2)
                
                # Add regularization
                reg_strength = 0.1
                reg_beta = reg_strength * (beta - 0.5) ** 2
                reg_nu = reg_strength * nu ** 2
                
                return mse + reg_beta + reg_nu
                
            except (ValueError, RuntimeError):
                return 1e6
        
        # Initial guess from current parameters
        x0 = [
            self._params.alpha,
            self._params.beta,
            self._params.rho,
            self._params.nu
        ]
        
        # Run optimization
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8}
        )
        
        # Update parameters with optimal values
        self._params = SABRParameters(
            alpha=result.x[0],
            beta=result.x[1],
            rho=result.x[2],
            nu=result.x[3]
        )
        
        return {
            "success": result.success,
            "final_error": result.fun,
            "iterations": result.nit,
            "message": result.message,
            "parameters": {
                "alpha": self._params.alpha,
                "beta": self._params.beta,
                "rho": self._params.rho,
                "nu": self._params.nu
            }
        }

    def _hagan_implied_vol(self, F: float, K: float, T: float) -> float:
        """
        Hagan (2002) lognormal SABR implied-volatility approximation.
        """
        alpha = self._params.alpha
        beta = self._params.beta
        rho = self._params.rho
        nu = self._params.nu

        eps = 1e-10
        one_minus_beta = 1.0 - beta

        # At-the-money limit (F == K)
        if abs(F - K) < eps:
            f_pow = F ** one_minus_beta
            atm = alpha / f_pow
            term1 = one_minus_beta ** 2 / 24 * (alpha / f_pow) ** 2
            term2 = rho * beta * nu * alpha / (4 * f_pow)
            term3 = (2 - 3 * rho ** 2) * nu ** 2 / 24
            return atm * (1 + (term1 + term2 + term3) * T)

        log_fk = np.log(F / K)
        fk_pow = (F * K) ** (one_minus_beta / 2)

        z = (nu / alpha) * fk_pow * log_fk
        if abs(z) < eps:
            z_over_x = 1.0
        else:
            chi = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
            z_over_x = z / chi

        denominator = (
            1
            + (one_minus_beta ** 2 / 24) * log_fk ** 2
            + (one_minus_beta ** 4 / 1920) * log_fk ** 4
        )
        prefactor = alpha / (fk_pow * denominator)

        term1 = one_minus_beta ** 2 / 24 * (alpha / fk_pow) ** 2
        term2 = rho * beta * nu * alpha / (4 * fk_pow)
        term3 = (2 - 3 * rho ** 2) * nu ** 2 / 24

        implied_vol = prefactor * z_over_x * (1 + (term1 + term2 + term3) * T)
        return max(0.001, min(5.0, implied_vol))
        
    def _mc_implied_vol(self, F: float, K: float, T: float) -> float:
        """
        Monte Carlo based implied volatility for extreme strikes
        """
        option_price = self._monte_carlo_price(F, K, T)
        
        # Use a more robust initial guess
        initial_sigma = self._params.alpha * (F / K)**(self._params.beta - 1.0)
        
        bsm_inputs = BSMInputs(
            S=float(F),
            K=float(K),
            T=float(T),
            r=0.0,
            sigma=max(0.001, min(5.0, initial_sigma))
        )
        
        try:
            implied_vol, _ = self.iv_calculator.calculate(
                market_price=float(option_price),
                inputs=bsm_inputs,
                option_type=OptionType.CALL
            )
            return implied_vol
        except ValueError as e:
            if "Vega too close to zero" in str(e):
                return max(0.001, min(5.0, initial_sigma))
            raise
    
    def _monte_carlo_price(self, 
                          F: float, 
                          K: float, 
                          T: float,
                          n_steps: int = 500,
                          n_paths: int = 50000,
                          seed: Optional[int] = None) -> float:
        """
        Price option using Monte Carlo simulation of SABR dynamics
        """
        # Convert inputs to float64
        F = np.float64(F)
        K = np.float64(K)
        T = np.float64(T)
        
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths with explicit float64 dtype
        F_t = np.full(n_paths, F, dtype=np.float64)
        alpha_t = np.full(n_paths, self._params.alpha, dtype=np.float64)
        
        # Generate correlated Brownian motions
        dW1 = np.random.normal(0, 1, (n_steps, n_paths))
        dW2 = self._params.rho * dW1 + np.sqrt(1 - self._params.rho**2) * \
              np.random.normal(0, 1, (n_steps, n_paths))
        
        # Pre-allocate array for variance reduction
        price_paths = np.zeros(n_paths, dtype=np.float64)
        
        # Simulate paths with careful numerical handling
        for i in range(n_steps):
            # Handle potential numerical issues
            F_t = np.maximum(F_t, 1e-10)
            alpha_t = np.maximum(alpha_t, 1e-10)
            
            # SABR dynamics with stability controls
            F_power = np.power(F_t, self._params.beta)
            F_power = np.minimum(F_power, 1e10)  # Prevent explosion
            
            # Calculate increments with scaling
            dF = alpha_t * F_power * dW1[i] * sqrt_dt
            dalpha = self._params.nu * alpha_t * dW2[i] * sqrt_dt
            
            # Update with stability checks
            F_t_new = F_t + dF
            alpha_t_new = alpha_t + dalpha
            
            # Only update if values are reasonable
            valid_F = np.abs(F_t_new) < 1e10
            valid_alpha = np.abs(alpha_t_new) < 1e10
            
            F_t = np.where(valid_F, F_t_new, F_t)
            alpha_t = np.where(valid_alpha, alpha_t_new, alpha_t)
            
            # Store intermediate prices for variance reduction
            price_paths += np.maximum(F_t - K, 0)
        
        # Average over time steps and paths for variance reduction
        option_price = np.mean(price_paths) / n_steps
        
        # Ensure we return a positive price
        return max(1e-10, float(option_price))
    
    def calibrate(self, 
                 market_vols: np.ndarray, 
                 strikes: np.ndarray, 
                 forwards: np.ndarray, 
                 times: np.ndarray,
                 initial_guess: Optional[SABRParameters] = None) -> SABRParameters:
        """
        Calibrate SABR parameters to market data using least squares optimization
        
        Parameters:
        -----------
        market_vols : np.ndarray
            Market implied volatilities
        strikes : np.ndarray
            Option strike prices
        forwards : np.ndarray
            Forward prices for each expiry
        times : np.ndarray
            Time to expiry for each option
        initial_guess : Optional[SABRParameters]
            Initial parameter guess, if None uses reasonable defaults
            
        Returns:
        --------
        SABRParameters
            Calibrated parameters
        """
        if initial_guess is None:
            # Set reasonable default initial parameters
            initial_guess = SABRParameters(
                alpha=np.mean(market_vols),
                beta=0.5,
                rho=-0.2,
                nu=0.4
            )
            
        def objective(x):
            # Unpack parameters
            alpha, beta, rho, nu = x
            
            # Add penalty for parameters near bounds
            if not (0 < beta < 1 and -1 < rho < 1 and alpha > 0 and nu > 0):
                return 1e6
            
            params = SABRParameters(alpha, beta, rho, nu)
            self._params = params  # Update model parameters
            
            # Calculate model vols
            try:
                model_vols = np.array([
                    self.implied_volatility(f, k, t)
                    for f, k, t in zip(forwards, strikes, times)
                ])
                
                # Weighted MSE with emphasis on ATM options
                weights = np.exp(-0.5 * ((strikes - forwards) / forwards) ** 2)
                mse = np.mean(weights * (market_vols - model_vols) ** 2)
                
                # Add regularization terms
                reg_strength = 0.1
                reg_beta = reg_strength * (beta - 0.5) ** 2  # Prefer beta near 0.5
                reg_nu = reg_strength * nu ** 2  # Prefer smaller nu
                
                return mse + reg_beta + reg_nu
                
            except (ValueError, RuntimeError):
                return 1e6
            
        # Initial parameters
        x0 = [
            initial_guess.alpha,
            initial_guess.beta,
            initial_guess.rho,
            initial_guess.nu
        ]
        
        # Run optimization
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8}
        )
        
        # Return calibrated parameters
        return SABRParameters(
            alpha=result.x[0],
            beta=result.x[1],
            rho=result.x[2],
            nu=result.x[3]
        )