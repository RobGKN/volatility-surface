from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, Union
import warnings
import numpy as np
import scipy.stats
from ..core.black_scholes import BSMInputs, BlackScholes, OptionType, ImpliedVolatility

@dataclass
class SABRParameters:
    """SABR model parameters"""
    alpha: float  # Initial volatility (similar to Black-Scholes sigma)
    beta: float   # CEV parameter (0 <= beta <= 1)
    rho: float    # Correlation between spot and vol (-1 <= rho <= 1)
    nu: float     # Volatility of volatility (volvol)
    
    def validate(self) -> None:
        """Validate SABR parameters are within valid ranges"""
        if not 0 <= self.beta <= 1:
            raise ValueError("Beta must be between 0 and 1")
        if not -1 <= self.rho <= 1:
            raise ValueError("Rho must be between -1 and 1")
        if self.alpha <= 0 or self.nu <= 0:
            raise ValueError("Alpha and nu must be positive")

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    n_paths: int = 50000          # Number of simulation paths
    steps_per_year: int = 252     # Time steps per year (daily simulation)
    min_steps: int = 50           # Minimum number of steps for short maturities
    antithetic: bool = True       # Use antithetic variates for variance reduction
    seed: Optional[int] = None    # Random seed for reproducibility

class SABRModel:
    """
    Stochastic Alpha Beta Rho (SABR) model implementation using Monte Carlo simulation
    
    limit moneyness to 0.85 to 1.15 to avoid numerical instability
    """
    
    def __init__(self, 
                 params: SABRParameters,
                 mc_config: Optional[MonteCarloConfig] = None):
        """
        Initialize SABR model
        
        Args:
            params: SABR model parameters
            mc_config: Monte Carlo simulation configuration
        """
        self.params = params
        self.params.validate()
        self.mc_config = mc_config or MonteCarloConfig()
        self.rng = np.random.default_rng(self.mc_config.seed)
        
        # Constants for numerical stability
        self._MIN_VOL = 0.001
        self._MIN_PRICE = 1e-12
        self._MAX_PRICE = 10000
        self._MAX_VOL = 1.0
        
        # Initialize Black-Scholes calculator for implied vol
        self.iv_calculator = ImpliedVolatility(tolerance=1e-7, max_iterations=200)
        
    def _get_n_steps(self, T: float) -> int:
        """Calculate number of time steps based on maturity"""
        return max(self.mc_config.min_steps, 
                  int(self.mc_config.steps_per_year * T))
    
    def _generate_correlated_paths(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate correlated Brownian paths with variance reduction techniques
        """
        n_steps = self._get_n_steps(T)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Generate base random numbers
        Z1 = self.rng.normal(0, 1, (self.mc_config.n_paths // 2, n_steps))
        Z2 = self.rng.normal(0, 1, (self.mc_config.n_paths // 2, n_steps))
        
        if self.mc_config.antithetic:
            # Add antithetic paths for variance reduction
            Z1 = np.vstack([Z1, -Z1])
            Z2 = np.vstack([Z2, -Z2])
        
        # Create correlated increments
        dW = sqrt_dt * Z1
        dZ = sqrt_dt * (self.params.rho * Z1 + 
                       np.sqrt(1 - self.params.rho**2) * Z2)
        
        return dW, dZ
    
    
    def _simulate_paths(self, F0: float, T: float, return_vol: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simulate price and volatility paths with enhanced numerical stability and debugging.
        """
        n_steps = self._get_n_steps(T)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Generate Brownian increments
        dW, dZ = self._generate_correlated_paths(T)
        dW = np.clip(dW, -3 * sqrt_dt, 3 * sqrt_dt)
        dZ = np.clip(dZ, -3 * sqrt_dt, 3 * sqrt_dt)
        
        # Initialize paths
        F = np.zeros((self.mc_config.n_paths, n_steps + 1))
        alpha = np.zeros_like(F)
        F[:, 0] = F0
        alpha[:, 0] = self.params.alpha
        
        # Debugging parameters
        DEBUG_PATH = 0  # Focus on path 0 for detailed outputs
        DEBUG_STEPS = 5  # Number of steps to output debug info
        
        for t in range(n_steps):
            F_t = np.maximum(F[:, t], self._MIN_PRICE)
            alpha_t = np.maximum(alpha[:, t], self._MIN_VOL)
            
            # Debugging: Print initial values
            if t < DEBUG_STEPS:
                print(f"\nStep {t}:")
                print(f"  Initial F_t[{DEBUG_PATH}] = {F_t[DEBUG_PATH]:.6f}")
                print(f"  Initial alpha_t[{DEBUG_PATH}] = {alpha_t[DEBUG_PATH]:.6f}")
            
            # CEV local volatility
            local_vol = alpha_t * np.sqrt(F_t)
            local_vol = np.minimum(local_vol, self._MAX_VOL)
            
            # Debugging: Print local volatility and increments
            if t < DEBUG_STEPS:
                print(f"  Local volatility (local_vol[{DEBUG_PATH}]) = {local_vol[DEBUG_PATH]:.6f}")
                print(f"  Brownian increment (dW[{DEBUG_PATH}, {t}]) = {dW[DEBUG_PATH, t]:.6f}")
            
            # Log-Euler dynamics
            dF = local_vol * dW[:, t]
            drift = -0.5 * local_vol**2 * dt
            F[:, t + 1] = F_t * np.exp(dF + drift)
            
            # Debugging: Print increments and updated F_t
            if t < DEBUG_STEPS:
                print(f"  Drift term (drift[{DEBUG_PATH}]) = {drift[DEBUG_PATH]:.6f}")
                print(f"  Price increment (dF[{DEBUG_PATH}]) = {dF[DEBUG_PATH]:.6f}")
                print(f"  Updated F[{DEBUG_PATH}, {t+1}] = {F[DEBUG_PATH, t+1]:.6f}")
            
            # Volatility dynamics (lognormal)
            dalpha = self.params.nu * dZ[:, t]
            alpha[:, t + 1] = alpha_t * np.exp(dalpha)
            
            # Debugging: Print volatility dynamics
            if t < DEBUG_STEPS:
                print(f"  Brownian increment (dZ[{DEBUG_PATH}, {t}]) = {dZ[DEBUG_PATH, t]:.6f}")
                print(f"  Volatility increment (dalpha[{DEBUG_PATH}]) = {dalpha[DEBUG_PATH]:.6f}")
                print(f"  Updated alpha[{DEBUG_PATH}, {t+1}] = {alpha[DEBUG_PATH, t+1]:.6f}")
            
            # Apply bounds
            F[:, t + 1] = np.clip(F[:, t + 1], self._MIN_PRICE, self._MAX_PRICE)
            alpha[:, t + 1] = np.clip(alpha[:, t + 1], self._MIN_VOL, self._MAX_VOL)
            
            # Debugging: Print values after applying bounds
            if t < DEBUG_STEPS:
                print(f"  After bounds:")
                print(f"    F[{DEBUG_PATH}, {t+1}] = {F[DEBUG_PATH, t+1]:.6f}")
                print(f"    alpha[{DEBUG_PATH}, {t+1}] = {alpha[DEBUG_PATH, t+1]:.6f}")
        
        # Return the simulated paths
        if return_vol:
            return F, alpha
        return F, None

    
    def price_option(self, F0: float, K: float, T: float, r: float,
                option_type: OptionType, return_std: bool = False) -> Union[float, Tuple[float, float]]:
        """Price European option using Monte Carlo with error estimate"""
        F_paths, _ = self._simulate_paths(F0, T)
        F_T = F_paths[:, -1]
        
        # Add diagnostics
        print(f"\nDiagnostics for K={K}:")
        print(f"F_T statistics:")
        print(f"Mean: {np.mean(F_T):.4f}")
        print(f"Std: {np.std(F_T):.4f}")
        print(f"Min: {np.min(F_T):.4f}")
        print(f"Max: {np.max(F_T):.4f}")
        
        if option_type == OptionType.CALL:
            payoffs = np.maximum(F_T - K, 0)
        else:
            payoffs = np.maximum(K - F_T, 0)
        
        # Add payoff diagnostics
        print(f"Payoff statistics:")
        print(f"Mean payoff: {np.mean(payoffs):.4f}")
        print(f"Std payoff: {np.std(payoffs):.4f}")
        print(f"Positive payoffs: {np.sum(payoffs > 0)} out of {len(payoffs)}")
        
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        std_err = discount * np.std(payoffs) / np.sqrt(len(payoffs))
        
        print(f"Final price: {price:.4f} (±{std_err:.4f})")
        
        if return_std:
            return price, std_err
        return price
    
    def implied_volatility(self,
                      F0: float,
                      K: float,
                      T: float,
                      r: float,
                      option_type: OptionType,
                      mc_price: Optional[float] = None
                      ) -> Tuple[float, Dict]:
        """
        Calculate implied volatility from SABR Monte Carlo price
        
        Returns:
            Tuple[float, Dict]: (implied_vol, metadata)
            metadata contains diagnostic information like number of iterations,
            standard error, etc.
        """
        # Initialize metadata dictionary
        metadata = {
            'convergence': False,
            'iterations': 0,
            'error': None,
            'mc_price': None,
            'standard_error': None
        }
        
        # Get Monte Carlo price if not provided
        if mc_price is None:
            mc_price, std_err = self.price_option(F0, K, T, r, option_type,
                                                return_std=True)
            metadata['standard_error'] = std_err
        else:
            std_err = None
            
        # Clean price for numerical stability
        mc_price = max(mc_price, self._MIN_PRICE)
        metadata['mc_price'] = mc_price
        
        # Prepare inputs for implied vol calculation
        bsm_inputs = BSMInputs(
            S=F0,  # Using forward as spot since we're in forward measure
            K=K,
            T=T,
            r=r,
            sigma=self.params.alpha  # Initial guess using SABR's alpha
        )
        
        try:
            impl_vol, n_iter = self.iv_calculator.calculate(
                mc_price, bsm_inputs, option_type)
            
            # Bound result for stability
            impl_vol = np.clip(impl_vol, self._MIN_VOL, self._MAX_VOL)
            
            metadata.update({
                'convergence': True,
                'iterations': n_iter
            })
                
            return impl_vol, metadata
                
        except (RuntimeError, ValueError) as e:
            metadata.update({
                'error': str(e),
                'convergence': False
            })
            return self._MIN_VOL, metadata
    
    def volatility_surface(self,
                          F0: float,
                          strikes: np.ndarray,
                          maturities: np.ndarray,
                          r: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete implied volatility surface
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (vols, errors)
            Arrays of shape (len(maturities), len(strikes))
        """
        vols = np.zeros((len(maturities), len(strikes)))
        errors = np.zeros_like(vols)
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                impl_vol, metadata = self.implied_volatility(
                    F0, K, T, r, OptionType.CALL)
                vols[i, j] = impl_vol
                errors[i, j] = metadata.get('standard_error', np.nan)
                
        return vols, errors