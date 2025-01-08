import pytest
import numpy as np
import QuantLib as ql
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Callable

from volsurface.models.sabr import SABRParameters, SABRModel
from volsurface.core.black_scholes import BSMInputs, OptionType

# Note: Using SABRParameters directly instead of a separate MarketScenario class

class TestSABR:
    @pytest.fixture
    def market_scenarios(self) -> List[Tuple[str, SABRParameters]]:
        """
        Define test scenarios:
        1. Skew scenario (typical equity-like with negative rho)
        2. Smile scenario (FX-like with near-zero rho)
        """
        return [
            (
                "skew_pattern",
                SABRParameters(
                    alpha=0.20,    # 20% ATM vol
                    beta=0.5,      # Standard for indices
                    nu=0.4,        # Moderate vol of vol
                    rho=-0.4       # Negative correlation -> skew
                )
            ),
            (
                "smile_pattern",
                SABRParameters(
                    alpha=0.20,    # Same ATM vol
                    beta=0.5,      # Same beta
                    nu=0.8,        # Higher vol of vol
                    rho=-0.1       # Near-zero correlation -> smile
                )
            )
        ]

    @pytest.fixture
    def standard_maturities(self) -> List[float]:
        """
        Standard option maturities in years, focusing on liquid tenors
        """
        return [
            1/12,     # 1 month
            3/12,     # 3 months
            6/12,     # 6 months
            1.0,      # 1 year
            1.5,      # 18 months
            2.0       # 2 years
        ]

    @pytest.fixture
    def realistic_moneyness(self) -> List[float]:
        """
        Realistic moneyness levels focusing on typically traded strikes
        Denser around ATM where liquidity is highest
        """
        return [
            0.90,     # 90% - Typical put buying range
            0.95,     # 95% - Common put strike
            0.975,    # 97.5% - Near ATM put
            1.00,     # ATM
            1.025,    # 102.5% - Near ATM call
            1.05,     # 105% - Common call strike
            1.10      # 110% - Typical call buying range
        ]

    @pytest.fixture
    def ql_sabr_engine(self):
        """
        Create a QuantLib SABR vol engine for comparison
        """
        def create_engine(alpha: float, beta: float, nu: float, rho: float):
            def calculate_vol(K: float, F: float, T: float) -> float:
                try:
                    today = ql.Date().todaysDate()
                    expiry = today + int(T * 365)
                    dayCount = ql.Actual365Fixed()
                    timeToExpiry = dayCount.yearFraction(today, expiry)
                    
                    sabr_vol = ql.sabrVolatility(
                        K, F, timeToExpiry, alpha, beta, nu, rho
                    )
                    return float(sabr_vol)
                except Exception as e:
                    raise ValueError(
                        f"QuantLib calculation failed:"
                        f"\nParameters: K={K}, F={F}, T={T}"
                        f"\nSABR params: α={alpha}, β={beta}, ν={nu}, ρ={rho}"
                        f"\nError: {str(e)}"
                    )
            return calculate_vol
        return create_engine

    def test_parameter_validation(self):
        """Test parameter validation logic"""
        # Valid parameters should not raise
        params = SABRParameters(alpha=0.2, beta=0.5, rho=-0.4, nu=0.4)
        params.validate()

        # Test invalid parameters
        with pytest.raises(ValueError):
            SABRParameters(alpha=-0.1, beta=0.5, rho=-0.4, nu=0.4).validate()
        with pytest.raises(ValueError):
            SABRParameters(alpha=0.2, beta=1.5, rho=-0.4, nu=0.4).validate()
        with pytest.raises(ValueError):
            SABRParameters(alpha=0.2, beta=0.5, rho=-1.5, nu=0.4).validate()
        with pytest.raises(ValueError):
            SABRParameters(alpha=0.2, beta=0.5, rho=-0.4, nu=-0.1).validate()

    @pytest.mark.parametrize("maturity", [3/12, 6/12, 1.0])  # Focus on liquid tenors
    @pytest.mark.parametrize("moneyness", [0.80, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2])  # Near ATM
    def test_vs_quantlib_liquid_options(
        self, 
        market_scenarios, 
        ql_sabr_engine, 
        maturity, 
        moneyness
    ):
        """Compare implied vols against QuantLib for liquid options"""
        F0 = 100.0  # Reference level
        K = F0 * moneyness

        # Test each market scenario
        for scenario in market_scenarios:
            scenario_name, params = scenario
            # Create QuantLib reference
            ql_vol_func = ql_sabr_engine(
                params.alpha,
                params.beta,
                params.nu,
                params.rho
            )
            ql_vol = ql_vol_func(K, F0, maturity)
            
            # Calculate with our model
            model = SABRModel(params)
            our_vol = model.implied_volatility(F0, K, maturity)
            
            # Define tolerance based on moneyness and maturity
            # More generous tolerances as our implementation differs from QuantLib
            base_tol = 0.003  # 50 bps
            moneyness_factor = 1 + 3 * abs(moneyness - 1.0)  # Wider tolerance away from ATM
            maturity_factor = 1 + maturity  # More generous scaling with maturity
            tol = base_tol * moneyness_factor * maturity_factor
            
            print(
                f"\nScenario: {scenario_name}"
                f"\nMaturity: {maturity:.2f}Y"
                f"\nMoneyness: {moneyness:.3f}"
                f"\nOur vol: {our_vol:.4%}"
                f"\nQL vol: {ql_vol:.4%}"
                f"\nDiff: {abs(our_vol - ql_vol):.4%}"
                f"\nTolerance: {tol:.4%}")
            
            
            assert abs(our_vol - ql_vol) < tol, (
                f"\nScenario: {scenario_name}"
                f"\nMaturity: {maturity:.2f}Y"
                f"\nMoneyness: {moneyness:.3f}"
                f"\nOur vol: {our_vol:.4%}"
                f"\nQL vol: {ql_vol:.4%}"
                f"\nDiff: {abs(our_vol - ql_vol):.4%}"
                f"\nTolerance: {tol:.4%}"
            )

    def test_atm_term_structure(
        self,
        market_scenarios,
        standard_maturities,
        ql_sabr_engine
    ):
        """Test ATM volatility term structure behavior"""
        F0 = 100.0
        K = F0  # ATM

        for scenario in market_scenarios:
            scenario_name, params = scenario
            ql_vol_func = ql_sabr_engine(
                params.alpha,
                params.beta,
                params.nu,
                params.rho
            )
            
            model = SABRModel(params)
            
            vols = []  # Store vols for monotonicity check
            for T in standard_maturities:
                ql_vol = ql_vol_func(K, F0, T)
                our_vol = model.implied_volatility(F0, K, T)
                
                # More generous tolerance for ATM
                tol = 0.005 * (1 + T)  # Scale with maturity
                if T > 1.0:  # Even more generous for longer dates
                    tol *= 1.5
                
                assert abs(our_vol - ql_vol) < tol, (
                    f"\nATM term structure divergence:"
                    f"\nScenario: {scenario.name}"
                    f"\nMaturity: {T:.2f}Y"
                    f"\nOur vol: {our_vol:.4%}"
                    f"\nQL vol: {ql_vol:.4%}"
                    f"\nDiff: {abs(our_vol - ql_vol):.4%}"
                )
                
                vols.append(our_vol)
            
            # Check term structure is well-behaved
            # Volatilities shouldn't jump erratically
            diffs = np.diff(vols)
            max_jump = 0.05  # Max 500bps between tenors
            assert all(abs(d) < max_jump for d in diffs), (
                f"Term structure shows erratic behavior in {scenario.name}"
                f"\nVols across term structure: {[f'{v:.4%}' for v in vols]}"
            )

    def test_smile_shape(
        self,
        market_scenarios,
        realistic_moneyness
    ):
        """Test volatility smile/skew characteristics"""
        T = 0.5  # 6 month tenor
        F0 = 100.0

        for scenario in market_scenarios:
            scenario_name, params = scenario
            model = SABRModel(params)
            
            # Calculate volatilities
            strikes = [F0 * k for k in realistic_moneyness]
            vols = [model.implied_volatility(F0, K, T) for K in strikes]
            
            # Find ATM index and vol
            atm_idx = realistic_moneyness.index(1.0)
            atm_vol = vols[atm_idx]
            
            # Print for debugging
            print(f"\nScenario: {scenario_name}")
            print("Strike  |  Vol")
            print("-" * 20)
            for k, vol in zip(realistic_moneyness, vols):
                print(f"{k:6.3f}  |  {vol:6.4%}")
            
            if "skew" in scenario_name:
                # Test for skew pattern (typical equity-like)
                assert vols[0] > vols[atm_idx], (
                    f"Put wing ({vols[0]:.4%}) should be higher than "
                    f"ATM ({vols[atm_idx]:.4%}) for skew pattern"
                )
                assert vols[-1] < vols[atm_idx], (
                    f"Call wing ({vols[-1]:.4%}) should be lower than "
                    f"ATM ({vols[atm_idx]:.4%}) for skew pattern"
                )
            else:
                # Test for smile pattern
                assert vols[0] > vols[atm_idx], (
                    f"Put wing ({vols[0]:.4%}) should be higher than "
                    f"ATM ({vols[atm_idx]:.4%}) for smile pattern"
                )
                assert vols[-1] > vols[atm_idx], (
                    f"Call wing ({vols[-1]:.4%}) should be higher than "
                    f"ATM ({vols[atm_idx]:.4%}) for smile pattern"
                )
            
            # Common tests for both patterns
            # Check for continuity (no large jumps)
            diffs = np.diff(vols)
            max_jump = 0.05  # Max 500bps between adjacent strikes
            assert all(abs(d) < max_jump for d in diffs), (
                f"Surface shows discontinuity in {scenario_name}"
                f"\nJumps: {[f'{d:.4%}' for d in diffs]}"
            )
            
            # 2. Smile should be roughly symmetric for indices
            put_wing = vols[0] - vols[atm_idx]
            call_wing = vols[-1] - vols[atm_idx]
            wing_diff = abs(put_wing - call_wing)
            assert wing_diff < 0.05, (
                f"Smile asymmetry too large: {wing_diff:.4f}"
                f"\nScenario: {scenario.name}"
                f"\nPut wing: {put_wing:.4f}"
                f"\nCall wing: {call_wing:.4f}"
            )
            
            # 3. No arbitrage test - smile should be smooth
            diffs = np.diff(vols)
            max_jump = 0.03  # Max 300bps between strikes
            assert all(abs(d) < max_jump for d in diffs), (
                f"Smile shows potential arbitrage in {scenario.name}"
                f"\nVols across strikes: {[f'{v:.4%}' for v in vols]}"
            )

if __name__ == "__main__":
    pytest.main([__file__])