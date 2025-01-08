import pytest
import numpy as np
from volsurface.core.black_scholes import BlackScholes, ImpliedVolatility, BSMInputs, OptionType

def test_bsm_inputs_validation():
    """Test BSMInputs validation"""
    with pytest.raises(ValueError):
        BSMInputs(S=-100, K=100, T=1, r=0.05, sigma=0.2).validate()
    with pytest.raises(ValueError):
        BSMInputs(S=100, K=100, T=-1, r=0.05, sigma=0.2).validate()
    with pytest.raises(ValueError):
        BSMInputs(S=100, K=100, T=1, r=0.05, sigma=-0.2).validate()

def test_black_scholes_put_call_parity():
    """Test put-call parity relationship"""
    inputs = BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.2)
    
    call_price = BlackScholes.price(inputs, OptionType.CALL)
    put_price = BlackScholes.price(inputs, OptionType.PUT)
    
    # Put-call parity: C - P = S - K*e^(-rT)
    lhs = call_price - put_price
    rhs = inputs.S - inputs.K * np.exp(-inputs.r * inputs.T)
    
    assert np.abs(lhs - rhs) < 1e-10

def test_implied_vol_recovers_input():
    """Test that implied vol calculation recovers the input volatility"""
    inputs = BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.2)
    true_price = BlackScholes.price(inputs, OptionType.CALL)
    
    # Calculate implied vol from the price
    iv_calc = ImpliedVolatility(tolerance=1e-6)
    implied_vol, _ = iv_calc.calculate(true_price, inputs, OptionType.CALL)
    
    assert np.abs(implied_vol - inputs.sigma) < 1e-4

def test_vega_positivity():
    """Test that vega is always positive"""
    inputs = BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.2)
    vega = BlackScholes.vega(inputs)
    assert vega > 0

@pytest.mark.parametrize("S,K,T,r,sigma,expected_d1", [
    (100, 100, 1, 0.05, 0.2, 0.35),  # Updated from 0.15
    (100, 90, 1, 0.05, 0.2, 0.8768),  # Updated from 0.6614
    (100, 110, 1, 0.05, 0.2, -0.1266)  # Updated from -0.3614
])
def test_d1_calculation(S, K, T, r, sigma, expected_d1):
    """Test d1 calculation with known values"""
    inputs = BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma)
    d1 = BlackScholes.d1(inputs)
    assert np.abs(d1 - expected_d1) < 1e-4

def test_implied_vol_extreme_cases():
    """Test implied volatility calculation for extreme cases"""
    inputs = BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.2)
    iv_calc = ImpliedVolatility()
    
    # Deep ITM call should have high implied vol
    deep_itm_price = BlackScholes.price(
        BSMInputs(S=100, K=50, T=1, r=0.05, sigma=0.5),
        OptionType.CALL
    )
    implied_vol, _ = iv_calc.calculate(deep_itm_price, 
                                     BSMInputs(S=100, K=50, T=1, r=0.05, sigma=0.2),
                                     OptionType.CALL)
    assert implied_vol > 0.4  # Should be close to 0.5