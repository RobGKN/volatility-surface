import numpy as np
from scipy.stats import norm
from numba import jit
import math
import time
import QuantLib as ql

class SABRQuantLibValidator:
    def __init__(self, alpha, beta, rho, nu):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        
        print("\nInitializing SABR Model with QuantLib validation")
        print(f"α (alpha) = {alpha:.4f}: Initial volatility level")
        print(f"β (beta) = {beta:.4f}: CEV parameter")
        print(f"ρ (rho) = {rho:.4f}: Price-volatility correlation")
        print(f"ν (nu) = {nu:.4f}: Volatility of volatility")

    def price_with_quantlib(self, S, K, T, r, is_call=True):
        """
        Price option using QuantLib's SABR implementation
        """
        print("\nCalculating QuantLib SABR price")
        
        # Set up QuantLib dates
        calendar = ql.NullCalendar()
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        
        # Calculate forward price
        F = S * math.exp(r * T)
        print(f"Forward price = {F:.4f}")
        
        # Calculate SABR implied volatility
        try:
            impl_vol = ql.sabrVolatility(K, F, T, 
                                       self.alpha, self.beta, 
                                       self.nu, self.rho)
            
            print(f"SABR implied volatility from QuantLib = {impl_vol:.4f}")
            
            # Set up the option for pricing
            riskFreeTS = ql.YieldTermStructureHandle(
                ql.FlatForward(today, r, ql.Actual365Fixed()))
            
            dividendTS = ql.YieldTermStructureHandle(
                ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
            
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
            
            # Create volatility surface
            volTS = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, calendar, impl_vol, ql.Actual365Fixed())
            )
            
            # Set up the process
            process = ql.BlackScholesMertonProcess(spot_handle, dividendTS,
                                                 riskFreeTS, volTS)
            
            # Create the option
            maturity = today + int(T * 365)
            exercise = ql.EuropeanExercise(maturity)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
            option = ql.VanillaOption(payoff, exercise)
            
            # Set the pricing engine
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)
            
            # Calculate the price
            price = option.NPV()
            print(f"QuantLib option price = {price:.4f}")
            
            # Calculate Greeks if needed
            delta = option.delta()
            gamma = option.gamma()
            vega = option.vega()
            theta = option.theta()
            
            print("\nGreeks:")
            print(f"Delta: {delta:.4f}")
            print(f"Gamma: {gamma:.4f}")
            print(f"Vega:  {vega:.4f}")
            print(f"Theta: {theta:.4f}")
            
            return price, impl_vol
            
        except RuntimeError as e:
            print(f"QuantLib calculation failed: {str(e)}")
            return None, None

    def price_option_mc(self, S, K, T, r, n_paths=10000, n_steps=252):
        """Monte Carlo simulation price"""
        print(f"\nRunning Monte Carlo simulation with {n_paths} paths")
        start_time = time.time()
        
        # Calculate forward
        F = S * math.exp(r*T)
        
        # Run simulation
        F_paths, _ = self._simulate_sabr_paths(
            F, self.alpha, self.beta, self.nu, self.rho, T, n_steps, n_paths
        )
        
        # Calculate payoffs
        payoffs = np.maximum(F_paths[:, -1] * math.exp(-r*T) - K, 0)
        
        # Get price and error estimates
        price = np.mean(payoffs)
        se = np.std(payoffs) / np.sqrt(n_paths)
        
        elapsed = time.time() - start_time
        print(f"Monte Carlo completed in {elapsed:.2f} seconds")
        print(f"Monte Carlo price = {price:.4f} (±{2*se:.4f} at 95% confidence)")
        
        return price, se

    @staticmethod
    @jit(nopython=True)
    def _simulate_sabr_paths(F0, alpha0, beta, nu, rho, T, n_steps, n_paths):
        """Simulate SABR paths using Euler scheme"""
        dt = T/n_steps
        sqrt_dt = np.sqrt(dt)
        
        F = np.zeros((n_paths, n_steps + 1))
        alpha = np.zeros((n_paths, n_steps + 1))
        
        F[:, 0] = F0
        alpha[:, 0] = alpha0
        
        for i in range(n_steps):
            dW1 = np.random.normal(0, sqrt_dt, n_paths)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, sqrt_dt, n_paths)
            
            F[:, i+1] = F[:, i] + alpha[:, i] * F[:, i]**beta * dW1
            alpha[:, i+1] = alpha[:, i] * np.exp(nu * dW2 - 0.5 * nu**2 * dt)
            
            # Absorbing boundary at zero
            F[:, i+1] = np.maximum(F[:, i+1], 0)
            
        return F, alpha

    def compare_all_methods(self, S, K, T, r):
        """Compare all available pricing methods"""
        print(f"\nComparing all pricing methods:")
        print(f"S = {S:.2f}, K = {K:.2f}, T = {T:.2f}, r = {r:.4f}")
        
        results = {}
        
        # QuantLib price
        ql_price, ql_vol = self.price_with_quantlib(S, K, T, r)
        if ql_price is not None:
            results['quantlib'] = {'price': ql_price, 'vol': ql_vol}
        
        # Monte Carlo price
        mc_price, mc_se = self.price_option_mc(S, K, T, r)
        results['monte_carlo'] = {'price': mc_price, 'std_error': mc_se}
        
        print("\nMethod Comparison:")
        print(f"{'Method':<15} {'Price':>10} {'Impl Vol':>10}")
        print("-" * 35)
        if ql_price is not None:
            print(f"{'QuantLib'::<15} {ql_price:>10.4f} {ql_vol:>10.4f}")
        else:
            print(f"{'QuantLib'::<15} {'Failed':>10} {'N/A':>10}")
        print(f"{'Monte Carlo'::<15} {mc_price:>10.4f} {'N/A':>10}")
        
        return results

if __name__ == "__main__":
    # Your parameters
    S = 100     # Stock price
    K = 110     # Strike
    T = 1.0     # Time to maturity
    r = 0.02    # Risk-free rate
    
    # SABR parameters
    alpha = 0.2  # Initial volatility
    beta = 0.5   # CEV parameter
    rho = 0.3    # Correlation
    nu = 0.4     # Vol of vol
    
    # Create validator and compare methods
    validator = SABRQuantLibValidator(alpha, beta, rho, nu)
    results = validator.compare_all_methods(S, K, T, r)