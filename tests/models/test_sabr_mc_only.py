#import pytest
#import numpy as np
#import QuantLib as ql
#from typing import Tuple, List
#import scipy.stats
#
#from volsurface.models.sabr_mc_only import SABRParameters, MonteCarloConfig, SABRModel
#
#from volsurface.core.black_scholes import BlackScholes, ImpliedVolatility, BSMInputs, OptionType
#
#@pytest.fixture
#def base_params() -> SABRParameters:
#    """Standard SABR parameters for testing"""
#    return SABRParameters(
#        alpha=0.2,  # Initial volatility
#        beta=0.5,   # CEV parameter
#        rho=-0.3,   # Correlation
#        nu=0.4      # Vol of vol
#    )
#
#@pytest.fixture
#def test_configs() -> List[MonteCarloConfig]:
#    """Different MC configurations for testing"""
#    return [
#        MonteCarloConfig(n_paths=1000, steps_per_year=52, seed=42, antithetic=True),
#        MonteCarloConfig(n_paths=10000, steps_per_year=252, seed=42, antithetic=True),
#    ]
#
#@pytest.fixture
#def ql_sabr_engine():
#    """Create QuantLib SABR engine for comparison"""
#    def create_engine(alpha: float, beta: float, nu: float, rho: float):
#        def get_vol(strike: float, forward: float = 100.0, T: float = 1.0) -> float:
#            return ql.sabrVolatility(
#                strike,    # strike
#                forward,   # forward
#                T,        # expiry time
#                alpha,    # alpha
#                beta,     # beta
#                nu,       # nu
#                rho       # rho
#            )
#        return get_vol
#    return create_engine
#
#class TestSABRModel:
#    @pytest.mark.parametrize("moneyness", [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
#    def test_vs_quantlib(self, base_params, test_configs, ql_sabr_engine, moneyness):
#        """Compare implied vols against QuantLib for standard moneyness range"""
#        F0 = 100.0
#        K = F0 * moneyness
#        T = 1.0
#        r = 0.02
#
#        # Get QuantLib reference value
#        ql_vol_func = ql_sabr_engine(
#            base_params.alpha,
#            base_params.beta,
#            base_params.nu,
#            base_params.rho
#        )
#        ql_vol = ql_vol_func(K, F0, T)
#        
#        # Get our model's value
#        model = SABRModel(base_params, test_configs[-1])
#        our_vol, metadata = model.implied_volatility(F0, K, T, r, OptionType.CALL)
#        
#        # Standard tolerance for practical strikes
#        tol = 0.002  # 20 bps
#        
#        assert metadata['convergence'], f"Failed to converge for moneyness {moneyness}"
#        assert abs(our_vol - ql_vol) < tol, \
#            f"Vol difference too large at K={K}: MC={our_vol:.4f}, QL={ql_vol:.4f}"
#
#    def test_parameter_validation(self):
#        """Test parameter validation"""
#        invalid_params = [
#            {'alpha': 0.2, 'beta': 1.5, 'rho': 0, 'nu': 0.4},    # beta > 1
#            {'alpha': 0.2, 'beta': -0.1, 'rho': 0, 'nu': 0.4},   # beta < 0
#            {'alpha': 0.2, 'beta': 0.5, 'rho': 1.5, 'nu': 0.4},  # |rho| > 1
#            {'alpha': -0.2, 'beta': 0.5, 'rho': 0, 'nu': 0.4},   # negative alpha
#            {'alpha': 0.2, 'beta': 0.5, 'rho': 0, 'nu': -0.4},   # negative nu
#        ]
#        
#        for params in invalid_params:
#            with pytest.raises(ValueError):
#                SABRParameters(**params).validate()
#
#    def test_surface_shape(self, base_params, test_configs):
#        """Test basic properties of vol surface"""
#        model = SABRModel(base_params, test_configs[-1])
#        F0 = 100.0
#        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
#        maturities = np.array([0.5, 1.0])
#        r = 0.02
#        
#        vols, _ = model.volatility_surface(F0, strikes, maturities, r)
#        
#        # Test volatility smile shape (should be convex)
#        for t_idx in range(len(maturities)):
#            vol_slice = vols[t_idx]
#            # Check convexity by computing second differences
#            second_diffs = np.diff(vol_slice, 2)
#            assert np.all(second_diffs > -0.001), "Volatility smile should be approximately convex"
#
#    @pytest.mark.parametrize("maturity", [0.5, 1.0])
#    def test_path_properties(self, base_params, test_configs, maturity):
#        """Test basic statistical properties of simulated paths"""
#        model = SABRModel(base_params, test_configs[0])  # Use faster config for path tests
#        F0 = 100.0
#        
#        F_paths, alpha_paths = model._simulate_paths(F0, maturity, return_vol=True)
#        
#        # Test no negative values
#        assert np.all(F_paths >= 0), "Forward prices should be non-negative"
#        assert np.all(alpha_paths >= 0), "Volatility should be non-negative"
#        
#        # Test initial values
#        assert np.allclose(F_paths[:, 0], F0), "Initial forward price should match F0"
#        assert np.allclose(alpha_paths[:, 0], base_params.alpha), "Initial vol should match alpha"
#
#    def test_convergence(self, base_params):
#        """Test MC convergence with increasing paths"""
#        F0, K, T, r = 100.0, 100.0, 1.0, 0.02
#        
#        configs = [
#            MonteCarloConfig(n_paths=1000, steps_per_year=52, seed=42),
#            MonteCarloConfig(n_paths=10000, steps_per_year=52, seed=42),
#        ]
#        
#        prices = []
#        for config in configs:
#            model = SABRModel(base_params, config)
#            price = model.price_option(F0, K, T, r, OptionType.CALL)
#            prices.append(price)
#        
#        # Check convergence
#        diff = abs(prices[1] - prices[0])
#        assert diff < 0.1, "Price should converge with increasing paths"
#    
#    def test_sabr_prices(self, base_params, test_configs, ql_sabr_engine):
#        """Test price levels directly before implied vol calculation"""
#        F0 = 100.0
#        moneyness_levels = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
#        T = 1.0
#        r = 0.02
#        
#        ql_engine = ql_sabr_engine(
#            base_params.alpha,
#            base_params.beta,
#            base_params.nu,
#            base_params.rho
#        )
#        
#        model = SABRModel(base_params, test_configs[-1])
#        
#        print("\nPrice comparison:")
#        print("Moneyness | MC Price | QL Vol | BS Price")
#        print("-" * 40)
#        
#        for m in moneyness_levels:
#            K = F0 * m
#            mc_price = model.price_option(F0, K, T, r, OptionType.CALL)
#            ql_vol = ql_engine(K, F0, T)
#            bs_price = BlackScholes.price(
#                BSMInputs(S=F0, K=K, T=T, r=r, sigma=ql_vol),
#                OptionType.CALL
#            )
#            print(f"{m:4.2f} | {mc_price:8.4f} | {ql_vol:7.4f} | {bs_price:8.4f}")
#        
#        
#    def test_path_statistics(self, base_params, test_configs):
#        """Test the statistical properties of the simulated paths"""
#        F0 = 100.0
#        T = 1.0
#        model = SABRModel(base_params, test_configs[-1])
#        
#        # Generate paths
#        F_paths, alpha_paths = model._simulate_paths(F0, T, return_vol=True)
#        
#        # Analyze final values
#        F_T = F_paths[:, -1]
#        alpha_T = alpha_paths[:, -1]
#        
#        print("\nPath statistics at T:")
#        print(f"Forward price:")
#        print(f"  Mean: {np.mean(F_T):.4f} (should be close to {F0})")
#        print(f"  Std:  {np.std(F_T):.4f}")
#        print(f"  Min:  {np.min(F_T):.4f}")
#        print(f"  Max:  {np.max(F_T):.4f}")
#        print(f"  Skew: {scipy.stats.skew(F_T):.4f}")
#        
#        print(f"\nVolatility:")
#        print(f"  Mean: {np.mean(alpha_T):.4f} (started at {base_params.alpha})")
#        print(f"  Std:  {np.std(alpha_T):.4f}")
#        print(f"  Min:  {np.min(alpha_T):.4f}")
#        print(f"  Max:  {np.max(alpha_T):.4f}")
#        
#        # Also look at path evolution
#        print("\nPath evolution:")
#        time_points = [0, int(F_paths.shape[1]/4), int(F_paths.shape[1]/2), -1]
#        for t in time_points:
#            F_t = F_paths[:, t]
#            alpha_t = alpha_paths[:, t]
#            print(f"\nAt time step {t}:")
#            print(f"F mean: {np.mean(F_t):.4f}, std: {np.std(F_t):.4f}")
#            print(f"α mean: {np.mean(alpha_t):.4f}, std: {np.std(alpha_t):.4f}")
#            
#    def test_ql_sabr_values(self, base_params, ql_sabr_engine):
#        """Verify QuantLib SABR implementation is giving sensible values"""
#        F0 = 100.0
#        T = 1.0
#        moneyness_levels = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
#        
#        ql_engine = ql_sabr_engine(
#            base_params.alpha,
#            base_params.beta,
#            base_params.nu,
#            base_params.rho
#        )
#        
#        print("\nQuantLib SABR values:")
#        print("Moneyness | Strike |   Vol   |")
#        print("-" * 35)
#        
#        for m in moneyness_levels:
#            K = F0 * m
#            vol = ql_engine(K, F0, T)
#            print(f"{m:9.2f} | {K:6.1f} | {vol:.4f} |")
#
#    def test_path_generation_details(self, base_params):
#        """Examine the details of path generation"""
#        import scipy.stats  # Add at top of file if not present
#        
#        F0 = 100.0
#        T = 1.0
#        model = SABRModel(base_params, MonteCarloConfig(n_paths=1000, steps_per_year=252, seed=42))
#        
#        # Generate paths
#        F, alpha = model._simulate_paths(F0, T, return_vol=True)
#        
#        # Examine evolution at key timepoints
#        check_points = [0, 63, 126, 251]  # start, quarter, half, end
#        
#        for t in check_points:
#            F_t = F[:, t]
#            alpha_t = alpha[:, t]
#            print(f"\nAt time step {t}:")
#            print(f"Forward:")
#            print(f"  Mean: {np.mean(F_t):.4f}")
#            print(f"  Std:  {np.std(F_t):.4f}")
#            print(f"  Skew: {scipy.stats.skew(F_t):.4f}")
#            print(f"  Kurt: {scipy.stats.kurtosis(F_t):.4f}")
#            print(f"Alpha:")
#            print(f"  Mean: {np.mean(alpha_t):.4f}")
#            print(f"  Std:  {np.std(alpha_t):.4f}")
#        
#        # Check increments
#        dW, dZ = model._generate_correlated_paths(T)
#        print("\nBrownian increment properties:")
#        print(f"dW mean: {np.mean(dW):.6f}, std: {np.std(dW):.6f}")
#        print(f"dZ mean: {np.mean(dZ):.6f}, std: {np.std(dZ):.6f}")
#        print(f"Correlation: {np.corrcoef(dW.flatten(), dZ.flatten())[0,1]:.4f}")
#    
#    def test_brownian_increments(self, base_params):
#        """Test properties of generated Brownian motion increments"""
#        model = SABRModel(base_params)
#        T = 1.0
#        dW, dZ = model._generate_correlated_paths(T)
#        
#        print("\nBrownian increment properties:")
#        print(f"dW mean: {np.mean(dW):.6f} (should be ≈ 0)")
#        print(f"dW std: {np.std(dW):.6f} (should be ≈ sqrt(dt))")
#        print(f"dZ mean: {np.mean(dZ):.6f} (should be ≈ 0)")
#        print(f"dZ std: {np.std(dZ):.6f} (should be ≈ sqrt(dt))")
#        print(f"Correlation: {np.corrcoef(dW.flatten(), dZ.flatten())[0,1]:.4f} (should be ≈ {base_params.rho})")
#
#    def test_volatility_evolution(self, base_params):
#        """Test volatility process evolution"""
#        model = SABRModel(base_params)
#        T = 1.0
#        _, alpha = model._simulate_paths(100.0, T, return_vol=True)
#        
#        print("\nVolatility process properties:")
#        print(f"Initial α: {np.mean(alpha[:,0]):.6f} (should be {base_params.alpha})")
#        print(f"Terminal α mean: {np.mean(alpha[:,-1]):.6f} (should be ≈ {base_params.alpha})")
#        print(f"Terminal α std: {np.std(alpha[:,-1]):.6f} (should be ≈ {base_params.alpha * base_params.nu * np.sqrt(T)})")
#
#    def test_forward_evolution(self, base_params):
#        """Test forward price evolution"""
#        model = SABRModel(base_params)
#        T = 1.0
#        F0 = 100.0
#        F, _ = model._simulate_paths(F0, T, return_vol=False)
#        
#        # For β=0.5, local vol should be α*sqrt(F)
#        print("\nForward process properties:")
#        print(f"Initial F: {np.mean(F[:,0]):.6f} (should be {F0})")
#        print(f"Terminal F mean: {np.mean(F[:,-1]):.6f} (should be ≈ {F0})")
#        
#        # Theoretical variance calculation for CEV
#        theoretical_std = None  # Need to derive this
#        print(f"Terminal F std: {np.std(F[:,-1]):.6f} (should be ≈ {theoretical_std})")