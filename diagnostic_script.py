import numpy as np
import src.volsurface.models.sabr as sabr_module

def detailed_sabr_diagnostics():
    # Original parameters
    true_params = sabr_module.SABRParameters(
        alpha=0.2, beta=0.5, rho=-0.2, nu=0.4
    )
    true_model = sabr_module.SABRModel(true_params)

    # Diagnostics for different strikes and maturities
    forward = 100
    maturities = [0.25, 0.5, 1.0, 2.0]
    strike_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]

    print("DETAILED SABR MODEL DIAGNOSTICS")
    print("==============================")

    # Volatility grid
    print("\nVolatility Grid:")
    vol_grid = []
    for T in maturities:
        row = []
        for k_ratio in strike_ratios:
            strike = forward * k_ratio
            vol = true_model.implied_volatility(forward, strike, T)
            row.append(vol)
            print(f"T={T}, K={strike}: Vol = {vol}")
        vol_grid.append(row)
    
    vol_grid = np.array(vol_grid)

    # Detailed parameter analysis
    print("\nModel Parameter Analysis:")
    print(f"Alpha: {true_params.alpha}")
    print(f"Beta:  {true_params.beta}")
    print(f"Rho:   {true_params.rho}")
    print(f"Nu:    {true_params.nu}")

    # Calibration test setup
    forwards = np.array([100, 100, 100])
    strikes = np.array([90, 100, 110])
    maturities = np.array([1.0, 1.0, 1.0])

    market_vols = np.array([
        true_model.implied_volatility(f, k, t)
        for f, k, t in zip(forwards, strikes, maturities)
    ])

    print("\nMarket Vols for Calibration:")
    print(market_vols)

    # Initial calibration guess
    initial_guess = sabr_module.SABRParameters(
        alpha=0.25, beta=0.6, rho=0.0, nu=0.3
    )
    model = sabr_module.SABRModel(initial_guess)

    print("\nInitial Calibration Guess:")
    print(f"Alpha: {initial_guess.alpha}")
    print(f"Beta:  {initial_guess.beta}")
    print(f"Rho:   {initial_guess.rho}")
    print(f"Nu:    {initial_guess.nu}")

    # Attempt calibration
    try:
        calibrated_params = model.calibrate(
            market_vols, strikes, forwards, maturities, initial_guess
        )
        
        print("\nCalibration Results:")
        print(f"Calibrated Alpha: {calibrated_params.alpha}")
        print(f"Calibrated Beta:  {calibrated_params.beta}")
        print(f"Calibrated Rho:   {calibrated_params.rho}")
        print(f"Calibrated Nu:    {calibrated_params.nu}")
    except Exception as e:
        print("\nCalibration Failed:")
        print(str(e))

if __name__ == '__main__':
    detailed_sabr_diagnostics()