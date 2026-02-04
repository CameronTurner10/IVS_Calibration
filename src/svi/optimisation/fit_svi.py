# SVI fitting using scipy
# Goal: find the 5 params (a, b, rho, m, sigma) that best match market data

import numpy as np
from scipy.optimize import minimize
from src.svi.implementation.svi_model import SVI


def fit_svi_slice(strikes, market_vols, T, forward):
    """
    Fit SVI to a single time slice. Returns dict of fitted params.
    """
    # Convert to log-moneyness and total variance
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T
    
    # Starting guess - just reasonable values to start from
    atm_var = np.mean(w_market)
    x0 = [atm_var, 0.1, 0.0, 0.0, 0.1]  # [a, b, rho, m, sigma]
    
    # Bounds to prevent arbitrage (b >= 0, |rho| < 1, sigma > 0)
    bounds = [
        (0.001, 1.0),      # a
        (0.001, 2.0),      # b
        (-0.99, 0.99),     # rho
        (-0.5, 0.5),       # m
        (0.01, 1.0),       # sigma
    ]
    
    def objective(params):
        # TODO A2: calculate squared error between model and market
        # svi = SVI(*params)
        # w_model = svi.total_variance(k_values)
        # return np.sum((w_model - w_market) ** 2)
        pass
    
    # TODO A2: use scipy.optimize.minimize with SLSQP method
    # result = minimize(objective, x0, method='SLSQP', bounds=bounds)
    # return dict(zip(['a','b','rho','m','sigma'], result.x))
    
    raise NotImplementedError("Not done yet")


def fit_svi_surface(strikes, maturities, total_variances):
    # TODO: loop over maturities and call fit_svi_slice for each
    raise NotImplementedError("Do slice fitting first")