# SVI fitting using scipy
# Goal: find the 5 params (a, b, rho, m, sigma) that best match market data
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from src.svi.implementation.svi_model import SVI
from src.svi.optimisation.constraints import c_nonneg_min_total_var, c_wing_right, c_wing_left, c_butterfly_grid 

def total_variance(k, a, b, rho, m, sigma): 
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi_slice(strikes, market_vols, T, forward):

   # Convert to log-moneyness and total variance
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T
    
    # Starting guess - just reasonable values to start from
    atm_var = np.mean(w_market)
    x0 = [atm_var, 0.1, 0.0, 0.0, 0.1]  # [a, b, rho, m, sigma]
    
    # Bounds to prevent arbitrage
    bounds = [
        (0.001, 1.0),      # a
        (0.001, 2.0),      # b
        (-0.99, 0.99),     # rho
        (-0.5, 0.5),       # m
        (0.01, 1.0),       # sigma
    ]

    def svi_slice_objective(parameters, k, w_market):
        a, b, rho, m, sigma = parameters
        w_model = total_variance(k, a, b, rho, m, sigma)
        return np.sum((w_model - w_market) ** 2)

    # Builds a grid that covers observed k and extends a bit to discourage edge arbitrage
    k_min, k_max = float(np.min(k_values)), float(np.max(k_values))
    pad = 0.5  # can be tuned (between 0.25 and 1.0 is common)
    k_grid = np.linspace(k_min - pad, k_max + pad, 201)

    constraints = [
        {'type': 'ineq', 'fun': c_nonneg_min_total_var},
        {'type': 'ineq', 'fun': c_wing_right},
        {'type': 'ineq', 'fun': c_wing_left},
        {'type': 'ineq', 'fun': c_butterfly_grid, 'args': (k_grid, 1e-8)},
    ]    

    result = minimize(             # Minimisation of objective function
        svi_slice_objective,       # Objective function f(k,parameters)
        x0,                        # Initial Guess Parameters [a, b, rho, m, sigma]
        args=(k_values, w_market), # Extra arguments for the objective function 
        method='SLSQP',            # Method of Optimisation
        bounds=bounds,             # Bounds for numerical stability 
        constraints=constraints    # Inequality constraints (basic no-arbitrage conditions) 
    )
    
    return dict(zip(['a','b','rho','m','sigma'], result.x))


def fit_svi_surface(strikes, maturities, total_variances):
    # TODO: loop over maturities and call fit_svi_slice for each
    raise NotImplementedError("Do slice fitting first")