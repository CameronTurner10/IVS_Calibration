# SVI fitting using scipy - local optimisation methods
# Goal: find the 5 params (a, b, rho, m, sigma) that best match market data
import numpy as np
from scipy.optimize import minimize
from src.svi.implementation.svi_model import SVI

# Bounds for the 5 SVI parameters - shared across all local methods
SVI_BOUNDS = [
    (0.001, 1.0),    # a
    (0.001, 0.99),   # b
    (-0.99, 0.99),   # rho
    (-0.5,  0.5),    # m
    (0.01,  1.0),    # sigma
]

def total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_objective(parameters, k, w_market): #Objective function shared by all local methods - minimises sum of squared errors.
    a, b, rho, m, sigma = parameters
    w_model = total_variance(k, a, b, rho, m, sigma)
    return np.sum((w_model - w_market) ** 2)

#below are the different local optimisers

def fit_svi_slsqp(strikes, market_vols, T, forward): #Sequential Least Squares Programming
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    atm_var = np.mean(w_market)
    x0 = [atm_var, 0.1, 0.0, 0.0, 0.1]  # [a, b, rho, m, sigma]

    result = minimize(
        svi_objective,
        x0,
        args=(k_values, w_market),
        method='SLSQP',
        bounds=SVI_BOUNDS
    )

    return dict(zip(['a', 'b', 'rho', 'm', 'sigma'], result.x))


def fit_svi_trust_region(strikes, market_vols, T, forward): #Trust Region Constrained optimisation
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    atm_var = np.mean(w_market)
    x0 = [atm_var, 0.1, 0.0, 0.0, 0.1]  # [a, b, rho, m, sigma]

    result = minimize(
        svi_objective,
        x0,
        args=(k_values, w_market),
        method="trust-constr",
        bounds=SVI_BOUNDS
    )

    return dict(zip(["a", "b", "rho", "m", "sigma"], result.x))


def fit_svi_cobyqa(strikes, market_vols, T, forward): #Constrained Optimisation BY Quadratic Approximations
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    atm_var = np.mean(w_market)
    x0 = [atm_var, 0.1, 0.0, 0.0, 0.1]  # [a, b, rho, m, sigma]

    result = minimize(
        svi_objective,
        x0,
        args=(k_values, w_market),
        method='COBYQA',
        bounds=SVI_BOUNDS
    )

    return dict(zip(["a", "b", "rho", "m", "sigma"], result.x))


# register new local optimisers here
# To add a new local method: write the function above, then add it below

LOCAL_METHODS = {
    "slsqp": fit_svi_slsqp,
    "trust_region": fit_svi_trust_region,
    "cobyqa": fit_svi_cobyqa,
}

# Backwards-compatible alias so plotting.py and other existing files don't need to change
fit_svi_slice = fit_svi_cobyqa
