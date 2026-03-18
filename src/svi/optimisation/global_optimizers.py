# SVI fitting using scipy - global optimisation methods
# Goal: find the 5 params (a, b, rho, m, sigma) that best match market data
# Global methods explore the full parameter space and are less sensitive to the
# choice of initial guess compared to local methods.
import numpy as np
from scipy.optimize import differential_evolution, shgo, basinhopping, dual_annealing
from src.svi.implementation.svi_model import SVI

# Note: 'a' upper bound is typically set to max(w_market) dynamically during fit
SVI_BOUNDS = [
    (1e-8, 10.0),    # a (placeholder upper bound, usually overridden)
    (1e-8, 5.0),     # b
    (-0.99, 0.99),   # rho
    (-10.0, 10.0),   # m
    (0.0, 10.0),     # sigma
]

def total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_objective(parameters, k, w_market): # Objective function shared by all global methods - minimises sum of squared errors.
    a, b, rho, m, sigma = parameters
    w_model = total_variance(k, a, b, rho, m, sigma)
    return np.sum((w_model - w_market) ** 2)


# below are the different global optimisers

def fit_svi_de(strikes, market_vols, T, forward): # Differential Evolution doesnt need initial guess

    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    bounds = list(SVI_BOUNDS)
    bounds[0] = (1e-8, float(np.max(w_market)))

    result = differential_evolution(
        svi_objective,
        bounds=bounds,
        args=(k_values, w_market),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=1e-7,
        polish=True,
        seed=42, # for reproducibility 
    )

    return dict(zip(["a", "b", "rho", "m", "sigma"], result.x))


def fit_svi_shgo(strikes, market_vols, T, forward): # Simplicial Homology Global Optimisation
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    bounds = list(SVI_BOUNDS)
    bounds[0] = (1e-8, float(np.max(w_market)))

    result = shgo(
        svi_objective,
        bounds=bounds,
        args=(k_values, w_market),
        n=200, # number of sampling points per iteration
        iters=3, # number of iterations of the homology growth
        sampling_method='simplicial',
    )

    return dict(zip(["a", "b", "rho", "m", "sigma"], result.x))


def fit_svi_basinhopping(strikes, market_vols, T, forward): # Basin Hopping with SLSQP local optimiser
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    atm_var = np.mean(w_market)
    x0 = [atm_var, 0.1, 0.0, 0.0, 0.1]  # [a, b, rho, m, sigma]

    bounds = list(SVI_BOUNDS)
    bounds[0] = (1e-8, float(np.max(w_market)))

    minimizer_kwargs = {
        "method": "SLSQP",
        "args": (k_values, w_market),
        "bounds": bounds,
    }

    result = basinhopping(
        svi_objective,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=200, # number of basin-hopping iterations
        stepsize=0.05, # step size for the random perturbation
        seed=42, # for reproducibility
    )

    return dict(zip(["a", "b", "rho", "m", "sigma"], result.x))


def fit_svi_dual_annealing(strikes, market_vols, T, forward): # Dual Annealing
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    bounds = list(SVI_BOUNDS)
    bounds[0] = (1e-8, float(np.max(w_market)))

    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
    }

    result = dual_annealing(
        svi_objective,
        bounds=bounds,
        args=(k_values, w_market),
        maxiter=1000,
        minimizer_kwargs=minimizer_kwargs,
        seed=42, # for reproducibility
    )

    return dict(zip(["a", "b", "rho", "m", "sigma"], result.x))


# register new global optimisers here
# To add a new global method: write the function above, then add it below

GLOBAL_METHODS = {
    "de": fit_svi_de,
    "shgo": fit_svi_shgo,
    "basinhopping": fit_svi_basinhopping,
    "dual_annealing": fit_svi_dual_annealing,
}
