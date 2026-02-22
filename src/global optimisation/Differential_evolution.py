#The key idea is to generate a population of candidate solutions through successive generations using vector differences for deviations.
#  This approach allows DE to adapt its search methodology to the objective function 

import numpy as np
from scipy.optimize import differential_evolution
from svi.implementation.svi_model import SVI


def svi_objective(params, k, market_iv, T):
    a, b, rho, m, sigma = params

    # basic parameter constraints
    if b <= 0 or sigma <= 0 or abs(rho) >= 1:
        return 1e6  # penalise invalid regions

    model = SVI(a, b, rho, m, sigma)
    model_iv = model.svi_implied_vol(k, T)

    return np.mean((model_iv - market_iv) ** 2)



def fit_svi_de(k, market_iv, T):

    bounds = [
        (0.0, 1.0),      # a
        (0.0, 5.0),      # b
        (-0.999, 0.999), # rho
        (-1.0, 1.0),     # m
        (0.001, 2.0)     # sigma
    ]

    result = differential_evolution(
        svi_objective,
        bounds=bounds,
        args=(k, market_iv, T),
        strategy='best1bin',
        maxiter=500,
        popsize=15,
        tol=1e-6,
        polish=True
    )

    return result.x
