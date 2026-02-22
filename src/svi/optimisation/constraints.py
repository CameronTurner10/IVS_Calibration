# src/svi/optimisation/constraints.py
import numpy as np

def c_nonneg_min_total_var(params):
    # a + b*sigma*sqrt(1-rho^2) >= 0
    a, b, rho, m, sigma = params
    return a + b * sigma * np.sqrt(max(0.0, 1.0 - rho*rho))

def c_wing_right(params):
    # 2 - b(1+rho) >= 0
    a, b, rho, m, sigma = params
    return 2.0 - b * (1.0 + rho)

def c_wing_left(params):
    # 2 - b(1-rho) >= 0
    a, b, rho, m, sigma = params
    return 2.0 - b * (1.0 - rho)
