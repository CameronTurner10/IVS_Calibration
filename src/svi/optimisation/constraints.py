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

def _w_wp_wpp(k, params):
    """
    Returns w(k), w'(k), w''(k) for the raw SVI total variance:
    w(k)=a+b*(rho*(k-m)+sqrt((k-m)^2+sigma^2))
    """
    a, b, rho, m, sigma = params
    x = k - m
    s = np.sqrt(x*x + sigma*sigma)
    w = a + b * (rho * x + s)
    wp = b * (rho + x / s)
    wpp = b * (sigma*sigma) / (s**3)
    return w, wp, wpp

def g_butterfly(k, params):
    """
    Butterfly condition function g(k). 
    No butterfly arbitrage requires g(k) >= 0.
    """
    w, wp, wpp = _w_wp_wpp(k, params)
    w_safe = np.maximum(w, 1e-14)
    term1 = (1.0 - (k * wp) / (2.0 * w_safe))**2
    term2 = (wp*wp)/4.0 * (1.0 / w_safe + 0.25)
    term3 = 0.5 * wpp
    return term1 - term2 + term3

def c_butterfly_grid(params, k_grid, eps=1e-8):
    """
    Inequality constraint for SLSQP: must return >= 0.
    Enforces min_k g(k) - eps >= 0 over a fixed grid.
    """
    g_vals = g_butterfly(k_grid, params)
    return np.min(g_vals) - eps