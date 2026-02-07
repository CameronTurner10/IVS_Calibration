import numpy as np
from src.utils.black_scholes import bs_call, bs_put
from scipy.optimize import brentq


def implied_vol(F, K, T, r, market_price, option_type="call"):

    disc_factor = np.exp(-r * T)
    
    #switching
    if option_type == "call" and F > K:
        # if ITM call -> use OTM put P = C - disc_factor * (F - K)
        solve_price = market_price - disc_factor * (F - K)
        solve_type = "put"
    elif option_type == "put" and F < K:
        # if ITM put -> use OTM call C = P + disc_factor * (F - K)
        solve_price = market_price + disc_factor * (F - K)
        solve_type = "call"
    else:
        solve_price = market_price
        solve_type = option_type
    if solve_price < 1e-9:
        return np.nan
    
    def f_sigma(sigma):
        if solve_type == "call":
            return bs_call(F, K, T, sigma, r) - solve_price
        else:
            return bs_put(F, K, T, sigma, r) - solve_price
    try:
        return brentq(f_sigma, 1e-10, 5.0)
    except ValueError:
        return np.nan

# We switchin between calls and puts using call–put parity to avoid numerically unstable
#deep ITM or deep OTM prices and to ensure a robust implied volatility root-finding.

