# Implied vol solver using Brent's method
# 
# TODO B1: Fix for deep ITM options using OTM switching
# Problem: when F >> K (deep ITM call), the solver fails because 
# the price is mostly intrinsic value and not sensitive to vol.
# Solution: use put-call parity to get the OTM put price instead:
#   P = C - exp(-r*T) * (F - K)
# Then solve for vol using the put.

import numpy as np
from src.utils.black_scholes import bs_call, bs_put
from scipy.optimize import brentq


def implied_vol(F, K, T, r, market_price, option_type="call"):
    
    # TODO B1: Add OTM switching logic here
    # If call is ITM (F > K), switch to put using parity
    # If put is ITM (F < K), switch to call using parity
    #
    # df = np.exp(-r * T)
    # if option_type == "call" and F > K:
    #     # deep ITM call - use put instead
    #     put_price = market_price - df * (F - K)
    #     solve_type = "put"
    #     solve_price = put_price
    # elif option_type == "put" and F < K:
    #     # deep ITM put - use call instead  
    #     call_price = market_price + df * (F - K)
    #     solve_type = "call"
    #     solve_price = call_price
    # else:
    #     solve_type = option_type
    #     solve_price = market_price
    
    def f_sigma(sigma):
        if option_type == "call":
            return bs_call(F, K, T, sigma, r) - market_price
        else:
            return bs_put(F, K, T, sigma, r) - market_price
    
    return brentq(f_sigma, 1e-10, 5)