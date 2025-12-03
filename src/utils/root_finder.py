import numpy as np
from src.utils.black_scholes import bs_call, bs_put
from scipy.optimize import brentq

def implied_vol(F,K,T,r,market_price,option_type="call"):
    
    def f_sigma(sigma,):
        if option_type=="call":
            return bs_call(F,K,T,sigma,r)-market_price
        else:
            return bs_put(F,K,T,sigma,r)-market_price
    
    return brentq(f_sigma,1e-10,5)