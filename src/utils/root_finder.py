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
#For sigma on the interval [a,b] brent solver finds f(sigma)=0
# Here a = 1e-8 and b = 5  (f(a)*f(b)<0 else -> exit)