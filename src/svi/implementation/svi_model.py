# SVI model - the maths for fitting volatility smiles
# See Ferhati paper for details on the formula

import numpy as np

class SVI:
    def __init__(self, a, b, rho, m, sigma):
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def total_variance(self, k):
        # w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))
    
    def svi_implied_vol(self, k, T):
        # IV = sqrt(w(k) / T)
        total_var = self.total_variance(k)
        return np.sqrt(total_var / T)


# TODO A1: Double check the maths above matches the Ferhati paper
# Try some hand calculations to verify:
#   - With a=0.04, b=0.1, rho=0, m=0, sigma=0.1, k=0
#   - w(0) should = 0.04 + 0.1*sqrt(0.01) = 0.05
#   - IV with T=1 should be sqrt(0.05) = 0.2236