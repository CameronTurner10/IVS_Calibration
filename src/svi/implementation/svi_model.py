"""SVI implementation
- SVI class: class which will store the raw svi parameters
- Total Variance Calculation: returns total implied variance w(k) based on SVI parameters.
- Implied Volatility Calculation: computes implied volatility from SVI 
"""


class SVI:
    def __init__(self, a, b, rho, m, sigma):
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma
        # shoudl be standard svi parameters

    def total_variance(self, k):
        """
        Calculate total implied variance w(k) using the SVI parameterization.
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        """
        return self.a + self.b * (self.rho * (k - self.m) + ((k - self.m) ** 2 + self.sigma ** 2) ** 0.5)
    

    def svi_implied_vol(self, k, T):
        """
        Calculate implied volatility from total variance.
        IV(k, T) = sqrt(w(k) / T)
        """
        total_var = self.total_variance(k)
        return (total_var / T) ** 0.5