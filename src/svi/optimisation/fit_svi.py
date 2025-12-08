"""
SVI fitting 
"""

import numpy as np
from src.svi.implementation.svi_model import total_variance, implied_volatility

def fit_svi_slice(strikes, total_variances):
    """
    Fit SVI parameters to a single slice of total variances at given strikes
    
    """
    assert NotImplementedError("SVI fitting not implemented yet")

def fit_svi_surface(strikes, maturities, total_variances):
    """
    Fit SVI parameters across all maturities.
    strikes: list of strike prices
    maturities: list of maturities
    total_variances: 2D list/array of total variances corresponding to strikes and maturities
    Returns a list of SVI parameters for each maturity slice.
    """

    assert NotImplementedError("SVI surface fitting not implemented yet")

    