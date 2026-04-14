"""
Diagnostics to verify the fitted smoothing spline is arbitrage-free and smooth.

Fengler, M.R. (2009). Arbitrage-Free Smoothing of the Implied Volatility Surface. Quantitative Finance, 9(4), 417-428.
"""

import numpy as np


def check_arbitrage(result: dict) -> dict:
    """
    Verify if the fitted smoothing spline satisfies the theoretical no-arbitrage bounds.

    Parameters
    ----------
    result : dict
        Dict returned by fit_smoothing_spline

    Returns
    -------
    dict
        Dictionary with keys 'pass' (bool), 'convexity_ok' (bool), 'monotonicity_ok' (bool), 'price_bounds_ok' (bool), 'violations' (list of str)

    Notes
    -----
    Checks Fengler eqs 25 (gamma >= 0), 26 (slope bounds), 27 (price bounds). Mirror structure of svi/testing/svi_diagnostics.py
    """
    raise NotImplementedError("Not yet implemented")


def check_smoothness(result: dict, market_prices: np.ndarray = None) -> dict:
    """
    Calculate the roughness penalty and optional RMSE fit error.

    Parameters
    ----------
    result : dict
        Dict returned by fit_smoothing_spline
    market_prices : np.ndarray, optional
        Array of observed call prices for RMSE calculation, by default None

    Returns
    -------
    dict
        Dictionary with keys 'roughness' (float), 'rmse' (float or None), 'max_abs_error' (float or None)

    Notes
    -----
    roughness = gamma @ R @ gamma = integral of g''(u)^2 du, Fengler Proposition 3.1.
    """
    raise NotImplementedError("Not yet implemented")
