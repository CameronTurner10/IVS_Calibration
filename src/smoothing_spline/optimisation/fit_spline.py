"""
Spline fitting automatic λ selection and batch maturity processing.

Fengler, M.R. (2009). Arbitrage-Free Smoothing of the Implied Volatility Surface. Quantitative Finance, 9(4), 417-428.
"""

import numpy as np


def choose_lambda(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    S: float,
    r: float,
    T: float,
    delta: float = 0.0,
    lam_grid: np.ndarray = None,
    plot_aic: bool = False,
) -> float:
    """
    Select the optimal smoothing parameter lambda by minimizing GCV / AIC.

    Parameters
    ----------
    strikes : np.ndarray
        1D sorted array of n strike prices
    call_prices : np.ndarray
        1D array of observed call option prices
    S : float
        Current spot price
    r : float
        Continuously compounded risk-free rate
    T : float
        Time to maturity in years
    delta : float, optional
        Continuous dividend yield, by default 0.0
    lam_grid : np.ndarray, optional
        Grid of lambda values to search over, by default np.logspace(-2,6,50)
    plot_aic : bool, optional
        If True plot the AIC curve, by default False

    Returns
    -------
    float
        Optimal lambda value

    Notes
    -----
    Fengler Section 3.4. AIC = n*log(RSS/n) + 2*df. Reproduces Fengler Figure 3 if plot_aic=True.
    """
    raise NotImplementedError("Not yet implemented")


def fit_all_splines(market_surfaces: dict, S_dict: dict, r: float = 0.05, delta: float = 0.0) -> dict:
    """
    Fit a smoothing spline for each maturity in the dataset.

    Parameters
    ----------
    market_surfaces : dict
        Dict {T: (strikes, call_prices)}
    S_dict : dict
        Dict {T: spot_price}
    r : float, optional
        Continuously compounded risk-free rate, by default 0.05
    delta : float, optional
        Continuous dividend yield, by default 0.0

    Returns
    -------
    dict
        Dict {T: result_dict} where result_dict is the output of fit_smoothing_spline

    Notes
    -----
    Calls choose_lambda then fit_smoothing_spline for each maturity. Print progress per slice.
    """
    raise NotImplementedError("Not yet implemented")