"""
Spline fitting automatic λ selection and batch maturity processing.

Fengler, M.R. (2009). Arbitrage-Free Smoothing of the Implied Volatility Surface. Quantitative Finance, 9(4), 417-428.
"""

# poetry run python -m src.smoothing_spline.optimisation.fit_spline

import numpy as np
import matplotlib.pyplot as plt
from src.smoothing_spline.implementation.spline_model import build_Q_matrix, build_R_matrix, load_spline_slice


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
    
    
    if lam_grid is None:
        lam_grid = np.linspace(0, 1000000, 50)

    # Pre-calculate matrices
    Q = build_Q_matrix(strikes)
    R = build_R_matrix(strikes)
    K = Q @ np.linalg.solve(R, Q.T)
    n = len(strikes)

    aic_scores = []
    #cant call lambda a variable bc its a python keyword, same as cant call a variable 'def'
    for lam in lam_grid:
        # FENGLER EQ 31: H = (I + lambda * Q * R^-1 * Q^T)^-1
        A = np.eye(n) + lam * K
        g_hat = np.linalg.solve(A, call_prices)
        
        # Error (RSS)
        rss = np.sum((g_hat - call_prices)**2)
        
        # Complexity (Degrees of Freedom)
        H_lam = np.linalg.inv(A)
        degrees_of_freedom = np.trace(H_lam)
        
        # FENGLER EQ 30: Xi(lambda) = RSS + 2 * Trace(H)   could mayvbe use gcv instead?
        aic = rss + 2 * degrees_of_freedom
        aic_scores.append(aic)
        
    # Find the lambda that gave the lowest AIC score
    best_idx = np.argmin(aic_scores)
    best_lam = lam_grid[best_idx]
    
    if plot_aic:
        plt.figure(figsize=(8, 5))
        plt.plot(lam_grid, aic_scores, 'b.-', markersize=4, linewidth=0.5, label="AIC Score")
        plt.axvline(best_lam, color='red', linestyle='--', label=f"Best lambda = {best_lam:.2f}")
        plt.title("Choosing Lambda Eq 30")
        plt.xlabel("Lambda")
        plt.ylabel("AIC Score Xi(lambda)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    return best_lam


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


if __name__ == "__main__":
    # Quick self-test to see everything working
    
    print("Running Lambda Selection Test")
    try:
        # Load a default slice from our test data
        strikes, prices, S, r, F = load_spline_slice("Surface4", 0.043836)
        best_lam = choose_lambda(strikes, prices, S, r, 0.043836, plot_aic=True)
        
        print(f"Best lambda found is {best_lam:.4f}")
    except Exception as e:
        print(f"TEST FAILED: {e}")