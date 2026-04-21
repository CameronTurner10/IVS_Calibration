
# poetry run python -m src.smoothing_spline.optimisation.fit_spline

import numpy as np
import matplotlib.pyplot as plt
from src.smoothing_spline.implementation.spline_model import (
    build_observation_matrix,
    build_Q_matrix,
    build_R_matrix,
    load_spline_slice,
)


def choose_lambda(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    S: float,
    r: float,
    T: float,
    delta: float = 0.0,
    lam_grid: np.ndarray = None,
    plot_aic: bool = False,
    #Brandon 20/04/2026: can be gcv or aic, fengler did aic but gcv is another i fount works well
    criterion: str = "aic",
    #criterion: str = "gcv",
    fit_mode: str = "weighted",
    min_price: float = 1e-4,
) -> float:
    """
    Select the optimal smoothing parameter lambda.

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
        If True plot the selected score curve, by default False
    criterion : str, optional
        Score used to choose lambda. Either "gcv" or "aic", by default "gcv"

    Returns
    -------
    float
        Optimal lambda value

    """
    # Brandon 20/04/2026: can be gcv or aic, fengler did aic but gcv is another i found works well
    assert criterion in {"gcv", "aic"}, "Criterion must be 'gcv' or 'aic'"

    if lam_grid is None:
        lam_grid = np.logspace(-2, 10, 80)

    Q = build_Q_matrix(strikes)
    R = build_R_matrix(strikes)
    K = Q @ np.linalg.solve(R, Q.T)
    observation_matrix = build_observation_matrix(call_prices, fit_mode, min_price)
    rhs = observation_matrix @ call_prices
    n = len(strikes)

    scores = []
    for lam in lam_grid:
        # Once the fitter switches from I to W, the lambda search has to use
        # the same observation matrix or the two steps optimise different models.
        A = observation_matrix + lam * K
        g_hat = np.linalg.solve(A, rhs)

        residual = g_hat - call_prices
        rss = float(residual.T @ observation_matrix @ residual)

        # Degrees of freedom are the trace of the hat matrix here.
        H_lam = np.linalg.solve(A, observation_matrix)
        degrees_of_freedom = np.trace(H_lam)

        if criterion == "gcv":
            denom = (n - degrees_of_freedom) ** 2
            if denom < 1e-12:
                scores.append(np.inf)
            else:
                scores.append((n * rss) / denom)
        else:
            # Standard AIC: n*log(RSS/n) + 2*df
            scale = max(rss / n, 1e-12)
            scores.append(n * np.log(scale) + 2 * degrees_of_freedom)

    best_idx = np.argmin(scores)
    best_lam = lam_grid[best_idx]

    if plot_aic:
        plt.figure(figsize=(8, 5))
        lbl = criterion.upper()
        plt.plot(lam_grid, scores, 'b.-', markersize=4, linewidth=0.5, label=f"{lbl} Score")
        plt.axvline(best_lam, color='red', linestyle='--', label=f"Best lambda = {best_lam:.2e}")
        plt.title(f"Lambda Search ({fit_mode}, {lbl})")
        plt.xlabel("Lambda")
        plt.ylabel(f"{lbl} Score")
        plt.xscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    return best_lam


def fit_all_splines(
    market_surfaces: dict,
    S_dict: dict,
    r: float = 0.05,
    delta: float = 0.0,
    fit_mode: str = "weighted",
    min_price: float = 1e-4,
) -> dict:
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
    fit_mode : str, optional
        Observation weighting mode passed to fit_smoothing_spline.
    min_price : float, optional
        Floor used in weighted fitting.

    Returns
    -------
    dict
        Dict {T: result_dict} where result_dict is the output of fit_smoothing_spline

    Notes
    -----
    Calls choose_lambda then fit_smoothing_spline for each maturity. Prints progress per slice.
    """
    if not isinstance(market_surfaces, dict):
        raise TypeError("market_surfaces must be a dict {T: (strikes, call_prices)}")
    if not isinstance(S_dict, dict):
        raise TypeError("S_dict must be a dict {T: spot_price}")

    results = {}

    for T in sorted(market_surfaces):
        if T not in S_dict:
            raise KeyError(f"Missing spot price in S_dict for maturity T={T}")

        slice_data = market_surfaces[T]
        if not isinstance(slice_data, (tuple, list)) or len(slice_data) != 2:
            raise ValueError(
                f"market_surfaces[{T}] must be a tuple/list of (strikes, call_prices)"
            )

        strikes, call_prices = slice_data
        strikes = np.asarray(strikes, dtype=float)
        call_prices = np.asarray(call_prices, dtype=float)
        S = float(S_dict[T])

        if strikes.ndim != 1 or call_prices.ndim != 1:
            raise ValueError(f"T={T}: strikes and call_prices must be 1D arrays")
        if len(strikes) != len(call_prices):
            raise ValueError(
                f"T={T}: strikes and call_prices must have the same length"
            )
        if len(strikes) < 3:
            raise ValueError(
                f"T={T}: need at least 3 strike points to fit the spline"
            )
        if np.any(~np.isfinite(strikes)) or np.any(~np.isfinite(call_prices)):
            raise ValueError(f"T={T}: strikes/call_prices contain NaN or inf")
        if np.any(np.diff(strikes) <= 0):
            raise ValueError(f"T={T}: strikes must be strictly increasing")

        print(f"Processing maturity T={T:.6f} ...")

        lam = choose_lambda(
            strikes=strikes,
            call_prices=call_prices,
            S=S,
            r=r,
            T=T,
            delta=delta,
            fit_mode=fit_mode,
            min_price=min_price,
        )

        print(f"  chosen lambda = {lam}")

        result = fit_smoothing_spline(
            strikes=strikes,
            call_prices=call_prices,
            lam=lam,
            S=S,
            r=r,
            T=T,
            delta=delta,
            fit_mode=fit_mode,
            min_price=min_price,
        )

        result["lambda"] = float(lam)
        results[T] = result

        if result.get("success", False):
            print(f"  fit successful for T={T:.6f}")
        else:
            print(f"  fit failed for T={T:.6f}: {result.get('message', 'unknown error')}")

    return results

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
