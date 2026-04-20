"""SPLine diagnostics tests

Contains:
check_arbitrage: Function to check for arbitrage in the fitted spline
check_smoothness: Function to check the smoothness of the spline"""

import numpy as np


def check_arbitrage(result: dict) -> dict:
    """
    Verify if the fitted smoothing spline satisfies Fengler's
    theoretical no-arbitrage bounds.

    Parameters
    ----------
    result : dict
        Dict returned by fit_smoothing_spline

    Returns
    -------
    dict
        Dictionary with keys:
        - 'pass' (bool)
        - 'convexity_ok' (bool)
        - 'monotonicity_ok' (bool)
        - 'price_bounds_ok' (bool)
        - 'violations' (list of str)

    Notes
    -----
    Checks Fengler eq. 25 (gamma >= 0),
    eq. 26 (slope bounds),
    eq. 27 (price bounds).
    """
    violations = []
    tol = 1e-8

    required = ["g", "gamma", "S", "r", "T", "strikes"]
    missing = [key for key in required if key not in result]
    if missing:
        return {
            "pass": False,
            "convexity_ok": False,
            "monotonicity_ok": False,
            "price_bounds_ok": False,
            "violations": [f"Missing required keys: {missing}"],
        }

    g = np.asarray(result["g"], dtype=float)
    gamma = np.asarray(result["gamma"], dtype=float)
    strikes = np.asarray(result["strikes"], dtype=float)
    S = float(result["S"])
    r = float(result["r"])
    T = float(result["T"])
    delta = float(result.get("delta", 0.0))

    if len(strikes) < 2:
        return {
            "pass": False,
            "convexity_ok": False,
            "monotonicity_ok": False,
            "price_bounds_ok": False,
            "violations": ["Need at least two strikes"],
        }

    if len(g) != len(strikes):
        return {
            "pass": False,
            "convexity_ok": False,
            "monotonicity_ok": False,
            "price_bounds_ok": False,
            "violations": ["g and strikes must have the same length"],
        }

    # gamma may be stored as:
    # - full length n with gamma[0] = gamma[-1] = 0
    # - interior only length n-2
    n = len(strikes)
    if len(gamma) == n - 2:
        gamma_full = np.concatenate(([0.0], gamma, [0.0]))
    elif len(gamma) == n:
        gamma_full = gamma
    else:
        return {
            "pass": False,
            "convexity_ok": False,
            "monotonicity_ok": False,
            "price_bounds_ok": False,
            "violations": ["gamma must have length n or n-2"],
        }

    convexity_ok = np.all(gamma_full[1:-1] >= -tol)
    if not convexity_ok:
        bad_idx = np.where(gamma_full[1:-1] < -tol)[0] + 1
        violations.append(
            f"Eq. 25 failed: negative gamma at interior indices {bad_idx.tolist()}"
        )

    monotonicity_ok = True
    disc_r = np.exp(-r * T)
    h = np.diff(strikes)

    left_slope = (g[1] - g[0]) / h[0] - (h[0] / 6.0) * gamma_full[1]
    right_slope = (g[-1] - g[-2]) / h[-1] - (h[-1] / 6.0) * gamma_full[-2]

    if left_slope < -disc_r - tol:
        monotonicity_ok = False
        violations.append(
            f"Eq. 26 failed: left derivative {left_slope:.6f} below lower bound {-disc_r:.6f}"
        )

    if right_slope > tol:
        monotonicity_ok = False
        violations.append(
            f"Eq. 26 failed: right derivative {right_slope:.6f} is positive"
        )

    # Eq. 27: price bounds
    # exp(-delta T)S - exp(-rT)u1 <= g1 <= exp(-delta T)S
    # g_n >= 0
    price_bounds_ok = True
    disc_delta = np.exp(-delta * T)

    lower_g1 = disc_delta * S - disc_r * strikes[0]
    upper_g1 = disc_delta * S

    if g[0] < lower_g1 - tol:
        price_bounds_ok = False
        violations.append(
            f"Eq. 27 failed: g[0]={g[0]:.6f} below lower bound {lower_g1:.6f}"
        )

    if g[0] > upper_g1 + tol:
        price_bounds_ok = False
        violations.append(
            f"Eq. 27 failed: g[0]={g[0]:.6f} above upper bound {upper_g1:.6f}"
        )

    if g[-1] < -tol:
        price_bounds_ok = False
        violations.append(
            f"Eq. 27 failed: g[-1]={g[-1]:.6f} is negative"
        )

    return {
        "pass": bool(convexity_ok and monotonicity_ok and price_bounds_ok),
        "convexity_ok": bool(convexity_ok),
        "monotonicity_ok": bool(monotonicity_ok),
        "price_bounds_ok": bool(price_bounds_ok),
        "violations": violations,
    }


def check_smoothness(spline):
    """
    Check the smoothness of the fitted spline.
    Placeholder function, actual implementation will analyze the spline's derivatives.
    """
    raise NotImplementedError("Smoothness check not implemented yet")


