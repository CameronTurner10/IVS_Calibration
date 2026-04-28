"""
Visual verification driver for plot_spline_slice.

Run from the project root with:

    poetry run python -m tests.verify_plot_spline_slice
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt

from src.smoothing_spline.implementation.spline_model import (
    load_spline_slice,
    fit_smoothing_spline,
    evaluate_spline,
    second_derivative,
    prices_to_iv,
)
from src.smoothing_spline.optimisation.fit_spline import choose_lambda
from src.utils.plotting import plot_spline_slice

# Configuration. Change these to verify a different slice.
SHEET_NAME = "Surface4"
T = 0.043836            # The short-dated slice
FILEPATH = "tests/data/Surfaces.xlsx"
PLOT_TYPE = "total_var" # Runs with "iv" & "total_var"

def check_residuals(market_y: np.ndarray, fitted_y_at_k: np.ndarray,
                    label: str = "") -> dict:
    """Residual diagnostics at the market points."""
    residuals = fitted_y_at_k - market_y
    finite = np.isfinite(residuals)
    n_nan = int((~finite).sum())

    if finite.sum() == 0:
        return {"ok": False, "reason": "all residuals are NaN/inf",
                "rmse": np.nan, "max_abs": np.nan, "n_nan": n_nan}

    r = residuals[finite]
    rmse = float(np.sqrt(np.mean(r ** 2)))
    max_abs = float(np.max(np.abs(r)))

    n_pos = int((r > 0).sum())
    n_neg = int((r < 0).sum())
    biased = (n_pos == 0) or (n_neg == 0)

    return {
        "ok": (n_nan == 0) and (not biased),
        "rmse": rmse,
        "max_abs": max_abs,
        "n_nan": n_nan,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "biased": biased,
        "label": label,
    }

def check_monotone_extrapolation(result: dict, forward: float,
                                 k_range: tuple[float, float]) -> dict:
    """
    In the wings, the spline is extrapolated linearly in strike. The call
    price should be non-negative and non-increasing in K (a basic no-arb
    requirement). This checks a dense grid outside the knot range.
    """
    k_lo, k_hi = k_range
    k_ext_lo = np.linspace(k_lo - 0.5, k_lo, 50)
    k_ext_hi = np.linspace(k_hi, k_hi + 0.5, 50)
    K_ext_lo = forward * np.exp(k_ext_lo)
    K_ext_hi = forward * np.exp(k_ext_hi)

    C_lo = np.array([evaluate_spline(result, float(K)) for K in K_ext_lo])
    C_hi = np.array([evaluate_spline(result, float(K)) for K in K_ext_hi])

    nonneg = bool(np.all(C_lo >= 0) and np.all(C_hi >= 0))
    nonincreasing = bool(np.all(np.diff(C_lo) <= 1e-9)
                         and np.all(np.diff(C_hi) <= 1e-9))

    return {
        "ok": nonneg and nonincreasing,
        "nonneg": nonneg,
        "nonincreasing": nonincreasing,
    }

def check_iv_inversion(result: dict, market_strikes: np.ndarray,
                       forward: float) -> dict:
    """
    prices_to_iv returns NaN for failed Black--Scholes inversions. If any of
    the market-point IVs are NaN, the plot residuals and RMSE will silently
    become NaN, so we need to flag it explicitly.
    """
    ivs = prices_to_iv(result, market_strikes, forward)
    n_nan = int(np.isnan(ivs).sum())
    return {
        "ok": n_nan == 0,
        "n_nan": n_nan,
        "n_total": len(ivs),
    }

def check_second_derivative(result: dict, n_samples: int = 200) -> dict:
    """
    Convexity: a no-arbitrage call-price surface has g''(K) >= 0. This is
    Alesha's complement to Claire and Gaby's check_smoothness, not a
    replacement for it.
    """
    strikes = np.asarray(result["strikes"], dtype=float)
    K_grid = np.linspace(strikes[0], strikes[-1], n_samples)
    g2 = np.array([second_derivative(result, float(K)) for K in K_grid])

    min_g2 = float(np.min(g2))
    max_g2 = float(np.max(g2))
    median_g2 = float(np.median(np.abs(g2)))

    tol = 1e-10
    convex = bool(min_g2 >= -tol)

    return {
        "ok": convex,
        "min_g2": min_g2,
        "max_g2": max_g2,
        "median_abs_g2": median_g2,
    }

def main() -> int:
    print(f"\nVerifying plot_spline_slice for {SHEET_NAME}, T = {T}\n")

    strikes, call_prices, S, r, F = load_spline_slice(
        SHEET_NAME, T, filepath=FILEPATH
    )

    from src.utils.root_finder import implied_vol
    market_vols = np.array([
        implied_vol(F, float(K), T, r, float(C), option_type="call")
        for K, C in zip(strikes, call_prices)
    ])
    if np.any(np.isnan(market_vols)):
        print("WARNING: some market vols could not be inverted from the "
              "quoted call prices. Verification will proceed but residuals "
              "at those strikes will be unreliable.\n")

    lam = choose_lambda(strikes, call_prices, S, r, T)
    print(f"Chosen lambda = {lam:.4e}")

    result = fit_smoothing_spline(
        strikes, call_prices, lam, S, r, T
    )
    print(f"Optimiser success: {result['success']}  status={result['status']}")
    print(f"Optimiser message: {result['message']}")

    print("\nRendering plot_spline_slice ...")
    plot_spline_slice(
        result=result,
        T=T,
        market_strikes=strikes,
        market_vols=market_vols,
        sheet_name=SHEET_NAME,
        plot_type=PLOT_TYPE,
    )

    print("\n--- Automated checks ---")

    iv_check = check_iv_inversion(result, strikes, F)
    print(f"[IV inversion]    {iv_check['n_nan']}/{iv_check['n_total']} NaNs "
          f"at market strikes  -> {'OK' if iv_check['ok'] else 'FAIL'}")

    iv_at_k = prices_to_iv(result, strikes, F)
    if PLOT_TYPE == "iv":
        market_y = market_vols
        fitted_y_at_k = iv_at_k
    else:
        market_y = market_vols ** 2 * T
        fitted_y_at_k = iv_at_k ** 2 * T
    res_check = check_residuals(market_y, fitted_y_at_k, label=PLOT_TYPE)
    print(f"[Residuals]       RMSE={res_check['rmse']:.3e}  "
          f"max|r|={res_check['max_abs']:.3e}  "
          f"pos/neg={res_check['n_pos']}/{res_check['n_neg']}  "
          f"-> {'OK' if res_check['ok'] else 'FAIL (biased or NaN)'}")

    conv_check = check_second_derivative(result)
    print(f"[Convexity]       min g''={conv_check['min_g2']:.3e}  "
          f"-> {'OK' if conv_check['ok'] else 'FAIL'}")

    k_values = np.log(strikes / F)
    ext_check = check_monotone_extrapolation(
        result, F, (float(k_values[0]), float(k_values[-1]))
    )
    print(f"[Wing extrapolation]  nonneg={ext_check['nonneg']}, "
          f"nonincreasing={ext_check['nonincreasing']}  "
          f"-> {'OK' if ext_check['ok'] else 'FAIL'}")

    all_ok = (iv_check["ok"] and res_check["ok"]
              and conv_check["ok"] and ext_check["ok"])
    print("\n" + ("ALL AUTOMATED CHECKS PASSED"
                  if all_ok else "ONE OR MORE CHECKS FAILED -- see above"))
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
    
