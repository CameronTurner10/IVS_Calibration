import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.svi.optimisation.local_optimizers import fit_svi_slice, total_variance
from src.svi.optimisation.arbitrage import calibrate_surface, fit_single_slice_with_bound
from mpl_toolkits.mplot3d import Axes3D
from src.smoothing_spline.implementation.spline_model import (
    evaluate_spline,
    prices_to_iv,
    fit_smoothing_spline,
)
from src.smoothing_spline.optimisation.fit_spline import choose_lambda,fit_all_splines

"""
This section is responsible for plotting the SVI calibration results.
The plotting can be tested by running "poetry run python -m src.utils.plotting" in the terminal.
Incorporated is a CLI style interface to allow for easy plotting of the SVI calibration results 
    so just follow the messages in the terminal.
Close the plot manually to continue with the next plot.
"""


def get_slice_from_data(T, sheet_name, filepath="tests/data/Surfaces.xlsx"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    slice_df = df[np.isclose(df["Year Fraction"], T, atol=1e-8)]

    strikes = slice_df["Strike"].values
    market_vols = slice_df["Volatility"].values
    forward = slice_df["Forward"].iloc[0]
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    return strikes, market_vols, forward, k_values, w_market

def get_spline_data(T, sheet_name, filepath="tests/data/Surfaces.xlsx"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    slice_df = df[np.isclose(df["Year Fraction"], T, atol=1e-8)]

    strikes = slice_df["Strike"].values
    call_prices=slice_df["Call Price"].values
    market_vols = slice_df["Volatility"].values
    spot=slice_df["Spot"].iloc[0]
    forward = slice_df["Forward"].iloc[0]
    r=slice_df["Discount Rate"].iloc[0]

    return strikes, call_prices,market_vols,spot,forward,r


def _finite_mask(values):
    return np.isfinite(np.asarray(values, dtype=float))


def _plot_finite_line(ax, x, y, **kwargs):
    y_plot = np.asarray(y, dtype=float).copy()
    mask = _finite_mask(y_plot)
    y_plot[~mask] = np.nan
    line = ax.plot(x, y_plot, **kwargs)
    return mask, line


def _mark_invalid_x(ax, x, mask, label="Invalid IV inversion"):
    invalid_x = np.asarray(x)[~mask]
    if invalid_x.size == 0:
        return
    ax.scatter(
        invalid_x,
        np.full(invalid_x.shape, 0.03),
        transform=ax.get_xaxis_transform(),
        color="red",
        marker="x",
        s=18,
        linewidths=0.9,
        alpha=0.8,
        label=label,
        zorder=4,
    )


def _annotate_invalid_iv(ax, invalid_count, total_count, loc="upper left"):
    if total_count <= 0:
        return
    text = f"Invalid IV inversions: {invalid_count} / {total_count}"
    y = 0.97 if loc.startswith("upper") else 0.03
    va = "top" if loc.startswith("upper") else "bottom"
    ax.text(
        0.02,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va=va,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _annotate_invalid_iv_3d(ax, invalid_count, total_count):
    if total_count <= 0:
        return
    ax.text2D(
        0.02,
        0.96,
        f"Invalid IV inversions: {invalid_count} / {total_count}",
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _finite_rmse(residuals):
    residuals = np.asarray(residuals, dtype=float)
    mask = _finite_mask(residuals)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean(residuals[mask] ** 2)))


def _fit_mode_label(fit_mode):
    return f"{str(fit_mode).replace('_', ' ').title()} Spline"


def _spline_title(sheet_name, fit_mode, description):
    return f"{sheet_name} - {_fit_mode_label(fit_mode)} {description}"


def _is_shortest_maturity(T, sheet_name, filepath="tests/data/Surfaces.xlsx"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = np.asarray(sorted(df["Year Fraction"].unique()), dtype=float)
    return bool(expiries.size and np.isclose(T, expiries[0], atol=1e-8))


def plot_single_slice(T, sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="total_var"):
    strikes, market_vols, forward, k_values, w_market = get_slice_from_data(T, sheet_name, filepath)

    fitted_params = fit_single_slice_with_bound(k_values, w_market)

    k_grid = np.linspace(min(k_values), max(k_values), 200)
    w_fitted_grid = total_variance(k_grid, **fitted_params)
    w_fitted_at_k = total_variance(k_values, **fitted_params)

    if plot_type == "iv":
        market_y = market_vols
        fitted_y_grid = np.sqrt(w_fitted_grid / T)
        fitted_y_at_k = np.sqrt(w_fitted_at_k / T)
        y_label = "Implied Volatility (σ)"
    else:
        market_y = w_market
        fitted_y_grid = w_fitted_grid
        fitted_y_at_k = w_fitted_at_k
        y_label = "Total Implied Variance (w)"

    residuals = fitted_y_at_k - market_y

    #we want 2 plots one for the fit and one for the residuals
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.suptitle(f"SVI Calibration for T = {T:.6f}y", fontsize=14, fontweight="bold")

    #top plot, market data vs fitted data
    ax_main.scatter(k_values, market_y, color="black", s=30, zorder=3, label="Market Data")
    ax_main.plot(k_grid, fitted_y_grid, color="red", linewidth=2, label="SVI Fit")
    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper right")
    ax_main.grid(True, alpha=0.3)

    #param text box 
    param_text = "  |  ".join([f"{p}={v:.5f}" for p, v in fitted_params.items()])
    ax_main.text(
        0.5, 0.97, param_text,
        transform=ax_main.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )

    #bottom plot, residuals
    ax_res.bar(k_values, residuals, width=0.01, color="steelblue", alpha=0.7)
    ax_res.axhline(y=0, color="black", linewidth=0.8)
    ax_res.set_xlabel("Log-moneyness (k)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, alpha=0.3)

    #RMSE annotation
    rmse = np.sqrt(np.mean(residuals ** 2))
    ax_res.text(
        0.98, 0.90, f"RMSE = {rmse:.2e}",
        transform=ax_res.transAxes,
        fontsize=9, ha="right", va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fitted_params,k_values,k_grid,market_y,fitted_y_grid,residuals


def plot_multi_slice(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="total_var"):
    
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    colours = cm.viridis(np.linspace(0.1, 0.9, len(expiries)))

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.suptitle(f"SVI Multi-Slice Calibration — {sheet_name}", fontsize=14, fontweight="bold")

    fitted = calibrate_surface(sheet_name, filepath)

    for i, T in enumerate(expiries):
        strikes, market_vols, forward, k_values, w_market = get_slice_from_data(T, sheet_name, filepath)

        fitted_params = fitted[T]

        k_grid = np.linspace(min(k_values), max(k_values), 200)
        w_fitted_grid = total_variance(k_grid, **fitted_params)
        w_fitted_at_k = total_variance(k_values, **fitted_params)

        if plot_type == "iv":
            market_y = market_vols
            fitted_y_grid = np.sqrt(w_fitted_grid / T)
            fitted_y_at_k = np.sqrt(w_fitted_at_k / T)
        else:
            market_y = w_market
            fitted_y_grid = w_fitted_grid
            fitted_y_at_k = w_fitted_at_k

        residuals = fitted_y_at_k - market_y
        colour = colours[i]
        label = f"T={T:.4f}"

        #top plot
        ax_main.scatter(k_values, market_y, color=colour, s=15, zorder=3)
        ax_main.plot(k_grid, fitted_y_grid, color=colour, linewidth=1.5, label=label)

        #bottom plot
        ax_res.scatter(k_values, residuals, color=colour, s=10, alpha=0.7)

    y_label = "Implied Volatility (σ)" if plot_type == "iv" else "Total Implied Variance (w)"
    ax_main.set_ylabel(y_label)
    if plot_type != "iv":
        ax_main.set_yscale("log")
    ax_main.legend(loc="upper right", fontsize=7, ncol=3)
    ax_main.grid(True, alpha=0.3)

    ax_res.axhline(y=0, color="black", linewidth=0.8)
    ax_res.set_xlabel("Log-moneyness (k)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_interpolated_surface(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="total_var"):
    
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())

    all_k = []
    fitted = calibrate_surface(sheet_name, filepath)
    for T in expiries:
        strikes, market_vols, forward, k_values, _ = get_slice_from_data(T, sheet_name, filepath)
        all_k.extend(k_values.tolist())
        
    k_grid = np.linspace(min(all_k), max(all_k), 200)
    T_grid = np.linspace(min(expiries), max(expiries), 1000)
    T_knots = np.array(expiries)

    param_names = ["a", "b", "rho", "m", "sigma"]
    param_curves = {}
    for p in param_names:
        column = []
        for T in T_knots:
            value = fitted[T][p]
            column.append(value)
        param_curves[p] = np.array(column)

    W = np.zeros((len(T_grid), len(k_grid)))
    for i, T in enumerate(T_grid):
        a = np.interp(T, T_knots, param_curves["a"])
        b = np.interp(T, T_knots, param_curves["b"])
        rho = np.interp(T, T_knots, param_curves["rho"])
        m = np.interp(T, T_knots, param_curves["m"])
        sigma = np.interp(T, T_knots, param_curves["sigma"])
        W[i, :] = total_variance(k_grid, a, b, rho, m, sigma)

    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid) #convert 1D arrays to 2D grids


    fig= plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111,projection="3d")
    surface = ax.plot_surface(K_mesh,T_mesh,W)

    ax.set_xlabel("Log-moneyness k = log(K / F)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Total implied variance w(k,T)")
    ax.set_title(f"SVI Variance Surface — {sheet_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_surface(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="total_var"):
    
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())

    all_k = []
    fitted = calibrate_surface(sheet_name, filepath)
    for T in expiries:
        strikes, market_vols, forward, k_values, _ = get_slice_from_data(T, sheet_name, filepath)
        all_k.extend(k_values.tolist())
        
    k_grid = np.linspace(min(all_k), max(all_k), 1000)
    T_knots = np.array(expiries)


    W_svi = np.zeros((len(T_knots), len(k_grid)))
    for i, T in enumerate(T_knots):
        params=fitted[T]
        if plot_type == "iv":
            W_svi[i, :] = np.sqrt(total_variance(k_grid, **params)/T)
            z_label = "Implied Volatility"
        else:
            W_svi[i, :] = total_variance(k_grid, **params)
            z_label = "Total Variance"

    K_mesh, T_mesh = np.meshgrid(k_grid, T_knots) #convert 1D arrays to 2D grids


    fig= plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111,projection="3d")
    surface = ax.plot_surface(K_mesh,T_mesh,W_svi)

    ax.set_xlabel("Log-moneyness k = log(K / F)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Total implied variance w(k,T)")
    ax.set_title(f"SVI Variance Surface — {sheet_name}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_variance_heatmap(sheet_name, filepath="tests/data/Surfaces.xlsx"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())

    all_k = []
    fitted = calibrate_surface(sheet_name, filepath)
    for T in expiries:
        strikes, market_vols, forward, k_values, _ = get_slice_from_data(T, sheet_name, filepath)
        all_k.extend(k_values.tolist())
        
    k_grid = np.linspace(min(all_k), max(all_k), 1000)
    T_grid = np.linspace(min(expiries), max(expiries), 1000)
    T_knots = np.array(expiries)

    param_names = ["a", "b", "rho", "m", "sigma"]
    param_curves = {}
    for p in param_names:
        column = []
        for T in T_knots:
            value = fitted[T][p]
            column.append(value)
        param_curves[p] = np.array(column)

    W = np.zeros((len(T_grid), len(k_grid)))
    for i, T in enumerate(T_grid):
        a = np.interp(T, T_knots, param_curves["a"])
        b = np.interp(T, T_knots, param_curves["b"])
        rho = np.interp(T, T_knots, param_curves["rho"])
        m = np.interp(T, T_knots, param_curves["m"])
        sigma = np.interp(T, T_knots, param_curves["sigma"])
        W[i, :] = total_variance(k_grid, a, b, rho, m, sigma)

    fig, ax = plt.subplots(figsize=(11, 6))
    cf = ax.contourf(k_grid, T_grid, W, levels=40, cmap="RdYlGn_r")
    fig.colorbar(cf, ax=ax, label="Total implied variance w(k, T)")

    cl = ax.contour(k_grid, T_grid, W, levels=12, colors="black", linewidths=0.6, alpha=0.45)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%.4f", inline_spacing=4)

    ax.set_xlabel("Log-moneyness k = log(K / F)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_title(f"SVI Variance Surface — {sheet_name}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def _compute_log10_rmse_errors(sheet_name, filepath, sequential_init):
    """Helper: calibrate the surface and return (expiry list, log10 RMSE list)."""
    fitted = calibrate_surface(sheet_name, filepath, sequential_init=sequential_init)

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())

    ts, errors = [], []
    for T in expiries:
        _, market_vols, _, k_values, _ = get_slice_from_data(T, sheet_name, filepath)
        if len(k_values) == 0 or T not in fitted or T <= 0:
            continue
        params = fitted[T]
        w_fit = total_variance(k_values, **params)
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
        rmse_val = np.sqrt(np.mean((iv_fit - market_vols) ** 2))
        errors.append(np.log10(max(rmse_val, 1e-16)))
        ts.append(T)
    return ts, errors


def plot_error_log10_rmse(sheet_name, filepath="tests/data/Surfaces.xlsx", mode="both"):
    """
    Plot log10(RMSE) of implied volatility across expiries.
    mode: 'sequential' = warm-start from prev slice (skips DE after slice 1)
          'de'         = full DE global search on every slice independently
          'both'       = overlay both curves for comparison
    """
    plt.figure(figsize=(12, 6))

    if mode in ("sequential", "both"):
        ts_seq, err_seq = _compute_log10_rmse_errors(sheet_name, filepath, sequential_init=True)
        plt.plot(ts_seq, err_seq,
                 color="darkblue", marker="+", markersize=8,
                 linestyle="-", linewidth=0.9,
                 label="Sequential init (warm-start)")

    if mode in ("de", "both"):
        ts_de, err_de = _compute_log10_rmse_errors(sheet_name, filepath, sequential_init=False)
        plt.plot(ts_de, err_de,
                 color="darkviolet", marker="x", markersize=8,
                 linestyle="-", linewidth=0.9,
                 label="Full DE per slice")

    plt.xlabel("Expiry (year fraction)", fontsize=11)
    plt.ylabel("log\u2081\u2080(RMSE of implied volatility)", fontsize=11)
    plt.title(
        f"SVI Fit Quality: log\u2081\u2080(RMSE) \u2014 {sheet_name}",
        fontsize=13, fontweight="bold"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.gca().tick_params(direction="in", length=4)
    plt.tight_layout()
    plt.show()

def list_available_data(filepath="tests/data/Surfaces.xlsx"):
    xl = pd.ExcelFile(filepath)
    data = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet)
        if "Year Fraction" in df.columns:
            expiries = sorted(df["Year Fraction"].unique())
            data[sheet] = expiries
    return data


def plot_spline_slice(T, sheet_name, plot_type, filepath="tests/data/Surfaces.xlsx", fit_mode=None):
    
    strikes, call_prices,market_vols,spot,forward,r= get_spline_data(T, sheet_name, filepath)
    lam = choose_lambda(strikes, call_prices, spot, r, T, fit_mode=fit_mode)

 
    result = fit_smoothing_spline(strikes, call_prices, lam, spot, r, T, forward=forward, fit_mode=fit_mode)
    fit_mode = result.get("fit_mode", "unknown")

    k_values = np.log(strikes / forward)

    k_grid = np.linspace(min(k_values), max(k_values), 200)
    K_grid = forward * np.exp(k_grid)    
    iv_grid = prices_to_iv(result, K_grid, forward)
    iv_at_k = prices_to_iv(result, strikes, forward)
    grid_iv_mask = _finite_mask(iv_grid)
    knot_iv_mask = _finite_mask(iv_at_k)
    invalid_grid_count = int(np.sum(~grid_iv_mask))
    invalid_knot_count = int(np.sum(~knot_iv_mask))

    if plot_type == "iv":
        fitted_y_grid = iv_grid
        fitted_y_at_k = iv_at_k
        market_y = market_vols
        y_label = "Implied Volatility"
    else:
        fitted_y_grid = iv_grid ** 2 * T
        fitted_y_at_k = iv_at_k ** 2 * T
        market_y = market_vols ** 2 * T
        y_label = "Total Implied Variance"
    
    residuals = fitted_y_at_k - market_y
    show_price_panel = (
        _is_shortest_maturity(T, sheet_name, filepath)
        or invalid_grid_count > 0
        or invalid_knot_count > 0
    )

    # copied from plot_single_slice
    if show_price_panel:
        fig, (ax_main, ax_res, ax_price) = plt.subplots(
            3, 1, figsize=(10, 9),
            gridspec_kw={"height_ratios": [3, 1, 2]},
            sharex=True
        )
    else:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )
        ax_price = None
    fig.suptitle(
        _spline_title(sheet_name, fit_mode, f"for T = {T:.6f}y"),
        fontsize=14,
        fontweight="bold",
    )
    
    ax_main.scatter(k_values, market_y, color="black", s=30, zorder=3, label="Market Data")
    plot_mask, _ = _plot_finite_line(
        ax_main,
        k_grid,
        fitted_y_grid,
        color="red",
        linewidth=2,
        label="Smoothing Spline Fit",
    )
    _mark_invalid_x(ax_main, k_grid, plot_mask)
    _annotate_invalid_iv(ax_main, invalid_grid_count, len(iv_grid))
    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper right")
    ax_main.grid(True, alpha=0.3)

    finite_res = _finite_mask(residuals)
    ax_res.bar(k_values[finite_res], residuals[finite_res], width=0.01, color="steelblue", alpha=0.7)
    _mark_invalid_x(ax_res, k_values, finite_res, label="Invalid knot IV")
    ax_res.axhline(y=0, color="black", linewidth=0.8)
    ax_res.set_xlabel("Log-moneyness (k)" if ax_price is None else "")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, alpha=0.3)
    
    rmse = _finite_rmse(residuals)
    ax_res.text(0.98, 0.90, f"RMSE = {rmse:.2e}", transform=ax_res.transAxes, fontsize=9, ha="right", va="top")

    if ax_price is not None:
        fitted_call_grid = np.array([evaluate_spline(result, float(K)) for K in K_grid])
        ax_price.scatter(k_values, call_prices, color="black", s=25, zorder=3, label="Market Calls")
        ax_price.plot(k_grid, fitted_call_grid, color="red", linewidth=1.8, label="Spline Call Price")
        ax_price.set_xlabel("Log-moneyness (k)")
        ax_price.set_ylabel("Call Price")
        ax_price.grid(True, alpha=0.3)
        ax_price.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()

def plot_multi_spline_slice(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv", fit_mode=None):
    requested_fit_mode = fit_mode
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())

    colours = cm.viridis(np.linspace(0.1, 0.9, len(expiries)))

    fig, ax_main = plt.subplots(figsize=(12, 8))
    total_invalid = 0
    total_points = 0
    display_fit_mode = "unknown"

    for i, T in enumerate(expiries):
        strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)

        lam = choose_lambda(strikes, call_prices, S, r, T, fit_mode=requested_fit_mode)

        result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward, fit_mode=requested_fit_mode)
        display_fit_mode = result.get("fit_mode", display_fit_mode)

        k_values = np.log(strikes / forward)

        k_grid = np.linspace(k_values.min(), k_values.max(), 200)
        K_grid = forward * np.exp(k_grid)

        iv_grid = prices_to_iv(result, K_grid, forward)
        iv_at_k = prices_to_iv(result, strikes, forward)
        iv_mask = _finite_mask(iv_grid)
        total_invalid += int(np.sum(~iv_mask))
        total_points += len(iv_grid)

        if plot_type == "iv":
            market_y = market_vols
            fitted_grid = iv_grid
            y_label = "Implied Volatility"
        else:
            market_y = market_vols ** 2 * T
            fitted_grid = iv_grid ** 2 * T
            y_label = "Total Variance"

        _plot_finite_line(
            ax_main,
            k_grid,
            fitted_grid,
            color=colours[i],
            alpha=0.8,
            label=f"T={T:.3f}",
        )
        invalid_label = "Invalid IV inversion" if total_invalid == int(np.sum(~iv_mask)) else "_nolegend_"
        _mark_invalid_x(ax_main, k_grid, iv_mask, label=invalid_label)

        ax_main.scatter(k_values,market_y,color=colours[i],s=20)

    ax_main.set_ylabel(y_label)
    ax_main.set_xlabel("Log-moneyness (k)")
    _annotate_invalid_iv(ax_main, total_invalid, total_points)
    if plot_type != "iv":
        ax_main.set_yscale("log")
    ax_main.set_title(_spline_title(sheet_name, display_fit_mode, "multi-maturity fit"))
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    plt.tight_layout()
    plt.show()

def plot_spline_surface(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv", fit_mode=None):
    requested_fit_mode = fit_mode
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    colours = cm.viridis(np.linspace(0.1, 0.9, len(expiries)))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_k = []

    slice_data = {}
    display_fit_mode = "unknown"
    for i, T in enumerate(expiries):
        strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)
        lam = choose_lambda(strikes, call_prices, S, r, T, fit_mode=requested_fit_mode)
        result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward, fit_mode=requested_fit_mode)
        display_fit_mode = result.get("fit_mode", display_fit_mode)

        k_values = np.log(strikes / forward)
        slice_data[T] = (result, forward)

        all_k.extend(k_values.tolist())

    k_grid = np.linspace(min(all_k), max(all_k), 200)
    T_grid = np.array(expiries)

    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    W = np.full_like(K_mesh, np.nan, dtype=float)
    invalid_k = []
    invalid_t = []

    for i, T in enumerate(expiries):
        result, forward = slice_data[T]

        K_grid = forward * np.exp(k_grid)
        iv_grid = prices_to_iv(result, K_grid, forward)
        iv_mask = _finite_mask(iv_grid)
        invalid_k.extend(k_grid[~iv_mask].tolist())
        invalid_t.extend([T] * int(np.sum(~iv_mask)))

        if plot_type == "iv":
            W[i, :] = iv_grid
            z_label = "Implied Volatility"
        else:
            W[i, :] = iv_grid**2 * T
            z_label = "Total Variance"

    surface = ax.plot_surface(K_mesh, T_mesh, np.ma.masked_invalid(W), cmap=cm.viridis, alpha=0.9)
    finite_w = W[np.isfinite(W)]
    z_marker = float(np.nanmin(finite_w)) if finite_w.size else 0.0
    if invalid_k:
        ax.scatter(
            invalid_k,
            invalid_t,
            np.full(len(invalid_k), z_marker),
            color="red",
            s=12,
            depthshade=False,
            label="Invalid IV inversion",
        )
    _annotate_invalid_iv_3d(ax, len(invalid_k), W.size)

    ax.set_xlabel("Log-moneyness (k)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel(z_label)
    ax.set_title(_spline_title(sheet_name, display_fit_mode, "surface"))
    if invalid_k:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def plot_spline_heatmap(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv", fit_mode=None):
    requested_fit_mode = fit_mode
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())

    all_k = []

    slice_data = {}
    display_fit_mode = "unknown"
    for i, T in enumerate(expiries):
        strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)
        lam = choose_lambda(strikes, call_prices, S, r, T, fit_mode=requested_fit_mode)
        result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward, fit_mode=requested_fit_mode)
        display_fit_mode = result.get("fit_mode", display_fit_mode)

        k_values = np.log(strikes / forward)
        slice_data[T] = (result, forward)

        all_k.extend(k_values.tolist())

    k_grid = np.linspace(min(all_k), max(all_k), 200)
    T_grid = np.array(expiries)

    W = np.full((len(T_grid), len(k_grid)), np.nan)
    invalid_k = []
    invalid_t = []

    for i, T in enumerate(expiries):
        result, forward = slice_data[T]

        K_grid = forward * np.exp(k_grid)
        iv_grid = prices_to_iv(result, K_grid, forward)
        iv_mask = _finite_mask(iv_grid)
        invalid_k.extend(k_grid[~iv_mask].tolist())
        invalid_t.extend([T] * int(np.sum(~iv_mask)))

        if plot_type == "iv":
            W[i, :] = iv_grid
            label = "Implied Volatility"
        else:
            W[i, :] = iv_grid**2 * T
            label = "Total Variance"

    fig, ax = plt.subplots(figsize=(11, 6))
    masked_w = np.ma.masked_invalid(W)
    cf = ax.contourf(k_grid, T_grid, masked_w, levels=40, cmap="RdYlGn_r")
    fig.colorbar(cf, ax=ax, label=label)

    cl = ax.contour(k_grid, T_grid, masked_w, levels=12, colors="black", linewidths=0.6, alpha=0.45)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%.4f", inline_spacing=4)
    if invalid_k:
        ax.scatter(invalid_k, invalid_t, color="red", marker="x", s=12, linewidths=0.8, label="Invalid IV inversion")
        ax.legend(loc="upper right")
    _annotate_invalid_iv(ax, len(invalid_k), W.size)

    ax.set_xlabel("Log-moneyness k = log(K / F)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_title(
        _spline_title(sheet_name, display_fit_mode, label),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()    


def _spline_slice_plot_data(T, sheet_name, filepath, plot_type, fit_mode):
    strikes, call_prices, market_vols, spot, forward, r = get_spline_data(T, sheet_name, filepath)
    lam = choose_lambda(strikes, call_prices, spot, r, T, fit_mode=fit_mode)
    result = fit_smoothing_spline(
        strikes,
        call_prices,
        lam,
        spot,
        r,
        T,
        forward=forward,
        fit_mode=fit_mode,
    )

    k_values = np.log(strikes / forward)
    k_grid = np.linspace(min(k_values), max(k_values), 200)
    K_grid = forward * np.exp(k_grid)
    iv_grid = prices_to_iv(result, K_grid, forward)
    iv_at_k = prices_to_iv(result, strikes, forward)

    if plot_type == "iv":
        fitted_grid = iv_grid
        fitted_at_k = iv_at_k
        market_y = market_vols
        y_label = "Implied Volatility"
    else:
        fitted_grid = iv_grid**2 * T
        fitted_at_k = iv_at_k**2 * T
        market_y = market_vols**2 * T
        y_label = "Total Implied Variance"

    return {
        "fit_mode": result.get("fit_mode", fit_mode),
        "strikes": strikes,
        "call_prices": call_prices,
        "result": result,
        "k_values": k_values,
        "k_grid": k_grid,
        "K_grid": K_grid,
        "market_y": market_y,
        "fitted_grid": fitted_grid,
        "residuals": fitted_at_k - market_y,
        "iv_grid": iv_grid,
        "iv_at_k": iv_at_k,
        "y_label": y_label,
    }


def compare_spline_weightings_slice(T, sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):
    rows = [
        _spline_slice_plot_data(T, sheet_name, filepath, plot_type, "unweighted"),
        _spline_slice_plot_data(T, sheet_name, filepath, plot_type, "weighted"),
    ]
    show_price_panel = any(
        _is_shortest_maturity(T, sheet_name, filepath)
        or np.any(~_finite_mask(row["iv_grid"]))
        or np.any(~_finite_mask(row["iv_at_k"]))
        for row in rows
    )
    nrows = 3 if show_price_panel else 2
    height_ratios = [3, 1, 2] if show_price_panel else [3, 1]
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(16, 9 if show_price_panel else 7),
        gridspec_kw={"height_ratios": height_ratios},
        sharex="col",
    )

    for col, row in enumerate(rows):
        grid_mask = _finite_mask(row["iv_grid"])
        residual_mask = _finite_mask(row["residuals"])
        title = _spline_title(sheet_name, row["fit_mode"], f"T = {T:.6f}y")

        axes[0, col].scatter(row["k_values"], row["market_y"], color="black", s=25, label="Market Data")
        plot_mask, _ = _plot_finite_line(
            axes[0, col],
            row["k_grid"],
            row["fitted_grid"],
            color="red",
            linewidth=2,
            label="Smoothing Spline Fit",
        )
        _mark_invalid_x(axes[0, col], row["k_grid"], plot_mask)
        _annotate_invalid_iv(axes[0, col], int(np.sum(~grid_mask)), len(row["iv_grid"]))
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")
        axes[0, col].set_ylabel(row["y_label"])
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(loc="upper right")

        axes[1, col].bar(
            row["k_values"][residual_mask],
            row["residuals"][residual_mask],
            width=0.01,
            color="steelblue",
            alpha=0.7,
        )
        _mark_invalid_x(axes[1, col], row["k_values"], residual_mask, label="Invalid knot IV")
        axes[1, col].axhline(0, color="black", linewidth=0.8)
        axes[1, col].set_ylabel("Residual")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].text(
            0.98,
            0.90,
            f"RMSE = {_finite_rmse(row['residuals']):.2e}",
            transform=axes[1, col].transAxes,
            fontsize=9,
            ha="right",
            va="top",
        )

        if show_price_panel:
            fitted_calls = np.array([evaluate_spline(row["result"], float(K)) for K in row["K_grid"]])
            axes[2, col].scatter(row["k_values"], row["call_prices"], color="black", s=22, label="Market Calls")
            axes[2, col].plot(row["k_grid"], fitted_calls, color="red", linewidth=1.8, label="Spline Call Price")
            axes[2, col].set_ylabel("Call Price")
            axes[2, col].set_xlabel("Log-moneyness (k)")
            axes[2, col].grid(True, alpha=0.3)
            axes[2, col].legend(loc="upper right")
        else:
            axes[1, col].set_xlabel("Log-moneyness (k)")

    fig.suptitle(f"{sheet_name} - Spline Weighting Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def compare_spline_weightings_multi(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    colours = cm.viridis(np.linspace(0.1, 0.9, len(expiries)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, fit_mode in zip(axes, ["unweighted", "weighted"]):
        total_invalid = 0
        total_points = 0
        y_label = "Implied Volatility" if plot_type == "iv" else "Total Variance"
        for i, T in enumerate(expiries):
            strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)
            lam = choose_lambda(strikes, call_prices, S, r, T, fit_mode=fit_mode)
            result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward, fit_mode=fit_mode)
            k_values = np.log(strikes / forward)
            k_grid = np.linspace(k_values.min(), k_values.max(), 200)
            K_grid = forward * np.exp(k_grid)
            iv_grid = prices_to_iv(result, K_grid, forward)
            iv_mask = _finite_mask(iv_grid)
            total_invalid += int(np.sum(~iv_mask))
            total_points += len(iv_grid)

            if plot_type == "iv":
                market_y = market_vols
                fitted_grid = iv_grid
            else:
                market_y = market_vols**2 * T
                fitted_grid = iv_grid**2 * T

            _plot_finite_line(ax, k_grid, fitted_grid, color=colours[i], alpha=0.8, label=f"T={T:.3f}")
            invalid_label = "Invalid IV inversion" if total_invalid == int(np.sum(~iv_mask)) else "_nolegend_"
            _mark_invalid_x(ax, k_grid, iv_mask, label=invalid_label)
            ax.scatter(k_values, market_y, color=colours[i], s=18)

        ax.set_title(_spline_title(sheet_name, fit_mode, "multi-maturity fit"))
        ax.set_xlabel("Log-moneyness (k)")
        ax.set_ylabel(y_label)
        _annotate_invalid_iv(ax, total_invalid, total_points)
        if plot_type != "iv":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(f"{sheet_name} - Spline Weighting Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def _spline_surface_matrix(sheet_name, filepath, plot_type, fit_mode, grid_points=200):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    all_k = []
    slice_data = {}
    for T in expiries:
        strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)
        lam = choose_lambda(strikes, call_prices, S, r, T, fit_mode=fit_mode)
        result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward, fit_mode=fit_mode)
        k_values = np.log(strikes / forward)
        all_k.extend(k_values.tolist())
        slice_data[T] = (result, forward)

    k_grid = np.linspace(min(all_k), max(all_k), grid_points)
    T_grid = np.array(expiries)
    W = np.full((len(T_grid), len(k_grid)), np.nan)
    invalid_k = []
    invalid_t = []

    for i, T in enumerate(expiries):
        result, forward = slice_data[T]
        K_grid = forward * np.exp(k_grid)
        iv_grid = prices_to_iv(result, K_grid, forward)
        iv_mask = _finite_mask(iv_grid)
        invalid_k.extend(k_grid[~iv_mask].tolist())
        invalid_t.extend([T] * int(np.sum(~iv_mask)))
        W[i, :] = iv_grid if plot_type == "iv" else iv_grid**2 * T

    label = "Implied Volatility" if plot_type == "iv" else "Total Variance"
    return k_grid, T_grid, W, invalid_k, invalid_t, label


def compare_spline_weightings_surface(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):
    fig = plt.figure(figsize=(16, 7))
    for idx, fit_mode in enumerate(["unweighted", "weighted"], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        k_grid, T_grid, W, invalid_k, invalid_t, label = _spline_surface_matrix(
            sheet_name,
            filepath,
            plot_type,
            fit_mode,
        )
        K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
        ax.plot_surface(K_mesh, T_mesh, np.ma.masked_invalid(W), cmap=cm.viridis, alpha=0.9)
        finite_w = W[np.isfinite(W)]
        z_marker = float(np.nanmin(finite_w)) if finite_w.size else 0.0
        if invalid_k:
            ax.scatter(
                invalid_k,
                invalid_t,
                np.full(len(invalid_k), z_marker),
                color="red",
                s=8,
                depthshade=False,
                label="Invalid IV inversion",
            )
            ax.legend(loc="upper right")
        _annotate_invalid_iv_3d(ax, len(invalid_k), W.size)
        ax.set_xlabel("Log-moneyness (k)")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel(label)
        ax.set_title(_spline_title(sheet_name, fit_mode, "surface"))

    fig.suptitle(f"{sheet_name} - Spline Weighting Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def compare_spline_weightings_heatmap(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, fit_mode in zip(axes, ["unweighted", "weighted"]):
        k_grid, T_grid, W, invalid_k, invalid_t, label = _spline_surface_matrix(
            sheet_name,
            filepath,
            plot_type,
            fit_mode,
        )
        masked_w = np.ma.masked_invalid(W)
        cf = ax.contourf(k_grid, T_grid, masked_w, levels=40, cmap="RdYlGn_r")
        fig.colorbar(cf, ax=ax, label=label)
        cl = ax.contour(k_grid, T_grid, masked_w, levels=12, colors="black", linewidths=0.6, alpha=0.45)
        ax.clabel(cl, inline=True, fontsize=7, fmt="%.4f", inline_spacing=4)
        if invalid_k:
            ax.scatter(invalid_k, invalid_t, color="red", marker="x", s=10, linewidths=0.8, label="Invalid IV inversion")
            ax.legend(loc="upper right")
        _annotate_invalid_iv(ax, len(invalid_k), W.size)
        ax.set_xlabel("Log-moneyness k = log(K / F)")
        ax.set_title(_spline_title(sheet_name, fit_mode, "surface"))
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Maturity T (years)")
    fig.suptitle(f"{sheet_name} - Spline Weighting Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def single_slice_comparison(T, sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):

     # SVI 
    strikes, market_vols, forward, k_values, w_market = get_slice_from_data(T, sheet_name, filepath)
    svi_params = fit_single_slice_with_bound(k_values, w_market)
    k_grid = np.linspace(min(k_values), max(k_values), 200)
    w_svi_grid = total_variance(k_grid, **svi_params)
    w_svi_at_k = total_variance(k_values, **svi_params)

    if plot_type == "iv":
        svi_y_grid = np.sqrt(w_svi_grid / T)
        svi_y_at_k = np.sqrt(w_svi_at_k / T)
        market_y = market_vols
        y_label = "Implied Volatility"
    else:
        svi_y_grid = w_svi_grid
        svi_y_at_k = w_svi_at_k
        market_y = w_market
        y_label = "Total Variance"

    svi_res = svi_y_at_k - market_y

    #Splines
    strikes, call_prices, market_vols, spot, forward, r = get_spline_data(T, sheet_name, filepath)
    lam = choose_lambda(strikes, call_prices, spot, r, T)
    spline_result = fit_smoothing_spline(strikes, call_prices, lam, spot, r, T, forward=forward)
    fit_mode = spline_result.get("fit_mode", "unknown")
    K_grid = forward * np.exp(k_grid)
    iv_grid = prices_to_iv(spline_result, K_grid, forward)
    iv_at_k = prices_to_iv(spline_result, strikes, forward)
    grid_iv_mask = _finite_mask(iv_grid)
    knot_iv_mask = _finite_mask(iv_at_k)
    invalid_grid_count = int(np.sum(~grid_iv_mask))

    if plot_type == "iv":
        spline_y_grid = iv_grid
        spline_y_at_k = iv_at_k
    else:
        spline_y_grid = iv_grid**2 * T
        spline_y_at_k = iv_at_k**2 * T

    spline_res = spline_y_at_k - market_y
    show_price_panel = (
        _is_shortest_maturity(T, sheet_name, filepath)
        or invalid_grid_count > 0
        or np.any(~knot_iv_mask)
    )

    if show_price_panel:
        fig, axes = plt.subplots(
            3, 2, figsize=(12, 9),
            gridspec_kw={"height_ratios": [3, 1, 2]},
            sharex="col"
        )
    else:
        fig, axes = plt.subplots(
            2, 2, figsize=(12, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex="col"
        )

    fig.suptitle(
        f"{sheet_name} - SVI vs {_fit_mode_label(fit_mode)} (T={T:.3f})",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].scatter(k_values, market_y, color="black", s=25)
    axes[0, 0].plot(k_grid, svi_y_grid, color="blue", label="SVI")
    axes[0, 0].set_title("SVI")
    axes[0, 0].set_ylabel(y_label)
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].bar(k_values, svi_res, width=0.01, color="blue")
    axes[1, 0].axhline(0, color="black")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Residual")

    axes[0, 1].scatter(k_values, market_y, color="black", s=25)
    plot_mask, _ = _plot_finite_line(axes[0, 1], k_grid, spline_y_grid, color="red", label="Spline")
    _mark_invalid_x(axes[0, 1], k_grid, plot_mask)
    _annotate_invalid_iv(axes[0, 1], invalid_grid_count, len(iv_grid))
    axes[0, 1].set_title(_fit_mode_label(fit_mode))
    axes[0, 1].grid(True, alpha=0.3)

    finite_spline_res = _finite_mask(spline_res)
    axes[1, 1].bar(k_values[finite_spline_res], spline_res[finite_spline_res], width=0.01, color="red")
    _mark_invalid_x(axes[1, 1], k_values, finite_spline_res, label="Invalid knot IV")
    axes[1, 1].axhline(0, color="black")
    axes[1, 1].set_xlabel("k" if not show_price_panel else "")

    if show_price_panel:
        axes[2, 0].axis("off")
        fitted_call_grid = np.array([evaluate_spline(spline_result, float(K)) for K in K_grid])
        axes[2, 1].scatter(k_values, call_prices, color="black", s=25, label="Market Calls")
        axes[2, 1].plot(k_grid, fitted_call_grid, color="red", linewidth=1.8, label="Spline Call Price")
        axes[2, 1].set_xlabel("k")
        axes[2, 1].set_ylabel("Call Price")
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def multi_slice_comparison(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    colours = cm.viridis(np.linspace(0.1, 0.9, len(expiries)))
    fitted = calibrate_surface(sheet_name, filepath)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    spline_invalid = 0
    spline_total = 0
    spline_fit_mode = "unknown"

    for i, T in enumerate(expiries):
        colour = colours[i]
        label = f"T={T:.3f}"

        # SVI
        strikes, market_vols, forward, k_values, w_market = get_slice_from_data(T, sheet_name, filepath)
        params = fitted[T]
        k_grid = np.linspace(min(k_values), max(k_values), 200)
        w_grid = total_variance(k_grid, **params)
        if plot_type == "iv":
            market_y = market_vols
            svi_y = np.sqrt(w_grid / T)
            y_label = "Implied Volatility"
        else:
            market_y = w_market
            svi_y = w_grid
            y_label = "Total Variance"

        axes[0].scatter(k_values, market_y, color=colour, s=15)
        axes[0].plot(k_grid, svi_y, color=colour, label=label)

        #Splines
        strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)
        lam = choose_lambda(strikes, call_prices, S, r, T)
        result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward)
        spline_fit_mode = result.get("fit_mode", spline_fit_mode)
        K_grid = forward * np.exp(k_grid)
        iv_grid = prices_to_iv(result, K_grid, forward)
        iv_mask = _finite_mask(iv_grid)
        previous_invalid = spline_invalid
        spline_invalid += int(np.sum(~iv_mask))
        spline_total += len(iv_grid)
        if plot_type == "iv":
            spline_y = iv_grid
        else:
            spline_y = iv_grid**2 * T
        axes[1].scatter(k_values, market_y, color=colour, s=15)
        _plot_finite_line(axes[1], k_grid, spline_y, color=colour, label=label)
        invalid_label = "Invalid IV inversion" if previous_invalid == 0 and np.any(~iv_mask) else "_nolegend_"
        _mark_invalid_x(axes[1], k_grid, iv_mask, label=invalid_label)

    axes[0].set_title("SVI")
    axes[1].set_title(_fit_mode_label(spline_fit_mode))

    for ax in axes:
        ax.set_xlabel("Log-moneyness (k)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(y_label)
    _annotate_invalid_iv(axes[1], spline_invalid, spline_total)
    if plot_type != "iv":
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    fig.suptitle(
        f"{sheet_name} - SVI vs {_fit_mode_label(spline_fit_mode)} multi-maturity fit",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def Surface_comparison(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    all_k = []
    fitted = calibrate_surface(sheet_name, filepath)

    for T in expiries:
        strikes, market_vols, forward, k_values, _ = get_slice_from_data(T, sheet_name, filepath)
        all_k.extend(k_values.tolist())

    k_grid = np.linspace(min(all_k), max(all_k), 1000)
    T_knots = np.array(expiries)
    W_svi= np.zeros((len(T_knots), len(k_grid)))
    for i, T in enumerate(T_knots):
        params=fitted[T]
        if plot_type == "iv":
            W_svi[i, :] = np.sqrt(total_variance(k_grid, **params)/T)
            z_label = "Implied Volatility"
        else:
            W_svi[i, :] = total_variance(k_grid, **params)
            z_label = "Total Variance"
    K_mesh, T_mesh = np.meshgrid(k_grid, T_knots) #convert 1D arrays to 2D grids


    slice_data = {}
    spline_fit_mode = "unknown"
    for i, T in enumerate(expiries):
        strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)
        lam = choose_lambda(strikes, call_prices, S, r, T)
        result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward)
        spline_fit_mode = result.get("fit_mode", spline_fit_mode)

        k_values = np.log(strikes / forward)
        slice_data[T] = (result, forward)

        all_k.extend(k_values.tolist())



    K_mesh, T_mesh = np.meshgrid(k_grid, T_knots)
    W_spline = np.full_like(K_mesh, np.nan, dtype=float)
    invalid_k = []
    invalid_t = []
    for i, T in enumerate(expiries):
        result, forward = slice_data[T]

        K_grid = forward * np.exp(k_grid)
        iv_grid = prices_to_iv(result, K_grid, forward)
        iv_mask = _finite_mask(iv_grid)
        invalid_k.extend(k_grid[~iv_mask].tolist())
        invalid_t.extend([T] * int(np.sum(~iv_mask)))

        if plot_type == "iv":
            W_spline[i, :] = iv_grid
            z_label = "Implied Volatility"
        else:
            W_spline[i, :] = iv_grid**2 * T
            z_label = "Total Variance"

    fig = plt.figure(figsize=(16, 7))
    #SVI
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(K_mesh, T_mesh, W_svi, cmap=cm.viridis, alpha=0.9)
    ax1.set_xlabel("Log-moneyness (k)")
    ax1.set_ylabel("Maturity T (years)")
    ax1.set_zlabel(z_label)
    ax1.set_title(f"SVI Surface — {sheet_name}")
    #SPline
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(K_mesh, T_mesh, np.ma.masked_invalid(W_spline), cmap=cm.viridis, alpha=0.9)
    finite_w = W_spline[np.isfinite(W_spline)]
    z_marker = float(np.nanmin(finite_w)) if finite_w.size else 0.0
    if invalid_k:
        ax2.scatter(
            invalid_k,
            invalid_t,
            np.full(len(invalid_k), z_marker),
            color="red",
            s=8,
            depthshade=False,
            label="Invalid IV inversion",
        )
    ax2.set_xlabel("Log-moneyness (k)")
    ax2.set_ylabel("Maturity T (years)")
    ax2.set_zlabel(z_label)
    ax2.set_title(_spline_title(sheet_name, spline_fit_mode, "surface"))
    _annotate_invalid_iv_3d(ax2, len(invalid_k), W_spline.size)
    if invalid_k:
        ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def Heatmap_comparison(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="iv"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())

    all_k = []
    fitted = calibrate_surface(sheet_name, filepath)
    for T in expiries:
        strikes, market_vols, forward, k_values, _ = get_slice_from_data(T, sheet_name, filepath)
        all_k.extend(k_values.tolist())
        
    k_grid = np.linspace(min(all_k), max(all_k), 200)
    T_grid = np.array(expiries)

    param_names = ["a", "b", "rho", "m", "sigma"]
    param_curves = {}
    for p in param_names:
        column = []
        for T in T_grid:
            value = fitted[T][p]
            column.append(value)
        param_curves[p] = np.array(column)

    W_svi = np.zeros((len(T_grid), len(k_grid)))
    for i, T in enumerate(T_grid):
        a = np.interp(T, T_grid, param_curves["a"])
        b = np.interp(T, T_grid, param_curves["b"])
        rho = np.interp(T, T_grid, param_curves["rho"])
        m = np.interp(T, T_grid, param_curves["m"])
        sigma = np.interp(T, T_grid, param_curves["sigma"])
        if plot_type == "iv":
            W_svi[i, :] = np.sqrt(total_variance(k_grid,  a, b, rho, m, sigma)/T)
            z_label = "Implied Volatility"
        else:
            W_svi[i, :] = total_variance(k_grid,  a, b, rho, m, sigma)
            z_label = "Total Variance"

    slice_data = {}
    spline_fit_mode = "unknown"
    for i, T in enumerate(expiries):
        strikes, call_prices, market_vols, S, forward, r = get_spline_data(T, sheet_name, filepath)
        lam = choose_lambda(strikes, call_prices, S, r, T)
        result = fit_smoothing_spline(strikes, call_prices, lam, S, r, T, forward=forward)
        spline_fit_mode = result.get("fit_mode", spline_fit_mode)
        slice_data[T] = (result, forward)


    W_spline = np.full((len(T_grid), len(k_grid)), np.nan)
    invalid_k = []
    invalid_t = []

    for i, T in enumerate(expiries):
        result, forward = slice_data[T]

        K_grid = forward * np.exp(k_grid)
        iv_grid = prices_to_iv(result, K_grid, forward)
        iv_mask = _finite_mask(iv_grid)
        invalid_k.extend(k_grid[~iv_mask].tolist())
        invalid_t.extend([T] * int(np.sum(~iv_mask)))

        if plot_type == "iv":
            W_spline[i, :] = iv_grid
            label = "Implied Volatility"
        else:
            W_spline[i, :] = iv_grid**2 * T
            label = "Total Variance"
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    #SVI
    cf1 = axes[0].contourf(k_grid, T_grid, W_svi, levels=40, cmap="RdYlGn_r")
    fig.colorbar(cf1, ax=axes[0], label=label)

    cl1 = axes[0].contour(k_grid, T_grid, W_svi, levels=12,
                      colors="black", linewidths=0.6, alpha=0.45)
    axes[0].clabel(cl1, inline=True, fontsize=7, fmt="%.4f", inline_spacing=4)

    axes[0].set_xlabel("Log-moneyness k = log(K / F)")
    axes[0].set_ylabel("Maturity T (years)")
    axes[0].set_title(f"SVI Surface — {sheet_name}")

    # Splines
    masked_spline = np.ma.masked_invalid(W_spline)
    cf2 = axes[1].contourf(k_grid, T_grid, masked_spline, levels=40, cmap="RdYlGn_r")
    fig.colorbar(cf2, ax=axes[1], label=label)

    cl2 = axes[1].contour(k_grid, T_grid, masked_spline, levels=12,
                      colors="black", linewidths=0.6, alpha=0.45)
    axes[1].clabel(cl2, inline=True, fontsize=7, fmt="%.4f", inline_spacing=4)
    if invalid_k:
        axes[1].scatter(invalid_k, invalid_t, color="red", marker="x", s=10, linewidths=0.8, label="Invalid IV inversion")
        axes[1].legend(loc="upper right")
    _annotate_invalid_iv(axes[1], len(invalid_k), W_spline.size)

    axes[1].set_xlabel("Log-moneyness k = log(K / F)")
    axes[1].set_title(_spline_title(sheet_name, spline_fit_mode, "surface"))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filepath = "tests/data/Surfaces.xlsx"
    data = list_available_data(filepath)

    #choose surface
    sheets = list(data.keys())
    print("\nChoose a surface to plot, press enter.")
    print("\nAvailable surfaces:")
    for i, name in enumerate(sheets):
        print(f"[{i + 1}] {name}  ({len(data[name])} expiries)")

    choice = input("\nSelect surface: ").strip()
    sheet_idx = int(choice) - 1 if choice else 0
    sheet_name = sheets[sheet_idx]

    #Choose Method
    method_choice = input("\nSelect Method [1 = SVI, 2 = Cubic Splines, 3 = Compare]: ").strip()
   
    if method_choice.upper()=="1":
         #choose expiry
        expiries = data[sheet_name]
        print(f"\nExpiries in {sheet_name}:")
        for i, T in enumerate(expiries):
            days = T * 365
            print(f"[{i + 1:>2}] T = {T:.6f}  (~{days:.0f} days)")
        print(f"[ 0] Plot ALL slices")
        choice = input("\nSelect expiry [0 = all, H = heatmap, S = surface, E = error]: ").strip()
        if choice.upper() == "H":
            print(f"\nPlotting variance heatmap for {sheet_name}")
            plot_variance_heatmap(sheet_name, filepath)
        elif choice.upper() == "S":
            print(f"\nPlotting variance surface for {sheet_name}")
            plot_surface(sheet_name, filepath)
        elif choice.upper() == "E":
            print("\nError plot mode:")
            print("[1] Sequential init (warm-start from previous slice)")
            print("[2] Full DE on every slice (slower, matches Jose's likely approach)")
            print("[3] Both overlaid for comparison")
            mode_choice = input("Select mode [1/2/3, default=3]: ").strip()
            mode_map = {"1": "sequential", "2": "de", "3": "both", "": "both"}
            mode = mode_map.get(mode_choice, "both")
            print(f"\nPlotting log10(RMSE) error for {sheet_name} [{mode}]")
            plot_error_log10_rmse(sheet_name, filepath, mode=mode)
        else:
            expiry_idx = int(choice) if choice else 0

            #choose plot type
            choice2 = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
            plot_type = "iv" if choice2 == "2" else "total_var"

            #plot
            if expiry_idx == 0:
                print(f"\nFitting all {len(expiries)} slices in {sheet_name}")
                plot_multi_slice(sheet_name, filepath, plot_type=plot_type)
            else:
                T = expiries[expiry_idx - 1]
                print(f"\nFitting for T = {T:.6f}")
                plot_single_slice(T, sheet_name, filepath, plot_type=plot_type)

    elif method_choice.upper()=="2":
        spline_mode_choice = input(
            "\nSpline mode — [1] Current default  [2] Compare weightings: "
        ).strip()
        compare_weightings = spline_mode_choice == "2"

         #choose expiry
        expiries = data[sheet_name]
        print(f"\nExpiries in {sheet_name}:")
        for i, T in enumerate(expiries):
            days = T * 365
            print(f"[{i + 1:>2}] T = {T:.6f}  (~{days:.0f} days)")
        print(f"[ 0] Plot ALL slices")
        choice = input("\nSelect expiry [0 = all, H = heatmap,  S = surface]: ").strip()
        if choice.upper() == "S":
            choice2 = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
            plot_type = "iv" if choice2 == "2" else "total_var"
            if compare_weightings:
                print(f"\nComparing spline weighting surfaces for {sheet_name}")
                compare_spline_weightings_surface(sheet_name, filepath, plot_type)
            else:
                print(f"\nPlotting current default spline surface for {sheet_name}")
                plot_spline_surface(sheet_name, filepath, plot_type)
        elif choice.upper() == "H":
            choice2 = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
            plot_type = "iv" if choice2 == "2" else "total_var"
            if compare_weightings:
                print(f"\nComparing spline weighting heatmaps for {sheet_name}")
                compare_spline_weightings_heatmap(sheet_name, filepath, plot_type)
            else:
                print(f"\nPlotting current default spline heatmap for {sheet_name}")
                plot_spline_heatmap(sheet_name, filepath, plot_type)
     
        else:
            expiry_idx = int(choice) if choice else 0

            #choose plot type
            choice2 = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
            plot_type = "iv" if choice2 == "2" else "total_var"

            #plot
            if expiry_idx == 0:
                if compare_weightings:
                    print(f"\nComparing all {len(expiries)} spline slices in {sheet_name}")
                    compare_spline_weightings_multi(sheet_name, filepath, plot_type)
                else:
                    print(f"\nFitting all {len(expiries)} current default spline slices in {sheet_name}")
                    plot_multi_spline_slice(sheet_name, filepath, plot_type)
            else:
                T = expiries[expiry_idx - 1]
                if compare_weightings:
                    print(f"\nComparing spline weightings for T = {T:.6f}")
                    compare_spline_weightings_slice(T, sheet_name, filepath, plot_type)
                else:
                    print(f"\nFitting current default spline for T = {T:.6f}")
                    plot_spline_slice(T, sheet_name, plot_type, filepath)
        
    elif method_choice.upper()=="3":
        #choose expiry
        expiries = data[sheet_name]
        print(f"\nExpiries in {sheet_name}:")
        for i, T in enumerate(expiries):
            days = T * 365
            print(f"[{i + 1:>2}] T = {T:.6f}  (~{days:.0f} days)")
        print(f"[ 0] Plot ALL slices")
        choice = input("\nSelect expiry [0 = all, H = heatmap,  S = surface]: ").strip()
        if choice.upper() == "S":
            choice2 = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
            plot_type = "iv" if choice2 == "2" else "total_var"
            print(f"\nPlotting surface for {sheet_name}")
            Surface_comparison(sheet_name, filepath, plot_type)
        elif choice.upper() == "H":
            choice2 = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
            plot_type = "iv" if choice2 == "2" else "total_var"
            print(f"\nPlotting heatmap for {sheet_name}")
            Heatmap_comparison(sheet_name, filepath, plot_type)
     
        else:
            expiry_idx = int(choice) if choice else 0

            #choose plot type
            choice2 = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
            plot_type = "iv" if choice2 == "2" else "total_var"

            #plot
            if expiry_idx == 0:
                print(f"\nFitting all {len(expiries)} slices in {sheet_name}")
                multi_slice_comparison(sheet_name, filepath, plot_type)
            else:
                T = expiries[expiry_idx - 1]
                print(f"\nFitting for T = {T:.6f}")
                single_slice_comparison(T, sheet_name, filepath,plot_type)


    
