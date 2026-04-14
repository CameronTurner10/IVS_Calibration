import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.svi.optimisation.local_optimizers import fit_svi_slice, total_variance
from src.svi.optimisation.arbitrage import calibrate_surface, fit_single_slice_with_bound
from mpl_toolkits.mplot3d import Axes3D

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

    plt.tight_layout()
    plt.show()

    return fitted_params


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
    ax_main.set_yscale('log')
    ax_main.legend(loc="upper right", fontsize=7, ncol=3)
    ax_main.grid(True, alpha=0.3)

    ax_res.axhline(y=0, color="black", linewidth=0.8)
    ax_res.set_xlabel("Log-moneyness (k)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_interpolated_surface(sheet_name, filepath="tests/data/Surfaces.xlsx", plot_type="total_var"):
    
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


    W = np.zeros((len(T_knots), len(k_grid)))
    for i, T in enumerate(T_knots):
        params=fitted[T]
        W[i, :] = total_variance(k_grid, **params)

    K_mesh, T_mesh = np.meshgrid(k_grid, T_knots) #convert 1D arrays to 2D grids


    fig= plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111,projection="3d")
    surface = ax.plot_surface(K_mesh,T_mesh,W)

    ax.set_xlabel("Log-moneyness k = log(K / F)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Total implied variance w(k,T)")
    ax.set_title(f"SVI Variance Surface — {sheet_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
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
    plt.tight_layout()

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


def plot_spline_slice(
    result: dict, T: float, market_strikes: np.ndarray, market_vols: np.ndarray, sheet_name: str = "", plot_type: str = "iv"
) -> None:
    """
    Plot a single slice of a fitted smoothing spline against market data.

    Parameters
    ----------
    result : dict
        Dict returned by fit_smoothing_spline
    T : float
        Maturity in years
    market_strikes : np.ndarray
        Array of market strike prices
    market_vols : np.ndarray
        Array of market implied volatilities
    sheet_name : str, optional
        Sheet name for the plot title, by default ""
    plot_type : str, optional
        Either "iv" or "total_var", by default "iv"

    Returns
    -------
    None

    Notes
    -----
    Mirror plot_single_slice() structure exactly. Replace SVI fit with evaluate_spline() + prices_to_iv().
    """
    raise NotImplementedError("Not yet implemented")


def plot_spline_surface(spline_dict: dict, sheet_name: str = "") -> None:
    """
    Plot the full 3D smoothing spline implied volatility surface and a 2D heatmap.

    Parameters
    ----------
    spline_dict : dict
        Dict {T: result_dict} output of fit_all_splines
    sheet_name : str, optional
        Sheet name for the plot title, by default ""

    Returns
    -------
    None

    Notes
    -----
    Mirror existing SVI surface plot structure. 3D surface and heatmap side by side.
    """
    raise NotImplementedError("Not yet implemented")


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
