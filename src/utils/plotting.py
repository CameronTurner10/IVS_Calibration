import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.svi.optimisation.SVI_SliceFit import fit_svi_slice, total_variance

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

    fitted_params = fit_svi_slice(
        strikes=strikes,
        market_vols=market_vols,
        T=T,
        forward=forward
    )

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

    for i, T in enumerate(expiries):
        strikes, market_vols, forward, k_values, w_market = get_slice_from_data(T, sheet_name, filepath)

        fitted_params = fit_svi_slice(
            strikes=strikes,
            market_vols=market_vols,
            T=T,
            forward=forward
        )

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
    ax_main.legend(loc="upper right", fontsize=7, ncol=3)
    ax_main.grid(True, alpha=0.3)

    ax_res.axhline(y=0, color="black", linewidth=0.8)
    ax_res.set_xlabel("Log-moneyness (k)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, alpha=0.3)

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


if __name__ == "__main__":
    filepath = "tests/data/Surfaces.xlsx"
    data = list_available_data(filepath)

    #choose surface
    sheets = list(data.keys())
    print("\nIndicate choice by entering a number and pressing enter.")
    print("\nAvailable surfaces:")
    for i, name in enumerate(sheets):
        print(f"  [{i + 1}] {name}  ({len(data[name])} expiries)")

    choice = input("\nSelect surface: ").strip()
    sheet_idx = int(choice) - 1 if choice else 0
    sheet_name = sheets[sheet_idx]

    #choose expiry
    expiries = data[sheet_name]
    print(f"\nExpiries in {sheet_name}:")
    for i, T in enumerate(expiries):
        days = T * 365
        print(f"  [{i + 1:>2}] T = {T:.6f}  (~{days:.0f} days)")
    print(f"  [ 0] Plot ALL slices")

    choice = input("\nSelect expiry [0 = all]: ").strip()
    expiry_idx = int(choice) if choice else 0

    #choose plot type
    choice = input("\nPlot type — [1] Total Variance  [2] Implied Vol: ").strip()
    plot_type = "iv" if choice == "2" else "total_var"

    #plot
    if expiry_idx == 0:
        print(f"\nFitting all {len(expiries)} slices in {sheet_name}")
        plot_multi_slice(sheet_name, filepath, plot_type=plot_type)
    else:
        T = expiries[expiry_idx - 1]
        print(f"\nFitting for T = {T:.6f}")
        plot_single_slice(T, sheet_name, filepath, plot_type=plot_type)
