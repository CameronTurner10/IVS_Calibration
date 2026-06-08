"""Make the weighting-ablation figures for the report.

This script is just for the smoothing-spline weighting investigation. It makes
the five figures we decided were useful enough to keep:

- an overall weighting matrix across all surfaces,
- a before/after carry-correction example,
- a Surface 4 weighting matrix,
- Surface 4 unweighted maturity plots,
- Surface 4 weighted maturity plots.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.smoothing_spline.implementation.spline_model import (  # noqa: E402
    build_Q_matrix,
    build_R_matrix,
    fit_smoothing_spline,
    prices_to_iv,
)
from src.smoothing_spline.optimisation.fit_spline import choose_lambda as choose_project_lambda  # noqa: E402
from src.smoothing_spline.testing.spline_diagnostics import check_arbitrage  # noqa: E402
from src.utils.black_scholes import vega  # noqa: E402
from src.utils.root_finder import implied_vol  # noqa: E402


WORKBOOK = PROJECT_ROOT / "tests/data/Surfaces.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data/weighting_ablation"
LAMBDA_GRID = np.logspace(-6, 10, 80)
LEFT_WING_LIMIT = -0.10

VARIANTS = {
    "Inverse call-price squared": "inverse_call_price_squared",
    "Unweighted": "unweighted",
    "Capped relative 10x": "capped_relative",
    "Capped vega 1000x": "capped_vega",
}


def infer_delta(spot, forward, rate, maturity):
    """Work out the carry rate from the market forward."""
    return rate - np.log(forward / spot) / maturity


def cap_weights(weights, limit):
    """Cap the weights so one point cannot dominate everything."""
    middle = np.median(weights)
    return np.clip(weights, middle / limit, middle * limit)


def make_weights(name, call_prices, vegas):
    """Make the weights for one method in the ablation."""
    if name == "unweighted":
        weights = np.ones_like(call_prices)
    elif name == "capped_vega":
        weights = cap_weights(1.0 / np.maximum(vegas, 1e-12) ** 2, 1000)
    else:
        weights = 1.0 / np.maximum(call_prices, 1e-4) ** 2
        if name == "capped_relative":
            weights = cap_weights(weights, 10)

    return weights / np.median(weights)


def choose_lambda(strikes, call_prices, weights):
    """Pick the best lambda using AIC for these weights."""
    Q = build_Q_matrix(strikes)
    R = build_R_matrix(strikes)
    roughness = Q @ np.linalg.solve(R, Q.T)
    W = np.diag(weights)
    rhs = W @ call_prices
    scores = []

    for lam in LAMBDA_GRID:
        system = W + lam * roughness
        fitted = np.linalg.solve(system, rhs)
        residual = fitted - call_prices
        rss = float(residual.T @ W @ residual)
        degrees_of_freedom = np.trace(np.linalg.solve(system, W))
        scale = max(rss / len(strikes), 1e-12)
        scores.append(len(strikes) * np.log(scale) + 2 * degrees_of_freedom)

    return float(LAMBDA_GRID[np.argmin(scores)])


def fit_weighted_slice(strikes, call_prices, weights, lam, spot, rate, maturity, delta):
    """Fit one maturity slice using the chosen custom weights."""
    n = len(strikes)
    Q = build_Q_matrix(strikes)
    R = build_R_matrix(strikes)
    A = np.vstack([Q, -R.T])
    W = np.diag(weights)
    B = block_diag(W, lam * R)
    y = np.concatenate([W @ call_prices, np.zeros(n - 2)])
    h = np.diff(strikes)

    x0 = np.zeros(2 * n - 2)
    x0[:n] = call_prices

    def objective(x):
        return float(-y.T @ x + 0.5 * x.T @ B @ x)

    def equality_constraint(x):
        return A.T @ x

    def slope_constraints(x):
        prices = x[:n]
        gamma = x[n:]
        left_slope = (prices[1] - prices[0]) / h[0] - h[0] * gamma[0] / 6
        right_slope = (prices[-1] - prices[-2]) / h[-1] + h[-1] * gamma[-1] / 6
        return np.array([left_slope + np.exp(-rate * maturity), -right_slope])

    disc_delta = np.exp(-delta * maturity)
    disc_r = np.exp(-rate * maturity)
    bounds = [(disc_delta * spot - disc_r * strikes[0], disc_delta * spot)]
    bounds += [(0.0, None)] * (n - 1)
    bounds += [(0.0, None)] * (n - 2)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": equality_constraint},
            {"type": "ineq", "fun": slope_constraints},
        ],
    )

    return {
        "g": result.x[:n],
        "gamma": result.x[n:],
        "strikes": strikes,
        "S": spot,
        "r": rate,
        "T": maturity,
        "delta": delta,
        "success": bool(result.success),
    }


def prepare_slice(slice_df):
    """Get the useful arrays from one maturity slice."""
    slice_df = slice_df.sort_values("Strike")
    strikes = slice_df["Strike"].to_numpy(float)
    call_prices = slice_df["Call Price"].to_numpy(float)
    market_iv = slice_df["Volatility"].to_numpy(float)
    spot = float(slice_df["Spot"].iloc[0])
    forward = float(slice_df["Forward"].iloc[0])
    rate = float(slice_df["Discount Rate"].iloc[0])
    maturity = float(slice_df["Year Fraction"].iloc[0])
    vegas = np.array(
        [vega(forward, strike, maturity, vol, rate) for strike, vol in zip(strikes, market_iv)]
    )
    return strikes, call_prices, market_iv, spot, forward, rate, maturity, vegas


def run_ablation_fit(slice_df, variant, use_inferred_carry=True):
    """Run one weighting method and collect the results we plot later."""
    strikes, calls, market_iv, spot, forward, rate, maturity, vegas = prepare_slice(slice_df)
    weights = make_weights(variant, calls, vegas)
    lam = choose_lambda(strikes, calls, weights)
    delta = infer_delta(spot, forward, rate, maturity) if use_inferred_carry else 0.0
    result = fit_weighted_slice(strikes, calls, weights, lam, spot, rate, maturity, delta)
    fitted_iv = np.array(
        [implied_vol(forward, strike, maturity, rate, price, "call") for strike, price in zip(strikes, result["g"])]
    )

    return {
        "log_moneyness": np.log(strikes / forward),
        "market_iv": market_iv,
        "fitted_iv": fitted_iv,
        "market_calls": calls,
        "fitted_calls": result["g"],
        "vegas": vegas,
        "success": result["success"],
        "arbitrage_pass": check_arbitrage(result)["pass"],
    }


def colour_table(ax, values, row_labels, column_labels, formats, higher_is_better, title):
    """Draw a simple red-to-green table, with green meaning better."""
    colours = np.zeros_like(values, dtype=float)

    for column, higher in enumerate(higher_is_better):
        data = values[:, column]
        spread = np.nanmax(data) - np.nanmin(data)
        if spread == 0 or not np.isfinite(spread):
            colours[:, column] = 1
        elif higher:
            colours[:, column] = (data - np.nanmin(data)) / spread
        else:
            colours[:, column] = (np.nanmax(data) - data) / spread

    image = ax.imshow(colours, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            colour = "black" if colours[row, column] > 0.45 else "white"
            ax.text(
                column,
                row,
                format(values[row, column], formats[column]),
                ha="center",
                va="center",
                color=colour,
                fontsize=9,
            )

    return image


def calculate_overall_summary():
    """Summarise how each weighting method performs across all surfaces."""
    records = {label: [] for label in VARIANTS}
    workbook = pd.ExcelFile(WORKBOOK)

    for surface in workbook.sheet_names:
        surface_df = pd.read_excel(workbook, sheet_name=surface)
        for _, slice_df in surface_df.groupby("Year Fraction"):
            for label, variant in VARIANTS.items():
                fit = run_ablation_fit(slice_df, variant)
                left = fit["log_moneyness"] < LEFT_WING_LIMIT
                valid = left & np.isfinite(fit["fitted_iv"])
                stable = valid & (fit["vegas"] >= 1.0)
                records[label].append(
                    {
                        "iv_errors": fit["fitted_iv"][valid] - fit["market_iv"][valid],
                        "stable_errors": fit["fitted_iv"][stable] - fit["market_iv"][stable],
                        "price_errors": fit["fitted_calls"][left] - fit["market_calls"][left],
                        "failures": int(np.sum(left & ~np.isfinite(fit["fitted_iv"]))),
                        "success": fit["success"],
                        "arbitrage_pass": fit["arbitrage_pass"],
                    }
                )

    summary = {}
    for label, rows in records.items():
        iv_errors = np.concatenate([row["iv_errors"] for row in rows])
        stable_errors = np.concatenate([row["stable_errors"] for row in rows])
        price_errors = np.concatenate([row["price_errors"] for row in rows])
        summary[label] = [
            np.sqrt(np.mean(stable_errors**2)),
            np.sqrt(np.mean(iv_errors**2)),
            sum(row["failures"] for row in rows),
            np.sqrt(np.mean(price_errors**2)),
            np.mean([row["success"] for row in rows]),
            np.mean([row["arbitrage_pass"] for row in rows]),
        ]

    return summary


def create_executive_matrix():
    """Create the main overall weighting comparison figure."""
    summary = calculate_overall_summary()
    values = np.array([summary[label] for label in VARIANTS], dtype=float)
    headings = [
        "Stable left-wing\nIV RMSE",
        "All finite left-wing\nIV RMSE",
        "IV inversion\nfailures",
        "Left-wing price\nRMSE",
        "Fit success\nrate",
        "Arbitrage pass\nrate",
    ]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = colour_table(
        ax,
        values,
        list(VARIANTS),
        headings,
        [".4f", ".4f", ".0f", ".2f", ".1%", ".1%"],
        [False, False, False, False, True, True],
        "Left-Wing Weighting Comparison After Correcting Carry\nGreen is better within each column",
    )
    fig.colorbar(image, ax=ax, label="Relative performance within column")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "executive_ablation_matrix.png", dpi=200)
    plt.close(fig)


def create_sample_slice_matrix():
    """Create the before/after plot for the carry correction."""
    surface_df = pd.read_excel(WORKBOOK, sheet_name="Surface4")
    maturity = surface_df["Year Fraction"].max()
    slice_df = surface_df[np.isclose(surface_df["Year Fraction"], maturity)]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    for row, use_carry in enumerate([False, True]):
        for column, (label, variant) in enumerate(VARIANTS.items()):
            fit = run_ablation_fit(slice_df, variant, use_inferred_carry=use_carry)
            valid = np.isfinite(fit["fitted_iv"])
            error = fit["fitted_iv"][valid] - fit["market_iv"][valid]
            rmse = np.sqrt(np.mean(error**2))
            failures = int(np.sum(~valid))
            ax = axes[row, column]

            ax.axvspan(fit["log_moneyness"].min(), LEFT_WING_LIMIT, color="#dbeafe", alpha=0.7)
            ax.plot(fit["log_moneyness"], fit["market_iv"], "o-", color="black", markersize=4, label="Market IV")
            ax.plot(
                fit["log_moneyness"][valid],
                fit["fitted_iv"][valid],
                "o-",
                color="#d62728",
                markersize=4,
                label="Spline IV",
            )
            ax.set_title(f"{label}\nIV RMSE={rmse:.4f}, failures={failures}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel("Log-moneyness")
            if column == 0:
                row_name = "Inferred carry" if use_carry else "Existing delta=0"
                ax.set_ylabel(f"{row_name}\nImplied volatility")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(
        f"Sample Slice Before and After Carry Correction: Surface4, T={maturity:.3f} years",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "sample_slice_before_after_matrix.png", dpi=200)
    plt.close(fig)


def calculate_surface4_summary():
    """Summarise the same weighting test, but only for Surface 4."""
    surface_df = pd.read_excel(WORKBOOK, sheet_name="Surface4")
    rows = []
    winners = []

    for maturity, slice_df in surface_df.groupby("Year Fraction"):
        maturity_rows = []
        for label, variant in VARIANTS.items():
            fit = run_ablation_fit(slice_df, variant)
            k = fit["log_moneyness"]
            finite = np.isfinite(fit["fitted_iv"])
            error = fit["fitted_iv"] - fit["market_iv"]
            left = k < LEFT_WING_LIMIT
            right = k > abs(LEFT_WING_LIMIT)
            row = {
                "method": label,
                "all_iv_rmse": np.sqrt(np.mean(error[finite] ** 2)),
                "left_iv_rmse": np.sqrt(np.mean(error[finite & left] ** 2)),
                "right_iv_rmse": np.sqrt(np.mean(error[finite & right] ** 2)),
                "iv_failures": int(np.sum(~finite)),
                "call_rmse": np.sqrt(np.mean((fit["fitted_calls"] - fit["market_calls"]) ** 2)),
                "success": fit["success"],
                "arbitrage_pass": fit["arbitrage_pass"],
            }
            rows.append(row)
            maturity_rows.append(row)

        best = min(maturity_rows, key=lambda item: item["all_iv_rmse"])
        winners.append(best["method"])

    summary = (
        pd.DataFrame(rows)
        .groupby("method")
        .agg(
            mean_iv_rmse=("all_iv_rmse", "mean"),
            worst_iv_rmse=("all_iv_rmse", "max"),
            mean_left_iv_rmse=("left_iv_rmse", "mean"),
            mean_right_iv_rmse=("right_iv_rmse", "mean"),
            iv_failures=("iv_failures", "sum"),
            mean_call_rmse=("call_rmse", "mean"),
            fit_success=("success", "mean"),
            arbitrage_pass=("arbitrage_pass", "mean"),
        )
        .loc[list(VARIANTS)]
    )
    summary["best_maturity_count"] = pd.Series(winners).value_counts().reindex(list(VARIANTS), fill_value=0)
    return summary


def create_surface4_weighting_matrix():
    """Create the Surface 4 weighting comparison figure."""
    summary = calculate_surface4_summary()
    metrics = [
        ("mean_iv_rmse", "Mean IV\nRMSE", False, ".5f"),
        ("worst_iv_rmse", "Worst IV\nRMSE", False, ".5f"),
        ("mean_left_iv_rmse", "Mean left\nIV RMSE", False, ".5f"),
        ("mean_right_iv_rmse", "Mean right\nIV RMSE", False, ".5f"),
        ("iv_failures", "IV inversion\nfailures", False, ".0f"),
        ("mean_call_rmse", "Mean call-price\nRMSE", False, ".2f"),
        ("fit_success", "Fit success\nrate", True, ".1%"),
        ("arbitrage_pass", "Arbitrage pass\nrate", True, ".1%"),
        ("best_maturity_count", "Best maturity\ncount", True, ".0f"),
    ]
    values = np.array([[summary.loc[label, key] for key, _, _, _ in metrics] for label in VARIANTS], dtype=float)

    fig, ax = plt.subplots(figsize=(15, 5.8))
    image = colour_table(
        ax,
        values,
        list(VARIANTS),
        [heading for _, heading, _, _ in metrics],
        [fmt for _, _, _, fmt in metrics],
        [higher for _, _, higher, _ in metrics],
        "Surface 4 Weighting Ablation, AIC Lambda\nGreen is better within each column",
    )
    fig.colorbar(image, ax=ax, label="Relative performance within each metric")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "surface4_weighting_ablation_matrix.png", dpi=200)
    plt.close(fig)


def fit_project_slice(slice_df, fit_mode):
    """Fit one Surface 4 slice with the normal project fitter."""
    strikes, calls, market_iv, spot, forward, rate, maturity, _ = prepare_slice(slice_df)
    lam = choose_project_lambda(strikes, calls, spot, rate, maturity, criterion="aic", fit_mode=fit_mode)
    result = fit_smoothing_spline(
        strikes,
        calls,
        lam,
        spot,
        rate,
        maturity,
        forward=forward,
        fit_mode=fit_mode,
    )
    grid = np.linspace(strikes.min(), strikes.max(), 160)
    fitted_iv_grid = prices_to_iv(result, grid, forward)
    fitted_iv_knots = np.array(
        [implied_vol(forward, strike, maturity, rate, price, "call") for strike, price in zip(strikes, result["g"])]
    )
    valid = np.isfinite(fitted_iv_knots)

    return {
        "T": maturity,
        "days": round(maturity * 365),
        "log_moneyness": np.log(strikes / forward),
        "market_iv": market_iv,
        "grid_log_moneyness": np.log(grid / forward),
        "fitted_iv_grid": fitted_iv_grid,
        "rmse": np.sqrt(np.mean((fitted_iv_knots[valid] - market_iv[valid]) ** 2)),
        "failures": int(np.sum(~valid)),
        "success": result["success"],
    }


def create_surface4_maturity_snapshot(fit_mode, colour, filename):
    """Plot all Surface 4 maturities for one fitting mode."""
    surface_df = pd.read_excel(WORKBOOK, sheet_name="Surface4")
    rows = [
        fit_project_slice(slice_df, fit_mode)
        for _, slice_df in surface_df.groupby("Year Fraction")
    ]
    rows = sorted(rows, key=lambda item: item["T"])

    cols = 5
    fig_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(fig_rows, cols, figsize=(22, 17))
    axes = axes.ravel()

    ylims = {}
    for row in rows:
        valid_grid = np.isfinite(row["fitted_iv_grid"])
        all_y = np.concatenate([row["market_iv"], row["fitted_iv_grid"][valid_grid]])
        ylims[row["T"]] = (max(0.0, float(all_y.min()) - 0.03), float(all_y.max()) + 0.03)

    for ax, row in zip(axes, rows):
        valid_grid = np.isfinite(row["fitted_iv_grid"])
        ax.plot(row["log_moneyness"], row["market_iv"], "o", color="black", markersize=3.5, label="Market IV")
        ax.plot(
            row["grid_log_moneyness"][valid_grid],
            row["fitted_iv_grid"][valid_grid],
            "-",
            color=colour,
            linewidth=1.8,
            label="Spline IV",
        )
        ax.axvline(0.0, color="grey", linewidth=0.8, alpha=0.5)
        ax.set_ylim(*ylims[row["T"]])
        status = "ok" if row["success"] else "warn"
        ax.set_title(
            f"{row['days']}d, T={row['T']:.3f}\nRMSE={row['rmse']:.5f}, fails={row['failures']}, {status}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)

    for ax in axes[len(rows):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Surface4: {fit_mode.title()} Smoothing Splines by Maturity\nAIC lambda, forward-aware bounds",
        fontsize=18,
        fontweight="bold",
    )
    fig.supxlabel("Log-moneyness, log(K/F)", y=0.035)
    fig.supylabel("Implied volatility", x=0.008)
    fig.tight_layout(rect=[0.02, 0.055, 1, 0.94])
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_executive_matrix()
    create_sample_slice_matrix()
    create_surface4_weighting_matrix()
    create_surface4_maturity_snapshot("unweighted", "#1f77b4", "surface4_unweighted_all_maturities.png")
    create_surface4_maturity_snapshot("weighted", "#d62728", "surface4_weighted_all_maturities.png")

    print(f"Created five images in {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
