"""
Benchmarking script: compare local optimisers (SLSQP, Trust Region, COBYQA) for the moment
using real market data from Surfaces.xlsx.
Prints a per-slice RMSE table for each method.
"""

# poetry run python -m src.svi.testing.benchmark_local_optimizers

import numpy as np
import pandas as pd
import sys
import datetime
from src.svi.optimisation.local_optimizers import LOCAL_METHODS, total_variance

FILEPATH = "tests/data/Surfaces.xlsx"

def load_slice(filepath, sheet_name, T):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    slice_df = df[np.isclose(df["Year Fraction"], T, atol=1e-8)]
    strikes = slice_df["Strike"].values
    market_vols = slice_df["Volatility"].values
    forward = slice_df["Forward"].iloc[0]
    return strikes, market_vols, forward

def rmse(params, k_values, market_vols, T):
    w_fit = total_variance(k_values, **params)
    iv_fit = np.sqrt(np.maximum(w_fit, 0) / T)
    return float(np.sqrt(np.mean((iv_fit - market_vols) ** 2)))

# load all sheets and expiries
xl = pd.ExcelFile(FILEPATH)
sheets = []
for s in xl.sheet_names:
    df_temp = pd.read_excel(xl, sheet_name=s)
    if "Year Fraction" in df_temp.columns:
        sheets.append(s)

method_names = list(LOCAL_METHODS.keys())

# list to accumulate all results for the CSV
all_results = []

for sheet in sheets:
    df = pd.read_excel(FILEPATH, sheet_name=sheet)
    expiries = sorted(df["Year Fraction"].unique())

    print(f"\n{'='*70}")
    print(f"Surface: {sheet}  ({len(expiries)} expiries)")
    print(f"{'='*70}")

    # header row
    header = f"{'Expiry (T)':>14}" + "".join(f"{m:>18}" for m in method_names) + f"  {'Best':>14}"
    print(header)
    print("-" * len(header))

    for T in expiries:
        strikes, market_vols, forward = load_slice(FILEPATH, sheet, T)
        k_values = np.log(strikes / forward)
        w_market = market_vols ** 2 * T

        rmse_scores = {}
        for method_name, fit_function in LOCAL_METHODS.items():
            try:
                params = fit_function(strikes, market_vols, T, forward)
                score = rmse(params, k_values, market_vols, T)
                rmse_scores[method_name] = score
                
                # Store full data for Team 2's instability analysis
                all_results.append({
                    "Date_Run": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "Surface": sheet,
                    "Expiry": T,
                    "Method": method_name,
                    "RMSE": score,
                    **params
                })
                
            except Exception as e:
                rmse_scores[method_name] = float("nan")

        best = min(rmse_scores, key=lambda x: rmse_scores[x] if not np.isnan(rmse_scores[x]) else float("inf"))
        row = f"{T:>14.6f}" + "".join(f"{rmse_scores[m]:>18.2e}" for m in method_names) + f"  {best:>14}"
        print(row)

    print()
    print("Average RMSE across all slices:")

    all_rmses = {}
    for m in method_names:
        all_rmses[m] = []
    for T in expiries:
        strikes, market_vols, forward = load_slice(FILEPATH, sheet, T)
        k_values = np.log(strikes / forward)
        w_market = market_vols ** 2 * T
        for method_name, fit_fn in LOCAL_METHODS.items():
            try:
                params = fit_fn(strikes, market_vols, T, forward)
                score = rmse(params, k_values, market_vols, T)
                all_rmses[method_name].append(score)
            except:
                pass
    for method_name in method_names:
        avg = np.mean(all_rmses[method_name]) if all_rmses[method_name] else float("nan")
        print(f"{method_name:>20}: {avg:.4e}")

# Save full results to CSV in the data/ folder with a dynamic name to avoid overwriting
suffix = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"data/benchmarks/local_optimizers_RMSE_{suffix}.csv"
pd.DataFrame(all_results).to_csv(out_path, index=False)

print(f"\n{'='*70}")
print(f"Done. Full parameter data exported to: {out_path}")
print(f"{'='*70}\n")
