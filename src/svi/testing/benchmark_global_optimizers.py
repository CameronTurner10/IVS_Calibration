"""
Benchmarking script: compare global optimisers (DE, SHGO, Basin Hopping)
using real market data from Surfaces.xlsx.
Prints a per-slice RMSE table for each method, along with wall-clock runtime.

Runtime is included because global methods are significantly slower than local
ones and speed is a relevant trade-off when choosing between them.
"""

# poetry run python -m src.svi.testing.benchmark_global_optimizers

import time
import numpy as np
import pandas as pd
import sys
import datetime
from src.svi.optimisation.global_optimizers import GLOBAL_METHODS, total_variance

FILEPATH = "tests/data/Surfaces.xlsx"

def load_slice(filepath, sheet_name, T):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    slice_df = df[np.isclose(df["Year Fraction"], T, atol=1e-8)]
    strikes = slice_df["Strike"].values
    market_vols = slice_df["Volatility"].values
    forward = slice_df["Forward"].iloc[0]
    return strikes, market_vols, forward

def rmse(params, k_values, w_market):
    w_fit = total_variance(k_values, **params)
    return float(np.sqrt(np.mean((w_fit - w_market) ** 2)))

# load all sheets and expiries
xl = pd.ExcelFile(FILEPATH)
sheets = []
for s in xl.sheet_names:
    df_temp = pd.read_excel(xl, sheet_name=s)
    if "Year Fraction" in df_temp.columns:
        sheets.append(s)

method_names = list(GLOBAL_METHODS.keys())

# list to accumulate all results for the CSV
all_results = []

for sheet in sheets:
    df = pd.read_excel(FILEPATH, sheet_name=sheet)
    expiries = sorted(df["Year Fraction"].unique())

    print(f"\n{'='*80}")
    print(f"Surface: {sheet}  ({len(expiries)} expiries)")
    print(f"{'='*80}")

    # header row — includes runtime (s) column per method
    col_width = 20
    header = f"{'Expiry (T)':>14}"
    for m in method_names:
        header += f"{m + ' RMSE':>{col_width}}" + f"{'time (s)':>12}"
    header += f"  {'Best (RMSE)':>14}"
    print(header)
    print("-" * len(header))

    for T in expiries:
        strikes, market_vols, forward = load_slice(FILEPATH, sheet, T)
        k_values = np.log(strikes / forward)
        w_market = market_vols ** 2 * T

        rmse_scores = {}
        runtimes = {}

        for method_name, fit_function in GLOBAL_METHODS.items():
            try:
                t_start = time.perf_counter()
                params = fit_function(strikes, market_vols, T, forward)
                t_end = time.perf_counter()

                score = rmse(params, k_values, w_market)
                elapsed = t_end - t_start
                rmse_scores[method_name] = score
                runtimes[method_name] = elapsed

                # store full data for CSV export
                all_results.append({
                    "Date_Run": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "Surface": sheet,
                    "Expiry": T,
                    "Method": method_name,
                    "RMSE": score,
                    "Runtime_s": round(elapsed, 4),
                    **params
                })

            except Exception as e:
                rmse_scores[method_name] = float("nan")
                runtimes[method_name] = float("nan")

        best = min(rmse_scores, key=lambda x: rmse_scores[x] if not np.isnan(rmse_scores[x]) else float("inf"))
        row = f"{T:>14.6f}"
        for m in method_names:
            row += f"{rmse_scores[m]:>{col_width}.2e}" + f"{runtimes[m]:>12.3f}"
        row += f"  {best:>14}"
        print(row)

    print()
    print("Average RMSE and runtime across all slices:")

    all_rmses = {m: [] for m in method_names}
    all_times = {m: [] for m in method_names}

    for T in expiries:
        strikes, market_vols, forward = load_slice(FILEPATH, sheet, T)
        k_values = np.log(strikes / forward)
        w_market = market_vols ** 2 * T
        for method_name, fit_fn in GLOBAL_METHODS.items():
            try:
                t_start = time.perf_counter()
                params = fit_fn(strikes, market_vols, T, forward)
                t_end = time.perf_counter()
                score = rmse(params, k_values, w_market)
                all_rmses[method_name].append(score)
                all_times[method_name].append(t_end - t_start)
            except:
                pass

    for method_name in method_names:
        avg_rmse = np.mean(all_rmses[method_name]) if all_rmses[method_name] else float("nan")
        avg_time = np.mean(all_times[method_name]) if all_times[method_name] else float("nan")
        print(f"{method_name:>20}: RMSE = {avg_rmse:.4e}   Avg time = {avg_time:.3f}s")

# Save full results to CSV in the data/ folder with a dynamic name to avoid overwriting
suffix = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"data/benchmarks/global_optimizers_RMSE_{suffix}.csv"
pd.DataFrame(all_results).to_csv(out_path, index=False)

print(f"\n{'='*80}")
print(f"Done. Full parameter data exported to: {out_path}")
print(f"{'='*80}\n")
