
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.svi.optimisation.global_optimizers import GLOBAL_METHODS
from src.svi.optimisation.local_optimizers import svi_objective, SVI_BOUNDS, total_variance

# poetry run python -m src.svi.testing.prototype_pipeline

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

def run_prototype():
    print("Started")

    xl = pd.ExcelFile(FILEPATH)
    sheets = [s for s in xl.sheet_names if "Year Fraction" in pd.read_excel(xl, sheet_name=s).columns]

    # Only using the 3 best globals to save time
    global_methods = ["de", "basinhopping", "shgo"]
    local_methods = ["SLSQP", "trust-constr", "COBYQA"]

    results = []

    for sheet in sheets:
        print(f"\nProcessing {sheet}")
        df = pd.read_excel(FILEPATH, sheet_name=sheet)
        expiries = sorted(df["Year Fraction"].unique())

        for idx, T in enumerate(expiries):
            strikes, market_vols, forward = load_slice(FILEPATH, sheet, T)
            k_values = np.log(strikes / forward)
            w_market = market_vols ** 2 * T
            
            for g_name in global_methods:
                g_fn = GLOBAL_METHODS[g_name]
                
                # 1. Global Step
                t0 = time.perf_counter()
                try:
                    g_params = g_fn(strikes, market_vols, T, forward)
                except:
                    continue
                t1 = time.perf_counter()
                
                g_time = t1 - t0
                g_rmse = rmse(g_params, k_values, market_vols, T)
                x0 = [g_params['a'], g_params['b'], g_params['rho'], g_params['m'], g_params['sigma']]
                
                # 2. Local Step
                bounds = list(SVI_BOUNDS)
                bounds[0] = (1e-8, float(np.max(w_market)))
                
                for l_method in local_methods:
                    t2 = time.perf_counter()
                    try:
                        res = minimize(svi_objective, x0, args=(k_values, w_market), method=l_method, bounds=bounds)
                        l_params = dict(zip(['a', 'b', 'rho', 'm', 'sigma'], res.x))
                        l_rmse = rmse(l_params, k_values, market_vols, T)
                    except:
                        l_rmse = float('nan')
                    t3 = time.perf_counter()
                    
                    l_time = t3 - t2
                    
                    results.append({
                        "Surface": sheet,
                        "Global": g_name,
                        "Local": l_method,
                        "G_Time(s)": g_time,
                        "G_RMSE": g_rmse,
                        "L_Time(s)": l_time,
                        "Final_RMSE": l_rmse,
                        "Total_Time(s)": g_time + l_time
                    })

    df_res = pd.DataFrame(results)
    
    # Calculate pipeline averages
    summary = df_res.groupby(["Global", "Local"]).agg({
        "G_RMSE": "mean",
        "Final_RMSE": "mean",
        "Total_Time(s)": "mean"
    }).reset_index().sort_values("Final_RMSE")
    
    print("\n\nPipeline Benchmark Summary (Averaged across ALL Surfaces):")
    print("=" * 80)
    print(f"{'Global':<15} {'Local':<15} {'G_RMSE':<12} {'Final_RMSE':<12} {'Total Time(s)':<15}")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"{row['Global']:<15} {row['Local']:<15} {row['G_RMSE']:<12.3e} {row['Final_RMSE']:<12.3e} {row['Total_Time(s)']:<15.3f}")

if __name__ == "__main__":
    run_prototype()
