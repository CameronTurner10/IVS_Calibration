from src.smoothing_spline.optimisation.fit_spline import choose_lambda
from src.smoothing_spline.implementation.spline_model import fit_smoothing_spline
from src.utils.plotting import (
    plot_single_slice,
    plot_surface,
    plot_spline_slice,
    plot_spline_surface,
    list_available_data,
    get_slice_from_data,
)
import pandas as pd, numpy as np

filepath = "tests/data/Surfaces.xlsx"
data = list_available_data(filepath)
sheet_name = list(data.keys())[3]       # choose the sheet for testing
T = data[sheet_name][9]                # choose the maturity for testing 

df = pd.read_excel(filepath, sheet_name=sheet_name)
slice_df = df[np.isclose(df["Year Fraction"], T)]
strikes = slice_df["Strike"].values
call_prices = slice_df["Call Price"].values
market_vols = slice_df["Volatility"].values
S = slice_df["Spot"].iloc[0]
r = float(slice_df["Discount Rate"].iloc[0])
forward = float(slice_df["Forward"].iloc[0])

implied_r = np.log(forward / S) / T
print(f"T = {T:.6f},  S = {S},  F = {forward}")
print(f"  r (spreadsheet) = {r:.4%}")
print(f"  r (implied from F = S*exp(rT), δ=0) = {implied_r:.4%}")

lam = choose_lambda(
    strikes=strikes,
    call_prices=call_prices,
    S=S,
    r=r,
    T=T,
)

result = fit_smoothing_spline(
    strikes=strikes,
    call_prices=call_prices,
    lam=lam,
    S=S,
    r=r,
    T=T,
)
result["lambda"] = float(lam)
result["forward"] = forward

# --- slice check --- 
plot_single_slice(T, sheet_name, filepath, plot_type="iv")
plot_spline_slice(result, T, strikes, market_vols,
                  sheet_name=sheet_name, plot_type="iv")

# --- surface check ---
plot_surface(sheet_name, filepath, plot_type="iv")
plot_spline_surface(sheet_name, filepath, plot_type="iv")
