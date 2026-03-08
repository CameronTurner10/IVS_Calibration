import numpy as np
from src.svi.optimisation.SVI_SliceFit import fit_svi_slice
from src.svi.testing import check_basic_constraints  # adjust if different path

# --- Fake synthetic market data ---

# True parameters (safe, non-pathological)
true_params = {
    "a": 0.04,
    "b": 0.5,
    "rho": -0.3,
    "m": 0.0,
    "sigma": 0.2
}

def total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# Generate synthetic strike grid
forward = 100
strikes = np.linspace(70, 130, 25)
k = np.log(strikes / forward)
T = 1.0

# Generate synthetic market vols
w = total_variance(k, **true_params)
market_vols = np.sqrt(w / T)

# --- Run fit ---
fitted_params = fit_svi_slice(strikes, market_vols, T, forward)

print("Fitted parameters:")
print(fitted_params)

print("\nConstraints satisfied?")
print(check_basic_constraints(fitted_params))