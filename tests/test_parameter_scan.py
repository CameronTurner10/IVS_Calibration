# # more advanced stretch target
# no data
# think how we can eval BS pricing formula over ranges of all parameters
# behaviour at extremes
# similar to fuzzing
#     wrong data
#     automated
#     look for crashes
# dimensional rescaling


# tests/test_parameter_scan.py
from itertools import product
import numpy as np
import os
import sys

# Add ../src to the Python module search path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from utils.black_scholes import bs_call, bs_put



def test_bs_call_parameter_scan_no_crash_and_bounds():
    """
    Evaluate the Black–Scholes call price over a grid of parameters
    and check that:
      - the function doesn't crash
      - the returned price is finite
      - basic theoretical bounds are satisfied
    """

    # We define reasonable but quite wide ranges = "fuzzing"
    F_values     = np.linspace(10.0, 200.0, 4)   # forward
    K_values     = np.linspace(10.0, 200.0, 4)   # strike
    T_values     = np.linspace(0.01, 5.0, 4)     # years to maturity
    r_values     = np.array([-0.02, 0.0, 0.05])  # interest rates
    sigma_values = np.linspace(0.01, 1.5, 4)     # volatilities

# Iterate over all combinations of parameters (so fuzz the model over a grid of possibiliities)
    for F, K, T, r, sigma in product(
        F_values, K_values, T_values, r_values, sigma_values
    ):
        #call price function
        price = bs_call(F, K, T, sigma, r)

        # 1) check returns sane number so no NaNs or infinities
        assert np.isfinite(price)

        # 2) check no-arbitrage bounds for a call on a forward
        discount = np.exp(-r * T)
        lower_bound = max(F - K * discount, 0.0)
        upper_bound = F  # call price cannot exceed F

        assert lower_bound - 1e-10 <= price <= upper_bound + 1e-10


def test_bs_put_parameter_scan_no_crash_and_bounds():
   
    F_values     = np.linspace(10.0, 200.0, 4)
    K_values     = np.linspace(10.0, 200.0, 4)
    T_values     = np.linspace(0.01, 5.0, 4)
    r_values     = np.array([-0.02, 0.0, 0.05])
    sigma_values = np.linspace(0.01, 1.5, 4)

    for F, K, T, r, sigma in product(
        F_values, K_values, T_values, r_values, sigma_values
    ):
        price = bs_put(F, K, T, sigma, r)

        assert np.isfinite(price)

        discount = np.exp(-r * T)
        # For a put on a forward: 0 <= P <= K * discount
        lower_bound = 0.0
        upper_bound = K * discount

        assert lower_bound - 1e-10 <= price <= upper_bound + 1e-10

print("Test ran successfully!")