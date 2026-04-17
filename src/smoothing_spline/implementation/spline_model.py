import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import block_diag

# poetry run python -m src.smoothing_spline.implementation.spline_model

FILEPATH = "tests/data/Surfaces.xlsx"


def load_spline_slice(sheet_name, T, filepath=FILEPATH):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    slice_df = df[np.isclose(df["Year Fraction"], T, atol=1e-8)]
    if slice_df.empty:
        raise ValueError(f"Maturity T={T} not found in {sheet_name}")
        
    return (
        slice_df["Strike"].values,
        slice_df["Call Price"].values,
        slice_df["Spot"].iloc[0],
        slice_df["Discount Rate"].iloc[0],
        slice_df["Forward"].iloc[0],
    )


def build_Q_matrix(strikes: np.ndarray) -> np.ndarray:
    """
    Build the n x (n-2) tridiagonal Q matrix. Fengler eq 13-14.
    Encodes the geometry of the strike grid.
    Returns np.ndarray shape (n, n-2).
    """
    n = len(strikes)
    h = np.diff(strikes)
    Q = np.zeros((n, n - 2))
    for j in range(n - 2):
        Q[j, j] = 1 / h[j]
        Q[j + 1, j] = -1 / h[j] - 1 / h[j + 1]
        Q[j + 2, j] = 1 / h[j + 1]
    return Q


def build_R_matrix(strikes: np.ndarray) -> np.ndarray:
    """
    Build the (n-2) x (n-2) symmetric tridiagonal R matrix. Fengler eq 14.
    Key property: gamma.T @ R @ gamma = integral of g''(u)^2 du.
    Returns np.ndarray shape (n-2, n-2).
    """
    n = len(strikes)
    h = np.diff(strikes)
    R = np.zeros((n - 2, n - 2))
    for i in range(n - 2):
        R[i, i] = (h[i] + h[i + 1]) / 3
        if i < n - 3:
            R[i, i + 1] = h[i + 1] / 6
            R[i + 1, i] = h[i + 1] / 6
    return R

def fit_smoothing_spline(strikes: np.ndarray, call_prices: np.ndarray, lam: float, S: float, r: float, T: float, delta: float = 0.0):
    """
    Fit a cubic smoothing spline to call prices subject to no-arbitrage constraints.


    Returns
    -------
    dict
        Dictionary containing keys 'g', 'gamma', 'h', 'S', 'r', 'T', 'strikes'

    Notes
    -----
    Fengler eqs 17-19 (QP formulation), eqs 25-27 (no-arbitrage constraints). Use scipy.optimize.minimize method='SLSQP' probably can exp tho
    """
    R=build_R_matrix(strikes)
    Q=build_Q_matrix(strikes)
    A=np.vstack([Q,-R.T]) 
    n = len(call_prices)
    I=np.identity(n)
    #say you have (Q;R) this would be 
    # (Q;R)=(Q11,Q12,Q13;.....Q51,Q52,Q53;R11,R12,R13;....R31,R32,R33)
    #Q stacks on top of R --> np.vstack does this
    B=block_diag(I,lam*R)

    y=[]

    for yi in call_prices:
        y.append(yi)
    for _ in range(n,2*n-2):
        y.append(0)

    y=np.array(y)

    
    x0=np.zeros(2*n - 2) #defining shape of x
    x0[:n] = y[:n] # better initial guess for g

    def objectivefunction(x):
        return -y.T @ x + (1/2)* x.T @ B @ x
    
    def eq_constraint(x):
        return A.T @ x

    disc_r = np.exp(-r * T)
    disc_delta = np.exp(-delta * T)

    def ineq_constraint(x):
        # Fengler Eq. 26: slope constraints for no-arbitrage
        # left_slope >= -exp(-rT) -> g_1 - g_0 + exp(-rT)*(u_1 - u_0) >= 0
        # right_end -> g_{n-1} - g_n >= 0
        g_current = x[:n]
        c1 = g_current[1] - g_current[0] + disc_r * (strikes[1] - strikes[0])
        c2 = g_current[-2] - g_current[-1]
        return np.array([c1, c2])

    bounds = []
    # Fengler Eq. 27: Price bounds bounds
    lower_g1 = disc_delta * S - disc_r * strikes[0]
    upper_g1 = disc_delta * S
    bounds.append((lower_g1, upper_g1)) # bounds on g_1

    for _ in range(1, n - 1):
        bounds.append((0.0, None)) # bounds on interior g_i

    bounds.append((0.0, None)) # bounds on g_n (Eq. 27)

    # Fengler Eq. 25: Convexity constraint (gamma_i >= 0)
    for _ in range(n - 2):
        bounds.append((0.0, None)) 

    result = minimize(
        objectivefunction,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": eq_constraint},
            {"type": "ineq", "fun": ineq_constraint}
        ]
    )
    x = result.x
    g = x[:n] # x elements 1....n 
    gamma = x[n:] # x elements n+1....end

    

    check = Q.T @ g - R @ gamma
    return {
        "g": g, 
        "gamma": gamma, 
        "strikes": strikes, 
        "x": x, 
        "check": check,
        "S": S,
        "r": r,
        "T": T,
        "delta": delta
    }

"""
call_prices = np.array([
    5200.0, 4500.0, 3900.0, 3400.0,
    3000.0, 2800.0, 2600.0, 2400.0,
    2200.0, 1800.0, 1400.0, 900.0, 400.0
])

y = []
n = len(call_prices)

for yi in call_prices:
    y.append(yi)

for _ in range(n, 2*n - 2):
    y.append(0)

y = np.array(y)[:, None]

y = np.array(y)
print(y.T)
print("2n-2 =", 2*n-2)                  
print()
print("len(y) =", len(y))   
print("yshape",y.shape)
print("y.T shape",(y.T).shape)
I=np.identity(3)     
R=np.array([[1,2,3],[4,5,6],[7,8,9]])
B=block_diag(I,R)
print(B)
"""
def evaluate_spline(result: dict, u: float) -> float:
    """
    Evaluate the fitted spline at a given strike u.

    Parameters
    ----------
    result : dict
        Dictionary returned by fit_smoothing_spline
    u : float
        Strike price to evaluate at

    Returns
    -------
    float
        Estimated call price at strike u, minimum 0.0

    Notes
    -----
    Fengler eq 20 (interior), eqs 21-24 (boundary derivatives and extrapolation).
    """
    # Pulling out the relevant data from the result dictionary
    g = np.asarray(result["g"], dtype=float)
    gamma = np.asarray(result["gamma"], dtype=float)
    strikes = np.asarray(result["strikes"], dtype=float)
    n = len(strikes)
    h = np.diff(strikes)

    # Reconstructing the full gamma vector from Fengler section 3.1 (γ_1 = γ_n = 0 by natural spline definition)
    if len(gamma) == n - 2:
        gamma_full = np.concatenate(([0.0], gamma, [0.0]))
    elif len(gamma) == n:
        gamma_full = gamma.copy()
    else:
        raise ValueError("gamma must have length n or n-2")

    # Computing the boundary derivatives: Fengler's eqs 21-22
    g_prime_left  = (g[1] - g[0]) / h[0] - (h[0] / 6) * gamma_full[1] # Slope of the leftmost knot
    g_prime_right = (g[-1] - g[-2]) / h[-1] + (h[-1] / 6) * gamma_full[-2] # Slope of the rightmost knot

    # Left extrapolation: Fengler eq 23
    if u <= strikes[0]:
        return max(float(g[0] + (u - strikes[0]) * g_prime_left), 0.0)

    # Right extrapolation: Fengler eq 24 (mirror of left extrapolation, note the sign change in g_prime_right)
    if u >= strikes[-1]:
        return max(float(g[-1] + (u - strikes[-1]) * g_prime_right), 0.0)

    # Interior evaluation: Fengler eq 20
    i = np.searchsorted(strikes, u, side="right") - 1
    i = int(np.clip(i, 0, n - 2))

    hi = h[i]
    gi, gi1 = g[i], g[i + 1]
    yi, yi1 = gamma_full[i], gamma_full[i + 1]

    val = (
        (u - strikes[i]) * gi1 + (strikes[i + 1] - u) * gi
    ) / hi - ( # Linear interpolation of g between strikes[i] and strikes[i+1]
        (1 / 6) * (u - strikes[i]) * (strikes[i + 1] - u) * (
            (1 + (u - strikes[i]) / hi) * yi1 +
            (1 + (strikes[i + 1] - u) / hi) * yi
        ) # Cubic correction term based on the second derivatives at the knots
    )

    return max(float(val), 0.0) # Call price must be non-negative


def second_derivative(result: dict, u: float) -> float:
    """
    Evaluate the second derivative of the fitted spline at a given strike u.

    Parameters
    ----------
    result : dict
        Dictionary returned by fit_smoothing_spline
    u : float
        Strike price to evaluate at

    Returns
    -------
    float
        g''(u). Returns 0.0 outside data range.

    Notes
    -----
    Fengler Proposition 3.1. g'' is linear between knots. Used to verify convexity.
    """
    strikes = np.asarray(result["strikes"], dtype=float)
    gamma_internal = np.asarray(result["gamma"], dtype=float)
    n = len(strikes)

    # reconstruct full gamma vector (length n)
    # gamma_1 = gamma_n = 0 for a natural smoothing spline
    if len(gamma_internal) == n - 2:
        gamma = np.concatenate(([0.0], gamma_internal, [0.0]))
    elif len(gamma_internal) == n:
        gamma = gamma_internal
    else:
        raise ValueError(f"gamma must have length n or n-2, got {len(gamma_internal)}")

    # outside the domain, second derivative is zero (natural spline extrapolation is linear)
    if u < strikes[0] or u > strikes[-1]:
        return 0.0

    # find the interval [strikes[i], strikes[i+1]] containing u
    i = np.searchsorted(strikes, u, side="right") - 1
    i = int(np.clip(i, 0, n - 2))

    # formula for second derivative g''(u) which is linear between knots
    h_i = strikes[i + 1] - strikes[i]
    if h_i == 0:
        return float(gamma[i])
        
    g_2 = ((strikes[i + 1] - u) / h_i * gamma[i] + (u - strikes[i]) / h_i * gamma[i + 1])

    return float(g_2)


def prices_to_iv(result: dict, strikes_grid: np.ndarray, F: float) -> np.ndarray:
    """
    Convert spline estimated call prices to implied volatilities via BS inversion.

    Parameters
    ----------
    result : dict
        Dictionary returned by fit_smoothing_spline
    strikes_grid : np.ndarray
        Array of strikes to evaluate at
    F : float
        Forward price

    Returns
    -------
    np.ndarray
        Array of implied volatilities, np.nan for any failed inversions

    Notes
    -----
    Evaluates spline on grid then inverts Black-Scholes using existing solver in src/utils/black_scholes.py
    """
    raise NotImplementedError("Not yet implemented")


if __name__ == "__main__":
    strikes, call_prices, S, r, F = load_spline_slice("Surface4", 0.043836)
    Q = build_Q_matrix(strikes)
    R = build_R_matrix(strikes)
    print(f"Q shape: {Q.shape}  — expected (13, 11)")
    print(f"R shape: {R.shape}  — expected (11, 11)")
    print(f"Q[0,0]: {Q[0,0]:.6f}  — expected  0.000202")
    print(f"Q[1,0]: {Q[1,0]:.6f}  — expected -0.000404")
    print(f"R[0,0]: {R[0,0]:.2f}  — expected  3303.94")
    print(f"R[0,1]: {R[0,1]:.2f}  — expected   825.98")
    lam=1
    # T=0.043836 based on the load_spline_slice call above
    smoothinsplineresults=fit_smoothing_spline(strikes, call_prices, lam, S, r, 0.043836)
 
    print("g =",smoothinsplineresults["g"])
    print("gamma =",smoothinsplineresults["gamma"])
    print("x =",smoothinsplineresults["x"])
    print("Natural Spline Test =",smoothinsplineresults["check"])