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


"""#Numbers for testing
strikes = np.array([24779.535, 29735.442, 34691.349, 39647.256,
                    44603.163, 47081.117, 49559.070, 52037.024,
                    54514.977, 59470.884, 64426.791, 69382.698, 74338.605])

Q = build_Q_matrix(strikes)
R = build_R_matrix(strikes)

print(Q.shape) # (13, 11)
print(R.shape) # (11, 11)
print(Q[0, 0]) # ~  0.000202
print(Q[1, 0]) # ~ -0.000404
print(R[0, 0]) # ~ 3303.94
print(R[0, 1]) # ~  825.98
"""

def fit_smoothing_spline(strikes: np.ndarray, call_prices: np.ndarray, lam: float):
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

    
    x0=np.zeros(2*n - 2) #initial guess

    def objectivefunction(x):
        return -y.T @ x + (1/2)* x.T @ B @ x
    
    def constraint(x):
        return A.T @ x
    
    result = minimize(
        objectivefunction,
        x0,
        method="SLSQP",
        constraints={"type": "eq", "fun": constraint}
    )
    x =result.x
    g =x[:n] # x elements 1....n 
    gamma = x[n:] # x elements n+1....end

    """
    Fit a cubic smoothing spline to call prices subject to no-arbitrage constraints.


    Returns
    -------
    dict
        Dictionary containing keys 'g', 'gamma', 'h', 'S', 'r', 'T', 'strikes'

    Notes
    -----
    Fengler eqs 17-19 (QP formulation), eqs 25-27 (no-arbitrage constraints). Use scipy.optimize.minimize method='SLSQP'.
    """

    return {"g":g,"gamma":gamma,"x":x}

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
    raise NotImplementedError("Not yet implemented")


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
    raise NotImplementedError("Not yet implemented")


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
    smoothinsplineresults=fit_smoothing_spline(strikes, call_prices, lam)
 
    print("g =",smoothinsplineresults["g"])
    print("gamma =",smoothinsplineresults["gamma"])
    print("x =",smoothinsplineresults["x"])