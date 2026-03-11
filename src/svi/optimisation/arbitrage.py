import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from src.svi.optimisation.constraints import c_nonneg_min_total_var, c_wing_right, c_wing_left, c_butterfly_grid
from src.svi.optimisation.local_optimizers import svi_objective, total_variance, SVI_BOUNDS

# poetry run python -m src.svi.optimisation.arbitrage

# SVI CALIBRATION

def get_slice_from_data(T, sheet_name, filepath="tests/data/Surfaces.xlsx"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    slice_df = df[np.isclose(df["Year Fraction"], T, atol=1e-8)]
    
    strikes = slice_df["Strike"].values
    market_vols = slice_df["Volatility"].values
    if len(slice_df) == 0:
        return np.array([]), np.array([]), 1.0, np.array([]), np.array([])
    forward = slice_df["Forward"].iloc[0]
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T
    return strikes, market_vols, forward, k_values, w_market

def fit_single_slice_with_bound(k_values, w_market, w_longer_bound=None, k_grid=None, initial_params=None):
    atm_var = np.mean(w_market)
    
    # Build a reasonable k grid if not provided
    if k_grid is None:
        k_min = float(np.min(k_values))
        k_max = float(np.max(k_values))
        pad = 0.5
        k_grid = np.linspace(k_min - pad, k_max + pad, 201)

    constraints = []
    
    # 1. Calendar Constraint: physical boundary from the strictly longer maturity slice
    if w_longer_bound is not None and k_grid is not None:
        def calendar_constraint_fun(p):
            w_curr = total_variance(k_grid, *p)
            return w_longer_bound - w_curr
        constraints.append(NonlinearConstraint(calendar_constraint_fun, lb=0.0, ub=np.inf))
        
    # 2. Basic slice-wise no-arbitrage constraints
    constraints.append(
        NonlinearConstraint(c_nonneg_min_total_var, lb=0.0, ub=np.inf)
    )
    constraints.append(
        NonlinearConstraint(c_wing_right, lb=0.0, ub=np.inf)
    )
    constraints.append(
        NonlinearConstraint(c_wing_left, lb=0.0, ub=np.inf)
    )

    # 3. Butterfly arbitrage constraint: enforce g(k) >= 0 over the k_grid
    eps= 1e-8
    constraints.append(
        NonlinearConstraint(
            lambda p: c_butterfly_grid(p, k_grid, eps),
            lb=0.0, 
            ub=np.inf
        )
    )

    if initial_params is not None:
        x0 = [initial_params['a'], initial_params['b'], initial_params['rho'], initial_params['m'], initial_params['sigma']]
    else:
        x0 = [atm_var, 0.1, 0.0, 0.0, 0.1]
        
        try:
            de_res = differential_evolution(
                svi_objective, 
                bounds=SVI_BOUNDS, 
                args=(k_values, w_market), 
                constraints=constraints,
                seed=42,
                polish=False
            )
            if hasattr(de_res, 'x') and len(de_res.x) == 5:
                x0 = de_res.x
        except Exception as e:
            # Fallback to standard x0 if global fails for some reason
            pass
        
    # Polishing with SLSQP
    res = minimize(
        svi_objective,
        x0,
        args=(k_values, w_market),
        method="SLSQP",
        bounds=SVI_BOUNDS,
        constraints=constraints
    )
    
    return dict(zip(["a", "b", "rho", "m", "sigma"], res.x))

def calibrate_surface(sheet_name, filepath="tests/data/Surfaces.xlsx"):
    print(f"Iterating Surface: {sheet_name}")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    expiries = sorted(df["Year Fraction"].unique(), reverse=True) #this flips the order so we start from the longest maturity
    
    fitted_slices = {} 
    
    k_min, k_max = 0, 0
    k_all = []
    for T in expiries:
         _, _, _, k, _ = get_slice_from_data(T, sheet_name, filepath)
         if len(k) > 0:
            k_all.extend(k)
            
    if len(k_all) > 0:
        k_min = np.min(k_all) - 0.2
        k_max = np.max(k_all) + 0.2
        
    k_grid = np.linspace(k_min, k_max, 200)
    w_longer_bound = None
    prev_params = None
    
    for T in expiries:
        _, _, _, k_values, w_market = get_slice_from_data(T, sheet_name, filepath)
        
        if len(k_values) == 0:
            continue
            
        print(f"Plotting T={T:.4f}")
        params_dict = fit_single_slice_with_bound(k_values, w_market, w_longer_bound, k_grid, initial_params=prev_params)
        fitted_slices[T] = params_dict
        
        params_list = [params_dict["a"], params_dict["b"], params_dict["rho"], params_dict["m"], params_dict["sigma"]]
        w_longer_bound = total_variance(k_grid, *params_list)
        prev_params = params_dict

    return fitted_slices


""""

def plot_calibrated_surface(sheet_name, filepath="tests/data/Surfaces.xlsx"):
    fitted_slices = calibrate_surface(sheet_name, filepath)
    
    plt.figure(figsize=(12,7))
    cmap = plt.get_cmap("viridis")
    
    expiries = sorted(list(fitted_slices.keys()))
    num_expiries = len(expiries)
    
    if num_expiries == 0:
        print("No slices to plot")
        return
        
    for i, T in enumerate(expiries):
        strikes, market_vols, _, k_values, w_market = get_slice_from_data(T, sheet_name, filepath)
        params = fitted_slices[T]
        
        k_plot_grid = np.linspace(min(k_values), max(k_values), 200)#200 k points to plot the curve
        w_fittedGrid = total_variance(k_plot_grid, **params)
        
        color = cmap(i / max(1, num_expiries - 1))
        
        plt.scatter(k_values, np.log(np.maximum(w_market, 1e-12)), s=15, color=color, alpha=0.6)
        plt.plot(k_plot_grid, np.log(np.maximum(w_fittedGrid, 1e-12)), linewidth=2, color=color, label=f"T={T:.2f}")

    plt.xlabel("Log-Moneyness (k)")
    plt.ylabel("Log Variance log(w)")
    plt.title(f"Calibrated SVI Surface — {sheet_name}", fontsize=13, fontweight="bold")
    
    if num_expiries <= 12:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_calibrated_surface("Surface1")








# backup: cameron's original simultaneous multi-slice calibration script


# SVI fitting using scipy - local optimisation methods
# Goal: find the 5 params (a, b, rho, m, sigma) that best match market data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize
from src.svi.implementation.svi_model import SVI

# poetry run python -m src.svi.optimisation.calender_arbitrage

# Bounds for the 5 SVI parameters - shared across all local methods
SVI_BOUNDS = [
    (0.001, 1.0),    # a
    (0.001, 0.99),   # b
    (-0.99, 0.99),   # rho
    (-0.5,  0.5),    # m
    (0.01,  1.0),    # sigma
]

def total_variance_cam(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def get_slice_from_data_cam(T, sheet_name, filepath="tests/data/Surfaces.xlsx"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    slice_df = df[np.isclose(df["Year Fraction"], T, atol=1e-8)]

    strikes = slice_df["Strike"].values
    market_vols = slice_df["Volatility"].values
    forward = slice_df["Forward"].iloc[0]
    k_values = np.log(strikes / forward)
    w_market = market_vols ** 2 * T

    return strikes, market_vols, forward, k_values, w_market

def svi_full_objective(parameters, k_list, w_market_list): #Objective function shared by all local methods - minimises sum of squared errors.
    total_sum=0
    for i in range(len(k_list)):
        slice_params = parameters[i*5:(i+1)*5] 
        w_model = total_variance_cam(k_list[i], *slice_params)
        total_sum += np.sum((w_model - w_market_list[i])**2)
    return total_sum

def calendar_constraints(parameters,k_list):
    constraints=[]
    for i in range(1,len(k_list)):     #parameters[i*5:(i+1)*5] -> i=0 sliceT[0] parameters [0:5] = [a0,b0,rho0,m0,sigma0] if i=1 parameters [5,10]=[a1,b1,rho1,m1,sigma1]
        w_current=total_variance_cam(k_list[i],*parameters[i*5:(i+1)*5]) #k_list[i] ->k's for T[i]
        w_previous=total_variance_cam(k_list[i-1],*parameters[(i-1)*5:i*5]) #k_list[i-1] ->k's for T[i-1]

        # interpolate previous slice to current ks -> Need to do this because w_current and w_previous are evaluated at same k
        w_prev_at_curr_ks = np.interp(k_list[i], k_list[i-1], w_previous)

        constraints.append(w_current - w_prev_at_curr_ks)
        #constraints.append(w_current - w_previous)
    return np.concatenate(constraints) #returns a single flat array of all contsraints

def fit_svi_multi_slsqp(sheet_name,filepath="tests/data/Surfaces.xlsx"): #Sequential Least Squares Programming

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    
    x0=[]
    k_list=[] 
    w_market_list=[]

    for T in expiries:
        strikes, market_vols, forward, k_values, w_market = get_slice_from_data_cam(T, sheet_name)
        k_list.append(k_values) 
        w_market_list.append(w_market)

    #print(k_list)
    #print("length k_list")
    #print(len(k_list))
    #print(len(expiries))

    for w_market in w_market_list:
        atm_var = np.mean(w_market)
        x0.extend([atm_var, 0.1, 0.0, 0.0, 0.1] ) # [a, b, rho, m, sigma] initial guess 

    number_of_slices = len(expiries)
    bounds = SVI_BOUNDS * number_of_slices # bounds for each slice 

    result = minimize(
        svi_full_objective,
        x0,
        args=(k_list, w_market_list),
        method='SLSQP',
        bounds=bounds,
                              #>=0         anonymous function            constraints
        constraints=[{'type':'ineq', 'fun': lambda p:          calendar_constraints(p, k_list)}] 
    )
    fitted_slices = []
    for i in range(len(expiries)):
        slice_params = result.x[i*5:(i+1)*5]
        fitted_slices.append(dict(zip(['a','b','rho','m','sigma'], slice_params))) #fitted parameters for each slice

    #print(len(fitted_slices))
    #print(len(x0))
    #print(len(bounds))

    return fitted_slices

def plot_multi_slice(sheet_name, filepath="tests/data/Surfaces.xlsx"):
    df = pd.read_excel("tests/data/Surfaces.xlsx", sheet_name)
    expiries = sorted(df["Year Fraction"].unique())
    fitted_slices=fit_svi_multi_slsqp(sheet_name,filepath="tests/data/Surfaces.xlsx")

    plt.figure(figsize=(10,6))

    for i, T in enumerate(expiries):

        strikes,market_vols,forward,k_values,w_market=get_slice_from_data_cam(T, sheet_name, filepath)
        params = fitted_slices[i]   

        k_grid = np.linspace(min(k_values), max(k_values), 200)
        w_fitted_grid = total_variance_cam(k_grid, **params)

        plt.scatter(k_values, np.log(w_market), s=15)

        plt.plot(k_grid, np.log(w_fitted_grid), linewidth=2)
        plt.xlabel("k")
        plt.ylabel("log(w)")
        plt.title(f"SVI Variance Surface — {sheet_name}", fontsize=13, fontweight="bold")
   
    plt.tight_layout()
    plt.show()

# LOCAL_METHODS = {
#     "slsqp": fit_svi_multi_slsqp,
# }
# fit_svi_slice = fit_svi_multi_slsqp
# plot_multi_slice("Surface3", filepath="tests/data/Surfaces.xlsx")
"""
