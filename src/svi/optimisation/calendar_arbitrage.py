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

def total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def get_slice_from_data(T, sheet_name, filepath="tests/data/Surfaces.xlsx"):
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
        w_model = total_variance(k_list[i], *slice_params)
        total_sum += np.sum((w_model - w_market_list[i])**2)
    return total_sum

def calendar_constraints(parameters,k_list):
    constraints=[]
    for i in range(1,len(k_list)):     #parameters[i*5:(i+1)*5] -> i=0 sliceT[0] parameters [0:5] = [a0,b0,rho0,m0,sigma0] if i=1 parameters [5,10]=[a1,b1,rho1,m1,sigma1]
        w_current=total_variance(k_list[i],*parameters[i*5:(i+1)*5]) #k_list[i] ->k's for T[i]
        w_previous=total_variance(k_list[i-1],*parameters[(i-1)*5:i*5]) #k_list[i-1] ->k's for T[i-1]

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
        strikes, market_vols, forward, k_values, w_market = get_slice_from_data(T, sheet_name)
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

        strikes,market_vols,forward,k_values,w_market=get_slice_from_data(T, sheet_name, filepath)
        params = fitted_slices[i]   

        k_grid = np.linspace(min(k_values), max(k_values), 200)
        w_fitted_grid = total_variance(k_grid, **params)

        plt.scatter(k_values, np.log(w_market), s=15)

        plt.plot(k_grid, np.log(w_fitted_grid), linewidth=2)
        plt.xlabel("k")
        plt.ylabel("log(w)")
        plt.title(f"SVI Variance Surface — {sheet_name}", fontsize=13, fontweight="bold")
   
   
    plt.tight_layout()
    plt.show()

LOCAL_METHODS = {
    "slsqp": fit_svi_multi_slsqp,
}

# Backwards-compatible alias so plotting.py and other existing files don't need to change
fit_svi_slice = fit_svi_multi_slsqp

plot_multi_slice("Surface4", filepath="tests/data/Surfaces.xlsx")