# SVI Multi Slice Plot
from svi.optimisation.SVI_SliceFit import fit_svi_slice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.Plot_Single_Slice import svi_single_slice

def svi_multi_slice(sheet_name):
    df = pd.read_excel("tests/data/Surfaces.xlsx", sheet_name="Surface1")
    expiries = sorted(df['Year Fraction'].unique()) # Picks all the different expiries - no repeates only unique 

    plt.figure(figsize=(10,6))

    sheet_name="Surface1"
    for T in expiries: #loop over all expiries for multi slice plot
        k_values, w_market, k_grid, w_SVI = svi_single_slice(T, sheet_name)

        plt.scatter(k_values, w_market) # Market data plot
        
        plt.plot(k_grid, w_SVI) # Fitted SVI plot

    plt.xlabel("log-moneyness k")
    plt.ylabel("Total Implied Variance")
    plt.title("SVI Fit vs Market Data for All Expiries")

    plt.grid(True)
    plt.show()

svi_multi_slice("Surface1")




