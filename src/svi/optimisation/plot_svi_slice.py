from svi.optimisation.SVI_SliceFit import fit_svi_slice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_slice_from_data_sheet(T,sheet_name,tol=1e-8):
    excel = pd.ExcelFile("tests/data/Surfaces.xlsx")
    df = pd.read_excel(excel, sheet_name)
    slice_df= df[np.isclose(df['Year Fraction'],T,atol=1e-8)]
    # ^^^ For specific expiry we use a tolerance as the sheet rounds up value 1e-8 is enough
    strikes=slice_df['Strike'].values #strikes from data sheet
    market_vols= slice_df['Volatility'].values # market volatility from data sheet
    forward=slice_df['Forward'].iloc[0] # Forward should be constant for constant expiry
    k_values = np.log(strikes / forward)   #log-moneyness-k from data-sheet
    w_market = market_vols ** 2 * T        # Market Total implied variance from data sheet
    return strikes,market_vols,forward,T,k_values,w_market

strikes,market_vols,forward,T,k_values,w_market=get_slice_from_data_sheet(0.01369863,"Surface1")

FittedSVIparameters= fit_svi_slice(
    strikes=strikes,
    market_vols=market_vols,
    T=T,
    forward=forward
)

a = FittedSVIparameters['a']
b = FittedSVIparameters['b']
rho = FittedSVIparameters['rho']
m = FittedSVIparameters['m']
sigma = FittedSVIparameters['sigma']


def total_variance(k, a, b, rho, m, sigma): 
    w_SVI= a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    return  w_SVI

k_grid = np.linspace(min(k_values), max(k_values), 200)
w_SVI=total_variance(k_grid,a,b,rho,m,sigma)

plt.figure()
plt.scatter(k_values, w_market, label="Market Data", color="black")
plt.plot(k_grid, w_SVI, label="SVI fitted ", color="red")
plt.xlabel("log-moneyness-k")
plt.ylabel("Total Implied Variance")
plt.legend()
plt.title("SVI Fit vs Market Data")
plt.grid(True)
plt.show()
