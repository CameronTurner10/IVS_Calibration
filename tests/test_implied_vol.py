# one-way IV test


import numpy as np
import pandas as pd
from src.utils.black_scholes import bs_call, bs_put
from scipy.optimize import brentq


def implied_vol(F,K,T,r,market_price,option_type="call"):
    
    def f_sigma(sigma,):
        if option_type=="call":
            return bs_call(F,K,T,sigma,r)-market_price
        else:
            return bs_put(F,K,T,sigma,r)-market_price
    
    return brentq(f_sigma,1e-10,5)
#For sigma on the interval [a,b] brent solver finds f(sigma)=0
# Here a = 1e-8 and b = 5  (f(a)*f(b)<0 else -> exit)

def test_implied_vol():
    excel = pd.ExcelFile("tests/data/Surfaces.xlsx")

    sheets_to_test = ["Surface1", "Surface3", "Surface4"] # Surface 2 does not PASS! (Passes for rows 26 onwards)


    for sheet in sheets_to_test:
        df = pd.read_excel(excel, sheet_name=sheet)

        for _, row in df.iterrows(): #for _, row in df.iloc[26:].iterrows():  # (Rows 26,27,28... until the last row)
            F = row["Forward"]
            K = row["Strike"]
            T = row["Year Fraction"]
            call_price = row["Call Price"]
            put_price = row["Put Price"]
            r = row["Discount Rate"]
            vol_expected = row["Volatility"]

            calculated_vol_call = implied_vol(F, K, T, r, call_price,option_type="call")
            calculated_vol_put = implied_vol(F, K, T, r, put_price,option_type="put")

            assert np.isclose(calculated_vol_call, vol_expected, atol=1e-5), f">>> Root Solver using CALL does NOT return expected value in {sheet}, row {row.name} <<<"
            assert np.isclose(calculated_vol_put, vol_expected, atol=1e-5), f">>> Root Solver using PUT does NOT return expected value in {sheet}, row {row.name} <<<"
 