# oneway BS
# Call-Put Parity


import numpy as np
import pandas as pd
from math import exp
from src.utils.black_scholes import bs_call, bs_put

def test_oneway_BS():
    excel = pd.ExcelFile("tests/data/Surfaces.xlsx")
    
    for sheet in excel.sheet_names:
        df = pd.read_excel(excel, sheet_name=sheet)

        for _, row in df.iterrows():
            F = row["Forward"]
            K = row["Strike"]
            T = row["Year Fraction"]
            sigma = row["Volatility"]
            call_expected = row["Call Price"]
            put_expected = row["Put Price"]
            r = row["Discount Rate"]
            our_call_model = bs_call(F,K,T,sigma,r) 
            
            our_put_model = bs_put(F,K,T,sigma,r)

            assert np.isclose(our_call_model, call_expected, atol=1e-5), f">>> CALL model does NOT return expected value in {sheet} <<<"
            assert np.isclose(our_put_model, put_expected, atol=1e-5), f">>> PUT model does NOT return expected value in {sheet} <<<"

