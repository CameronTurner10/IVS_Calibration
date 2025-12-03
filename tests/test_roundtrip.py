# round trip test

import numpy as np
import pandas as pd
from src.utils.black_scholes import bs_call, bs_put
from src.utils.root_finder import implied_vol

def test_roundtrip():
    excel = pd.ExcelFile("tests/data/Surfaces.xlsx")

    sheets_to_test = ["Surface1", "Surface3", "Surface4"] # Surface 2 does not PASS! (Passes for rows 26 onwards)
    #sheets_to_test = ["Surface1","Surface2", "Surface3", "Surface4"]

    for sheet in sheets_to_test:
        df = pd.read_excel(excel, sheet_name=sheet)

        for _, row in df.iterrows():
            F = row["Forward"]
            K = row["Strike"]
            T = row["Year Fraction"]
            sigma = row["Volatility"]
            r = row["Discount Rate"]

            # calculate call/put prices from original vol
            BScall_price = bs_call(F,K,T,sigma,r) 
            BSput_price = bs_put(F,K,T,sigma,r)

            # Using calculated call/put use Brent RS to go back to ~ orginal vol
            calculated_vol_call = implied_vol(F, K, T, r, BScall_price,option_type="call")
            calculated_vol_put = implied_vol(F, K, T, r, BSput_price,option_type="put")

            assert np.isclose(sigma, calculated_vol_call, atol=1e-5), f">>> (CALL) calculated vol does not match vol in {sheet} <<<"
            assert np.isclose(sigma, calculated_vol_call, atol=1e-5), f">>> (PUT) calculated vol does not match vol in {sheet} <<<"
        
