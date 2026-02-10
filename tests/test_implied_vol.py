# one-way IV test
import numpy as np
import pandas as pd
from src.utils.black_scholes import bs_call, bs_put
from src.utils.root_finder import implied_vol

def test_implied_vol():
    excel = pd.ExcelFile("tests/data/Surfaces.xlsx")

    sheets_to_test = ["Surface1", "Surface2", "Surface3", "Surface4"] # Surface 2 does not PASS! (Passes for rows 26 onwards)


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
            #check call vol
            if np.isnan(calculated_vol_call):
                intrinsic_call = np.exp(-r*T) * max(F-K, 0)
                if abs(call_price - intrinsic_call) < 1e-9:
                    pass
                else:
                    assert False, f">>> Root Solver returned NaN for CALL in {sheet}, row {row.name} (Calculated={calculated_vol_call}, Expected={vol_expected}) <<<"
            else:
                assert np.isclose(calculated_vol_call, vol_expected, atol=1e-5), f">>> Root Solver using CALL does NOT return expected value in {sheet}, row {row.name} (Got {calculated_vol_call}, Expected {vol_expected}) <<<"
            #check put vol
            if np.isnan(calculated_vol_put):
                intrinsic_put = np.exp(-r*T) * max(K-F, 0)
                if abs(put_price - intrinsic_put) < 1e-9:
                    pass
                else:
                    assert False, f">>> Root Solver returned NaN for PUT in {sheet}, row {row.name} <<<"
            else:
                assert np.isclose(calculated_vol_put, vol_expected, atol=1e-5), f">>> Root Solver using PUT does NOT return expected value in {sheet}, row {row.name} <<<"
 