import pandas as pd
import numpy as np
from src.utils.root_finder import implied_vol
from src.utils.black_scholes import bs_call

def debug_surface2():
    df = pd.read_excel("tests/data/Surfaces.xlsx", sheet_name="Surface2")
    print(f"Total rows in Surface2: {len(df)}")
    for i, row in df.head(10).iterrows():
        F = row["Forward"]
        K = row["Strike"]
        T = row["Year Fraction"]
        r = row["Discount Rate"]
        market_price = row["Call Price"]
        expected_vol = row["Volatility"]
        print(f"\nRow {i}: F={F:.2f}, K={K:.2f}, T={T:.6f}, r={r:.4f}, price={market_price:.4f}, expected_vol={expected_vol:.4f}")
        print(f"  Moneyness (F/K): {F/K:.4f}")
        check_price = bs_call(F, K, T, expected_vol, r)
        print(f"  BS price with expected vol: {check_price:.4f}")
        try:
            iv = implied_vol(F, K, T, r, market_price, option_type="call")
            print(f"  Calculated IV: {iv}")
            if np.isclose(iv, expected_vol, atol=1e-5):
                print(f"PASS")
            else:
                print(f"FAIL (diff: {abs(iv - expected_vol):.6f})")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    debug_surface2()
