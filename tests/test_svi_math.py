import numpy as np
from src.svi.implementation.svi_model import SVI

def verify_svi_math():
    a = 0.04
    b = 0.1
    rho = 0
    m = 0
    sigma = 0.1
    k = 0
    T = 1

    svi = SVI(a, b, rho, m, sigma)

    w = svi.total_variance(k)
    
    # w = 0.04 + 0.1 * sqrt(0.01) = 0.04 + 0.01 = 0.05
    expected_w = 0.05
    
    print(f"Total Variance w(k=0):")
    print(f"Calculated: {w}")
    print(f"Expected: {expected_w}")
    print(f"Match: {np.isclose(w, expected_w)}")
    
    iv = svi.svi_implied_vol(k, T)
    
    #IV = sqrt(0.05 / 1) = sqrt(0.05) ≈ 0.2236
    expected_iv = np.sqrt(0.05)
    
    print(f"\nImplied Volatility (T=1):")
    print(f"Calculated:{iv:.6f}")
    print(f"Expected: {expected_iv:.6f}")
    print(f"Match: {np.isclose(iv, expected_iv)}")
    

    if np.isclose(w, expected_w) and np.isclose(iv, expected_iv):
        print("\n SVI MATH GOOD, all calculations same as hand calculations")
        return True
    else:
        print("\n SVI MATH FAILED, calculations do not match")
        return False

if __name__ == "__main__":
    verify_svi_math()
