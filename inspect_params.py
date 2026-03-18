import pandas as pd
import numpy as np
from src.svi.optimisation.arbitrage import calibrate_surface

def inspect_calibration(sheet_name="Surface1"):
    fitted = calibrate_surface(sheet_name)
    print(f"\nCalibrated Parameters for {sheet_name}:")
    print(f"{'T':<10} | {'a':<10} | {'b':<10} | {'rho':<10} | {'m':<10} | {'sigma':<10}")
    print("-" * 65)
    for T in sorted(fitted.keys()):
        p = fitted[T]
        print(f"{T:<10.6f} | {p['a']:<10.6f} | {p['b']:<10.6f} | {p['rho']:<10.6f} | {p['m']:<10.6f} | {p['sigma']:<10.6f}")

if __name__ == "__main__":
    inspect_calibration()
