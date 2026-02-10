import numpy as np
import matplotlib.pyplot as plt

def plot_svi_fit(k, market_iv, svi_model, T):
    k_grid = np.linspace(min(k), max(k), 200)
    iv_fit = svi_model.svi_implied_vol(k_grid, T)

    plt.figure()
    plt.scatter(k, market_iv, label="Market IV", color="black")
    plt.plot(k_grid, iv_fit, label="SVI fit", color="red")
    plt.xlabel("Strike k")
    plt.ylabel("Implied volatility")
    plt.legend()
    plt.title("SVI Fit vs Market Data")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import numpy as np
    from svi.implementation.svi_model import SVI

    k_data = np.linspace(-0.3, 0.3, 20)
    market_iv = 0.2 + 0.05 * k_data**2   # dummy smile
    T = 0.5

    svi = SVI(a=0.04, b=0.2, rho=-0.4, m=0.0, sigma=0.2)

    plot_svi_fit(k_data, market_iv, svi, T)


