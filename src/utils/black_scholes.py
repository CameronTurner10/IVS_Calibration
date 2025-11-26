import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



# F -> forward price at time T  F>0
# K -> strike price             K>0
# T -> time to maturity         T>0
# sigma -> implied volatility   sigma>0

D = 1
N = norm.cdf

def d1(F, K, T, sigma):
    return (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))

def bs_call(F,K,T,sigma):

    if T<=0: 
        return max(F-K,0)
    if sigma<=0:
        return max(F-K,0)
    
    d1_value = d1(F,K,T,sigma)
    d2_value = d1_value - sigma * np.sqrt(T)

    C = D*(F * N(d1_value) - K * N(d2_value))

    return C

def vega(F, K, T, sigma):
    d1_value = d1(F, K, T, sigma)
    return F * np.sqrt(T) * norm.pdf(d1_value)

def f_sigma(sigma, F, K, T, market_price):
    return bs_call(F, K, T, sigma) - market_price


if __name__ == "__main__":
    F = 5000
    T = 0.25
    sigma = 0.20

    example_p = bs_call(F, 5500, T, sigma)
    print(f"The call price at a strike price of 5500 is {example_p}")

    strikes = np.linspace(4500, 6000, 100)
    prices = []
    for K in strikes:
        price = bs_call(F, K, T, sigma)
        prices.append(price)

    plt.plot(strikes, prices)
    plt.title("Call price vs Strike")
    plt.xlabel("Strike K")
    plt.ylabel("Call price")
    plt.grid(True)
    plt.show()
        



    