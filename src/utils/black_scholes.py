import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



# F -> forward price at time T  F>0
# K -> strike price             K>0
# T -> time to maturity         T>0
# sigma -> implied volatility   sigma>0

D = 1
N = norm.cdf

def bs_call(F,K,T,sigma):

    if T<=0: 
        return max(F-K,0)
    if sigma<=0:
        return max(F-K,0)
    
    d1 = (np.log(F/K) + (sigma**2 * T)/2)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T) 

    C = D*(F * N(d1) - K * N(d2))

    return C

if __name__ == "__main__":
    F = 5000
    T = 0.25
    sigma = 0.20
    example_p = bs_call(F,5500,T,sigma)
    print(f"The call price at a strike price of 5500 is {example_p}")

    strikes = np.linspace(4500, 6000, 100)
    prices = []
    for K in strikes:
        price = bs_call(F,K,T,sigma)
        prices.append(price)
    
    plt.plot(strikes, prices)
    plt.title("Call price vs Strike")
    plt.xlabel("Strike K")
    plt.ylabel("Call price")
    plt.grid(True)
    plt.show()
        



    