import numpy as np

def f(x):
    return x**2 -2

def df(x):
    return 2*x

def Newton_Raphson(f,df,x0,tol,iterations):
    x=x0
    for i in range(iterations):

        if df(x) == 0:
            return None
        
        x = x - f(x)/df(x)
        
        if abs(f(x))<tol:
            return x
    return x

Root=Newton_Raphson(f,df,1,1e-6,100)
print(Root)
