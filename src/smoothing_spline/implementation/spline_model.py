


class SmoothingSpline:
    """Smoothing Spline Model for Implied Volatility Surface
    Stores:
        strikes (list): List of strike prices
        imp_vol (list): List of implied volatilities corresponding to the strikes
        lam (float): Smoothing parameter lambda
        coef (list): Spline coefficients calculated during fittin
        """

    def __init__(self, strikes, imp_vol, lam, coef=None):
        self.strikes = strikes
        self.imp_vol = imp_vol
        self.lam= lam
        self.coef = coef

    def evaluate(self, k):
        """
        Evaluate spline at strike k, will implement later
        """
        raise NotImplementedError("Spline evaluation not implemented yet")

    def second_derivative(self, k):
        """
        Calculate second derivative (curvature) at strike k, could be useful to check convexity? 
        Butterfly arbirtrage happens when second derivative is negative when it shouldnt be
        Smoothness penalty is based on integral of square of second derivative
        """
        raise NotImplementedError("Second derivative calculation not implemented yet")
    
def fit_smoothing_spline(strikes, imp_vol, lam):
    """
    Creates the SmoothingSpline object.
    Fit a smoothing spline to the given strikes and vols with smoothing parameter lam.
    This is a placeholder function, actual implementation will involve solving a system of equations.
    Solve system of linear eqns:
        (K + λP) * c = y
    where K is the kernel matrix, P is the roughness penalty matrix, c is the spline coefficients we solve for, and y is the observed implied volatilities.
    ️
    
    """
    coef = None 
    spline = SmoothingSpline(strikes, imp_vol, lam, coef)
    return spline