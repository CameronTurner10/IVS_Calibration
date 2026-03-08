"""
Docstring for svi surfaces
- butterfly arbitrage testing
- calendar arbitrage testing


"""
import numpy as np
from src.svi.optimisation.constraints import c_nonneg_min_total_var, c_wing_right, c_wing_left

def check_basic_constraints(params):
    p = np.array([params["a"], params["b"], params["rho"], params["m"], params["sigma"]])
    return (c_nonneg_min_total_var(p) >= -1e-8
            and c_wing_right(p) >= -1e-8
            and c_wing_left(p) >= -1e-8)


def check_butterfly_arbitrage(surface):
    """
    Check for butterfly arbitrage in the given surface.
    Placeholder function - implement actual logic.
    """
    return True

def check_calendar_arbitrage(surface):
    """
    Check for calendar arbitrage in the given surface.
    Placeholder function - implement actual logic.
    """
    return True