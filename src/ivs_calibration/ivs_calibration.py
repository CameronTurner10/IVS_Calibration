
from src.smoothing_spline.optimisation.fit_spline import fit_all_splines
from src.svi.optimisation.fit_svi import fit_svi_surface


def calibrate_spline_surface(market_data, lam):
    """
    Calibrate smoothing spline surface to market data.
    market_data: dict with keys as maturities and values as (strikes, implied_vols)
    lam: smoothing parameter
    Returns: dict of SmoothingSpline objects keyed by maturity
    """
    spline_surface = fit_all_splines(market_data, lam)
    return spline_surface

def calibrate_svi_surface(market_data):
    """
    Calibrate SVI surface to market data.
    market_data: dict with keys as maturities and values as (strikes, implied_vols)
    Returns: dict of SVI objects keyed by maturity
    """
    svi_surface = fit_svi_surface(market_data)
    return svi_surface


def compare_spline_vs_svi(market_data: dict, svi_fitted: dict, spline_fitted: dict) -> pd.DataFrame:
    """
    Compare the fit quality of smoothing spline vs SVI on the same surface.

    Parameters
    ----------
    market_data : dict
        Dict {T: (strikes, market_vols)}
    svi_fitted : dict
        Dict {T: svi_params_dict}
    spline_fitted : dict
        Dict {T: result_dict}

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['maturity', 'spline_rmse', 'svi_rmse', 'rmse_diff', 'spline_arb_free', 'winner']

    Notes
    -----
    winner = 'spline' if spline_rmse < svi_rmse else 'svi'. spline_arb_free from check_arbitrage(result)['pass'].
    """
    raise NotImplementedError("Not yet implemented")


def calibration_pipeline(market_data, method='spline', lam=0.1):
    """
    Full calibration pipeline to fit either smoothing spline or SVI surface.
    market_data: dict with keys as maturities and values as (strikes, implied_vols)
    method: 'spline' or 'svi'
    lam: smoothing parameter for spline method
    Returns: dict of calibrated models keyed by maturity
    """
    if method == 'spline':
        return calibrate_spline_surface(market_data, lam)
    elif method == 'svi':
        return calibrate_svi_surface(market_data)
    else:
        raise ValueError("Unknown calibration method: choose 'spline' or 'svi'")
    