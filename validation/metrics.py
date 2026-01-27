"""Module to align and subset IFS and observation data"""

import numpy as np
import xarray as xr

def add_gof_stats(
    ds: xr.Dataset,
    var1: str,
    var2: str,
    output_prefix: str | None = None,
    ) -> xr.Dataset:
    """
    Compute goodness-of-fit (GOF) statistics between two variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variables to compare.
    var1 : str
        Name of the first variable (e.g., observed).
    var2 : str
        Name of the second variable (e.g., modeled).
    output_prefix : str, optional
        Prefix for output variable names. If None, uses "{var1}_vs_{var2}".

    Returns
    -------
    xarray.Dataset
        The input dataset with new variables: RMSE, MBE, SDE, and residuals.
    """
    data1 = ds[var1]
    data2 = ds[var2]

    valid = np.isfinite(data1) & np.isfinite(data2)
    res = (data2 - data1).where(valid)

    rmse = np.sqrt((res**2).mean(dim="time", skipna=True))
    mbe = res.mean(dim="time", skipna=True)
    sde = res.std(dim="time", skipna=True, ddof=1)

    prefix = output_prefix if output_prefix else f"{var1}_vs_{var2}"
    
    ds[f"{prefix}_RMSE"] = rmse
    ds[f"{prefix}_MBE"] = mbe
    ds[f"{prefix}_SDE"] = sde
    ds[f"{prefix}_residuals"] = res

    return ds