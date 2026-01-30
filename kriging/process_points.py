"""Module to compute residuals between IFS and point obs"""

import os
import xarray as xr
import glob
import numpy as np
import sys
import importlib
import pandas as pd

# Import modules from git repo 
sys.path.insert(0, "/home/lucasnorth/")
processing = importlib.import_module("2026_nvhackathon.validation.processing")
subset_grid_to_point_xy = processing.subset_grid_to_point_xy
forecast_obs_merge = processing.forecast_obs_merge
convert_1d_to_2d_latlon = processing.convert_1d_to_2d_latlon
metrics = importlib.import_module("2026_nvhackathon.validation.metrics")
add_gof_stats = metrics.add_gof_stats


# Define file paths
OBSERVATIONS = '/project/cowy-nvhackathon/cowy-wildfire/data/observations/cowy_madis_metar_mesonet_2024.nc'
IFS_PATHS = sorted(glob.glob("/project/cowy-nvhackathon/cowy-wildfire/data/nwp/ifs_yearly/*"))


def calc_met_gof(
    path_ifs: str,
    path_obs: str = OBSERVATIONS,
    path_save: str = None,
    option_save: bool = False
):
    """
    Compute goodness of fit statistcs between IFS and met observations

    Adapted from K. Bazlen
    
    Parameters
    ----------
    path_ifs: str
        Full filepath to the IFS data
    path_obs: str
        Full filepath to the observation data
    path_save: str | None
        Full filepath to the saved ouput. Must have .nc file extension.
    option_save: bool
        True writes file to path_save

    Return
    ------
    all_data_gof: xr.Dataset
        Dataset with point residuals and goodness of fit
    
    """
    if path_save is None & option_save:
        msg = "Save path must be specified to save."
        raise ValueError(msg)
    else:
        if not os.path.isdir(dir_save):
            msg = f"Save directory does not exist: {dir_save}"
            #logger.error(msg)
            raise NotADirectoryError(msg)

    # Open the observations file
    obs_ds = xr.open_dataset(path_obs) #space, time diminsions
    
    # Open the first IFS file
    ifs_f72 = xr.open_dataset(IFS_PATHS[0]) #time, lat, lon dims
    
    #drop stations with all NA windspeed data
    has_data_mask = ~obs_ds['windspeed_10m'].isnull().all(dim='time')
    obs_ds_clean = obs_ds.sel(space=has_data_mask)
    
    # # debug 
    # print(f"Original stations: {obs_ds.dims['space']}")
    # print(f"Stations with data: {obs_ds_clean.dims['space']}")
    
    # align data
    ds = convert_1d_to_2d_latlon(ds = ifs_f72)
    ifs_subset = subset_grid_to_point_xy(ds=ds, point_ds=obs_ds_clean)
    all_data = forecast_obs_merge(
        ds1=ifs_subset,
        ds2=obs_ds_clean,
        ds1_timevar="valid_time",
        ds2_timevar="time")
    
    # add windspeed error as a variable
    all_data['ws_error'] = all_data['ws_10'] - all_data['obs_windspeed_10m']
    
    # add GOF statistics
    all_data_gof = add_gof_stats(
        ds = all_data,
        var1 = "ws_10",
        var2 = "obs_windspeed_10m")
    
    if option_write:
        all_data_gof.to_netcdf(path_save)
    
    return(all_data_gof)


def filter_sufficient_data(
    path_gof: str,
    valid_thresh: float = 0.5,
    var_to_assess: str = "ws_error",
    dim_time: str = "time",
    path_save: str = None,
    option_save: bool = False
):
    """
    Filter data based on a fraction of existing timesteps

    
    Parameters
    ----------
    path_gof: str
        Full filepath to point GOF data, from calc_met_gof
    valid_thresh: float
        Fraction of valid data, must be greater than. Range from 0 to 1
    path_save: str | None
        Full filepath to the saved ouput. Must have .nc file extension.
    option_save: bool
        True writes file to path_save

    Return
    ------
    ds_filtered: xr.Dataset
        Dataset filtered to sufficient observation stations
    
    """
    if path_save is None & option_save:
        msg = "Save path must be specified to save."
        raise ValueError(msg)
    else:
        if not os.path.isdir(dir_save):
            msg = f"Save directory does not exist: {dir_save}"
            #logger.error(msg)
            raise NotADirectoryError(msg)
    
    ds = xr.open_dataset(path_gof)
    
    # Fraction of valid (non-NaN) values per space
    valid_frac = ds[var_to_assess].notnull().mean(dim=dim_time)
    
    # Mask space dimension
    ds_filtered = ds.where(valid_frac >= valid_thresh, drop=True)

    if option_write:
        ds_filtered.to_netcdf(path_save)
    
    return(ds_filtered)    
