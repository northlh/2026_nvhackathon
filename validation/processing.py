"""Module to align and subset IFS and observation data"""

import xarray as xr
import numpy as np


def forecast_obs_merge(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    ds1_timevar: str,
    ds2_timevar: str,
    ds_2_prefix: str = "obs"
    ):
    """
    Merge forecast and observation datasets on matching time and space.
    """
    
    # Find common times
    common_times = np.intersect1d(ds1[ds1_timevar].values, ds2[ds2_timevar].values)
    
    if len(common_times) == 0:
        raise ValueError(f"No common times found between {ds1_timevar} and {ds2_timevar}")
    
    # Create boolean masks for filtering
    mask1 = ds1[ds1_timevar].isin(common_times)
    mask2 = ds2[ds2_timevar].isin(common_times)
    
    # Subset using the time dimension
    ds1_subset = ds1.isel(time=mask1)
    ds2_subset = ds2.isel(time=mask2)
    
    # Assign common_times as the time coordinate
    ds1_subset = ds1_subset.assign_coords(time=common_times)
    ds2_subset = ds2_subset.assign_coords(time=common_times)
    
    # Align space coordinate types
    ds_1_aligned = ds1_subset.assign_coords(space=np.array(ds1_subset.space.values, dtype=object))
    ds_2_aligned = ds2_subset.assign_coords(space=np.array(ds2_subset.space.values, dtype=object))
    
    # Optionally rename variables in ds2
    if ds_2_prefix:
        ds_2_aligned = ds_2_aligned.rename({var: f"{ds_2_prefix}_{var}" for var in ds_2_aligned.data_vars})
    
    # Merge datasets
    merged = xr.merge([ds_1_aligned, ds_2_aligned], join="exact")
    merged = merged.apply(lambda x: x.astype("float64"))
    
    return merged

def subset_grid_to_point_xy(
    ds: xr.Dataset,
    point_ds: xr.Dataset,  
        ) -> xr.Dataset:
    """
    Match each observation site to the nearest grid cell (with y/x dimensions).

    Parameters
    ----------
    ds : xr.Dataset
        Gridded data with dims ('time', 'y', 'x') and 2D 'latitude',
        'longitude'.

    point_ds : xr.Dataset
        Point datat with dims ('time', 'space') and 1D 'latitude', 'longitude'
        for each site.

    Returns
    -------
    xr.Dataset
        Subset of gridded data with dims ('time', 'space') containing values
        from the grid cell nearest to each observation location.
    """
    # Ensure overlapping time range
    point_ds = point_ds.where(point_ds.time.isin(ds.time), drop=True)

    # Extract grid coordinate arrays
    hrrr_lat = ds["latitude"].values  # shape: (y, x)
    hrrr_lon = ds["longitude"].values  # shape: (y, x)

    nearest_y = []
    nearest_x = []

    # Loop over each observation site
    for i in range(point_ds.sizes["space"]):
        lat_obs = float(point_ds["latitude"].isel(space=i))
        lon_obs = float(point_ds["longitude"].isel(space=i))

        # Compute 2D distance field (simple Euclidean in lat/lon space)
        dist = np.sqrt((hrrr_lat - lat_obs) ** 2 + (hrrr_lon - lon_obs) ** 2)

        # Find nearest grid cell index
        flat_idx = dist.argmin()
        y_idx, x_idx = np.unravel_index(flat_idx, dist.shape)

        nearest_y.append(y_idx)
        nearest_x.append(x_idx)

    # Create index arrays
    nearest_y = xr.DataArray(nearest_y, dims="space")
    nearest_x = xr.DataArray(nearest_x, dims="space")

    # Subset HRRR at those points
    hrrr_subset = ds.isel(y=nearest_y, x=nearest_x)

    # Assign obs metadata back onto the new dataset
    hrrr_subset = hrrr_subset.assign_coords(
        space=np.arange(point_ds.sizes["space"]),
        latitude=("space", point_ds["latitude"].values),
        longitude=("space", point_ds["longitude"].values),
    )

    # Drop elevation if present
    hrrr_subset = hrrr_subset.drop_vars("elevation", errors="ignore")

    return hrrr_subset

def convert_1d_to_2d_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert a dataset with 1D lat/lon coordinates to 2D (y, x) grid format.
    """
    lat_2d, lon_2d = np.meshgrid(ds["latitude"], ds["longitude"], indexing="ij")

    ds = ds.assign_coords({
        "latitude": (("y", "x"), lat_2d),
        "longitude": (("y", "x"), lon_2d),
    })

    ds = ds.swap_dims({"latitude": "y", "longitude": "x"})
    ds = ds.drop_dims(["lat", "lon"], errors="ignore")

    return ds