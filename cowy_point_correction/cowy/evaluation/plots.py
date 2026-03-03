
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_train_test_map(train_ds, test_ds, dem_path: str = None) -> plt.Axes:
    """
    Plot train/test observation locations, optionally on top of a DEM.
    Returns the axis for further customization or saving.
    """
    train_lon = train_ds.obs_lookup["longitude"].values
    train_lat = train_ds.obs_lookup["latitude"].values
    test_lon  = test_ds.obs_lookup["longitude"].values
    test_lat  = test_ds.obs_lookup["latitude"].values

    # fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=38
    )})

    if dem_path is not None:
        dem = xr.open_dataset(dem_path)
        im = ax.pcolormesh(
            dem["longitude"], dem["latitude"], dem["HGT"],
            cmap="cubehelix", shading="auto", transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        plt.colorbar(im, ax=ax, label="Elevation (m)")
        # lon_range = float(dem["longitude"].max() - dem["longitude"].min())
        # lat_range = float(ddef plot_train_test_map(train_ds, test_ds, dem_path: str = None) -> plt.Axes:
    """
    Plot train/test observation locations, optionally on top of a DEM.
    Returns the axis for further customization or saving.
    """
    train_lon = train_ds.obs_lookup["longitude"].values
    train_lat = train_ds.obs_lookup["latitude"].values
    test_lon  = test_ds.obs_lookup["longitude"].values
    test_lat  = test_ds.obs_lookup["latitude"].values

    fig, ax = plt.subplots(figsize=(10, 8))

    if dem_path is not None:
        dem = xr.open_dataset(dem_path)
        im = ax.pcolormesh(
            dem["longitude"], dem["latitude"], dem["HGT"],
            cmap="cubehelix", shading="auto",
            rasterized=True,
        )
        plt.colorbar(im, ax=ax, label="Elevation (m)")

    ax.scatter(train_lon, train_lat, s=15, c="blue", alpha=0.5, edgecolors="lightgray", linewidth=0.2, label="Train")
    ax.scatter(test_lon,  test_lat,  s=15, c="red",  alpha=0.5, edgecolors="lightgray", linewidth=0.2, label="Test")

    states_shp = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
    gpd.read_file(states_shp).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

    ax.set_xlim(test_lon.min(), test_lon.max())
    ax.set_ylim(test_lat.min(), test_lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    ax.legend(loc="upper right")

    return ax

def density_scatter(
    ds: xr.Dataset = None,
    x=None,
    y=None,
    bins=100,
    cmap="viridis",
    one_to_one_line=True,
    trend_line=False) -> plt.Axes:
    """
    Scatter plot with coloring by point density.

    Parameters
    ----------
    ds : xr.Dataset, optional
        Dataset containing x and y variables. If provided, x and y should be variable names (str).
    x : str or np.ndarray
        Name of x variable in the dataset (if ds is provided) or x values as a NumPy array.
    y : str or np.ndarray
        Name of y variable in the dataset (if ds is provided) or y values as a NumPy array.
    bins : int, optional
        Number of bins for the 2D histogram. Default is 100.
    cmap : str, optional
        Colormap for the scatter plot. Default is "viridis".

    Returns
    -------
    matplotlib.axes.Axes
        Axis of the created plot.
    """

    # Extract data from xarray.Dataset if provided and x/y are variable names
    if ds is not None and isinstance(x, str) and isinstance(y, str):
        x = ds[x].values.ravel()
        y = ds[y].values.ravel()
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        x = x.ravel()
        y = y.ravel()
    else:
        raise ValueError("Provide either a dataset with variable names (x, y as str) or x and y as NumPy arrays.")

    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # Compute 2D histogram for density
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Map each point to its bin density
    ix = np.searchsorted(xedges, x) - 1
    iy = np.searchsorted(yedges, y) - 1
    ix = np.clip(ix, 0, counts.shape[0] - 1)
    iy = np.clip(iy, 0, counts.shape[1] - 1)
    density = counts[ix, iy]

    # Create the scatter plot
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=density, cmap=cmap, s=5)
    fig.colorbar(sc, label="Density")
    # sc = ax.scatter(x, y, c=np.log1p(density), cmap=cmap, s=5)
    # fig.colorbar(sc, label="Log Density")

    # Add 1:1 line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    if one_to_one_line:
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=1.2,
            label="1:1 Line",
    )

    # Add trend line (linear regression)
    if trend_line:
        # Calculate linear regression
        coeffs = np.polyfit(x, y, 1)  # Returns [slope, intercept]
        trend_y = coeffs[0] * x + coeffs[1]
        
        # Plot trend line
        ax.plot(
            [x.min(), x.max()],
            [coeffs[0] * x.min() + coeffs[1], coeffs[0] * x.max() + coeffs[1]],
            "r--",
            linewidth=1.2,
            label=f"Trend (y = {coeffs[0]:.2f}x + {coeffs[1]:.2f})",
        )

    return ax

def confusion_matrix(obs, pred, threshold, labels=None):
    """
    Compute a 2×2 confusion matrix for >= threshold classification.

    Parameters
    ----------
    obs : np.ndarray
        Observed windspeed array.
    pred : np.ndarray
        Predicted/model windspeed array.
    threshold : float
        Threshold in same units as obs/pred.
    labels : list of str, optional
        Labels for the bins, e.g. ["0–15 m/s", "15–50 m/s"].

    Returns
    -------
    cm_df : pd.DataFrame
        2×2 confusion matrix with rows = predicted, columns = observed.
    """
    obs_bin = (obs >= threshold).astype(np.uint8).ravel()
    pred_bin = (pred >= threshold).astype(np.uint8).ravel()

    # bincount trick: 0=TN, 1=FP, 2=FN, 3=TP → reshape into [[TN,FP],[FN,TP]]
    cm = np.bincount(2 * obs_bin + pred_bin, minlength=4).reshape(2, 2).T

    if labels is None:
        labels = [f"< {threshold}", f">= {threshold}"]

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df

def plot_confusion_matrix(cm_df, title="", fontsize=16, number_fontsize=22):
    """
    2x2 confusion-matrix plot with outlined boxes and large numbers.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    nrows, ncols = cm_df.shape

    for i in range(nrows):
        for j in range(ncols):
            rect = Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=1
            )
            ax.add_patch(rect)

            # Numbers
            ax.text(
                j, i,
                f"{cm_df.values[i, j]:,}",
                ha="center", va="center",
                fontsize=number_fontsize
            )

    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(cm_df.columns, fontsize=fontsize)
    ax.set_yticklabels(cm_df.index, fontsize=fontsize)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Axis labels
    ax.set_xlabel("Observed", fontsize=fontsize)
    ax.set_ylabel("Forecasted", fontsize=fontsize)
    ax.xaxis.set_label_position("top")

    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    plt.tight_layout()
    plt.show()