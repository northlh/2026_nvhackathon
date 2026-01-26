
import os
import warnings

def plot_train_test_map(train_ds, test_ds, output_png: str):
    """
    Plot train/test observation locations on top of a DEM.
    Requires cartopy and geopandas. If not installed, no-op with warning.
    """
    try:
        import xarray as xr
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import geopandas as gpd
        import cartopy.io.shapereader as shpreader
    except Exception as e:
        warnings.warn(f"cartopy/geopandas not available: {e}")
        return

    # Attempt to discover a DEM from the dataset's topo list, or fallback to path guess
    dem_path = None
    if hasattr(train_ds, "dset_topo") and "HGT" in train_ds.dset_topo.data_vars:
        dem = train_ds.dset_topo
    else:
        # Fallback: try to open a likely DEM filename if present nearby
        dem_guess = "/projects/cowy/datasets/terrain_data/terrain_990m/conus_elev_reprojected_wgs84_cowy_990m.nc"
        if os.path.exists(dem_guess):
            dem = xr.open_dataset(dem_guess)
        else:
            warnings.warn("DEM not found; plotting without background.")
            dem = None

    train_lon = train_ds.obs_lookup["longitude"].values
    train_lat = train_ds.obs_lookup["latitude"].values
    test_lon = test_ds.obs_lookup["longitude"].values
    test_lat = test_ds.obs_lookup["latitude"].values

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    if dem is not None and "longitude" in dem and "latitude" in dem and "HGT" in dem:
        im = ax.pcolormesh(
            dem["longitude"],
            dem["latitude"],
            dem["HGT"],
            cmap="cubehelix",
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        plt.colorbar(im, ax=ax, label="Elevation (m)")

    # States outlines
    try:
        states_shp = shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_1_states_provinces"
        )
        gdf_states = gpd.read_file(states_shp)
        gdf_states.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, transform=ccrs.PlateCarree())
    except Exception as e:
        warnings.warn(f"Could not draw states outlines: {e}")

    ax.scatter(train_lon, train_lat, s=15, c="blue", alpha=.5, edgecolors="lightgray", linewidth=0.2, label="Train")
    ax.scatter(test_lon,  test_lat,  s=15, c="red",  alpha=.5, edgecolors="lightgray", linewidth=0.2, label="Test")

    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    ax.set_title("Training and Test Set Observations")
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close(fig)
