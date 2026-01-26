"""Module to create plots for validation results"""

import matplotlib.pyplot as plt
import seaborn as sns



COLORS = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
]

MARKERS = ["o", "s", "^", "D", "v", "p", "h", "X"]

def configure_style(
    style="whitegrid",
    font_family="sans-serif",
    font_size=12,
    axes_label_size=14,
    axes_title_size=16,
    tick_label_size=11,
    legend_font_size=11,
    figure_dpi=150,
    savefig_dpi=300,
    grid_alpha=0.7,
    grid_linewidth=0.8,
    axes_linewidth=1.2,
    show_spines=True,
    ):
    """
    Configure matplotlib and seaborn for publication-ready figures.

    Parameters
    ----------
    style : str
        Seaborn theme style.
    font_family : str
        Font family for all text.
    font_size : int
        Base font size.
    axes_label_size : int
        Font size for axis labels.
    axes_title_size : int
        Font size for titles.
    tick_label_size : int
        Font size for tick labels.
    legend_font_size : int
        Font size for legend text.
    figure_dpi : int
        DPI for figure display.
    savefig_dpi : int
        DPI for saved figures.
    grid_alpha : float
        Grid line transparency.
    grid_linewidth : float
        Grid line width.
    axes_linewidth : float
        Axes border line width.
    show_spines : bool
        Whether to show all spines.

    Returns
    -------
    dict
        The configuration dictionary applied to rcParams.
    """
    sns.set_theme(style=style)

    config = {
        "font.family": font_family,
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": font_size,
        "axes.labelsize": axes_label_size,
        "axes.titlesize": axes_title_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "legend.fontsize": legend_font_size,
        "figure.dpi": figure_dpi,
        "savefig.dpi": savefig_dpi,
        "axes.linewidth": axes_linewidth,
        "axes.spines.top": show_spines,
        "axes.spines.right": show_spines,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": grid_alpha,
        "grid.linewidth": grid_linewidth,
    }

    plt.rcParams.update(config)

    return config


def filter_df_by_location(df, latitude, longitude):
    """
    Filter a DataFrame to rows matching the specified latitude and longitude.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'latitude' and 'longitude' columns.
    latitude : float
        Latitude value to match.
    longitude : float
        Longitude value to match.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only rows with the specified latitude and longitude.
    """
    return df[(df["latitude"] == latitude) & (df["longitude"] == longitude)]

def filter_df_by_time(df, start, end, time_col="timestamp"):
    """
    Filter DataFrame by a time range.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a datetime column.
    start : str or pd.Timestamp
        Start time (inclusive).
    end : str or pd.Timestamp
        End time (exclusive).
    time_col : str, optional
        Name of the datetime column (default: "timestamp").

    Returns
    -------
    pandas.DataFrame
        Filtered and sorted DataFrame.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    return df[(df[time_col] >= start) & (df[time_col] < end)]

def plot_timeseries(
    df,
    lines,
    latitude=None,
    longitude=None,
    start=None,
    end=None,
    time_col="timestamp",
    xlabel="Time",
    ylabel="Value",
    title=None,
    figsize=(10, 5),
    save_path=None,
    show=True,
    rotate_xticks=45,
    legend_outside=True,
    ):
    """
    Create a publication-ready time series plot with multiple lines.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    lines : list of dict
        Each dict defines a line with keys:
        - 'col': str (required) - column name in df
        - 'label': str (required)
        - 'color': str (optional)
        - 'marker': str (optional)
        - 'linewidth': float (optional, default 2)
        - 'markersize': float (optional, default 6)
    latitude : float, optional
        If provided, filters df to this latitude.
    longitude : float, optional
        If provided, filters df to this longitude.
    start : str or pd.Timestamp, optional
        If provided, filters df to times >= start.
    end : str or pd.Timestamp, optional
        If provided, filters df to times < end.
    time_col : str, optional
        Name of the datetime column (default: "timestamp").
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        If provided, saves figure to this path (without extension).
    show : bool, optional
        Whether to display the plot.
    rotate_xticks : int or None, optional
        Rotation angle for x-tick labels. None for no rotation.
    legend_outside : bool, optional
        Whether to place legend outside the plot.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    import pandas as pd
    
    filtered = df.copy()

    # Apply location filter
    if latitude is not None and longitude is not None:
        filtered = filter_df_by_location(filtered, latitude, longitude)

    # Apply time filter
    if start is not None or end is not None:
        filtered[time_col] = pd.to_datetime(filtered[time_col])
        filtered = filtered.sort_values(time_col)
        if start is not None:
            filtered = filtered[filtered[time_col] >= start]
        if end is not None:
            filtered = filtered[filtered[time_col] < end]

    x = filtered[time_col]

    fig, ax = plt.subplots(figsize=figsize)

    for i, line in enumerate(lines):
        ax.plot(
            x,
            filtered[line["col"]],
            label=line["label"],
            color=line.get("color", COLORS[i % len(COLORS)]),
            marker=line.get("marker", MARKERS[i % len(MARKERS)]),
            linewidth=line.get("linewidth", 2),
            markersize=line.get("markersize", 6),
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    ax.set_xlabel(xlabel, fontweight="medium")
    ax.set_ylabel(ylabel, fontweight="medium")

    if title:
        ax.set_title(title, fontweight="bold", pad=15)

    if legend_outside:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fancybox=False,
            edgecolor="gray",
            framealpha=1,
        )
    else:
        ax.legend(frameon=True, fancybox=False, edgecolor="gray", framealpha=1)

    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)

    if rotate_xticks is not None:
        plt.xticks(rotation=rotate_xticks, ha="right")

    plt.tight_layout()

    if legend_outside:
        fig.subplots_adjust(right=0.82)

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()

    return fig, ax

def plot_station_gof_map(
    df, 
    rmse_col='rmse_prediction_vs_observation',
    lat_col='latitude', 
    lon_col='longitude',
    cmap='YlOrRd', 
    s=30, 
    title='Station RMSE Map',
    cbar_label='RMSE', 
    ax=None, 
    state_borders=None,
    dem=None, 
    dem_var='HGT', 
    dem_cmap='Greys_r', 
    dem_alpha=0.6,
    dem_vmin=None, 
    dem_vmax=None,
    vmin=None, 
    vmax=None,
    ):
    """
    Plot station RMSE values on a map with optional DEM and state borders.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing station coordinates and RMSE values.
    rmse_col : str, optional
        Column name for RMSE values. Default is 'rmse_prediction_vs_observation'.
    lat_col : str, optional
        Column name for latitude. Default is 'latitude'.
    lon_col : str, optional
        Column name for longitude. Default is 'longitude'.
    cmap : str, optional
        Colormap for RMSE scatter points. Default is 'YlOrRd'.
    s : int, optional
        Marker size for scatter points. Default is 30.
    title : str, optional
        Plot title. Default is 'Station RMSE Map'.
    cbar_label : str, optional
        Colorbar label. Default is 'RMSE'.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure. Default is None.
    state_borders : GeoDataFrame, optional
        State boundaries to overlay. Default is None.
    dem : xr.Dataset, optional
        Digital elevation model dataset. Default is None.
    dem_var : str, optional
        Variable name in DEM dataset. Default is 'HGT'.
    dem_cmap : str, optional
        Colormap for DEM. Default is 'Greys_r'.
    dem_alpha : float, optional
        Transparency for DEM layer (0-1). Default is 0.6.
    dem_vmin : float, optional
        Minimum value for DEM colormap normalization. Default is None.
    dem_vmax : float, optional
        Maximum value for DEM colormap normalization. Default is None.
    vmin : float, optional
        Minimum value for RMSE colormap normalization. Default is None.
    vmax : float, optional
        Maximum value for RMSE colormap normalization. Default is None.

    Returns
    -------
    dict
        Dictionary containing:
        - 'ax': matplotlib.axes.Axes
        - 'fig': matplotlib.figure.Figure or None
        - 'cbar': matplotlib.colorbar.Colorbar
        - 'scatter': matplotlib.collections.PathCollection
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Plot DEM if provided (zorder=1 ensures it's behind other elements)
    if dem is not None:
        im = ax.pcolormesh(
            dem['longitude'], dem['latitude'], dem[dem_var],
            cmap=dem_cmap, shading='auto', alpha=dem_alpha,
            vmin=dem_vmin, vmax=dem_vmax, zorder=2
        )

    if state_borders is not None:
        state_borders.boundary.plot(ax=ax, color='black', linewidth=1, zorder=1)

    sc = ax.scatter(
        df[lon_col], df[lat_col], c=df[rmse_col],
        cmap=cmap, s=s, edgecolor='k', linewidth=0.5, zorder=3,
        vmin=vmin, vmax=vmax,
    )
    cbar = plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8, zorder=0)

    return {'ax': ax, 'fig': fig, 'cbar': cbar, 'scatter': sc}