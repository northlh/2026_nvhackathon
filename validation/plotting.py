"""Module to create plots for validation results"""

import matplotlib.pyplot as plt
import seaborn as sns



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