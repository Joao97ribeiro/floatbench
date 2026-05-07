# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
"""Plots for visualizing SHAP feature importance and beeswarm plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from floatbench.colors import COLORS_DICT, CUSTOM_MAP
from . import general


def plot_shap_importance_bar(
    mean_abs_shap: np.ndarray,
    feature_names: list[str],
    plot_dir: str,
    filename: str = "shap_importance_bar",
    save_svg: bool = False,
    style: str = "fivethirtyeight",
) -> None:
    """Plot global feature importance from SHAP values in paper style.

    Args:
        mean_abs_shap: Array with mean absolute SHAP values per feature.
        feature_names: List of feature names.
        plot_dir: Directory to save the plot.
        filename: Base name of the saved file.
        save_svg: Whether to also save the SVG version of the figure.
        style: Matplotlib style to use.
    """

    # order features by mean absolute SHAP
    imp_df = (pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))

    # set global visual style
    plt.style.use(style)
    fig, axs = plt.subplots(1, 1, figsize=(7, 5), facecolor="white")

    # plot
    axs.barh(
        imp_df["feature"][::-1],
        imp_df["mean_abs_shap"][::-1],
        color=COLORS_DICT["blue_paper"],
    )

    # add value labels
    for i, value in enumerate(imp_df["mean_abs_shap"][::-1]):
        axs.text(
            value,
            i,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color=COLORS_DICT["dark_gray_paper"],
        )

    # axes + style
    axs.set_xlabel("Mean |SHAP|", fontsize=10)
    axs.set_title("Global Feature Importance (SHAP)", fontsize=12, pad=10)
    axs.grid(True,
             linestyle="-",
             color=COLORS_DICT["light_gray_paper"],
             zorder=0)
    general.style_ticks(axs, fontsize=9)

    plt.tight_layout()

    # save figure
    general.save_figure(fig, plot_dir, filename, save_svg)


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    x_val_scaled: np.ndarray,
    feature_names: list[str],
    plot_dir: str,
    filename: str = "shap_beeswarm",
    save_svg: bool = False,
    style: str = "fivethirtyeight",
) -> None:
    """Generate a paper-style SHAP beeswarm plot.

    Args:
        shap_values: Array with SHAP values for each sample and feature.
        x_val_scaled: Scaled validation features (shape: n_samples ×
          n_features).
        feature_names: List of feature names corresponding to column order.
        plot_dir: Directory where the plot will be saved.
        filename: Base name of the saved figure.
        save_svg: Whether to save the figure in SVG format as well.
        style: Matplotlib style sheet to apply before plotting.
    """

    # set global visual style
    plt.style.use(style)

    # Convert inputs to dataframe for SHAP
    df_val = pd.DataFrame(x_val_scaled, columns=feature_names)

    # Base SHAP plot (hidden display)
    shap.summary_plot(
        shap_values,
        df_val,
        show=False,
        plot_size=(7, 5),
        cmap=CUSTOM_MAP,
    )

    fig = plt.gcf()
    axs = fig.axes[0] if fig.axes else plt.gca()

    # Title
    axs.set_title("SHAP Beeswarm – input features", fontsize=12, pad=10)

    # Axis ticks
    axs.tick_params(
        axis="both",
        which="major",
        length=4,
        width=1,
        labelsize=9,
        color=COLORS_DICT["dark_gray_paper"],
    )

    # Make y-labels match color theme
    for label in axs.get_yticklabels():
        label.set_color(COLORS_DICT["dark_gray_paper"])
        label.set_fontsize(9)

    # X-axis grid
    axs.grid(
        True,
        axis="x",
        linestyle="-",
        linewidth=0.8,
        alpha=0.7,
        color=COLORS_DICT["light_gray_paper"],
    )

    # Custom spines (only bottom visible)
    for side in ["top", "left", "right"]:
        axs.spines[side].set_visible(False)

    axs.spines["bottom"].set_visible(True)
    axs.spines["bottom"].set_linewidth(1)
    axs.spines["bottom"].set_color(COLORS_DICT["light_gray_paper"])

    # Zero vertical line (thinner)
    for line in axs.lines:
        xdata = getattr(line, "get_xdata", lambda: [])()
        if len(xdata) == 2 and xdata[0] == xdata[1] == 0:
            line.set_linewidth(1.0)
            line.set_color(COLORS_DICT["grey_paper"])

    # Colorbar styling
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]

        cbar_ax.tick_params(
            axis="y",
            color=COLORS_DICT["dark_gray_paper"],
            labelsize=8,
            width=1,
            length=3,
        )

        for tick_label in cbar_ax.get_yticklabels():
            tick_label.set_color(COLORS_DICT["dark_gray_paper"])
            tick_label.set_fontsize(8)

        # Colorbar label
        if cbar_ax.yaxis.label:
            cbar_ax.yaxis.label.set_fontsize(9)
            cbar_ax.yaxis.label.set_color(COLORS_DICT["dark_gray_paper"])

        # Remove border
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    # save figure
    general.save_figure(fig, plot_dir, filename, save_svg)
