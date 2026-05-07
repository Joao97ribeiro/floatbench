# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
# pylint: disable=duplicate-code
"""Plots for visualizing SHAP feature importance and beeswarm plots."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from mpl_toolkits.axes_grid1 import make_axes_locatable

from floatbench.colors import COLORS_DICT, mix_colors, CUSTOM_MAP_SEQ
from floatbench.utils import resolve_group_order
from . import general


def _draw_identity_and_band(axs: plt.Axes, lower_lim: float,
                            upper_lim: float) -> None:
    """Draw the identity line (y=x) and ±10% error band on the given axis.

        Args:
            axs: Matplotlib Axes object where to draw.
            lower_lim: Lower limit for the axes.
            upper_lim: Upper limit for the axes.

        """
    axs.plot(
        [lower_lim, upper_lim],
        [lower_lim, upper_lim],
        "--",
        linewidth=1.0,
        color=COLORS_DICT["dark_red_paper"],
        label="y = x",
        zorder=3,
    )
    upper = 1.10 * np.array([lower_lim, upper_lim])
    lower = 0.90 * np.array([lower_lim, upper_lim])
    axs.fill_between(
        [lower_lim, upper_lim],
        lower,
        upper,
        color=COLORS_DICT["light_red_paper"],
        alpha=0.5,
        label="±10% error",
        zorder=1,
    )


def _compute_limits(y_a: np.ndarray,
                    y_b: np.ndarray = None,
                    margin: float = 0.01) -> tuple[float, float]:
    """Compute axis limits based on min/max of two arrays.

        Args:
            y_a: Array of values (e.g., y_true).
            y_b: Array of values (e.g., y_pred).
            margin: Fractional margin to add on each side of the limits
              (default 0.05 for 5%).

        Returns:
            A tuple (lower_lim, upper_lim) with the computed limits.
        """
    if y_b is None:
        y_b = y_a
    lower_lim = float(min(y_a.min(), y_b.min()))
    upper_lim = float(max(y_a.max(), y_b.max()))

    lower_lim -= margin * (upper_lim - lower_lim)
    upper_lim += margin * (upper_lim - lower_lim)
    return lower_lim, upper_lim


def sec_key(sec: str) -> str:
    """Key function to sort sections by numeric suffix if present.

        Args:
            sec: Section identifier.

        Returns:
            A key for sorting.
        """
    sec_str = str(sec)

    match = re.search(r"(\d+)$", sec_str)
    return int(match.group(1)) if match else sec_str


def plot_signed_error_vs_y_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_dir: str | Path,
    save_svg: bool = False,
    groups: np.ndarray = None,
    group_col: str = None,
    group_order: List[str] = None,
    split_by_group: bool = False,
    ncols: int = 3,
    style: str = "fivethirtyeight",
    label: str = "Damage",
    color_by: np.ndarray = None,
    color_by_label: str = None,
) -> None:
    """Scatter of signed error vs true target value, optionally by group.

    Args:
        y_true: Ground-truth damage values (array-like, shape (N,)).
        y_pred: Predicted damage values (array-like, shape (N,)).
        plot_dir: Directory where the figure will be saved.
        save_svg: If True, also saves SVG.
        groups: Optional group identifier per sample (same length as y_true).
        group_col: Logical name of the group column (e.g., "wind_group").
        group_order: Optional explicit order of groups to appear in the plots.
        split_by_group: If True, creates subplots for each group; otherwise,
            everything is drawn in a single axis (coloured by group if given).
        ncols: Number of columns in the subplot grid when split_by_group=True.
        style: Matplotlib style sheet (default "fivethirtyeight").
        label: Logical name of the damage type.
        color_by: Optional continuous array to color scatter points by
            (e.g., sections_id, std_mean_wind).
        color_by_label: Label for the colorbar (e.g., "Tower Section",
            "Std Mean Wind"). Defaults to "Color Variable" if not provided.

    Raises:
        ValueError: If input arrays have inconsistent lengths or invalid
            parameters.
        ValueError: If split_by_group=True but no groups are provided.
    """

    plt.style.use(style)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    signed_err = y_pred - y_true
    mae = float(np.mean(np.abs(signed_err)))
    bias = float(np.mean(signed_err))

    # Common limits help comparisons
    _, x_max = _compute_limits(y_true)
    y_abs_max = max(abs(float(signed_err.min())), abs(float(
        signed_err.max()))) * 1.05

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Group validation / ordering -
    unique_groups: list = []
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_true):
            raise ValueError("groups must have the same length as y_true.")

        unique_groups = resolve_group_order(np.unique(groups), group_order)

    if split_by_group and (groups is None or len(unique_groups) == 0):
        raise ValueError("split_by_group=True requires valid 'groups'.")

    if group_col is None:
        group_col = "group"

    # Color-by setup
    if color_by is not None:
        if color_by_label is None:
            color_by_label = "Color Variable"
        cb_min = color_by.min()
        cb_max = color_by.max()
        norm_cb = Normalize(vmin=cb_min, vmax=cb_max)
        cmap_custom = CUSTOM_MAP_SEQ
        scalar_map = ScalarMappable(norm=norm_cb, cmap=cmap_custom)
        scalar_map.set_array([])

    # base colors
    blue = COLORS_DICT["blue_paper"]
    grey = COLORS_DICT["grey_paper"]
    red = COLORS_DICT["red_paper"]
    base_colors = [blue, grey, red]
    if group_order is not None and len(group_order) == 9:
        base_colors = [
            COLORS_DICT["blue_paper"],
            mix_colors(blue, grey),
            mix_colors(blue, red),
            mix_colors(grey, blue),
            COLORS_DICT["grey_paper"],
            mix_colors(grey, red),
            mix_colors(red, blue),
            mix_colors(red, grey),
            COLORS_DICT["red_paper"],
        ]

    # Case 1: single axis
    if not split_by_group:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6), facecolor="white")

        if groups is None or len(unique_groups) == 0:
            if color_by is not None:
                scatter = axs.scatter(
                    y_true,
                    signed_err,
                    c=color_by,
                    s=10,
                    linewidth=0,
                    alpha=0.85,
                    cmap=cmap_custom,
                    norm=norm_cb,
                )
                divider = make_axes_locatable(axs)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                cbar = fig.colorbar(scatter, cax=cax)
                cbar.set_label(color_by_label, fontsize=10, labelpad=10)
                cbar.ax.tick_params(axis='both',
                                    which='major',
                                    length=4,
                                    width=1,
                                    labelsize=10,
                                    color=COLORS_DICT["dark_gray_paper"])
                cbar.outline.set_visible(False)
                cb_slug = color_by_label.lower().replace(" ", "_")
                title = (f"Signed Error vs True {label}"
                         f" colored by {color_by_label}")
                base_name = f"signed_error_vs_y_true_colored_by_{cb_slug}"
            else:
                axs.scatter(
                    y_true,
                    signed_err,
                    s=10,
                    linewidth=0.5,
                    alpha=0.85,
                    color=COLORS_DICT["blue_paper"],
                )
                title = f"Signed Error vs True {label}"
                base_name = "signed_error_vs_y_true"
        else:
            for i, group in enumerate(unique_groups):
                mask = groups == group
                axs.scatter(
                    y_true[mask],
                    signed_err[mask],
                    s=10,
                    linewidth=0.5,
                    alpha=0.85,
                    color=base_colors[i % len(base_colors)],
                    label=str(group),
                )
            title = f"Signed Error vs True {label} by {group_col}"
            base_name = f"signed_error_vs_y_true_by_{group_col}"

            axs.legend(frameon=False, fontsize=9)

        axs.axhline(0,
                    color="black",
                    linewidth=0.8,
                    linestyle="-",
                    alpha=0.5,
                    zorder=1)
        axs.set_title(f"{title}\nMAE={mae:.1e}, Bias={bias:.1e}",
                      fontsize=12,
                      pad=10)

        axs.set_xlim(0, x_max)
        axs.set_ylim(-y_abs_max, y_abs_max)

        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))

        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

        general.style_axes(axs, f"True {label}",
                           f"Signed Error ({label} units)")

        plt.tight_layout()

        png_path = os.path.join(plot_dir, f"{base_name}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

        if save_svg:
            svg_path = os.path.join(plot_dir, f"{base_name}.svg")
            plt.savefig(svg_path,
                        bbox_inches="tight",
                        transparent=True,
                        format="svg")

        plt.close(fig)
        return

    # Case 2: panels by group
    n_groups = len(unique_groups)
    ncols = min(max(1, ncols), n_groups)
    nrows = int(math.ceil(n_groups / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 6 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes = axes.ravel()

    for i, group in enumerate(unique_groups):
        axs = axes[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        mask = groups == group

        if color_by is not None:
            axs.scatter(
                y_true[mask],
                signed_err[mask],
                c=color_by[mask],
                s=10,
                linewidth=0,
                alpha=0.85,
                cmap=cmap_custom,
                norm=norm_cb,
            )
        else:
            axs.scatter(
                y_true[mask],
                signed_err[mask],
                s=10,
                linewidth=0.5,
                alpha=0.85,
                color=base_colors[i % len(base_colors)],
            )

        axs.axhline(0,
                    color="black",
                    linewidth=0.8,
                    linestyle="-",
                    alpha=0.5,
                    zorder=1)
        axs.set_xlim(0, x_max)
        axs.set_ylim(-y_abs_max, y_abs_max)

        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))

        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

        if i < ncols:
            try:
                wave_name = group.split("_")[1]
            except IndexError:
                wave_name = str(group)
            axs.set_title(wave_name, fontsize=11)

        xlabel = f"True {label}" if is_bottom_row else ""
        ylabel = f"Signed Error ({label} units)" if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)

        if not is_bottom_row:
            axs.set_xlabel("")
        if not is_left_col:
            axs.set_ylabel("")

    # Global titles for rows/columns
    if len(unique_groups) == 9:
        fig.text(0.5, 0.97, "Wave Group", ha="center", fontsize=12)
        fig.text(0,
                 0.5,
                 "Wind Group",
                 va="center",
                 rotation="vertical",
                 fontsize=12)

    else:
        group_name = str(group_col).replace("_", " ").title()
        fig.text(0.5, 0.925, group_name, ha="center", fontsize=12)

    # Turn off empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if color_by is not None:
        cb_slug = color_by_label.lower().replace(" ", "_")
        suptitle = (f"Signed Error vs True {label} "
                    f"(by {group_col}) colored by {color_by_label}")
        base_name = (f"signed_error_vs_y_true_panels_{group_col}"
                     f"_colored_by_{cb_slug}")
    else:
        suptitle = f"Signed Error vs True {label} (by {group_col})"
        base_name = f"signed_error_vs_y_true_panels_{group_col}"

    fig.suptitle(suptitle, fontsize=12, y=0.99)
    fig.subplots_adjust(
        top=0.875 + ((nrows - 1) * 0.0375),
        bottom=0.12 / (1 + (nrows - 1) * 1.5),
        left=0.055,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    # Row titles for 9 groups case
    fig.canvas.draw()
    for i, group in enumerate(unique_groups):
        is_left_col = (i % ncols) == 0
        if is_left_col and len(unique_groups) == 9:
            wind_name = group.split("_")[0]
            pos = axes[i].get_position()
            fig.text(0.01,
                     pos.y0 + pos.height / 2,
                     wind_name,
                     va="center",
                     rotation="vertical",
                     fontsize=11)

    # Colorbar for color_by if applicable
    if color_by is not None:
        last_row_start = (nrows - 1) * ncols
        last_row_axes = [
            axes[j] for j in range(
                last_row_start, min(last_row_start + ncols, len(unique_groups)))
        ]
        pos_left = last_row_axes[0].get_position()
        single_subplot_width = pos_left.width

        cax = fig.add_axes([
            pos_left.x0, pos_left.y0 - 0.75 / fig.get_figheight(),
            single_subplot_width, 0.1 / fig.get_figheight()
        ])

        cbar = fig.colorbar(scalar_map, cax=cax, orientation="horizontal")
        cbar.ax.text(1.02,
                     0,
                     color_by_label,
                     fontsize=9,
                     va='center',
                     ha='left',
                     transform=cbar.ax.transAxes)
        cbar.ax.tick_params(axis='both',
                            which='major',
                            length=2,
                            width=0.5,
                            labelsize=9,
                            color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)

    png_path = os.path.join(plot_dir, f"{base_name}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir, f"{base_name}.svg")
        plt.savefig(svg_path,
                    dpi=300,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)


def plot_y_true_vs_y_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_dir: str | Path,
    save_svg: bool = False,
    groups: np.ndarray = None,
    group_col: str = None,
    group_order: List[str] = None,
    split_by_group: bool = False,
    ncols: int = 3,
    style: str = "fivethirtyeight",
    label: str = "Damage",
    color_by: np.ndarray = None,
    color_by_label: str = None,
) -> None:
    """Plot y_true vs y_pred scatter, optionally by groups.

    Args:
        y_true: Ground-truth damage values (array-like, shape (N,)).
        y_pred: Predicted damage values (array-like, shape (N,)).
        plot_dir: Directory where the figure will be saved.
        save_svg: If True, also saves SVG.
        groups: Optional group identifier per sample (same length as y_true).
        group_col: Logical name of the group column (e.g., "wind_group").
        group_order: Optional explicit order of groups to appear in the plots.
        split_by_group: If True, creates subplots for each group; otherwise,
            everything is drawn in a single axis (coloured by group if given).
        ncols: Number of columns in the subplot grid when split_by_group=True.
        style: Matplotlib style sheet (default "fivethirtyeight").
        label: Logical name of the damage type.
        color_by: Optional continuous array to color scatter points by
            (e.g., sections_id, std_mean_wind).
        color_by_label: Label for the colorbar (e.g., "Tower Section",
            "Std Mean Wind"). Defaults to "Color Variable" if not provided.

    Raises:
        ValueError: If input arrays have inconsistent lengths or invalid
            parameters.
        ValueError: If split_by_group=True but no groups are provided.
    """
    # Style setup
    plt.style.use(style)

    # Validations and conversions
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if groups is not None:
        groups = np.asarray(groups)
        if len(y_true) != len(y_pred) or len(y_true) != len(groups):
            raise ValueError(
                "y_true, y_pred and groups must have the same length.")
    else:
        if split_by_group:
            raise ValueError(
                "split_by_group=True requires 'groups' to be provided.")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

    r2_global = r2_score(y_true, y_pred)

    err = y_pred - y_true
    abs_err = np.abs(err)
    rel_err = abs_err / np.abs(y_true) * 100.0

    mre_global = np.mean(rel_err)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Color groups setup - unique groups and order
    if groups is not None:
        unique_groups = resolve_group_order(np.unique(groups), group_order)
    else:
        unique_groups = []

    # Color-by setup
    if color_by is not None:
        if color_by_label is None:
            color_by_label = "Color Variable"

        cb_min = color_by.min()
        cb_max = color_by.max()
        norm = Normalize(vmin=cb_min, vmax=cb_max)

        cmap_custom = CUSTOM_MAP_SEQ
        scalar_map = ScalarMappable(norm=norm, cmap=cmap_custom)
        scalar_map.set_array([])

    n_groups = len(unique_groups)

    # base colors
    blue = COLORS_DICT["blue_paper"]
    grey = COLORS_DICT["grey_paper"]
    red = COLORS_DICT["red_paper"]
    base_colors = [blue, grey, red]
    if group_order is not None and len(group_order) == 9:
        base_colors = [
            COLORS_DICT["blue_paper"],
            mix_colors(blue, grey),
            mix_colors(blue, red),
            mix_colors(grey, blue),
            COLORS_DICT["grey_paper"],
            mix_colors(grey, red),
            mix_colors(red, blue),
            mix_colors(red, grey),
            COLORS_DICT["red_paper"],
        ]

    # 1. Case 1: single plot
    if not split_by_group:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6), facecolor="white")

        # Scatter points - no groups
        if n_groups == 0 and color_by is not None:
            scatter = axs.scatter(
                y_true,
                y_pred,
                c=color_by,
                s=10,
                linewidth=0,
                alpha=0.85,
                cmap=cmap_custom,
                norm=norm,
                zorder=2,
            )

            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = fig.colorbar(scatter, cax=cax)
            cbar.set_label(color_by_label, fontsize=10, labelpad=10)
            cbar.ax.tick_params(axis='both',
                                which='major',
                                length=4,
                                width=1,
                                labelsize=10,
                                color=COLORS_DICT["dark_gray_paper"])
            cbar.outline.set_visible(False)

        elif n_groups == 0 and color_by is None:
            axs.scatter(
                y_true,
                y_pred,
                s=10,
                linewidth=0.5,
                alpha=0.85,
                color=COLORS_DICT["blue_paper"],
                zorder=2,
            )
        else:
            # Scatter points - with groups
            for i, group in enumerate(unique_groups):
                mask = groups == group
                y_true_sec = y_true[mask]
                y_pred_sec = y_pred[mask]
                axs.scatter(
                    y_true_sec,
                    y_pred_sec,
                    s=10,
                    linewidth=0.5,
                    alpha=0.85,
                    color=base_colors[i],
                    label=str(group),
                    zorder=2,
                )

        _, upper_lim = _compute_limits(y_true, y_pred)
        _draw_identity_and_band(axs, 0, upper_lim)
        axs.set_xlim(0, upper_lim)
        axs.set_ylim(0, upper_lim)
        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))

        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

        axs.set_aspect("equal", adjustable="box")
        axs.xaxis.set_major_locator(axs.yaxis.get_major_locator())

        if group_col is None:
            if color_by is not None:
                cb_slug = color_by_label.lower().replace(" ", "_")
                title = f"True vs Predicted {label} colored by {color_by_label}"
                base_name = f"y_true_vs_y_pred_colored_by_{cb_slug}"
            else:
                title = f"True vs Predicted {label}"
                base_name = "y_true_vs_y_pred"
        else:
            title = f"True vs Predicted {label} by {group_col}"
            base_name = f"y_true_vs_y_pred_by_{group_col}"

        axs.set_title(f"{title}\nR²={r2_global:.3f},  MRE={mre_global:.1f}",
                      fontsize=12,
                      pad=10)
        general.style_axes(axs, f"True {label}", f"Predicted {label}")

        axs.legend(
            frameon=False,
            framealpha=0.5,
            fontsize=9,
            title_fontsize=10,
        )

        plt.tight_layout()

        # Save figure -
        png_path = os.path.join(plot_dir, f"{base_name}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

        if save_svg:
            svg_path = os.path.join(plot_dir, f"{base_name}.svg")
            plt.savefig(svg_path,
                        bbox_inches="tight",
                        transparent=True,
                        format="svg")

        plt.close(fig)
        return

    # Case 2: multi-panel per group
    if n_groups == 0:
        raise ValueError("split_by_group=True but no valid groups were found.")

    # Adjust ncols and nrows for the number of groups
    ncols = min(max(1, ncols), n_groups)
    nrows = math.ceil(n_groups / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 6 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes = axes.ravel()

    # Global limits for all panels
    _, hi_glob = _compute_limits(y_true, y_pred)

    # Plot each group in its own panel
    for i, group in enumerate(unique_groups):

        axs = axes[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        mask = groups == group
        y_true_sec = y_true[mask]
        y_pred_sec = y_pred[mask]

        if len(y_true_sec) == 0:
            axs.axis("off")
            continue

        r2_grp = r2_score(y_true_sec, y_pred_sec)

        err_grp = y_pred_sec - y_true_sec
        abs_err_grp = np.abs(err_grp)
        rel_err_grp = abs_err_grp / np.abs(y_true_sec) * 100.0

        mre_grp = np.mean(rel_err_grp)

        if color_by is not None:
            cb_vals = color_by[mask]
            scatter = axs.scatter(
                y_true_sec,
                y_pred_sec,
                c=cb_vals,
                s=10,
                linewidth=0,
                alpha=0.85,
                cmap=cmap_custom,
                norm=norm,
                zorder=2,
            )

        else:
            axs.scatter(
                y_true_sec,
                y_pred_sec,
                s=10,
                linewidth=0.5,
                alpha=0.85,
                color=base_colors[i],
                zorder=2,
            )

        _draw_identity_and_band(axs, 0, hi_glob)
        axs.set_xlim(0, hi_glob)
        axs.set_ylim(0, hi_glob)
        axs.set_aspect("equal", adjustable="box")
        axs.xaxis.set_major_locator(axs.yaxis.get_major_locator())

        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))

        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

        if i < ncols:
            try:
                wave_name = group.split("_")[1]
            except IndexError:
                wave_name = str(group)
            axs.set_title(wave_name, fontsize=11)

        axs.text(0,
                 1,
                 f"R²={r2_grp:.3f}, MRE={mre_grp:.1f}%",
                 transform=axs.transAxes,
                 fontsize=10,
                 va="top")

        xlabel = f"True {label}" if is_bottom_row else ""
        ylabel = f"Predicted {label}" if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)

    # Global titles for rows/columns
    if len(unique_groups) == 9:
        fig.text(0.5, 0.965, "Wave Group", ha="center", fontsize=12)
        fig.text(0,
                 0.5,
                 "Wind Group",
                 va="center",
                 rotation="vertical",
                 fontsize=12)

    else:
        group_name = str(group_col).replace("_", " ").title()
        fig.text(0.5, 0.925, group_name, ha="center", fontsize=12)

    # Turn off empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Legend global
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for handle, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            uniq_handles.append(handle)
            uniq_labels.append(lab)

    if color_by is not None:
        cb_slug = color_by_label.lower().replace(" ", "_")
        title = (f"Per-group {label} Predictions "
                 f"({group_col}) colored by {color_by_label}")
        base_name = f"y_true_vs_y_pred_panels_{group_col}_colored_by_{cb_slug}"

    else:
        title = f"Per-group {label} Predictions ({group_col})"
        base_name = f"y_true_vs_y_pred_panels_{group_col}"

    fig.suptitle(
        title,
        fontsize=12,
        y=0.99,
    )
    fig.subplots_adjust(
        top=0.875 + ((nrows - 1) * 0.0375),
        bottom=0.12 / (1 + (nrows - 1) * 1.5),
        left=0.06,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    # Row titles for 9 groups case
    fig.canvas.draw()
    for i, group in enumerate(unique_groups):
        is_left_col = (i % ncols) == 0
        if is_left_col and len(unique_groups) == 9:
            wind_name = group.split("_")[0]
            pos = axes[i].get_position()
            fig.text(0.015,
                     pos.y0 + pos.height / 2,
                     wind_name,
                     va="center",
                     rotation="vertical",
                     fontsize=11)

    # Get all subplots from the last row that have data
    last_row_start = (nrows - 1) * ncols
    last_row_axes = [
        axes[j] for j in range(last_row_start,
                               min(last_row_start + ncols, len(unique_groups)))
    ]

    # Position: from left of first to right of last subplot
    # of the last row
    pos_left = last_row_axes[0].get_position()
    pos_right = last_row_axes[0].get_position()

    single_subplot_width = pos_right.x0 + pos_right.width - pos_left.x0

    pos_left1 = last_row_axes[0].get_position()
    pos_right1 = last_row_axes[-1].get_position()
    total_width = pos_right1.x0 + pos_right1.width - pos_left1.x0

    # Colorbar for color_by if applicable
    if color_by is not None:

        cax = fig.add_axes([
            pos_left.x0, pos_left.y0 - 0.75 / fig.get_figheight(),
            single_subplot_width, 0.1 / fig.get_figheight()
        ])

        cbar = fig.colorbar(scalar_map, cax=cax, orientation="horizontal")
        cbar.ax.text(1.02,
                     0,
                     color_by_label,
                     fontsize=9,
                     va='center',
                     ha='left',
                     transform=cbar.ax.transAxes)

        cbar.ax.tick_params(axis='both',
                            which='major',
                            length=2,
                            width=0.5,
                            labelsize=9,
                            color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)

    if uniq_handles:
        legend_x = pos_left1.x0 + total_width / 2  # centro da colorbar
        # below the colorbar
        legend_y = pos_left.y0 - 0.75 / fig.get_figheight()

        fig.legend(
            uniq_handles,
            uniq_labels,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(legend_x, legend_y),
        )

    # Save figure
    png_path = os.path.join(plot_dir, f"{base_name}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir, f"{base_name}.svg")
        plt.savefig(svg_path,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)


def plot_relative_error_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_dir: str | Path,
    save_svg: bool = False,
    groups: np.ndarray = None,
    group_col: str = None,
    group_order: list[str] = None,
    split_by_group: bool = False,
    ncols: int = 3,
    bins: int = 40,
    style: str = "fivethirtyeight",
    label: str = "Damage",
) -> None:
    """Plot histogram of relative errors, optionally split/colored by groups.

    Args:
        y_true: Ground-truth damage values (array-like, shape (N,)).
        y_pred: Predicted damage values (array-like, shape (N,)).
        plot_dir: Directory where the figure will be saved.
        save_svg: If True, also saves SVG version of the figure.
        groups: Optional group identifier per sample (same length as y_true).
        group_col: Logical name of the group column (e.g., "wind_group").
        group_order: Optional explicit order of groups in the plots.
        split_by_group: If True, creates one histogram per group (subplots).
            If False, a single axis is used; if groups is given, each group
            gets its own color.
        ncols: Number of columns when split_by_group=True.
        bins: Number of histogram bins.
        style: Matplotlib style sheet (default "fivethirtyeight").
        label: Logical name of the damage type.

    Raises:
        ValueError: If input arrays have inconsistent lengths or invalid
            parameters.
        ValueError: If split_by_group=True but no groups are provided.
    """
    # Style setup
    plt.style.use(style)

    # Validations and conversions
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if groups is not None:
        groups = np.asarray(groups)
        if len(y_true) != len(y_pred) or len(y_true) != len(groups):
            raise ValueError(
                "y_true, y_pred and groups must have the same length.")
    else:
        if split_by_group:
            raise ValueError(
                "split_by_group=True requires 'groups' to be provided.")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

    err = y_pred - y_true
    abs_err = np.abs(err)
    rel_err = abs_err / np.abs(y_true) * 100.0

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Color groups setup
    if groups is not None:
        unique_groups = resolve_group_order(np.unique(groups), group_order)
    else:
        unique_groups = []

    n_groups = len(unique_groups)

    # Base colors
    blue = COLORS_DICT["blue_paper"]
    grey = COLORS_DICT["grey_paper"]
    red = COLORS_DICT["red_paper"]
    base_colors = [blue, grey, red]
    if group_order is not None and len(group_order) == 9:
        base_colors = [
            COLORS_DICT["blue_paper"],
            mix_colors(blue, grey),
            mix_colors(blue, red),
            mix_colors(grey, blue),
            COLORS_DICT["grey_paper"],
            mix_colors(grey, red),
            mix_colors(red, blue),
            mix_colors(red, grey),
            COLORS_DICT["red_paper"],
        ]

    err_min = float(np.min(rel_err))
    err_max = float(np.max(rel_err))
    bin_edges = np.linspace(err_min, err_max, bins + 1)

    # Global limits for all panels
    _, x_max = _compute_limits(rel_err)

    y_max_global = 0

    if n_groups == 0:
        counts, _ = np.histogram(rel_err, bins=bin_edges)
        y_max_global = float(counts.max())
    else:
        for group in unique_groups:
            mask = groups == group
            counts, _ = np.histogram(rel_err[mask], bins=bin_edges)
            y_max_global = max(y_max_global, float(counts.max()))

    if group_col is None:
        group_col = "group"

    # Case 1 : single axis, overlaid histograms per group
    if not split_by_group:
        fig, axs = plt.subplots(figsize=(6, 6), facecolor="white")

        # Histogram - no groups
        if n_groups == 0:

            axs.hist(
                rel_err,
                bins=bin_edges,
                edgecolor=COLORS_DICT["blue_paper"],
                color=COLORS_DICT["blue_paper"],
                alpha=0.85,
            )
            title = f"Relative Error Distribution ({label})"
            base_name = "relative_error_hist"
        else:
            # Histogram - with groups
            for i, group in enumerate(unique_groups):
                mask = groups == group
                color = base_colors[i % len(base_colors)]
                axs.hist(
                    rel_err[mask],
                    bins=bin_edges,
                    histtype="stepfilled",
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.85,
                    label=str(group),
                )
            title = f"Relative Error Distribution by {group_col} ({label})"
            base_name = f"rel_error_hist_by_{group_col}"

        axs.set_title(f"{title}\nMRE={np.mean(rel_err):.1f}%",
                      fontsize=12,
                      pad=10)
        axs.set_xlim(0, x_max)
        axs.set_ylim(0, y_max_global * 1.05)

        general.style_axes(axs, "Relative Error (%)", "Count")

        if n_groups > 0:
            legend = axs.legend(
                frameon=True,
                framealpha=0.5,
                fontsize=9,
                title_fontsize=10,
            )
            legend.get_frame().set_facecolor(COLORS_DICT["light_gray_paper"])
            legend.get_frame().set_linewidth(0)

        plt.tight_layout()

        # Save figure
        png_path = os.path.join(plot_dir, f"{base_name}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

        if save_svg:
            svg_path = os.path.join(plot_dir, f"{base_name}.svg")
            plt.savefig(svg_path,
                        bbox_inches="tight",
                        transparent=True,
                        format="svg")

        plt.close(fig)
        return

    # Case 2: multi-panel per group
    if n_groups == 0:
        raise ValueError("split_by_group=True but no valid groups were found.")

    ncols = min(max(1, ncols), n_groups)
    nrows = math.ceil(n_groups / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 6 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes = axes.ravel()

    # Plot each group in its own panel
    for i, group in enumerate(unique_groups):

        axs = axes[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        mask = groups == group

        axs.hist(
            rel_err[mask],
            bins=bin_edges,
            edgecolor=base_colors[i],
            color=base_colors[i],
            alpha=0.85,
        )

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

        if i < ncols:
            try:
                wave_name = group.split("_")[1]
            except IndexError:
                wave_name = str(group)
            axs.set_title(wave_name, fontsize=11)

        axs.text(0.45,
                 1,
                 f"MRE={np.mean(rel_err[mask]):.2f}%",
                 transform=axs.transAxes,
                 fontsize=8,
                 va="top")

        xlabel = "Relative Error (%)" if is_bottom_row else ""
        ylabel = "Count" if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)
        if not is_bottom_row:
            axs.set_xlabel("")
        if not is_left_col:
            axs.set_ylabel("")
        axs.set_xlim(0, x_max)
        axs.set_ylim(0, y_max_global * 1.01)
        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))

        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

    # Global titles for rows/columns
    if len(unique_groups) == 9:
        fig.text(0.5, 0.97, "Wave Group", ha="center", fontsize=12)
        fig.text(0,
                 0.5,
                 "Wind Group",
                 va="center",
                 rotation="vertical",
                 fontsize=12)

    else:
        group_name = str(group_col).replace("_", " ").title()
        fig.text(0.5, 0.925, group_name, ha="center", fontsize=12)

    # Turn off empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Relative error distribution by {group_col}",
                 fontsize=12,
                 y=0.99)
    fig.subplots_adjust(
        top=0.875 + ((nrows - 1) * 0.0375),
        bottom=0.12 / (1 + (nrows - 1) * 1.5),
        left=0.055,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    # Row titles for 9 groups case
    fig.canvas.draw()
    for i, group in enumerate(unique_groups):
        is_left_col = (i % ncols) == 0
        if is_left_col and len(unique_groups) == 9:
            wind_name = group.split("_")[0]
            pos = axes[i].get_position()
            fig.text(0.0075,
                     pos.y0 + pos.height / 2,
                     wind_name,
                     va="center",
                     rotation="vertical",
                     fontsize=11)

    # Save figure
    png_path = os.path.join(plot_dir, f"rel_error_hist_panels_{group_col}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir,
                                f"rel_error_hist_panels_{group_col}.svg")
        plt.savefig(svg_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.close(fig)


def plot_relative_error_hist_by_section(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    section_labels: np.ndarray,
    plot_dir: str | Path,
    ncols: int = 5,
    bins: int = 40,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
    label: str = "Damage",
    color: str = None,
    title_suffix: str = "",
) -> None:
    """Plot histogram of relative errors per section in a grid of subplots.

    Each subplot shows the distribution of relative error for one section,
    together with mean relative error (MRE) in percent.

    Args:
        y_true: Array of ground-truth damage values, shape (N,).
        y_pred: Array of predicted damage values, shape (N,).
        section_labels: Section identifier for each sample, shape (N,).
        plot_dir: Directory where the figure will be saved.
        ncols: Number of subplot columns in the grid.
        bins: Number of histogram bins.
        style: Matplotlib style to apply.
        save_svg: Whether to also save an SVG version.
        label: Logical name of the damage type.
        color: Optional color for the histogram bars.
        title_suffix: Optional suffix appended to the plot title.

    Raises:
        ValueError: If input arrays have inconsistent lengths.
    """
    # Style setup
    plt.style.use(style)
    if color is None:
        color = COLORS_DICT["blue_paper"]

    # Validations and conversions
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    section_labels = np.asarray(section_labels)

    if len(y_true) != len(y_pred) or len(y_true) != len(section_labels):
        raise ValueError(
            "y_true, y_pred and section_labels must have the same length.")

    # Metrics
    err = y_pred - y_true
    abs_err = np.abs(err)
    rel_err = abs_err / np.abs(y_true) * 100.0

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "rel_err": rel_err,
        "section": section_labels,
    })

    sections = sorted(df_plot["section"].unique(), key=sec_key)

    nsec = len(sections)
    ncols = max(1, min(ncols, nsec))
    nrows = int(np.ceil(nsec / ncols))

    # Common bin edges for all sections (to compare scales)
    err_min = float(np.min(rel_err))
    err_max = float(np.max(rel_err))
    bin_edges = np.linspace(err_min, err_max, bins + 1)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes_flat = axes.ravel()

    i = -1
    for i, sec in enumerate(sections):
        axs = axes_flat[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        gdf = df_plot[df_plot["section"] == sec]
        errs = gdf["rel_err"].values
        y_true_sec = gdf["y_true"].values
        y_pred_sec = gdf["y_pred"].values

        r2_val = r2_score(y_true_sec, y_pred_sec)

        if len(errs) == 0:
            axs.axis("off")
            continue

        mre = np.mean(errs)

        axs.hist(
            errs,
            bins=bin_edges,
            edgecolor=color,
            color=color,
            alpha=0.85,
        )

        sec_clean = sec.replace("_", " ").title()
        axs.text(0,
                 1,
                 f"{sec_clean}\nR²={r2_val:.3f}, MRE={mre:.1f}%",
                 transform=axs.transAxes,
                 fontsize=10,
                 va="top")

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

        xlabel = "Relative Error (%)" if is_bottom_row else ""
        ylabel = "Count" if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)

    # Turn off empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    title = f"Per-section Relative Error Distribution ({label})"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(
        top=0.96,
        bottom=0.06,
        left=0.07,
        right=0.96,
        hspace=0.075,
        wspace=0.075,
    )

    # Save figure
    png_path = os.path.join(plot_dir, "rel_error_hist_per_section.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir, "rel_error_hist_per_section.svg")
        plt.savefig(
            svg_path,
            dpi=300,
            bbox_inches="tight",
            transparent=True,
            format="svg",
        )

    plt.close(fig)


def plot_y_true_vs_y_pred_by_section(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    section_labels: np.ndarray,
    plot_dir: str | Path,
    ncols: int = 5,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
    label: str = "Damage",
    color: str = None,
    title_suffix: str = "",
    color_by: np.ndarray = None,
    color_by_label: str = None,
) -> None:
    """Plot y_true vs y_pred per section in a grid of subplots.

    Each subplot shows true vs predicted damage for one section, together with
    the ideal y = x line, R² and mean |relative error| (%).

    Args:
        y_true: Array of ground-truth damage values, shape (N,).
        y_pred: Array of predicted damage values, shape (N,).
        section_labels: Section identifier for each sample, shape (N,).
        plot_dir: Directory where the figure will be saved.
        ncols: Number of subplot columns in the grid.
        style: Matplotlib style to apply.
        save_svg: Whether to also save an SVG version.
        label: Logical name of the damage type.
        color: Optional color for the scatter points.
        title_suffix: Optional suffix appended to the plot title.
        color_by: Optional continuous array to color scatter points by
            (e.g., wind speed). Same length as y_true.
        color_by_label: Label for the colorbar.

    Raises:
        ValueError: If input arrays have inconsistent lengths.
    """
    # Style setup
    plt.style.use(style)
    use_colormap = color_by is not None
    if not use_colormap and color is None:
        color = COLORS_DICT["blue_paper"]

    # Validations and conversions
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    section_labels = np.asarray(section_labels)

    if len(y_true) != len(y_pred) or len(y_true) != len(section_labels):
        raise ValueError(
            "y_true, y_pred and section_labels must have the same length.")

    if use_colormap:
        color_by = np.asarray(color_by, dtype=float)
        if color_by_label is None:
            color_by_label = "Color Variable"
        cb_min, cb_max = color_by.min(), color_by.max()
        norm = Normalize(vmin=cb_min, vmax=cb_max)
        cmap_custom = CUSTOM_MAP_SEQ
        scalar_map = ScalarMappable(norm=norm, cmap=cmap_custom)
        scalar_map.set_array([])

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "section": section_labels,
    })
    if use_colormap:
        df_plot["color_by"] = color_by

    sections = sorted(df_plot["section"].unique(), key=sec_key)

    nsec = len(sections)
    nrows = int(np.ceil(nsec / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes_flat = axes.ravel()

    _, hi_glob = _compute_limits(y_true, y_pred)

    # plot each section
    i = -1
    for i, sec in enumerate(sections):
        axs = axes_flat[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        gdf = df_plot[df_plot["section"] == sec]

        y_true_sec = gdf["y_true"].values
        y_pred_sec = gdf["y_pred"].values

        if len(y_true_sec) == 0:
            axs.axis("off")
            continue

        # metrics
        r2_val = r2_score(y_true_sec, y_pred_sec)

        err = y_pred_sec - y_true_sec
        abs_err = np.abs(err)
        rel_err = abs_err / np.abs(y_true_sec) * 100.0

        mre = np.mean(rel_err)

        # Scatter
        if use_colormap:
            cb_sec = gdf["color_by"].values
            axs.scatter(
                y_true_sec,
                y_pred_sec,
                s=10,
                linewidth=0.5,
                alpha=0.85,
                c=cb_sec,
                cmap=cmap_custom,
                norm=norm,
                zorder=2,
            )
        else:
            axs.scatter(
                y_true_sec,
                y_pred_sec,
                s=10,
                linewidth=0.5,
                alpha=0.85,
                color=color,
                zorder=2,
            )

        # Identity line + ±10% band
        _draw_identity_and_band(axs, 0, hi_glob)
        axs.set_xlim(0, hi_glob)
        axs.set_ylim(0, hi_glob)
        axs.set_aspect("equal", adjustable="box")
        axs.xaxis.set_major_locator(axs.yaxis.get_major_locator())

        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))

        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)
        sec_clean = sec.replace("_", " ").title()
        axs.text(0,
                 1,
                 f"{sec_clean}\nR²={r2_val:.3f}, MRE={mre:.1f}%",
                 transform=axs.transAxes,
                 fontsize=10,
                 va="top")
        xlabel = f"True {label}" if is_bottom_row else ""
        ylabel = f"Predicted {label}" if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)

    # Axis off for empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    # Labels global
    title = f"Per-section {label} Predictions"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(
        top=0.96,
        bottom=0.02,
        left=0.07,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    # Determine last-row axes for positioning colorbar / legend
    last_row_start = (nrows - 1) * ncols
    last_row_sec_axes = [
        axes_flat[k]
        for k in range(last_row_start, min(last_row_start + ncols, nsec))
    ]

    fig.canvas.draw()

    pos_left = last_row_sec_axes[0].get_position()
    pos_right = last_row_sec_axes[-1].get_position()
    single_w = pos_left.width

    uniq_handles, uniq_labels = [], []
    if use_colormap:
        cax = fig.add_axes([
            pos_left.x0,
            pos_left.y0 - 0.75 / fig.get_figheight(),
            single_w,
            0.1 / fig.get_figheight(),
        ])
        cbar = fig.colorbar(scalar_map, cax=cax, orientation="horizontal")
        cbar.ax.text(1.02,
                     0,
                     color_by_label,
                     fontsize=9,
                     va='center',
                     ha='left',
                     transform=cbar.ax.transAxes)
        cbar.ax.tick_params(axis='both',
                            which='major',
                            length=2,
                            width=0.5,
                            labelsize=9,
                            color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)
    else:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        seen = set()
        for handle, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab)
                uniq_handles.append(handle)
                uniq_labels.append(lab)

    total_w = pos_right.x0 + pos_right.width - pos_left.x0
    if uniq_handles:
        legend_x = pos_left.x0 + total_w / 2
        legend_y = pos_left.y0 - 0.75 / fig.get_figheight()
        fig.legend(
            uniq_handles,
            uniq_labels,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(legend_x, legend_y),
        )

    # Save figure
    png_path = os.path.join(plot_dir, "y_true_vs_y_pred_per_section.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir, "y_true_vs_y_pred_per_section.svg")
        plt.savefig(svg_path,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)


def _sec_number(sec) -> int:
    """Extract trailing integer from a section label."""
    match = re.search(r"(\d+)$", str(sec))
    return int(match.group(1)) if match else 0


def plot_y_true_vs_y_pred_by_section_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    section_labels: np.ndarray,
    plot_dir: str | Path,
    sections_per_group: int = 5,
    ncols: int = 3,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
    label: str = "Damage",
    color_by: np.ndarray = None,
    color_by_label: str = None,
) -> None:
    """Plot y_true vs y_pred with sections grouped into ranges.

    Creates one subplot per group of sections (e.g. 1-5, 6-10, ...),
    giving larger, more readable subplots than the 30-panel version.

    Args:
        y_true: Ground-truth values, shape (N,).
        y_pred: Predicted values, shape (N,).
        section_labels: Section identifier per sample, shape (N,).
        plot_dir: Directory where the figure will be saved.
        sections_per_group: How many consecutive sections per subplot.
        ncols: Number of subplot columns.
        style: Matplotlib style sheet.
        save_svg: Whether to also save SVG.
        label: Logical name of the quantity (e.g. "Damage", "DEL").
        color_by: Optional continuous array for colormap.
        color_by_label: Label for the colorbar.

    Raises:
        ValueError: If input arrays have inconsistent lengths.
    """
    plt.style.use(style)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    section_labels = np.asarray(section_labels)

    if len(y_true) != len(y_pred) \
            or len(y_true) != len(section_labels):
        raise ValueError("y_true, y_pred and section_labels must have the "
                         "same length.")

    use_colormap = color_by is not None
    if use_colormap:
        color_by = np.asarray(color_by, dtype=float)
        if color_by_label is None:
            color_by_label = "Color Variable"
        norm = Normalize(vmin=color_by.min(), vmax=color_by.max())
        cmap_custom = CUSTOM_MAP_SEQ
        scalar_map = ScalarMappable(norm=norm, cmap=cmap_custom)
        scalar_map.set_array([])

    # Build per-sample section numbers and determine groups
    sec_nums = np.array([_sec_number(s) for s in section_labels])
    all_sec_sorted = sorted(set(sec_nums))

    spg = sections_per_group
    group_ranges = []
    for start_idx in range(0, len(all_sec_sorted), spg):
        chunk = all_sec_sorted[start_idx:start_idx + spg]
        group_ranges.append((chunk[0], chunk[-1]))

    n_groups = len(group_ranges)
    nrows = math.ceil(n_groups / ncols)

    _, hi_glob = _compute_limits(y_true, y_pred)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes_flat = axes.ravel()

    base_color = COLORS_DICT["blue_paper"]

    gi = -1
    for gi, (lo, hi) in enumerate(group_ranges):
        axs = axes_flat[gi]
        mask = (sec_nums >= lo) & (sec_nums <= hi)

        yt = y_true[mask]
        yp = y_pred[mask]

        if len(yt) == 0:
            axs.axis("off")
            continue

        r2_val = r2_score(yt, yp)
        rel_err = np.abs(yp - yt) / np.abs(yt) * 100.0
        mre = np.mean(rel_err)

        if use_colormap:
            cb = color_by[mask]
            axs.scatter(yt,
                        yp,
                        s=12,
                        linewidth=0.5,
                        alpha=0.85,
                        c=cb,
                        cmap=cmap_custom,
                        norm=norm,
                        zorder=2)
        else:
            axs.scatter(yt,
                        yp,
                        s=12,
                        linewidth=0.5,
                        alpha=0.85,
                        color=base_color,
                        zorder=2)

        _draw_identity_and_band(axs, 0, hi_glob)
        axs.set_xlim(0, hi_glob)
        axs.set_ylim(0, hi_glob)
        axs.set_aspect("equal", adjustable="box")

        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))
        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))
        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

        grp_label = (f"Sections {lo}-{hi}" if lo != hi else f"Section {lo}")
        axs.text(0,
                 1, f"{grp_label}\nR\u00b2={r2_val:.3f}, "
                 f"MRE={mre:.1f}%",
                 transform=axs.transAxes,
                 fontsize=11,
                 va="top")

        is_left_col = (gi % ncols) == 0
        is_bottom_row = (gi // ncols) == (nrows - 1)
        xlabel = f"True {label}" if is_bottom_row else ""
        ylabel = f"Predicted {label}" if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)
        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

    for j in range(gi + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    title = f"{label} Predictions by Section Group"
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(top=0.95,
                        bottom=0.04,
                        left=0.07,
                        right=0.98,
                        hspace=0.1,
                        wspace=0.1)

    # Determine last-row axes for positioning colorbar / legend
    last_row_start = (nrows - 1) * ncols
    last_row_axes = [
        axes_flat[k]
        for k in range(last_row_start, min(last_row_start + ncols, n_groups))
    ]

    fig.canvas.draw()

    pos_left = last_row_axes[0].get_position()
    pos_right = last_row_axes[-1].get_position()
    single_w = pos_left.width

    uniq_h, uniq_l = [], []
    if use_colormap:
        cax = fig.add_axes([
            pos_left.x0,
            pos_left.y0 - 0.75 / fig.get_figheight(),
            single_w,
            0.1 / fig.get_figheight(),
        ])
        cbar = fig.colorbar(scalar_map, cax=cax, orientation="horizontal")
        cbar.ax.text(1.02,
                     0,
                     color_by_label,
                     fontsize=10,
                     va='center',
                     ha='left',
                     transform=cbar.ax.transAxes)
        cbar.ax.tick_params(axis='both',
                            which='major',
                            length=2,
                            width=0.5,
                            labelsize=9,
                            color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)
    else:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        seen = set()
        for handle, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab)
                uniq_h.append(handle)
                uniq_l.append(lab)

    total_w = pos_right.x0 + pos_right.width - pos_left.x0
    if uniq_h:
        legend_x = pos_left.x0 + total_w / 2
        legend_y = pos_left.y0 - 0.75 / fig.get_figheight()
        fig.legend(uniq_h,
                   uniq_l,
                   loc="center",
                   ncol=2,
                   frameon=False,
                   fontsize=9,
                   bbox_to_anchor=(legend_x, legend_y))

    fname = "y_true_vs_y_pred_by_section_group"
    png_path = os.path.join(plot_dir, f"{fname}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    if save_svg:
        svg_path = os.path.join(plot_dir, f"{fname}.svg")
        plt.savefig(svg_path,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)


def plot_signed_error_vs_y_true_by_section(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    section_labels: np.ndarray,
    plot_dir: str | Path,
    ncols: int = 5,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
    label: str = "Damage",
    color: str = None,
    title_suffix: str = "",
) -> None:
    """Plot signed error vs y_true per section in a grid of subplots.

    Each subplot shows (y_pred - y_true) as a function of y_true,
    together with R² and MAE per section.

    Args:
        y_true: Array of ground-truth damage values, shape (N,).
        y_pred: Array of predicted damage values, shape (N,).
        section_labels: Section identifier for each sample, shape (N,).
        plot_dir: Directory where the figure will be saved.
        ncols: Number of subplot columns in the grid.
        style: Matplotlib style to apply.
        save_svg: Whether to also save an SVG version.
        label: Logical name of the damage type.
        color: Optional color for the scatter points.
        title_suffix: Optional suffix appended to the plot title.

    Raises:
        ValueError: If input arrays have inconsistent lengths.
    """
    # Style setup
    plt.style.use(style)
    if color is None:
        color = COLORS_DICT["blue_paper"]

    # Validations and conversions
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    section_labels = np.asarray(section_labels)

    signed_err = y_pred - y_true

    # Common limits help comparisons
    _, x_max = _compute_limits(y_true)
    y_abs_max = max(abs(float(signed_err.min())), abs(float(
        signed_err.max()))) * 1.05

    if len(y_true) != len(y_pred) or len(y_true) != len(section_labels):
        raise ValueError(
            "y_true, y_pred and section_labels must have the same length.")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "section": section_labels,
    })

    sections = sorted(df_plot["section"].unique(), key=sec_key)

    nsec = len(sections)
    nrows = int(np.ceil(nsec / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes_flat = axes.ravel()

    # plot each section
    i = -1
    for i, sec in enumerate(sections):
        axs = axes_flat[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        gdf = df_plot[df_plot["section"] == sec]
        y_true_sec = gdf["y_true"].values
        y_pred_sec = gdf["y_pred"].values

        if len(y_true_sec) == 0:
            axs.axis("off")
            continue

        sec_signed_err = y_pred_sec - y_true_sec

        r2_val = r2_score(y_true_sec, y_pred_sec)
        mae = float(np.mean(np.abs(sec_signed_err)))
        sec_bias = float(np.mean(sec_signed_err))

        # Scatter: signed error vs y_true
        axs.scatter(
            y_true_sec,
            sec_signed_err,
            s=10,
            linewidth=0.5,
            alpha=0.85,
            color=color,
            zorder=2,
        )
        axs.axhline(0,
                    color="black",
                    linewidth=0.8,
                    linestyle="-",
                    alpha=0.5,
                    zorder=1)

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))

        sec_clean = sec.replace("_", " ").title()
        axs.text(0,
                 1, f"{sec_clean}\nR²={r2_val:.3f}, MAE={mae:.1e}"
                 f", Bias={sec_bias:.1e}",
                 transform=axs.transAxes,
                 fontsize=10,
                 va="top")

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

        xlabel = f"True {label}" if is_bottom_row else ""
        ylabel = f"Signed Error ({label})" if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)

        axs.set_xlim(0, x_max)
        axs.set_ylim(-y_abs_max, y_abs_max)
        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_scientific(True)
        fmt_x.set_powerlimits((-2, 2))

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))

        axs.xaxis.set_major_formatter(fmt_x)
        axs.yaxis.set_major_formatter(fmt_y)

    # Turn off unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    title = f"Per-section Signed Error vs True {label}"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=12)

    fig.subplots_adjust(
        top=0.96,
        bottom=0.02,
        left=0.07,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    png_path = os.path.join(plot_dir, "signed_error_vs_y_true_per_section.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir,
                                "signed_error_vs_y_true_per_section.svg")
        plt.savefig(
            svg_path,
            bbox_inches="tight",
            transparent=True,
            format="svg",
        )

    plt.close(fig)


def plot_cumulative_damage_by_section(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    section_labels: np.ndarray,
    plot_dir: str | Path,
    mode: str = "cumulative",
    error_mode: str = "rel",
    normalize_cumulative: bool = True,
    n_cols: int = 5,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
    damage_type: str = "damage",
) -> None:
    """Plot cumulative damage or error over time by section.

    Args:
        y_true: Array of ground-truth damage values, shape (N,).
        y_pred: Array of predicted damage values, shape (N,).
        section_labels: Section identifier for each sample, shape (N,).
        plot_dir: Directory where the figure will be saved.
        mode: One of:
            - "cumulative": plot cumulative damage per section (True vs Pred).
            - "error": plot error of cumulative damage per section.
        error_mode: When mode="error", type of error:
            - "rel": |D_pred - D_true| / |D_true| * 100 [%]
            - "abs":      |D_pred - D_true| (same units as damage)
        normalize_cumulative: If True and mode="cumulative", normalize the
            cumulative curves by the final true cumulative value.
        n_cols: Number of subplot columns.
        style: Matplotlib style to apply.
        save_svg: Whether to also save an SVG version.
        damage_type: Logical name of the damage type:
            - "damage": generic damage
            - "DEL": fatigue damage equivalent load

    Raises:
        ValueError: If input arrays have inconsistent lengths or invalid
            parameters.
    """
    # Style setup
    plt.style.use(style)

    # Validations and conversions
    if len(y_true) != len(y_pred) or len(y_true) != len(section_labels):
        raise ValueError(
            "y_true, y_pred and section_labels must have the same length.")

    if mode not in {"cumulative", "error"}:
        raise ValueError("mode must be 'cumulative' or 'error'.")

    if error_mode not in {"rel", "abs"}:
        raise ValueError("error_mode must be 'rel' or 'abs'.")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame({
        "y_true": y_true.astype(float),
        "y_pred": y_pred.astype(float),
        "section": np.asarray(section_labels),
    })

    sections = sorted(df_plot["section"].dropna().unique(), key=sec_key)
    nsections = len(sections)

    ncols = min(n_cols, nsections)
    nrows = int(math.ceil(nsections / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes = axes.ravel()

    # Text labels
    if mode == "error":
        if error_mode == "rel":
            title_suffix = "Final MRE = {err:.2f}%"
        else:
            title_suffix = "Final MAE = {err:.3e}"
    else:
        title_suffix = ""

    # Compute y-axis labels once
    if mode == "cumulative":
        if normalize_cumulative:
            y_label = f"Normalized Cumulative {damage_type.capitalize()}"
        else:
            y_label = f"Cumulative {damage_type.capitalize()}"
    else:
        if error_mode == "rel":
            y_label = "Relative Error [%]"
        else:
            y_label = "Absolute Error"

    # Compute max days from first section for consistent x-axis
    first_sec = df_plot[df_plot["section"] == sections[0]]
    max_days = len(first_sec) / 24 / 6

    # plot each section
    i = -1
    for i, sec in enumerate(sections):
        axs = axes[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        gdf = df_plot[df_plot["section"] == sec]

        y_true_sec = gdf["y_true"].values
        y_pred_sec = gdf["y_pred"].values

        if len(y_true_sec) == 0:
            axs.axis("off")
            continue

        cum_true = np.cumsum(y_true_sec)
        cum_pred = np.cumsum(y_pred_sec)

        # metrics
        r2_sec = r2_score(y_true_sec, y_pred_sec)

        if mode == "cumulative":
            if normalize_cumulative:
                final_true = cum_true[-1]
                cum_true_plot = cum_true / final_true
                cum_pred_plot = cum_pred / final_true
            else:
                cum_true_plot = cum_true
                cum_pred_plot = cum_pred

            rel_err = (np.abs(cum_pred - cum_true) / np.abs(cum_true)) * 100.0
            last_err = float(rel_err[-1])

            if damage_type == "DEL":
                cum_true_plot = np.cbrt(cum_true_plot)
                cum_pred_plot = np.cbrt(cum_pred_plot)

                rel_err = (np.abs(np.cbrt(cum_pred) - np.cbrt(cum_true)) /
                           np.abs(np.cbrt(cum_true))) * 100.0
                last_err = float(rel_err[-1])

            axs.plot(
                np.arange(len(cum_true_plot)) / 24 / 6,
                cum_true_plot,
                label=f"True {damage_type.capitalize()}",
                alpha=1.0,
                color=COLORS_DICT["grey_paper"],
                linewidth=1.5,
            )
            axs.plot(
                np.arange(len(cum_pred_plot)) / 24 / 6,
                cum_pred_plot,
                label=f"Predicted {damage_type.capitalize()}",
                alpha=1.0,
                color=COLORS_DICT["blue_paper"],
                linestyle="-",
                linewidth=1.5,
            )

            sec_clean = sec.replace("_", " ").title()
            axs.text(0,
                     1, f"{sec_clean}\nR²={r2_sec:.3f}, "
                     f"Final MRE={last_err:.2f}%",
                     transform=axs.transAxes,
                     fontsize=10,
                     va="top")

        else:
            if damage_type == "DEL":
                cum_pred = np.cbrt(cum_pred)
                cum_true = np.cbrt(cum_true)

            if error_mode == "rel":
                rel_err = (np.abs(cum_pred - cum_true) /
                           np.abs(cum_true)) * 100.0
                series_to_plot = rel_err
                last_err = float(rel_err[-1])
            else:
                abs_err = np.abs(cum_pred - cum_true)
                series_to_plot = abs_err
                last_err = float(abs_err[-1])

            axs.plot(
                np.arange(len(series_to_plot)) / 24 / 6,
                series_to_plot,
                label="Error (Cumulative)",
                alpha=1.0,
                color=COLORS_DICT["red_paper"],
                linewidth=1.5,
            )

            sec_clean = sec.replace("_", " ").title()
            axs.text(0,
                     1,
                     f"{sec_clean}\n" + title_suffix.format(err=last_err),
                     transform=axs.transAxes,
                     fontsize=10,
                     va="top")

        axs.set_xlim(0, max_days)
        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)
        xlabel = "Time (days)" if is_bottom_row else ""
        ylabel = y_label if is_left_col else ""
        general.style_axes(axs, xlabel, ylabel)

    # Axis off for empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Legend global
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for handle, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            uniq_handles.append(handle)
            uniq_labels.append(lab)
    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Global labels and layout
    if mode == "cumulative":
        sup_title = ("Normalized cumulative damage over time by section"
                     if normalize_cumulative else
                     "Cumulative damage over time by section")
        filename = (f"cumulative_damage_by_section"
                    f"{'_normalized' if normalize_cumulative else ''}")
    else:
        sup_title = {
            "rel": "Relative error of cumulative damage by section",
            "abs": "Absolute error of cumulative damage by section",
        }[error_mode]
        filename = f"cumulative_damage_error_by_section_{error_mode}"

    if damage_type == "DEL":
        filename += "_DEL"

    fig.suptitle(sup_title, fontsize=12)
    fig.subplots_adjust(
        top=0.94,
        bottom=0.02,
        left=0.07,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    # Save figure
    png_path = os.path.join(plot_dir, f"{filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir, f"{filename}.svg")
        plt.savefig(svg_path,
                    dpi=300,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)


def plot_signed_error_vs_var_by_section(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    section_labels: np.ndarray,
    x_values: np.ndarray,
    x_label: str,
    plot_dir: str | Path,
    ncols: int = 5,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
    label: str = "Damage",
    color: str = None,
    title_suffix: str = "",
) -> None:
    """Plot signed error vs a variable per section in a grid.

    Each subplot shows (y_pred - y_true) as a function of *x_values*,
    together with MAE and Bias per section.

    Args:
        y_true: Ground-truth values, shape (N,).
        y_pred: Predicted values, shape (N,).
        section_labels: Section identifier per sample, shape (N,).
        x_values: Variable for the x-axis, shape (N,).
        x_label: Human-readable label for the x-axis variable.
        plot_dir: Directory where the figure will be saved.
        ncols: Number of subplot columns in the grid.
        style: Matplotlib style to apply.
        save_svg: Whether to also save an SVG version.
        label: Logical name of the damage type.
        color: Optional color for the scatter points.
        title_suffix: Optional suffix appended to the plot title.
    """
    plt.style.use(style)
    if color is None:
        color = COLORS_DICT["blue_paper"]

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    section_labels = np.asarray(section_labels)
    x_values = np.asarray(x_values, dtype=float)

    signed_err = y_pred - y_true
    y_abs_max = (
        max(abs(float(signed_err.min())), abs(float(signed_err.max()))) * 1.05)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    x_pad = (x_max - x_min) * 0.02
    x_lo, x_hi = x_min - x_pad, x_max + x_pad

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "section": section_labels,
        "x_var": x_values,
    })

    sections = sorted(df_plot["section"].unique(), key=sec_key)
    nsec = len(sections)
    nrows = int(np.ceil(nsec / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes_flat = axes.ravel()

    i = -1
    for i, sec in enumerate(sections):
        axs = axes_flat[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        gdf = df_plot[df_plot["section"] == sec]
        if len(gdf) == 0:
            axs.axis("off")
            continue

        sec_err = gdf["y_pred"].values - gdf["y_true"].values
        mae = float(np.mean(np.abs(sec_err)))
        sec_bias = float(np.mean(sec_err))

        axs.scatter(
            gdf["x_var"].values,
            sec_err,
            s=10,
            linewidth=0.5,
            alpha=0.85,
            color=color,
            zorder=2,
        )
        axs.axhline(0,
                    color="black",
                    linewidth=0.8,
                    linestyle="-",
                    alpha=0.5,
                    zorder=1)

        sec_clean = sec.replace("_", " ").title()
        axs.text(0,
                 1, f"{sec_clean}\nMAE={mae:.1e}"
                 f", Bias={sec_bias:.1e}",
                 transform=axs.transAxes,
                 fontsize=10,
                 va="top")

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

        x_txt = x_label if is_bottom_row else ""
        ylabel = (f"Signed Error ({label} units)" if is_left_col else "")
        general.style_axes(axs, x_txt, ylabel)

        axs.set_xlim(x_lo, x_hi)
        axs.set_ylim(-y_abs_max, y_abs_max)

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))
        axs.yaxis.set_major_formatter(fmt_y)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    title = f"Per-section Signed Error vs {x_label}"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=12)

    fig.subplots_adjust(
        top=0.92,
        bottom=0.02,
        left=0.07,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    slug = x_label.lower().replace(" ", "_")
    png_path = plot_dir / f"signed_error_vs_{slug}_per_section.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = plot_dir / f"signed_error_vs_{slug}_per_section.svg"
        plt.savefig(svg_path,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)


def plot_signed_error_vs_var_by_section_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    section_labels: np.ndarray,
    x_values: np.ndarray,
    x_label: str,
    plot_dir: str | Path,
    sections_per_group: int = 5,
    ncols: int = 3,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
    label: str = "Damage",
) -> None:
    """Plot signed error vs a variable per section group.

    Each subplot aggregates sections into groups (e.g. 1-5, 6-10).

    Args:
        y_true: Ground-truth values, shape (N,).
        y_pred: Predicted values, shape (N,).
        section_labels: Section identifier per sample, shape (N,).
        x_values: Variable for the x-axis, shape (N,).
        x_label: Human-readable label for the x-axis variable.
        plot_dir: Directory where the figure will be saved.
        sections_per_group: Sections per group subplot.
        ncols: Number of subplot columns in the grid.
        style: Matplotlib style to apply.
        save_svg: Whether to also save an SVG version.
        label: Logical name of the damage type.
    """
    plt.style.use(style)
    color = COLORS_DICT["blue_paper"]

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    section_labels = np.asarray(section_labels)
    x_values = np.asarray(x_values, dtype=float)

    signed_err = y_pred - y_true
    y_abs_max = (
        max(abs(float(signed_err.min())), abs(float(signed_err.max()))) * 1.05)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    x_pad = (x_max - x_min) * 0.02
    x_lo, x_hi = x_min - x_pad, x_max + x_pad

    sec_nums = np.array([_sec_number(s) for s in section_labels])
    all_sec_sorted = sorted(set(sec_nums))

    spg = sections_per_group
    group_ranges = []
    for start_idx in range(0, len(all_sec_sorted), spg):
        chunk = all_sec_sorted[start_idx:start_idx + spg]
        group_ranges.append((chunk[0], chunk[-1]))

    n_groups = len(group_ranges)
    nrows = math.ceil(n_groups / ncols)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes_flat = axes.ravel()

    i = -1
    for i, (lo, hi) in enumerate(group_ranges):
        axs = axes_flat[i]
        is_left_col = (i % ncols) == 0
        is_bottom_row = (i // ncols) == (nrows - 1)

        mask = (sec_nums >= lo) & (sec_nums <= hi)
        yt = y_true[mask]
        yp = y_pred[mask]
        xv = x_values[mask]

        if len(yt) == 0:
            axs.axis("off")
            continue

        grp_err = yp - yt
        mae = float(np.mean(np.abs(grp_err)))
        grp_bias = float(np.mean(grp_err))

        axs.scatter(
            xv,
            grp_err,
            s=10,
            linewidth=0.5,
            alpha=0.85,
            color=color,
            zorder=2,
        )
        axs.axhline(0,
                    color="black",
                    linewidth=0.8,
                    linestyle="-",
                    alpha=0.5,
                    zorder=1)

        grp_label = (f"Sections {lo}-{hi}" if lo != hi else f"Section {lo}")
        axs.text(0,
                 1, f"{grp_label}\nMAE={mae:.1e}"
                 f", Bias={grp_bias:.1e}",
                 transform=axs.transAxes,
                 fontsize=10,
                 va="top")

        if not is_left_col:
            axs.tick_params(labelleft=False)
        if not is_bottom_row:
            axs.tick_params(labelbottom=False)

        x_txt = x_label if is_bottom_row else ""
        ylabel = (f"Signed Error ({label} units)" if is_left_col else "")
        general.style_axes(axs, x_txt, ylabel)

        axs.set_xlim(x_lo, x_hi)
        axs.set_ylim(-y_abs_max, y_abs_max)

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits((-2, 2))
        axs.yaxis.set_major_formatter(fmt_y)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"{label} Signed Error vs {x_label} by Section Group",
                 fontsize=12)

    fig.subplots_adjust(
        top=0.96,
        bottom=0.02,
        left=0.07,
        right=0.98,
        hspace=0.075,
        wspace=0.075,
    )

    slug = x_label.lower().replace(" ", "_")
    png_path = plot_dir / f"signed_error_vs_{slug}_by_section_group.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = plot_dir / f"signed_error_vs_{slug}_by_section_group.svg"
        plt.savefig(svg_path,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)


def plot_signed_error_vs_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distances: np.ndarray,
    plot_dir: str | Path,
    save_svg: bool = False,
    groups: np.ndarray = None,
    group_col: str = None,
    group_order: List[str] = None,
    edges: np.ndarray = None,
    style: str = "fivethirtyeight",
    label: str = "Damage",
    color_by: np.ndarray = None,
    color_by_label: str = None,
    distance_label: str = "Distance",
) -> None:
    """Scatter of signed error vs distance to training set.

    Args:
        y_true: Ground-truth values (N,).
        y_pred: Predicted values (N,).
        distances: Distance of each sample to training set (N,).
        plot_dir: Directory where the figure will be saved.
        save_svg: If True, also saves SVG.
        groups: Optional group identifier per sample (e.g.,
            in-train, interpolate, extrapolate).
        group_col: Logical name of the group column.
        group_order: Optional explicit order of groups.
        edges: Boundary values between groups drawn as vertical
            dashed lines.
        style: Matplotlib style sheet.
        label: Name of the target variable.
        color_by: Optional continuous array to color points by.
        color_by_label: Label for the colorbar.
        distance_label: Label for the x-axis.
    """

    plt.style.use(style)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(distances) != len(y_true):
        raise ValueError("distances must have the same length as y_true.")

    signed_err = y_pred - y_true

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Group setup
    unique_groups: list = []
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_true):
            raise ValueError("groups must have the same length as y_true.")
        unique_groups = resolve_group_order(np.unique(groups), group_order)

    if group_col is None:
        group_col = "group"

    # Base colors
    blue = COLORS_DICT["blue_paper"]
    grey = COLORS_DICT["grey_paper"]
    red = COLORS_DICT["red_paper"]
    base_colors = [blue, grey, red]
    if group_order is not None and len(group_order) == 9:
        base_colors = [
            COLORS_DICT["blue_paper"],
            mix_colors(blue, grey),
            mix_colors(blue, red),
            mix_colors(grey, blue),
            COLORS_DICT["grey_paper"],
            mix_colors(grey, red),
            mix_colors(red, blue),
            mix_colors(red, grey),
            COLORS_DICT["red_paper"],
        ]

    # Color-by setup
    if color_by is not None:
        if color_by_label is None:
            color_by_label = "Color Variable"
        cb_min = color_by.min()
        cb_max = color_by.max()
        norm_cb = Normalize(vmin=cb_min, vmax=cb_max)
        cmap_custom = CUSTOM_MAP_SEQ

    # Infer edges from group boundaries when not provided
    if edges is None and len(unique_groups) > 1:
        inferred = []
        for k in range(len(unique_groups) - 1):
            g_cur = unique_groups[k]
            g_nxt = unique_groups[k + 1]
            d_max = distances[groups == g_cur].max()
            d_min = distances[groups == g_nxt].min()
            inferred.append((d_max + d_min) / 2.0)
        edges = np.array(inferred)

    # Wider figure when colorbar is present
    fig_w = 7.2 if color_by is not None else 6
    fig, axs = plt.subplots(1, 1, figsize=(fig_w, 6), facecolor="white")

    # Edge lines and group labels (only with color_by)
    if edges is not None and color_by is not None:
        band_edges = np.concatenate(
            ([distances.min()], edges, [distances.max()]))
        for i, edge in enumerate(edges):
            color = base_colors[i % len(base_colors)]
            axs.axvline(
                float(edge),
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                zorder=1,
            )
        for i, group in enumerate(unique_groups):
            mid = (band_edges[i] + band_edges[i + 1]) / 2.0
            axs.text(
                mid,
                0.97,
                str(group),
                ha="center",
                va="top",
                fontsize=8,
                color=base_colors[i % len(base_colors)],
                transform=axs.get_xaxis_transform(),
            )

    # Scatter
    if color_by is not None:
        scatter = axs.scatter(
            distances,
            signed_err,
            c=color_by,
            s=10,
            linewidth=0,
            alpha=0.85,
            cmap=cmap_custom,
            norm=norm_cb,
            zorder=2,
        )
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(color_by_label, fontsize=10, labelpad=10)
        cbar.ax.tick_params(axis='both',
                            which='major',
                            length=4,
                            width=1,
                            labelsize=10,
                            color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)
        cb_slug = color_by_label.lower().replace(" ", "_")
        title = (f"Signed Error vs {distance_label}"
                 f" colored by {color_by_label}")
        base_name = (f"signed_error_vs_distance_{group_col}"
                     f"_colored_by_{cb_slug}")
    elif len(unique_groups) > 0:
        for i, group in enumerate(unique_groups):
            mask = groups == group
            axs.scatter(
                distances[mask],
                signed_err[mask],
                s=10,
                linewidth=0.5,
                alpha=0.85,
                color=base_colors[i % len(base_colors)],
                label=str(group),
                zorder=2,
            )
        axs.legend(frameon=False, fontsize=9)
        title = (f"Signed Error vs {distance_label}"
                 f" by {group_col}")
        base_name = f"signed_error_vs_distance_{group_col}"
    else:
        axs.scatter(
            distances,
            signed_err,
            s=10,
            linewidth=0.5,
            alpha=0.85,
            color=COLORS_DICT["blue_paper"],
            zorder=2,
        )
        title = f"Signed Error vs {distance_label}"
        base_name = "signed_error_vs_distance"

    axs.axhline(0,
                color="black",
                linewidth=0.8,
                linestyle="-",
                alpha=0.5,
                zorder=1)
    axs.set_title(title, fontsize=12, pad=10)

    y_abs_max = max(abs(float(signed_err.min())), abs(float(
        signed_err.max()))) * 1.05
    axs.set_ylim(-y_abs_max, y_abs_max)

    fmt_y = ScalarFormatter(useMathText=True)
    fmt_y.set_scientific(True)
    fmt_y.set_powerlimits((-2, 2))
    axs.yaxis.set_major_formatter(fmt_y)

    general.style_axes(
        axs,
        distance_label,
        f"Signed Error ({label} units)",
    )

    plt.tight_layout()

    png_path = os.path.join(plot_dir, f"{base_name}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir, f"{base_name}.svg")
        plt.savefig(svg_path,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)
