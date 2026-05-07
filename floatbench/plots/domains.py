# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
"""Plots for visualizing training and test distributions."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import seaborn as sns

from floatbench.colors import COLORS_DICT, CUSTOM_MAP
from . import general


def plot_train_test_subplots(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str = None,
    pairs: List[Tuple[str, str]] = None,
    group_col: Optional[Union[str, Tuple[Optional[str], Optional[str]]]] = None,
    group_names: Optional[Union[List[str], Tuple[Optional[List[str]],
                                                 Optional[List[str]]]]] = None,
    filename: str = "domain_subplots_wind_wave",
    save_svg: bool = False,
    style: str = "fivethirtyeight",
    color_map: Optional[Union[Dict[str, str],
                              Tuple[Optional[Dict[str, str]],
                                    Optional[Dict[str, str]]]]] = None,
    marker_map: Optional[Union[Dict[str, str],
                               Tuple[Optional[Dict[str, str]],
                                     Optional[Dict[str, str]]]]] = None,
    save_separate: bool = False,
    save_combined: bool = True,
    separate_suffix: Tuple[str, str] = ("left", "right"),
    axis_labels: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None,
    titles: Optional[Union[str, Tuple[Optional[str], Optional[str]],
                           List[Optional[str]]]] = None,
    df_test_without_groups: Optional[List[pd.DataFrame]] = None,
    df_test_without_groups_name: Optional[List[str]] = None,
    train_label: str = "Train",
    domain_polygons: Optional[Tuple[Optional[list], Optional[list]]] = None,
) -> None:
    """Plot two side-by-side subplots comparing Train vs Test data.

    Supports per-subplot grouping (distinct group columns, names, colors,
    markers), global or local legends, custom labels and titles, and PNG/SVG
    saving.

    Args:
        df_train: Training dataset as a pandas DataFrame.
        df_test: Testing dataset as a pandas DataFrame.
        output_dir: Folder to save plots.
        pairs: List of two (x, y) column pairs for the left and right subplots.
        group_col: Grouping column name or (left, right). None disables
            grouping.
        group_names: List or (left_list, right_list) specifying legend order.
        filename: Base name for saved figures (no extension).
        save_svg: Whether to also save the figure as .svg.
        style: Matplotlib style to apply.
        color_map: Dict or (left, right) mapping groups to colors.
        marker_map: Dict or (left, right) mapping groups to markers.
        save_separate: Whether to save cropped per-subplot images.
        save_combined: Whether to save a single combined figure.
        separate_suffix: Suffixes for separate subplot filenames.
        axis_labels: Either list of (xlabel, ylabel) or dict {col: label}.
        titles: Either one suptitle, two subplot titles, or list of both.
        df_test_without_groups: Optional list of DataFrames for ungrouped test
          points to overlay on both subplots.
        df_test_without_groups_name: Optional list of names for the ungrouped
          test points in the legend.
        train_label: Label for the training data in the legend.

    Raises:
        ValueError: If `pairs` or tuple arguments are malformed.
    """
    if pairs is None:
        pairs = [("wind_speed", "wave_hs"), ("wave_hs", "wave_tp")]
    if not isinstance(pairs, (list, tuple)) or len(pairs) != 2:
        raise ValueError("`pairs` must contain exactly two (x, y) tuples.")
    if df_test_without_groups is None:
        df_test_without_groups = [None, None]

    # style setup
    plt.style.use(style)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    def _pairify(val):
        """Ensure a value is a (left, right) tuple.

        Args:
            val: A single value or tuple.

        Returns:
            A tuple (left, right), duplicating val if scalar.
        """
        if isinstance(val, tuple):
            if len(val) != 2:
                raise ValueError("Tuples must have length 2: (left, right).")
            return val
        return (val, val)

    gcol_left, gcol_right = _pairify(group_col)
    gnames_left, gnames_right = _pairify(group_names)
    cmap_left, cmap_right = _pairify(color_map)
    mmap_left, mmap_right = _pairify(marker_map)

    def _resolve_axis_labels(x_col: str, y_col: str,
                             idx: int) -> Tuple[str, str]:
        """Resolve axis labels for each subplot.

        Args:
            x_col: X-axis column name.
            y_col: Y-axis column name.
            idx: Subplot index (0 or 1).

        Returns:
            A tuple (xlabel, ylabel) as strings.
        """
        if isinstance(axis_labels,
                      list) and idx < len(axis_labels) and axis_labels[idx]:
            return axis_labels[idx]
        if isinstance(axis_labels, dict):
            x_lbl = axis_labels.get(x_col) or x_col.replace("_", " ").title()
            y_lbl = axis_labels.get(y_col) or y_col.replace("_", " ").title()
            return x_lbl, y_lbl
        return x_col.replace("_", " ").title(), y_col.replace("_", " ").title()

    def _prep_grouping(group_col, ordered_names, c_map, m_map):
        """Prepare per-side grouping configuration.

        Args:
            group_col: Column name for grouping, or None.
            ordered_names: Desired order of groups in legend.
            c_map: Optional color map dict.
            m_map: Optional marker map dict.

        Returns:
            A dict with keys {col, groups, colors, markers} or None.
        """
        if not group_col:
            return None
        df_test[group_col] = df_test[group_col].astype(str).str.strip()
        groups = ordered_names or list(
            dict.fromkeys(df_test[group_col].tolist()))

        if c_map is None:
            palette = [
                COLORS_DICT["blue_paper"], COLORS_DICT["grey_paper"],
                COLORS_DICT["red_paper"], COLORS_DICT["light_red_paper"],
                COLORS_DICT["dark2_red_paper"]
            ]
            c_map = {
                grp: palette[i % len(palette)] for i, grp in enumerate(groups)
            }
        else:
            c_map = {
                grp: c_map.get(grp, COLORS_DICT["grey_paper"]) for grp in groups
            }

        if m_map is None:
            markers = ["o", "P", "D", "v", "^"]
            m_map = {
                grp: markers[i % len(markers)] for i, grp in enumerate(groups)
            }
        else:
            m_map = {grp: m_map.get(grp, "o") for grp in groups}

        return {
            "col": group_col,
            "groups": groups,
            "colors": c_map,
            "markers": m_map
        }

    left_cfg = _prep_grouping(gcol_left, gnames_left, cmap_left, mmap_left)
    right_cfg = _prep_grouping(gcol_right, gnames_right, cmap_right, mmap_right)

    global_handles, global_labels = [], []

    def _scatter(axs, x_col: str, y_col: str, title: Optional[str],
                 cfg: Optional[Dict], idx: int) -> Tuple[List, List]:
        """Scatter both train and test data on a given subplot.

        Args:
            axs: Matplotlib axis to draw on.
            x_col: X-axis column name.
            y_col: Y-axis column name.
            title: Title for subplot.
            cfg: Grouping configuration dict or None.
            idx: Subplot index (0 or 1).

        Returns:
            A tuple (handles, labels) for legend creation.
        """

        # Train scatter
        train_data = df_train[[x_col, y_col]].dropna()
        h_train = axs.scatter(train_data[x_col],
                              train_data[y_col],
                              color=COLORS_DICT["dark_blue_paper"],
                              marker="x",
                              s=20,
                              linewidth=0.5,
                              alpha=1.0,
                              label=train_label,
                              zorder=2,
                              rasterized=True)
        local_handles, local_labels = [h_train], [train_label]
        if train_label not in global_labels:
            global_handles.append(h_train)
            global_labels.append(train_label)

        # Test scatter (with grouping)
        if cfg is None:
            test_data = df_test[[x_col, y_col]].dropna()
            h_test = axs.scatter(test_data[x_col],
                                 test_data[y_col],
                                 color=COLORS_DICT["red_paper"],
                                 marker="P",
                                 s=10,
                                 linewidth=0.5,
                                 alpha=1.0,
                                 label="Test",
                                 zorder=3,
                                 rasterized=True)
            local_handles.append(h_test)
            local_labels.append("Test")
            if "Test" not in global_labels:
                global_handles.append(h_test)
                global_labels.append("Test")
        else:
            col = cfg["col"]
            for group in cfg["groups"]:
                mask = df_test[col].astype(str).str.strip() == str(group)
                gdf = df_test.loc[mask, [x_col, y_col]].dropna()
                if gdf.empty:
                    continue
                handle = axs.scatter(gdf[x_col],
                                     gdf[y_col],
                                     color=cfg["colors"][group],
                                     marker=cfg["markers"][group],
                                     s=10,
                                     linewidth=0.5,
                                     alpha=1.0,
                                     label=str(group),
                                     zorder=3,
                                     rasterized=True)
                local_handles.append(handle)
                local_labels.append(str(group))

                if str(group) not in global_labels:
                    global_handles.append(handle)
                    global_labels.append(str(group))

        if any(data_df is not None for data_df in df_test_without_groups):
            markers = ["o", "x"]
            sizes = [10, 20]
            colors = [
                COLORS_DICT["light_red_paper"], COLORS_DICT["light_blue_paper"]
            ]
            for i, (data_df, df_name) in enumerate(
                    zip(df_test_without_groups, df_test_without_groups_name)):
                if data_df is None:
                    continue

                h_test_without_groups = axs.scatter(data_df[x_col],
                                                    data_df[y_col],
                                                    color=colors[i],
                                                    marker=markers[i],
                                                    s=sizes[i],
                                                    linewidth=0.5,
                                                    alpha=1.0,
                                                    label=df_name or
                                                    "Test with Groups",
                                                    zorder=2,
                                                    rasterized=True)

                local_handles.append(h_test_without_groups)
                local_labels.append(df_name)
                if df_name not in global_labels:
                    global_handles.append(h_test_without_groups)
                    global_labels.append(df_name)

        # axes and styling
        xlabel, ylabel = _resolve_axis_labels(x_col, y_col, idx)
        general.style_axes(axs, xlabel, ylabel, fontsize=10)
        axs.xaxis.labelpad = 10
        axs.yaxis.labelpad = 10

        if title:
            axs.set_title(title, fontsize=12)
        return local_handles, local_labels

    def _default_title(x_col: str, y_col: str) -> str:
        """Generate a default subplot title.

        Args:
            x_col: Name of the x-axis column.
            y_col: Name of the y-axis column.

        Returns:
            Formatted subplot title as 'X Col vs Y Col'.
        """
        x_name = x_col.replace("_", " ").title()
        y_name = y_col.replace("_", " ").title()
        return f"{x_name} vs {y_name}"

    # Determine subplot titles and suptitle
    if isinstance(titles, str):
        subplot_titles, suptitle = [None, None], titles
    elif isinstance(titles, (list, tuple)) and len(titles) == 2:
        subplot_titles, suptitle = [titles[0], titles[1]], None
    else:
        subplot_titles = [_default_title(*pairs[0]), _default_title(*pairs[1])]
        suptitle = ("Input Domains (Train vs Test Groups)" if
                    (gcol_left or
                     gcol_right) else "Input Domains (Train vs Test)")

    left_loc = _scatter(axs[0], *pairs[0], subplot_titles[0], left_cfg, 0)
    right_loc = _scatter(axs[1], *pairs[1], subplot_titles[1], right_cfg, 1)

    # Draw alpha shape domain polygon if provided
    if domain_polygons is not None:
        boundary_color = COLORS_DICT["dark_blue_paper"]
        for idx, (ax, polygon) in enumerate(zip(axs, domain_polygons)):
            if not polygon:
                continue
            poly = plt.Polygon(polygon,
                               closed=True,
                               facecolor=COLORS_DICT["light_blue_paper"],
                               alpha=0.3,
                               edgecolor=boundary_color,
                               linewidth=1,
                               zorder=1,
                               label="Domain" if idx == 0 else None)
            ax.add_patch(poly)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Global legend
    if save_combined:
        order = [lbl for lbl in ("Train", "Test") if lbl in global_labels]
        order += [lbl for lbl in global_labels if lbl not in ("Train", "Test")]
        handles = [global_handles[global_labels.index(lbl)] for lbl in order]
        if domain_polygons is not None:
            domain_handle = Patch(facecolor=COLORS_DICT["light_blue_paper"],
                                  edgecolor=COLORS_DICT["dark_blue_paper"],
                                  alpha=0.3,
                                  linewidth=1,
                                  label="Train Domain")
            handles.append(domain_handle)
            order.append("Train Domain")
        fig.legend(handles,
                   order,
                   ncol=10,
                   fontsize=8,
                   frameon=False,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.02))

    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    if save_combined:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"),
                    dpi=300,
                    bbox_inches="tight",
                    transparent=True)
        if save_svg:
            fig.savefig(os.path.join(output_dir, f"{filename}.svg"),
                        dpi=300,
                        bbox_inches="tight",
                        transparent=True,
                        format="svg")
    if save_separate:
        fig.canvas.draw()
        for subplot_ax, (handles, labels), suf in zip(axs,
                                                      (left_loc, right_loc),
                                                      separate_suffix):

            other_axes = [other for other in axs if other is not subplot_ax]
            old_vis = [other.get_visible() for other in other_axes]
            for other in other_axes:
                other.set_visible(False)

            global_legs = list(fig.legends)
            for leg in global_legs:
                leg.set_visible(False)

            if domain_polygons is not None:
                domain_h = Patch(facecolor=COLORS_DICT["light_blue_paper"],
                                 edgecolor=COLORS_DICT["dark_blue_paper"],
                                 alpha=0.3,
                                 linewidth=1)
                handles = list(handles) + [domain_h]
                labels = list(labels) + ["Train Domain"]
            subplot_ax.legend(handles,
                              labels,
                              ncol=10,
                              fontsize=8,
                              frameon=False,
                              loc="lower center",
                              bbox_to_anchor=(0.5, -0.22))

            fig.savefig(os.path.join(output_dir, f"{filename}_{suf}.png"),
                        dpi=300,
                        bbox_inches="tight",
                        transparent=True)
            if save_svg:
                fig.savefig(os.path.join(output_dir, f"{filename}_{suf}.svg"),
                            dpi=300,
                            bbox_inches="tight",
                            transparent=True,
                            format="svg")

            for other, vis in zip(other_axes, old_vis):
                other.set_visible(vis)

    plt.close(fig)


def plot_distance_histogram(
    values: np.ndarray,
    space: str,
    plot_dir: str,
    edges: np.ndarray = None,
    save_svg: bool = False,
    style: str = "fivethirtyeight",
    show_median: bool = False,
    show_mean: bool = False,
    group_names: Optional[List[str]] = None,
    band_colors: Optional[List[str]] = None,
    multipliers: Optional[List[float]] = None,
    train_reference: bool = False,
    title: Optional[str] = "Distance Histogram",
) -> None:
    """
    Plot histogram of (test|train)→train distances with optional domain bands.

    Args:
        values: Array of distances (M,).
        space: Space name ('wind' or 'wave').
        plot_dir: Directory to save PNG/SVG.
        edges: Array of absolute edges values (len = n_bands - 1).
        save_svg: If True, also saves SVG.
        style: Matplotlib style (default 'fivethirtyeight').
        show_median: Draw a line and label for the median of the distances.
        show_mean: Draw a line and label for the mean of the distances.
        group_names: Optional list of band names (len = len(edges)+1). If None,
          no bands are drawn.
        band_colors: Optional list of colors for each band (len = len(edges)+1).
          If None, uses 'jet' colormap.
        multipliers: Optional multipliers relative to the median train spacing
          for edge labels.
        train_reference: If True, labels the x-axis as 'Train→train'; otherwise
          'Test→train'.
        title: Optional title for the plot.
    """

    # style setup
    plt.style.use(style)
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), facecolor="white")

    # limits
    axs.set_xlim(0, values.max())

    # default edge colors
    if band_colors is None:
        band_colors = [
            COLORS_DICT["blue_paper"], COLORS_DICT["grey_paper"],
            COLORS_DICT["red_paper"], COLORS_DICT["light_red_paper"],
            COLORS_DICT["dark2_red_paper"]
        ]

    # plot histogram first (so y-limits are defined)
    axs.hist(values, bins=40, alpha=0.8, color=COLORS_DICT["dark_gray_paper"])

    # median/mean lines
    if show_median or show_mean:

        def draw_stat_line(label: str, value: float, align: str):
            axs.axvline(value,
                        color=COLORS_DICT["dark_gray_paper"],
                        lw=1.5,
                        ls="--")
            axs.text(value,
                     0,
                     f"{label}: {value:.2f}",
                     rotation=90,
                     color=COLORS_DICT["dark_gray_paper"],
                     fontsize=8,
                     va="bottom",
                     ha=align,
                     transform=axs.get_xaxis_transform())

        if show_median:
            draw_stat_line("Median", float(np.nanmedian(values)), "left")
        if show_mean:
            draw_stat_line("Mean", float(np.nanmean(values)), "right")

    # color bands (optional)
    if edges is not None and group_names is not None:
        band_edges = np.concatenate(([0.0], edges, [values.max()]))
        for i in range(len(band_edges) - 1):
            axs.axvspan(band_edges[i],
                        band_edges[i + 1],
                        ymin=0,
                        ymax=1,
                        color=band_colors[i],
                        alpha=0.25,
                        zorder=1,
                        transform=axs.get_xaxis_transform())

    # edge lines and labels
    if edges is not None:
        for i, edge in enumerate(edges):
            color = band_colors[i]
            axs.axvline(float(edge), color=color, linestyle="--", linewidth=1.5)
            label = f"{edge:.2f}"
            if multipliers is not None and i < len(multipliers):
                label = f"{edge:.2f} ({multipliers[i]}·median train)"
            axs.text(float(edge),
                     0,
                     label,
                     rotation=90,
                     color=color,
                     fontsize=8,
                     va="bottom",
                     ha="right",
                     transform=axs.get_xaxis_transform())

    # axes and styling
    prefix = "Train" if train_reference else "Test"
    xlabel = f"Normalized {prefix}\u2192train Distance ({space})"
    general.style_axes(axs, xlabel, "Count", fontsize=10)
    axs.xaxis.labelpad = 10
    axs.yaxis.labelpad = 10
    axs.set_title(f"{title} ({space})", fontsize=12)

    # legend (optional)
    if group_names:
        legend_elements = [
            Patch(facecolor=band_colors[i],
                  edgecolor='k',
                  alpha=1.0,
                  label=group_names[i]) for i in range(len(group_names))
        ]
        legend = axs.legend(handles=legend_elements,
                            title="Distance Groups",
                            fontsize=8,
                            title_fontsize=9,
                            loc="upper right",
                            frameon=True,
                            framealpha=0.5)
        legend.get_frame().set_facecolor(COLORS_DICT["light_gray_paper"])
        legend.get_frame().set_linewidth(0)

    plt.tight_layout()

    # save figure
    prefix_lower = prefix.lower()
    filename = f"{prefix_lower}_train_dist_hist_{space}"
    general.save_figure(fig, plot_dir, filename, save_svg)


def plot_feature_correlation(
    dataframe: pd.DataFrame,
    features: List[str],
    plot_dir: str,
    filename: str = "feature_correlation_matrix",
    save_svg: bool = False,
    style: str = "fivethirtyeight",
) -> None:
    """Plot a correlation matrix using the custom paper style.

    The function computes the correlation matrix for the given features and
    generates a heatmap with consistent styling used across the paper
    (custom colormap, spine formatting, transparent background, and
    optional SVG export).

    Args:
        dataframe (pd.DataFrame): Input dataframe containing the feature
          columns.
        features (List[str]): List of feature names to compute correlations for.
        plot_dir (str): Directory where the plot will be saved.
        filename (str, optional): Base name of the saved file (without
          extension).
          Defaults to "feature_correlation_matrix".
        save_svg (bool, optional): If True, also saves the plot as an SVG file.
          Defaults to False.
        style (str, optional): Matplotlib style to apply. Defaults to
          "fivethirtyeight".
    """
    corr = dataframe[features].corr()

    # style setup
    plt.style.use(style)
    fig, axs = plt.subplots(1, 1, figsize=(7, 5), facecolor="white")

    # heatmap
    heatmap = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=CUSTOM_MAP,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor=COLORS_DICT["light_gray_paper"],
        annot_kws={"size": 4},
        ax=axs,
        xticklabels=features,
        yticklabels=features,
    )

    # axes and styling
    axs.set_title("Correlation matrix – input features", fontsize=12, pad=10)
    general.style_ticks(axs, fontsize=9)

    # Colorbar styling
    cbar = heatmap.collections[0].colorbar
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(
        color=COLORS_DICT["dark_gray_paper"],
        labelsize=7,
        width=1,
        length=3,
    )
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_color(COLORS_DICT["dark_gray_paper"])
    cbar.ax.set_facecolor("white")

    plt.tight_layout()

    # save figure
    general.save_figure(fig, plot_dir, filename, save_svg)
