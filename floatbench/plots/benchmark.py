# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-statements
# pylint: disable=invalid-name
"""Benchmark analysis plots for model comparison across contexts."""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional dependency
    adjust_text = None
import numpy as np
import pandas as pd

from floatbench.colors import (COLORS_DICT, CUSTOM_MAP_SEQ, CUSTOM_MAP_RED_SEQ)
from . import general

# Register custom colormaps so matplotlib can find them by name.
if "custom_seq" not in plt.colormaps:
    plt.colormaps.register(CUSTOM_MAP_SEQ, name="custom_seq")
if "custom_red_seq" not in plt.colormaps:
    plt.colormaps.register(CUSTOM_MAP_RED_SEQ, name="custom_red_seq")
# Reversed red_seq: used for metrics where higher = better (e.g. R²)
# so that dark red means a low/bad value.
if "custom_red_seq_r" not in plt.colormaps:
    plt.colormaps.register(CUSTOM_MAP_RED_SEQ.reversed(),
                           name="custom_red_seq_r")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

FAMILY_COLORS: Dict[str, str] = {
    "CatBoost": COLORS_DICT["light_red_paper"],
    "Ensemble": COLORS_DICT["red_paper"],
    "LightGBM": COLORS_DICT["light_blue_paper"],
    "NeuralNet": COLORS_DICT["blue_paper"],
    "TabM": COLORS_DICT["grey_paper"],
    "XGBoost": COLORS_DICT["light_brown_paper"],
    "RandomForest": COLORS_DICT["brown_paper"],
    "ExtraTrees": COLORS_DICT["middle_blue_paper"],
}

FAMILY_MARKERS: Dict[str, str] = {
    "CatBoost": "o",
    "LightGBM": "s",
    "XGBoost": "D",
    "NeuralNet": "^",
    "TabM": "v",
    "Ensemble": "*",
}

FAMILY_ORDER: List[str] = [
    "CatBoost",
    "LightGBM",
    "XGBoost",
    "NeuralNet",
    "TabM",
    "Ensemble",
]

REGIME_COLORS: Dict[str, str] = {
    "In-train": COLORS_DICT["blue_paper"],
    "Interpolate": COLORS_DICT["grey_paper"],
    "Extrapolate": COLORS_DICT["red_paper"],
}

SECTION_HIGHLIGHT: Dict[str, str] = {
    "section_1": COLORS_DICT["blue_paper"],
    "section_15": COLORS_DICT["light_brown_paper"],
    "section_30": COLORS_DICT["red_paper"],
}

_FAMILIES_SEARCH = [
    "LightGBMXT",
    "LightGBMLarge",
    "LightGBM",
    "NeuralNetTorch",
    "NeuralNetFastAI",
    "CatBoost",
    "XGBoost",
    "TabM",
    "RandomForestMSE",
    "RandomForest",
    "ExtraTreesMSE",
    "ExtraTrees",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_model_family(name: str) -> Optional[str]:
    """Return the model family from a full AutoGluon model name.

    Args:
        name: Full model name (e.g. ``CatBoost_r42_BAG_L1``).

    Returns:
        Family string or ``None`` if unrecognised.
    """
    if "WeightedEnsemble" in name:
        return "Ensemble"
    for family in ("CatBoost", "LightGBM", "XGBoost", "NeuralNet", "TabM",
                   "RandomForest", "ExtraTrees"):
        if family in name:
            return family
    return None


def shorten_model_name(name: str) -> str:
    """Shorten a full model name to variant only.

    The family is conveyed by colour/marker, so the short name keeps
    only the variant suffix and the stacking level.

    Args:
        name: Full model name.

    Returns:
        Shortened name string.
    """
    if "WeightedEnsemble" in name:
        return name.replace("WeightedEnsemble_", "")

    level = ""
    if "_BAG_L2" in name:
        level = " (L2)"
        name = name.replace("_BAG_L2", "")
    elif "_BAG_L3" in name:
        level = " (L3)"
        name = name.replace("_BAG_L3", "")
    else:
        name = name.replace("_BAG_L1", "")

    for family in _FAMILIES_SEARCH:
        if name.startswith(family):
            rest = name[len(family):]
            if rest.startswith("_"):
                rest = rest[1:]
            if not rest:
                rest = "base"
            return rest + level
    return name + level


def _add_family_legend(
    axs: plt.Axes,
    families: Optional[Sequence[str]] = None,
    fontsize: int = 8,
    **kwargs,
) -> None:
    """Add a legend with one entry per model family.

    Args:
        axs: Target axes.
        families: Families to include. Defaults to ``FAMILY_ORDER``.
        fontsize: Legend font size.
        **kwargs: Forwarded to ``axs.legend``.
    """
    if families is None:
        families = FAMILY_ORDER
    for fam in families:
        axs.plot(
            [],
            [],
            "o-",
            color=FAMILY_COLORS[fam],
            label=fam,
            linewidth=2,
        )
    axs.legend(fontsize=fontsize, **kwargs)


def _heatmap_cell_text(
    axs: plt.Axes,
    data: np.ndarray,
    fmt: str,
    cmap_name: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Annotate every cell of a heatmap with its value.

    Text colour is chosen from the relative luminance of the colormap at
    each cell's value — works uniformly for sequential and diverging
    colormaps.

    Args:
        axs: Axes containing the heatmap image.
        data: 2-D array of cell values.
        fmt: Format string (e.g. ``.3f``).
        cmap_name: Colormap name.
        vmin: Minimum value of the colormap normalisation.
        vmax: Maximum value of the colormap normalisation.
    """
    n_rows, n_cols = data.shape
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            rgba = cmap(norm(val))
            # Perceptual luminance (Rec. 601) — white text on dark cells.
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if lum < 0.5 else "black"
            axs.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )


# ---------------------------------------------------------------------------
# 1. Heatmap — wind / wave groups (6 columns)
# ---------------------------------------------------------------------------


def plot_heatmap_groups(
    all_m: pd.DataFrame,
    all_g: pd.DataFrame,
    plot_dir: str | Path,
    top_n: int = 15,
    save_svg: bool = False,
    *,
    metric_prefix: str = "r2_damage",
    sort_col: str = "r2_damage",
    sort_ascending: bool = False,
    global_col: Optional[str] = "r2_damage",
    cmap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: str = "R² damage",
    fmt: str = ".3f",
    filename: str = "heatmap_groups_sorted_by_r2_damage",
) -> None:
    """Heatmap of a regime metric per wind/wave group.

    One subplot with six columns: Wave IT / IP / EX and Wind IT / IP / EX,
    optionally prefixed by a Global column.

    Args:
        all_m: Metrics DataFrame used to pick top models.
        all_g: Groups DataFrame with per-group columns
            ``{metric_prefix}_{wind|wave}_{IT|IP|EX}``.
        plot_dir: Directory where the figure is saved.
        top_n: Number of top models to display.
        save_svg: Whether to also save an SVG.
        metric_prefix: Column prefix for the 6 regime columns in *all_g*.
        sort_col: Column in *all_m* used to pick top models.
        sort_ascending: If True, smallest *sort_col* first (for errors).
        global_col: Column in *all_m* drawn as a left Global reference
            column. ``None`` omits the column.
        cmap: Matplotlib colormap name.
        vmin, vmax: Colour-scale bounds; cells are clipped for colouring
            only (annotations show the real value).
        title: Colorbar label.
        fmt: Annotation format spec (e.g. ``".3f"``).
        filename: Output filename (no extension).
    """
    regime_suffixes = [
        "wave_IT", "wave_IP", "wave_EX", "wind_IT", "wind_IP", "wind_EX"
    ]
    cols = [f"{metric_prefix}_{s}" for s in regime_suffixes]
    regime_labels = [
        "In-train", "Interpolation", "Extrapolation", "In-train",
        "Interpolation", "Extrapolation"
    ]

    # Join on (model, preset) so the same model name under both presets
    # survives as two distinct rows instead of colliding in .loc lookups.
    key_cols = ["model", "preset"] if "preset" in all_m.columns else ["model"]
    dup_models: set = set()
    if "preset" in all_m.columns:
        pcounts = all_m.groupby("model")["preset"].nunique()
        dup_models = set(pcounts[pcounts > 1].index)

    def _row_label(row):
        short = shorten_model_name(row["model"])
        family = get_model_family(row["model"])
        if family:
            short = f"{family} {short}"
        if "preset" in row and row["model"] in dup_models:
            short = f"{short} [{row['preset']}]"
        return short

    has_global = global_col is not None and global_col in all_m.columns
    top_cols = key_cols + ([global_col] if has_global else [])
    top = (all_m.nsmallest(top_n, sort_col) if sort_ascending else
           all_m.nlargest(top_n, sort_col))[top_cols].copy()
    # Pad missing regime cols with NaN so splits without extrapolation
    # data (e.g. random splits with no EX regimes) plot blank cells
    # instead of crashing.
    g_view = all_g.reindex(columns=key_cols + cols)
    data = top.merge(g_view, on=key_cols, how="left")
    y_labels = [_row_label(r) for _, r in data.iterrows()]
    if has_global:
        data_vals = data[[global_col] + cols].values
        x_labels = ["Global"] + regime_labels
        block_vlines = [0.5, 3.5]
    else:
        data_vals = data[cols].values
        x_labels = regime_labels
        block_vlines = [2.5]
    data_vals_clip = np.clip(data_vals, vmin, vmax)

    plt.style.use("fivethirtyeight")
    fig, axs = plt.subplots(1, 1, figsize=(10, 6), facecolor="white")
    axs.set_facecolor("white")

    im = axs.imshow(
        data_vals_clip,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axs.set_yticks(range(len(y_labels)))
    axs.set_yticklabels(y_labels, fontsize=10)
    axs.set_xticks(range(len(x_labels)))
    axs.set_xticklabels(x_labels, fontsize=11)
    _heatmap_cell_text(axs, data_vals, fmt, cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=axs)
    cbar.set_label(title, fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=COLORS_DICT["dark_gray_paper"],
                        labelsize=9,
                        width=0.7,
                        length=4)

    for x in block_vlines:
        axs.axvline(x=x, color=COLORS_DICT["dark_gray_paper"], linewidth=1.5)
    axs.grid(False)
    for spine in axs.spines.values():
        spine.set_visible(False)
    axs.tick_params(axis="both",
                    which="major",
                    length=4,
                    width=1,
                    labelsize=10,
                    color=COLORS_DICT["dark_gray_paper"])

    ax_top = axs.secondary_xaxis("top")
    offset = 1 if has_global else 0
    ax_top.set_xticks([1 + offset, 4 + offset])
    ax_top.set_xticklabels(["Wave", "Wind"], fontsize=12)
    ax_top.tick_params(length=0)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    general.save_figure(fig, str(plot_dir), filename, save_svg)


# ---------------------------------------------------------------------------
# 2. Heatmap — 9 combined wind x wave groups
# ---------------------------------------------------------------------------


def plot_heatmap_9groups(
    all_m: pd.DataFrame,
    all_g: pd.DataFrame,
    plot_dir: str | Path,
    top_n: int = 15,
    save_svg: bool = False,
    *,
    metric_prefix: str = "r2_damage",
    sort_col: str = "r2_damage",
    sort_ascending: bool = False,
    global_col: Optional[str] = "r2_damage",
    cmap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: str = "R² damage",
    fmt: str = ".3f",
    filename: str = "heatmap_9groups_sorted_by_r2_damage",
) -> None:
    """Heatmap of a regime metric for 9 wind x wave groups.

    Nine columns (IT_IT ... EX_EX) with vertical separators between
    wave blocks, optionally prefixed by a Global column.

    Args:
        all_m: Metrics DataFrame used to pick top models.
        all_g: Groups DataFrame with per-regime columns
            ``{metric_prefix}_{regime}`` for each regime label.
        plot_dir: Directory where the figure is saved.
        top_n: Number of top models to display.
        save_svg: Whether to also save an SVG.
        metric_prefix: Column prefix for the 9 regime columns in *all_g*.
        sort_col: Column in *all_m* used to pick top models.
        sort_ascending: If True, smallest *sort_col* first (for errors).
        global_col: Column in *all_m* drawn as a left Global reference
            column. ``None`` omits the column.
        cmap: Matplotlib colormap name.
        vmin, vmax: Colour-scale bounds; cells are clipped for colouring
            only (annotations show the real value).
        title: Colorbar label.
        fmt: Annotation format spec (e.g. ``".3f"``).
        filename: Output filename (no extension).
    """
    regimes = [
        "IT_IT",
        "IP_IT",
        "EX_IT",
        "IT_IP",
        "IP_IP",
        "EX_IP",
        "IT_EX",
        "IP_EX",
        "EX_EX",
    ]

    # Join on (model, preset) so the same model name under both presets
    # survives as two distinct rows instead of colliding in .loc lookups.
    key_cols = ["model", "preset"] if "preset" in all_m.columns else ["model"]
    dup_models: set = set()
    if "preset" in all_m.columns:
        pcounts = all_m.groupby("model")["preset"].nunique()
        dup_models = set(pcounts[pcounts > 1].index)

    def _row_label(row):
        short = shorten_model_name(row["model"])
        family = get_model_family(row["model"])
        if family:
            short = f"{family} {short}"
        if "preset" in row and row["model"] in dup_models:
            short = f"{short} [{row['preset']}]"
        return short

    cols = [f"{metric_prefix}_{r}" for r in regimes]
    has_global = global_col is not None and global_col in all_m.columns
    top_cols = key_cols + ([global_col] if has_global else [])
    top = (all_m.nsmallest(top_n, sort_col) if sort_ascending else
           all_m.nlargest(top_n, sort_col))[top_cols].copy()
    # Pad missing regime cols with NaN so splits without extrapolation
    # data (e.g. random splits with no EX regimes) plot blank cells
    # instead of crashing.
    g_view = all_g.reindex(columns=key_cols + cols)
    data = top.merge(g_view, on=key_cols, how="left")
    y_labels = [_row_label(r) for _, r in data.iterrows()]
    wind_labels = [
        "Wind\nIn-train", "Wind\nInterpolation", "Wind\nExtrapolation"
    ] * 3
    if has_global:
        data_vals = data[[global_col] + cols].values
        x_labels = ["Global"] + wind_labels
        block_vlines = [0.5, 3.5, 6.5]
    else:
        data_vals = data[cols].values
        x_labels = wind_labels
        block_vlines = [2.5, 5.5]
    data_vals_clip = np.clip(data_vals, vmin, vmax)

    plt.style.use("fivethirtyeight")
    fig, axs = plt.subplots(1, 1, figsize=(14, 6), facecolor="white")
    axs.set_facecolor("white")

    im = axs.imshow(
        data_vals_clip,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axs.set_yticks(range(len(y_labels)))
    axs.set_yticklabels(y_labels, fontsize=10)
    axs.set_xticks(range(len(x_labels)))
    axs.set_xticklabels(x_labels, fontsize=9)
    _heatmap_cell_text(axs, data_vals, fmt, cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=axs)
    cbar.set_label(title, fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=COLORS_DICT["dark_gray_paper"],
                        labelsize=9,
                        width=0.7,
                        length=4)

    for x in block_vlines:
        axs.axvline(x=x, color=COLORS_DICT["dark_gray_paper"], linewidth=1.5)
    axs.grid(False)
    for spine in axs.spines.values():
        spine.set_visible(False)
    axs.tick_params(axis="both",
                    which="major",
                    length=4,
                    width=1,
                    labelsize=10,
                    color=COLORS_DICT["dark_gray_paper"])

    ax_top = axs.secondary_xaxis("top")
    offset = 1 if has_global else 0
    ax_top.set_xticks([1 + offset, 4 + offset, 7 + offset])
    ax_top.set_xticklabels(
        ["Wave In-train", "Wave Interpolation", "Wave Extrapolation"],
        fontsize=11,
    )
    ax_top.tick_params(length=0)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    general.save_figure(fig, str(plot_dir), filename, save_svg)


# ---------------------------------------------------------------------------
# 3. Bubble chart — efficiency frontier
# ---------------------------------------------------------------------------

_BUBBLE_Y_LABELS = {
    "rel_l2_del": "Rel L\u00b2 DEL",
    "mre_del": "MRE DEL (%)",
    "rel_l2_damage": "Rel L\u00b2 damage",
    "mre_damage": "MRE damage (%)",
}


def plot_bubble_efficiency(
    all_full: pd.DataFrame,
    plot_dir: str | Path,
    save_svg: bool = False,
    metric: str = "rel_l2_del",
    filename: str | None = None,
    ylim: tuple | None = None,
) -> None:
    """Pareto bubble chart of *metric* vs capacity and latency.

    Two side-by-side panels (CarBench Fig. 2 style): Peak memory on the
    left, mean inference time on the right. Bubble size encodes fit
    (training) time. Colour encodes model family.

    Args:
        all_full: Merged DataFrame with *metric*,
            ``Peak Memory (GB)``, ``Mean Latency (ms)``,
            ``fit_time_marginal`` and ``family`` columns.
        plot_dir: Directory where the figure is saved.
        save_svg: Whether to also save an SVG.
        metric: Numeric column to plot on the Y axis.
        filename: Output filename (without extension). Defaults to
            ``f"bubble_efficiency_{metric}"``.
    """
    df = all_full.copy()

    # Bubble area = size_scale * fit_time. 1 min (60 s) → 30 pts² (base
    # Linear area-proportional scaling, half the previous scale so the
    # bubbles don't eat the plot. Anything faster than 1 min floors to
    # the 1 min bubble; anything slower than 20 min caps at 20 min.
    # 1 min -> 40 pts², 10 min -> 400, 20 min -> 800.
    size_scale = 40.0 / 60.0
    capped_fit = df["fit_time_marginal"].fillna(0).clip(lower=60, upper=1200)
    df["bubble_size"] = size_scale * capped_fit

    y_col = metric
    y_label = _BUBBLE_Y_LABELS.get(metric, metric)
    if filename is None:
        filename = f"bubble_efficiency_{metric}"
    panel_specs = [
        ("Peak Memory (GB)", "Peak memory (GB)"),
        ("Mean Latency (ms)", "Mean inference time (ms)"),
    ]
    panel_specs = [(c, lbl) for c, lbl in panel_specs if c in df.columns]
    if not panel_specs:
        logging.warning("Bubble chart: no capacity columns found; skipping.")
        return

    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(1,
                             len(panel_specs),
                             figsize=(7 * len(panel_specs), 6),
                             sharey=True,
                             squeeze=False,
                             facecolor="white")
    ax_left = axes[0, 0]
    panel_configs = list(
        zip(axes[0], [p[0] for p in panel_specs], [p[1] for p in panel_specs]))

    # Disambiguate labels for models that appear in multiple presets.
    dup_models = set()
    if "preset" in df.columns:
        counts = df.groupby("model")["preset"].nunique()
        dup_models = set(counts[counts > 1].index)

    def _label(row):
        short = shorten_model_name(row["model"])
        if row["model"] in dup_models and "preset" in row:
            short = f"{short} [{row['preset']}]"
        return short

    for axs, x_col, x_label in panel_configs:
        axs.set_facecolor("white")
        sub = df.dropna(subset=[x_col, y_col])
        texts = []
        for fam, grp in sub.groupby("family"):
            axs.scatter(
                grp[x_col],
                grp[y_col],
                s=grp["bubble_size"],
                label=fam,
                color=FAMILY_COLORS.get(fam, "grey"),
                alpha=0.6,
                edgecolors="white",
                linewidth=0.5,
            )
            for _, row in grp.iterrows():
                radius_pts = np.sqrt(max(row["bubble_size"], 1.0) / np.pi)
                # Tiny deterministic y jitter based on the model name so
                # labels in dense clusters fan out instead of stacking
                # perfectly on top of each other.
                jitter_y = ((hash(row["model"]) % 11) - 5) * 1.2
                texts.append(
                    axs.annotate(
                        _label(row),
                        xy=(row[x_col], row[y_col]),
                        xytext=(radius_pts + 2, jitter_y),
                        textcoords="offset points",
                        fontsize=5,
                        alpha=0.9,
                        color=FAMILY_COLORS.get(fam, "grey"),
                        ha="left",
                        va="center",
                    ))
        # Y label only on the leftmost panel (axes are shared).
        axis_y_label = y_label if axs is ax_left else ""
        general.style_axes(axs,
                           f"{x_label} [Log Scale]",
                           axis_y_label,
                           fontsize=10)
        axs.set_xscale("log")
        if ylim is not None:
            axs.set_ylim(*ylim)

    # Both legends stacked on the right panel (Y is shared, so the left
    # panel is free of annotations that would otherwise be duplicated).
    ax_right = axes[0, -1]
    # Use fixed-size family handles so the marker area doesn't read as
    # another training-time signal (that lives in the second legend).
    seen_families = {}
    for fam in sorted(df["family"].dropna().unique()):
        seen_families[fam] = Line2D([0], [0],
                                    marker="o",
                                    linestyle="",
                                    color=FAMILY_COLORS.get(fam, "grey"),
                                    markersize=7,
                                    alpha=0.8,
                                    label=fam)
    family_leg = ax_right.legend(handles=list(seen_families.values()),
                                 fontsize=8,
                                 title="Family",
                                 title_fontsize=9,
                                 loc="upper right",
                                 frameon=False)
    ax_right.add_artist(family_leg)
    fit_handles = []
    for fit_val, fit_label in [(60, "1 min"), (600, "10 min"),
                               (1200, "20 min")]:
        size = size_scale * fit_val
        handle = ax_right.scatter([], [],
                                  s=size,
                                  facecolors="none",
                                  edgecolors=COLORS_DICT["dark_gray_paper"],
                                  linewidth=1,
                                  label=fit_label)
        fit_handles.append(handle)
    ax_right.legend(handles=fit_handles,
                    fontsize=7,
                    markerscale=1,
                    title="Training time",
                    title_fontsize=9,
                    loc="center right",
                    frameon=False,
                    labelspacing=1.8,
                    handleheight=2.5,
                    borderpad=0.6)

    plt.tight_layout()
    for ax in fig.get_axes():
        general.style_ticks(ax)
    general.save_figure(fig, str(plot_dir), filename, save_svg)


# ---------------------------------------------------------------------------
# 4. Scatter — global vs context (EX_EX)
# ---------------------------------------------------------------------------


def plot_scatter_global_vs_context(
    all_m: pd.DataFrame,
    all_g: pd.DataFrame,
    plot_dir: str | Path,
    save_svg: bool = False,
) -> None:
    """Scatter of MRE DEL global vs MRE DEL in the EX_EX regime.

    Each point is a model, coloured by family. Diagonal y=x marks
    "no degradation under extrapolation"; points above it lose
    accuracy when both wind and wave are simultaneously extrapolated
    (the EX_EX corner of the regime grid). Uses MRE DEL (point-wise
    relative error) so cross-regime comparison is not distorted by
    Σy² artefacts that affect Rel L²-type metrics.

    Args:
        all_m: Metrics DataFrame with ``mre_del``.
        all_g: Groups DataFrame with ``mre_del_EX_EX``.
        plot_dir: Directory where the figure is saved.
        save_svg: Whether to also save an SVG.
    """
    key = [c for c in ("model", "preset") if c in all_m.columns]
    if "mre_del_EX_EX" not in all_g.columns:
        logging.warning(
            "Skipping global vs EX_EX scatter: no mre_del_EX_EX column "
            "(split has no extrapolation regime).")
        return
    merged = all_m.merge(all_g[key + ["mre_del_EX_EX"]], on=key)
    merged["family"] = merged["model"].apply(get_model_family)
    merged = merged[merged["family"].notna()]

    # Disambiguate labels for models that appear in multiple presets.
    dup_models: set = set()
    if "preset" in merged.columns:
        pcounts = merged.groupby("model")["preset"].nunique()
        dup_models = set(pcounts[pcounts > 1].index)

    def _label(row):
        short = shorten_model_name(row["model"])
        if row["model"] in dup_models and "preset" in row:
            short = f"{short} [{row['preset']}]"
        return short

    plt.style.use("fivethirtyeight")
    fig, axs = plt.subplots(1, 1, figsize=(14, 5), facecolor="white")
    axs.set_facecolor("white")

    texts = []
    for fam, grp in merged.groupby("family"):
        axs.scatter(
            grp["mre_del_EX_EX"],
            grp["mre_del"],
            color=FAMILY_COLORS.get(fam, "grey"),
            marker="o",
            alpha=0.6,
            s=70,
            edgecolors="white",
            linewidth=0.5,
        )
        for _, row in grp.iterrows():
            texts.append(
                axs.text(
                    row["mre_del_EX_EX"],
                    row["mre_del"],
                    _label(row),
                    fontsize=5,
                    alpha=0.9,
                    color=FAMILY_COLORS.get(fam, "grey"),
                    ha="left",
                    va="center",
                ))
    x_vals = merged["mre_del_EX_EX"].dropna()
    y_vals = merged["mre_del"].dropna()
    if not x_vals.empty and not y_vals.empty:
        diag_lo = min(x_vals.min(), y_vals.min())
        diag_hi = max(x_vals.max(), y_vals.max())
        axs.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
                 color=COLORS_DICT["dark_gray_paper"],
                 linestyle="--",
                 alpha=0.3,
                 linewidth=0.8,
                 zorder=1)
    axs.set_xscale("log")
    axs.set_yscale("log")

    if adjust_text is not None and texts:
        adjust_text(
            texts,
            ax=axs,
            expand_points=(1.4, 1.4),
            expand_text=(1.2, 1.2),
            arrowprops=dict(arrowstyle="-",
                            color=COLORS_DICT["dark_gray_paper"],
                            lw=0.3,
                            alpha=0.5),
        )
    general.style_axes(
        axs,
        "MRE DEL — wind Extrapolation ∧ wave Extrapolation (%) [Log]",
        "MRE DEL global (%) [Log]",
        fontsize=10)

    family_handles = [
        Line2D([0], [0],
               marker="o",
               linestyle="",
               color=FAMILY_COLORS.get(fam, "grey"),
               markersize=7,
               alpha=0.8,
               label=fam) for fam in sorted(merged["family"].dropna().unique())
    ]
    axs.legend(handles=family_handles,
               fontsize=8,
               title="Family",
               title_fontsize=9,
               loc="upper left",
               frameon=False)

    plt.tight_layout()
    general.style_ticks(axs)
    general.save_figure(fig, str(plot_dir), "scatter_global_vs_ex_ex_mre_del",
                        save_svg)


# ---------------------------------------------------------------------------
# 5. Bar chart — family x regime
# ---------------------------------------------------------------------------


def plot_bar_family_regime(
    all_m: pd.DataFrame,
    all_g: pd.DataFrame,
    plot_dir: str | Path,
    save_svg: bool = False,
    metric: str = "mre_del",
    metric_label: str | None = None,
    filename: str | None = None,
) -> None:
    """Bar chart of *metric* by family and regime.

    1 x 2 grid: columns = (Wind, Wave). Three bars per family:
    In-train, Interpolation, Extrapolation. Families sorted by
    Extrapolation value (best first for errors, worst first for R\u00b2).

    Args:
        all_m: Metrics DataFrame.
        all_g: Groups DataFrame.
        plot_dir: Directory where the figure is saved.
        save_svg: Whether to also save an SVG.
        metric: Column key in *all_g* (without the wind/wave suffix).
            Defaults to ``"rel_l2_del"``.
        metric_label: Y-axis label. Defaults to a nicely formatted form
            of *metric*.
        filename: Output filename (without extension). Defaults to
            ``f"bar_family_regime_{metric}"``.
    """
    key = [c for c in ("model", "preset") if c in all_m.columns]
    all_merged = all_m.merge(all_g, on=key)
    all_merged["family"] = all_merged["model"].apply(get_model_family)
    all_merged = all_merged[all_merged["family"].notna()]

    bar_colors = {
        "In-train": COLORS_DICT["blue_paper"],
        "Interpolation": COLORS_DICT["grey_paper"],
        "Extrapolation": COLORS_DICT["red_paper"],
    }
    default_labels = {
        "rel_l2_del": "Rel L\u00b2 DEL",
        "mre_del": "MRE DEL (%)",
        "r2_damage": "R\u00b2 damage",
        "mae_ratio_damage": "MAE damage / MAE IT_IT",
    }
    if metric_label is None:
        metric_label = default_labels.get(metric, metric)
    if filename is None:
        filename = f"bar_family_regime_{metric}"
    # R\u00b2 rewards higher values; every other metric is an error/ratio
    # where lower = better.
    sort_ascending = metric != "r2_damage"

    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    group_configs = [("Wind", "wind"), ("Wave", "wave")]
    regime_map = {
        "In-train": "IT",
        "Interpolation": "IP",
        "Extrapolation": "EX",
    }

    # Build per-group family tables once so we can both plot and export.
    fam_dfs: Dict[str, pd.DataFrame] = {}
    for group_type, cols_prefix in group_configs:
        group_cols: Dict[str, List[str]] = {}
        for label, code in regime_map.items():
            matching = [
                c for c in all_g.columns
                if c.startswith(f"{metric}_{cols_prefix}_{code}")
            ]
            if matching:
                group_cols[label] = matching

        fam_data: Dict[str, pd.Series] = {}
        for label, mcols in group_cols.items():
            tmp_col = f"tmp_{group_type}_{label}"
            all_merged[tmp_col] = all_merged[mcols].mean(axis=1)
            fam_data[label] = all_merged.groupby("family")[tmp_col].mean()

        fam_df = pd.DataFrame(fam_data)
        # Prefer Extrapolation as the sort anchor; fall back to the
        # worst available regime when the split has no EX data.
        sort_by = next(
            (r for r in ("Extrapolation", "Interpolation", "In-train")
             if r in fam_df.columns), None)
        if sort_by is not None:
            fam_df = fam_df.sort_values(sort_by, ascending=sort_ascending)
        fam_dfs[group_type] = fam_df

    for col_idx, (group_type, _) in enumerate(group_configs):
        axs = axes[col_idx]
        axs.set_facecolor("white")
        fam_df = fam_dfs[group_type]

        x = np.arange(len(fam_df))
        width = 0.25
        for offset, label in [(-width, "In-train"), (0, "Interpolation"),
                              (width, "Extrapolation")]:
            if label not in fam_df.columns:
                continue
            axs.bar(
                x + offset,
                fam_df[label],
                width,
                label=label,
                color=bar_colors[label],
            )
        axs.set_xticks(x)
        axs.set_xticklabels(fam_df.index, fontsize=10)
        if col_idx == 0:
            axs.set_ylabel(metric_label, fontsize=12)
        axs.set_title(f"{group_type} regime", fontsize=12)
        general.style_ticks(axs)

    # Aggregate handles from all panels — splits without one of the
    # regimes may leave that handle absent from the first panel.
    seen: set = set()
    handles, labels = [], []
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen:
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)
    fig.legend(handles,
               labels,
               loc="lower center",
               ncol=3,
               fontsize=10,
               frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for ax in fig.get_axes():
        general.style_ticks(ax)
    general.save_figure(fig, str(plot_dir), filename, save_svg)

    # Export the underlying family x regime values next to the figure.
    export = pd.concat(
        dict(fam_dfs.items()),
        axis=1,
        names=["group", "regime"],
    )
    export.index.name = "family"
    csv_path = Path(plot_dir) / f"{filename}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(csv_path)


# ---------------------------------------------------------------------------
# 7. Bump chart (values version)
# ---------------------------------------------------------------------------


def plot_bump_chart_values(
    all_m: pd.DataFrame,
    all_g: pd.DataFrame,
    all_s: pd.DataFrame,
    plot_dir: str | Path,
    top_n: int = 5,
    save_svg: bool = False,
    only_global: bool = False,
) -> Dict[tuple, List[str]]:
    """True bump chart of Rel L² rank across four contexts.

    Contexts: Global Rel L² DEL (anchor), EX_EX Rel L² damage,
    Section 1 (base) Rel L² damage, Section 30 (top) Rel L² damage.
    Y axis is rank (1 = best). Lines connect the same model across
    contexts so leapfrogs are immediately visible. Uses Rel L² DEL
    globally as the anchor and Rel L² damage in the regime/section
    contexts (where DEL becomes sensitive to low-damage normalisation
    artefacts).

    Args:
        all_m: Metrics DataFrame with ``rel_l2_del`` and
            ``rel_l2_damage``.
        all_g: Groups DataFrame with ``rel_l2_damage_EX_EX``.
        all_s: Sections DataFrame with ``rel_l2_damage_section_1`` /
            ``rel_l2_damage_section_30``.
        plot_dir: Directory where the figure is saved.
        top_n: Union of top-N per context drawn (1 = best).
        save_svg: Whether to also save an SVG.

    Returns:
        Dict of ``(model, preset) -> list[str]`` with the context names
        that picked each drawn model — used downstream to build the
        ``selected_by`` column of the filtered summary CSV.
    """
    key = [c for c in ("model", "preset") if c in all_m.columns]
    global_cols = [
        c for c in ("rel_l2_del", "rel_l2_damage") if c in all_m.columns
    ]
    merged = all_m[key + global_cols].copy()

    if "rel_l2_damage_EX_EX" in all_g.columns and not all_g.empty:
        merged = merged.merge(all_g[key + ["rel_l2_damage_EX_EX"]],
                              on=key,
                              how="left")
    s_cols = key + [
        c for c in ("rel_l2_damage_section_1", "rel_l2_damage_section_30")
        if c in all_s.columns
    ]
    if len(s_cols) > len(key) and not all_s.empty:
        merged = merged.merge(all_s[s_cols], on=key, how="left")

    candidates = [
        ("rel_l2_del", "Global\nRel L² DEL"),
        ("rel_l2_damage_EX_EX", "EX_EX\nRel L² damage"),
        ("rel_l2_damage_section_1", "Section 1 (base)\nRel L² damage"),
        ("rel_l2_damage_section_30", "Section 30 (top)\nRel L² damage"),
    ]
    if only_global:
        candidates = candidates[:1]
    mre_cols = [c for c, _ in candidates if c in merged.columns]
    context_names = [n for c, n in candidates if c in merged.columns]

    # Rank per context (1 = lowest MRE = best).
    for col in mre_cols:
        merged[f"rank_{col}"] = merged[col].rank(ascending=True,
                                                 method="min").astype(int)
    rank_cols = [f"rank_{c}" for c in mre_cols]

    # Use (model, preset) as the selection key so the same model name
    # under different presets doesn't drag in non-top rows via isin().
    # Also record which context names selected each (used downstream to
    # build a `selected_by` column in the filtered summary CSV).
    key_sel = [c for c in ("model", "preset") if c in merged.columns]
    top_reasons: dict = {}
    for rc, name in zip(rank_cols, context_names):
        label = name.replace("\n", " ").strip()
        for _, r in merged.nsmallest(top_n, rc)[key_sel].iterrows():
            tup = tuple(r[k] for k in key_sel)
            top_reasons.setdefault(tup, []).append(label)
    top_keys = set(top_reasons.keys())

    mask = merged[key_sel].apply(
        lambda r: tuple(r[k] for k in key_sel) in top_keys, axis=1)
    val_df = merged[mask].copy()
    val_df["family"] = val_df["model"].apply(get_model_family)
    # Disambiguate labels for models that appear in multiple presets.
    dup_models: set = set()
    if "preset" in val_df.columns:
        _pc = merged.groupby("model")["preset"].nunique()
        dup_models = set(_pc[_pc > 1].index)

    def _short_with_preset(row):
        short = shorten_model_name(row["model"])
        if "preset" in row and row["model"] in dup_models:
            short = f"{short} [{row['preset']}]"
        return short

    val_df["short"] = val_df.apply(_short_with_preset, axis=1)
    val_df["n_top_hits"] = val_df.apply(
        lambda r: sum(1 for rc in rank_cols if r[rc] <= top_n), axis=1)

    plt.style.use("fivethirtyeight")
    fig, axs = plt.subplots(1, 1, figsize=(14, 7), facecolor="white")
    axs.set_facecolor("white")

    n_ctx = len(mre_cols)
    for _, row in val_df.iterrows():
        fam = row["family"]
        if fam is None:
            continue
        ranks = [row[f"rank_{c}"] for c in mre_cols]
        alpha = 1.0
        linewidth = 1.0
        color = FAMILY_COLORS.get(fam, "grey")
        axs.plot(range(n_ctx),
                 ranks,
                 "-o",
                 color=color,
                 alpha=alpha,
                 linewidth=linewidth,
                 markersize=7)
        for x_pos, offset, ha in [(n_ctx - 1, 8, "left"), (0, -8, "right")]:
            axs.annotate(
                row["short"],
                xy=(x_pos, ranks[0] if x_pos == 0 else ranks[-1]),
                xytext=(offset, 0),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
                color=color,
                va="center",
                ha=ha,
            )

    axs.set_xticks(range(n_ctx))
    axs.set_xticklabels(context_names, fontsize=10)
    general.style_axes(axs, "", "Rank (1 = best)", fontsize=10)
    axs.invert_yaxis()
    family_handles = [
        Line2D([0], [0],
               marker="o",
               linestyle="-",
               linewidth=1.0,
               color=FAMILY_COLORS.get(fam, "grey"),
               markersize=8,
               alpha=1.0,
               label=fam) for fam in sorted(val_df["family"].dropna().unique())
    ]
    axs.legend(handles=family_handles,
               fontsize=8,
               title="Family",
               title_fontsize=9,
               loc="upper left",
               bbox_to_anchor=(1.02, 1.0),
               frameon=False)

    plt.tight_layout()
    general.save_figure(fig, str(plot_dir), "bump_chart", save_svg)
    return top_reasons


# ---------------------------------------------------------------------------
# 8. Combined scatter — wind / wave / section
# ---------------------------------------------------------------------------


def _plot_scatter_grid(
    df_test: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    models: Sequence[str],
    plot_dir: str | Path,
    filename: str,
    mode: str,
    save_svg: bool = False,
    n_cols: int = 4,
    presets: Optional[Dict[str, str]] = None,
) -> None:
    """Shared grid-scatter renderer used by the EX_EX and sections plots.

    Args:
        df_test: Test DataFrame with ``damage`` and (depending on mode)
            ``wind_group``/``wave_group`` or ``section_name`` columns.
        predictions: Model name -> predicted damage array.
        metrics: Model name -> ``{"r2": .., "mre": ..}`` for the title.
        models: Ordered list of model names (one panel each).
        plot_dir: Output directory.
        filename: Output filename (no extension).
        mode: ``"ex_ex"`` filters to wind=EX \u2227 wave=EX points only.
            ``"sections"`` shows all points as grey background and
            highlights section_1 (blue) and section_30 (red).
        save_svg: Whether to also save an SVG.
        n_cols: Number of grid columns.
    """
    y_true = df_test["damage"].astype(float).values
    n_models = len(models)
    if n_models == 0:
        return
    n_rows = int(np.ceil(n_models / n_cols))

    if mode == "ex_ex":
        if ("wind_group" not in df_test.columns or
                "wave_group" not in df_test.columns):
            logging.warning("scatter grid ex_ex: missing group columns")
            return
        mask_sel = ((df_test["wind_group"] == "Extrapolate") &
                    (df_test["wave_group"] == "Extrapolate")).values
        if mask_sel.sum() == 0:
            logging.warning("scatter grid ex_ex: no points match EX_EX")
            return
    elif mode == "sections":
        if "section_name" not in df_test.columns:
            logging.warning("scatter grid sections: missing section_name")
            return
    else:
        raise ValueError(f"Unknown mode: {mode}")

    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             facecolor="white",
                             sharex=True,
                             sharey=True)
    axes = np.atleast_2d(axes)

    # Global limits (shared across panels) based on the full test set.
    all_preds = np.concatenate([predictions[m] for m in models])
    lo = float(min(y_true.min(), all_preds.min()))
    hi = float(max(y_true.max(), all_preds.max()))
    lims = [lo, hi]
    section_targets = {
        "section_1": REGIME_COLORS["In-train"],
        "section_30": REGIME_COLORS["Extrapolate"]
    }

    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        axs = axes[row, col]
        axs.set_facecolor("white")
        y_pred = predictions[model]
        model_metrics = metrics.get(model, {})
        # Strip disambiguation suffix (_B/_E) so the title shows the
        # original model name, then append the preset explicitly.
        real_model = (model[:-2] if model.endswith(("_B", "_E")) else model)
        fam = get_model_family(real_model)
        preset = presets.get(model) if presets else None
        model_label = f"{fam} {real_model}" if fam else real_model
        if preset:
            model_label = f"{model_label} [{preset}]"
        rel_l2_del = model_metrics.get("rel_l2_del", float("nan"))
        r2 = model_metrics.get("r2", float("nan"))
        if mode == "ex_ex":
            rel_l2_del_ex = model_metrics.get("rel_l2_del_ex_ex", float("nan"))
            title = (f"{model_label}    R\u00b2={r2:.3f}\n"
                     f"Rel L\u00b2 DEL: global={rel_l2_del:.3f}, "
                     f"EX_EX={rel_l2_del_ex:.3f}")
        else:
            rel_l2_s1 = model_metrics.get("rel_l2_del_section_1", float("nan"))
            rel_l2_s30 = model_metrics.get("rel_l2_del_section_30",
                                           float("nan"))
            title = (f"{model_label}    R\u00b2={r2:.3f}\n"
                     f"Rel L\u00b2 DEL: global={rel_l2_del:.3f}, "
                     f"Sec1={rel_l2_s1:.3f}, "
                     f"Sec30={rel_l2_s30:.3f}")

        if mode == "ex_ex":
            mask_other = ~mask_sel
            axs.scatter(y_true[mask_other],
                        y_pred[mask_other],
                        color="#d0d0d0",
                        s=2,
                        alpha=0.2,
                        rasterized=True)
            axs.scatter(y_true[mask_sel],
                        y_pred[mask_sel],
                        color=REGIME_COLORS["Extrapolate"],
                        s=6,
                        alpha=0.6,
                        rasterized=True)
        else:  # sections
            mask_other = ~df_test["section_name"].isin(
                section_targets.keys()).values
            axs.scatter(y_true[mask_other],
                        y_pred[mask_other],
                        color="#d0d0d0",
                        s=2,
                        alpha=0.2,
                        rasterized=True)
            for sec, clr in section_targets.items():
                mask = (df_test["section_name"] == sec).values
                axs.scatter(y_true[mask],
                            y_pred[mask],
                            color=clr,
                            s=4,
                            alpha=0.6,
                            rasterized=True)

        axs.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
        axs.set_xlim(lims)
        axs.set_ylim(lims)
        axs.set_aspect("equal")
        axs.set_title(title, fontsize=9)
        general.style_ticks(axs, fontsize=7)
        # Shrink the "1e-5"-style exponent labels on both axes.
        axs.xaxis.get_offset_text().set_fontsize(6)
        axs.yaxis.get_offset_text().set_fontsize(6)
        if col == 0:
            axs.set_ylabel("Predicted damage", fontsize=9)
        # Show xlabel + tick labels when this is the last row, or when
        # the panel directly below is missing (last filled cell of its
        # column under sharex=True).
        is_last_row = row == n_rows - 1
        below_missing = (idx + n_cols) >= n_models
        if is_last_row or below_missing:
            axs.set_xlabel("True damage", fontsize=9)
            axs.tick_params(axis="x", labelbottom=True)

    # Hide unused panels when n_models < n_rows * n_cols.
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    # Single legend at the bottom of the figure (applies to every panel).
    if mode == "ex_ex":
        handles = [
            Line2D([0], [0],
                   marker="o",
                   color="w",
                   markerfacecolor="#d0d0d0",
                   markersize=8,
                   label="Other"),
            Line2D([0], [0],
                   marker="o",
                   color="w",
                   markerfacecolor=REGIME_COLORS["Extrapolate"],
                   markersize=8,
                   label="EX_EX (wind=EX \u2227 wave=EX)"),
        ]
        legend_title = "Regime"
    else:
        handles = [
            Line2D([0], [0],
                   marker="o",
                   color="w",
                   markerfacecolor="#d0d0d0",
                   markersize=8,
                   label="Other"),
        ] + [
            Line2D([0], [0],
                   marker="o",
                   color="w",
                   markerfacecolor=c,
                   markersize=8,
                   label=s) for s, c in section_targets.items()
        ]
        legend_title = "Section"

    fig.legend(handles=handles,
               fontsize=9,
               title=legend_title,
               title_fontsize=10,
               loc="lower center",
               ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01),
               frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.subplots_adjust(wspace=0.08, hspace=0.18)
    general.save_figure(fig, str(plot_dir), filename, save_svg)


def plot_scatter_ex_ex_grid(
    df_test: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    models: Sequence[str],
    plot_dir: str | Path,
    save_svg: bool = False,
    n_cols: int = 4,
    presets: Optional[Dict[str, str]] = None,
) -> None:
    """Grid of scatter plots restricted to the EX_EX regime only."""
    _plot_scatter_grid(df_test,
                       predictions,
                       metrics,
                       models,
                       plot_dir,
                       filename="scatter_ex_ex_top_models",
                       mode="ex_ex",
                       save_svg=save_svg,
                       n_cols=n_cols,
                       presets=presets)


def plot_scatter_sections_grid(
        df_test: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        metrics: Dict[str, Dict[str, float]],
        models: Sequence[str],
        plot_dir: str | Path,
        save_svg: bool = False,
        top_n: int = 3,
        presets: Optional[Dict[str, str]] = None,
        n_cols: int = 4,  # pylint: disable=unused-argument
) -> None:
    """One scatter figure per ranking criterion.

    Always emits ``global``, ``sec1`` and ``sec30`` orderings. When the
    ``rel_l2_del_ex_ex`` metric is available (datasets with wind/wave
    groups) a 4th ``exex`` figure is added too. Each figure is a
    1×``top_n`` row showing the best models for that criterion, saved
    as ``scatter_sections_top_models_<key>.png``.

    The legacy ``n_cols`` argument is accepted (and ignored) so callers
    that still pass it keep working.
    """
    if "section_name" not in df_test.columns:
        logging.warning("scatter sections grid: missing section_name")
        return

    candidates = [m for m in models if m in predictions]
    if not candidates:
        return

    criteria = [
        ("global", "rel_l2_del"),
        ("sec1", "rel_l2_del_section_1"),
        ("sec30", "rel_l2_del_section_30"),
    ]
    # Add EX_EX ordering when the metric is available (datasets that
    # carry wind/wave groups). Same sec1/sec30 highlight, just sorted
    # by EX_EX performance.
    if any("rel_l2_del_ex_ex" in metrics.get(m, {}) for m in candidates):
        criteria.append(("exex", "rel_l2_del_ex_ex"))

    section_targets = {
        "section_1": REGIME_COLORS["In-train"],
        "section_30": REGIME_COLORS["Extrapolate"],
    }
    mask_other = ~df_test["section_name"].isin(section_targets.keys()).values
    y_true = df_test["damage"].astype(float).values

    # Shared limits across all 3 figures so panels are visually
    # comparable even though they live in different files.
    used = sorted({
        m for _, key in criteria for m in sorted(
            candidates,
            key=lambda mm, k=key: metrics.get(mm, {}).get(k, float("inf")),
        )[:top_n]
    })
    if not used:
        return
    all_preds = np.concatenate([predictions[m] for m in used])
    lo = float(min(y_true.min(), all_preds.min()))
    hi = float(max(y_true.max(), all_preds.max()))
    lims = [lo, hi]

    for suffix, key in criteria:
        ranked = sorted(
            candidates,
            key=lambda m, k=key: metrics.get(m, {}).get(k, float("inf")),
        )
        row_models = ranked[:top_n]
        if not row_models:
            continue

        plt.style.use("fivethirtyeight")
        fig, axes = plt.subplots(1,
                                 len(row_models),
                                 figsize=(4 * len(row_models), 4),
                                 facecolor="white",
                                 sharex=True,
                                 sharey=True)
        axes = np.atleast_1d(axes)

        for c, model in enumerate(row_models):
            axs = axes[c]
            axs.set_facecolor("white")
            y_pred = predictions[model]
            model_metrics = metrics.get(model, {})
            real_model = (model[:-2] if model.endswith(("_B", "_E")) else model)
            fam = get_model_family(real_model)
            preset = presets.get(model) if presets else None
            model_label = f"{fam} {real_model}" if fam else real_model
            if preset:
                model_label = f"{model_label} [{preset}]"
            rel_l2_del = model_metrics.get("rel_l2_del", float("nan"))
            r2 = model_metrics.get("r2", float("nan"))
            rel_l2_s1 = model_metrics.get("rel_l2_del_section_1", float("nan"))
            rel_l2_s30 = model_metrics.get("rel_l2_del_section_30",
                                           float("nan"))
            title = (f"{model_label}    R²={r2:.3f}\n"
                     f"Rel L² DEL: global={rel_l2_del:.3f}, "
                     f"Sec1={rel_l2_s1:.3f}, "
                     f"Sec30={rel_l2_s30:.3f}")

            axs.scatter(y_true[mask_other],
                        y_pred[mask_other],
                        color="#d0d0d0",
                        s=2,
                        alpha=0.2,
                        rasterized=True)
            for sec, clr in section_targets.items():
                mask = (df_test["section_name"] == sec).values
                axs.scatter(y_true[mask],
                            y_pred[mask],
                            color=clr,
                            s=4,
                            alpha=0.6,
                            rasterized=True)

            axs.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
            axs.set_xlim(lims)
            axs.set_ylim(lims)
            axs.set_aspect("equal")
            axs.set_title(title, fontsize=9)
            general.style_ticks(axs, fontsize=7)
            axs.xaxis.get_offset_text().set_fontsize(6)
            axs.yaxis.get_offset_text().set_fontsize(6)
            axs.set_xlabel("True damage", fontsize=9)
            if c == 0:
                axs.set_ylabel("Predicted damage", fontsize=9)

        handles = [
            Line2D([0], [0],
                   marker="o",
                   color="w",
                   markerfacecolor="#d0d0d0",
                   markersize=8,
                   label="Other"),
        ] + [
            Line2D([0], [0],
                   marker="o",
                   color="w",
                   markerfacecolor=c,
                   markersize=8,
                   label=s) for s, c in section_targets.items()
        ]
        fig.legend(handles=handles,
                   fontsize=9,
                   title="Section",
                   title_fontsize=10,
                   loc="lower center",
                   ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.02),
                   frameon=False)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.subplots_adjust(wspace=0.08)
        general.save_figure(fig, str(plot_dir),
                            f"scatter_sections_top_models_{suffix}", save_svg)


# ---------------------------------------------------------------------------
# 9. Tower profiles comparison
# ---------------------------------------------------------------------------


def plot_tower_profiles_comparison(
    models: Sequence[str],
    model_base: "str | Path | Sequence[str | Path] | Dict[str, str | Path]",
    plot_dir: str | Path,
    top_n: int = 3,
    save_svg: bool = False,
    presets: Optional[Dict[str, str]] = None,
) -> None:
    """Tower damage profiles vs. reference for multiple models.

    Single panel: predicted damage profiles overlaid on the reference.
    Reference drawn in black; each model coloured by family.

    Args:
        models: List of model names to include. Names ending in ``_B``
            or ``_E`` are disambiguation suffixes for models present in
            multiple presets — they are stripped before path lookup.
        model_base: Either a single base directory, a list of base
            directories, or a ``{preset: dir}`` mapping. When a mapping
            is given together with *presets*, lookups are scoped to the
            matching preset directory.
        plot_dir: Directory where the figure is saved.
        top_n: Maximum number of models to display (lowest MRE).
        save_svg: Whether to also save an SVG.
        presets: Optional ``{model: preset}`` mapping used to scope
            lookups when *model_base* is a ``{preset: dir}`` dict.
    """
    if isinstance(model_base, dict):
        bases_by_preset: Dict[str, str] = {
            k: str(v) for k, v in model_base.items()
        }
        all_dirs = list(bases_by_preset.values())
    elif isinstance(model_base, (str, Path)):
        bases_by_preset = {}
        all_dirs = [str(model_base)]
    else:
        bases_by_preset = {}
        all_dirs = [str(b) for b in model_base]

    profiles: List[dict] = []

    for model in models:
        real_model = (model[:-2] if model.endswith(("_B", "_E")) else model)
        preset = presets.get(model) if presets else None
        if bases_by_preset and preset and preset in bases_by_preset:
            search_dirs = [bases_by_preset[preset]]
        else:
            search_dirs = all_dirs
        csv_path = None
        for base in search_dirs:
            candidate = os.path.join(base, real_model, "test",
                                     "tower_damage_profiles_vs_reference",
                                     "damage_profile_comparison.csv")
            if os.path.exists(candidate):
                csv_path = candidate
                break
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        mean_mre = df["mre"].mean()
        profiles.append({
            "model": model,
            "real_model": real_model,
            "preset": preset,
            "df": df,
            "mean_mre": mean_mre,
        })

    # Sort by MRE and take top_n
    profiles.sort(key=lambda p: p["mean_mre"])
    profiles = profiles[:top_n]

    if not profiles:
        return

    plt.style.use("fivethirtyeight")
    fig, ax_profile = plt.subplots(figsize=(9, 8), facecolor="white")
    ax_profile.set_facecolor("white")

    # Colour by family (consistent with the other plots). Inside a
    # family, cycle through line styles to keep models distinguishable.
    linestyle_cycle = ["-", "--", ":", "-."]
    fam_counters: Dict[str, int] = {}
    line_info = []  # (color, linestyle) per profile, for the legend
    ref_plotted = False
    for entry in profiles:
        df = entry["df"]
        model = entry["real_model"]
        height = df["height"].values
        fam = get_model_family(model)
        color = FAMILY_COLORS.get(fam, COLORS_DICT["dark_gray_paper"])
        idx_in_fam = fam_counters.get(fam, 0)
        fam_counters[fam] = idx_in_fam + 1
        linestyle = linestyle_cycle[idx_in_fam % len(linestyle_cycle)]
        line_info.append((color, linestyle))

        if not ref_plotted:
            ax_profile.plot(
                df["reference_damage"].values,
                height,
                color="black",
                linewidth=3,
                alpha=0.9,
                label="Reference",
                zorder=0,
            )
            ref_plotted = True

        ax_profile.plot(
            df["predicted_damage"].values,
            height,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.9,
        )

    general.style_axes(ax_profile,
                       "Weighted Fatigue Damage",
                       "Tower Height [m]",
                       fontsize=10)
    ax_profile.set_ylim(bottom=0)

    # Single legend on the right panel, external + frameless (same
    # style as the bump charts).
    def _legend_label(entry, color, ls):  # pylint: disable=unused-argument
        fam = get_model_family(entry["real_model"]) or ""
        preset = entry.get("preset")
        suffix = f" [{preset}]" if preset else ""
        return (f"{fam} {entry['real_model']}{suffix} "
                f"(MRE={entry['mean_mre']:.2f}%)")

    legend_handles = [
        Line2D([0], [0],
               linewidth=1.5,
               color=color,
               linestyle=ls,
               alpha=0.9,
               label=_legend_label(p, color, ls))
        for p, (color, ls) in zip(profiles, line_info)
    ]
    legend_handles.insert(
        0,
        Line2D([0], [0],
               linewidth=3,
               color="black",
               alpha=0.9,
               label="Reference"))
    ax_profile.legend(handles=legend_handles,
                      fontsize=8,
                      title="Model",
                      title_fontsize=9,
                      loc="upper left",
                      bbox_to_anchor=(1.02, 1.0),
                      frameon=False)

    plt.tight_layout()
    for ax in fig.get_axes():
        general.style_ticks(ax)
    general.save_figure(fig, str(plot_dir), "tower_profiles_comparison",
                        save_svg)


# ---------------------------------------------------------------------------
# 13. Family distribution — violin/box of R2 by family
# ---------------------------------------------------------------------------


def plot_family_distribution(
    all_m: pd.DataFrame,
    plot_dir: str | Path,
    save_svg: bool = False,
) -> None:
    """Violin plot of Rel L² DEL distribution by model family.

    Individual model points are overlaid as jittered scatter and the
    median line is highlighted in each violin.

    Args:
        all_m: Metrics DataFrame with ``rel_l2_del`` column.
        plot_dir: Directory where the figure is saved.
        save_svg: Whether to also save an SVG.
    """
    df = all_m[["model", "rel_l2_del"]].copy()
    df["family"] = df["model"].apply(get_model_family)
    df = df[df["family"].notna()]

    # Families in alphabetical order for legend/axis consistency.
    fam_order = sorted(df["family"].dropna().unique())

    # Build per-family data lists
    fam_data = [
        df[df["family"] == fam]["rel_l2_del"].dropna().values
        for fam in fam_order
    ]

    plt.style.use("fivethirtyeight")
    fig, axs = plt.subplots(1, 1, figsize=(10, 6), facecolor="white")
    axs.set_facecolor("white")

    parts = axs.violinplot(fam_data,
                           positions=range(len(fam_order)),
                           showmedians=False,
                           showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        fam = fam_order[i]
        color = FAMILY_COLORS.get(fam, "grey")
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)

    # Boxplot overlay (thin, no fill)
    bp = axs.boxplot(fam_data,
                     positions=range(len(fam_order)),
                     widths=0.15,
                     patch_artist=True,
                     showfliers=False,
                     zorder=3)
    for i, box in enumerate(bp["boxes"]):
        fam = fam_order[i]
        color = FAMILY_COLORS.get(fam, "grey")
        box.set_facecolor("white")
        box.set_edgecolor(color)
        box.set_linewidth(1.2)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(COLORS_DICT["dark_gray_paper"])
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color(COLORS_DICT["dark_gray_paper"])
        line.set_linewidth(2)

    # Jittered scatter
    rng = np.random.default_rng(42)
    for i, fam in enumerate(fam_order):
        vals = fam_data[i]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        color = FAMILY_COLORS.get(fam, "grey")
        axs.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            color=color,
            s=20,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.4,
            zorder=4,
        )

    axs.set_xticks(range(len(fam_order)))
    axs.set_xticklabels(fam_order, fontsize=11)
    axs.set_ylabel("Rel L² DEL", fontsize=12)
    axs.set_xlabel("Model Family", fontsize=12)

    general.style_ticks(axs)
    plt.tight_layout()
    general.save_figure(fig, str(plot_dir), "family_distribution_rel_l2_del",
                        save_svg)


# ---------------------------------------------------------------------------
# 14. Cross-geometry comparison — same models across geometries
# ---------------------------------------------------------------------------


def plot_cross_geometry_comparison(
    ranking_dirs: Dict[str, str],
    plot_dir: str | Path,
    save_svg: bool = False,
) -> None:
    """Compare models across different geometries/splits.

    Loads ``ranking_table.csv`` from each geometry directory, finds
    models appearing in the top 10 of any geometry, and draws two
    subplots (R2 top, MRE bottom) with lines connecting the same model
    across geometries.

    Args:
        ranking_dirs: Mapping from geometry label to path of the
            ``ranking_table.csv`` file.
        plot_dir: Directory where the figure is saved.
        save_svg: Whether to also save an SVG.
    """
    geom_labels = list(ranking_dirs.keys())
    geom_dfs: Dict[str, pd.DataFrame] = {}
    for label, csv_path in ranking_dirs.items():
        if not os.path.exists(csv_path):
            continue
        geom_dfs[label] = pd.read_csv(csv_path)
    if not geom_dfs:
        return

    # Collect models in top 10 of any geometry
    top_models: set = set()
    for label, gdf in geom_dfs.items():
        top_models.update(gdf.head(10)["model"].tolist())
    top_models_list = sorted(top_models)

    # Parse numeric values from formatted strings (e.g. "0.950", "12.3%")
    def _parse_val(val):
        if isinstance(val, str):
            return float(val.replace("%", ""))
        return float(val)

    # Detect column names (support both R2_Global and r2_damage)
    def _get_col(gdf, candidates):
        for c in candidates:
            if c in gdf.columns:
                return c
        return None

    r2_col_candidates = ["R2_Global", "r2_damage", "mean_r2"]
    mre_col_candidates = ["MRE_Global", "mre_del", "mean_mre"]

    plt.style.use("fivethirtyeight")
    fig, (ax_r2, ax_mre) = plt.subplots(2,
                                        1,
                                        figsize=(12, 10),
                                        facecolor="white")
    ax_r2.set_facecolor("white")
    ax_mre.set_facecolor("white")

    x_positions = np.arange(len(geom_labels))

    for model in top_models_list:
        fam = get_model_family(model)
        if fam is None:
            continue
        color = FAMILY_COLORS.get(fam, "grey")
        short = shorten_model_name(model)

        r2_vals = []
        mre_vals = []
        x_valid_r2 = []
        x_valid_mre = []

        for idx, label in enumerate(geom_labels):
            if label not in geom_dfs:
                continue
            gdf = geom_dfs[label]
            row = gdf[gdf["model"] == model]
            if row.empty:
                continue

            r2_col = _get_col(gdf, r2_col_candidates)
            mre_col = _get_col(gdf, mre_col_candidates)

            if r2_col is not None:
                r2_vals.append(_parse_val(row[r2_col].values[0]))
                x_valid_r2.append(idx)
            if mre_col is not None:
                mre_vals.append(_parse_val(row[mre_col].values[0]))
                x_valid_mre.append(idx)

        if r2_vals:
            ax_r2.plot(x_valid_r2,
                       r2_vals,
                       "-o",
                       color=color,
                       alpha=0.7,
                       linewidth=1.5,
                       markersize=5)
            ax_r2.annotate(short,
                           xy=(x_valid_r2[-1], r2_vals[-1]),
                           xytext=(8, 0),
                           textcoords="offset points",
                           fontsize=6,
                           alpha=0.8,
                           color=color,
                           va="center")
        if mre_vals:
            ax_mre.plot(x_valid_mre,
                        mre_vals,
                        "-o",
                        color=color,
                        alpha=0.7,
                        linewidth=1.5,
                        markersize=5)
            ax_mre.annotate(short,
                            xy=(x_valid_mre[-1], mre_vals[-1]),
                            xytext=(8, 0),
                            textcoords="offset points",
                            fontsize=6,
                            alpha=0.8,
                            color=color,
                            va="center")

    for axs, ylabel in [(ax_r2, "R\u00b2 damage"), (ax_mre, "MRE DEL (%)")]:
        axs.set_xticks(x_positions)
        axs.set_xticklabels(geom_labels, fontsize=11)
        axs.set_ylabel(ylabel, fontsize=12)
        axs.grid(True, alpha=0.2)
        general.style_ticks(axs)

    _add_family_legend(ax_r2, fontsize=8, loc="lower left")
    ax_mre.set_xlabel("Geometry / Split", fontsize=12)

    plt.tight_layout()
    for ax in fig.get_axes():
        general.style_ticks(ax)
    general.save_figure(fig, str(plot_dir), "cross_geometry_comparison",
                        save_svg)


# ---------------------------------------------------------------------------
# 15. Paired scatter — same model across two geometries
# ---------------------------------------------------------------------------


def plot_paired_scatter(
    ranking_dirs: Dict[str, str],
    plot_dir: str | Path,
    metric: str = "r2_damage",
    top_n: int = 15,
    save_svg: bool = False,
) -> None:
    """Paired scatter comparing model performance across geometries.

    For each pair of geometries, plots metric_x vs metric_y where
    each point is a model. Points on the diagonal are consistent;
    off-diagonal means performance changes between geometries.

    Args:
        ranking_dirs: Mapping from geometry label to ranking CSV path.
        plot_dir: Directory where the figure is saved.
        metric: Column name to compare (default ``r2_damage``).
        top_n: Top N models per geometry to include.
        save_svg: Whether to also save an SVG.
    """
    labels = list(ranking_dirs.keys())
    dfs = {}
    for label, csv_path in ranking_dirs.items():
        df = pd.read_csv(csv_path)
        df["family"] = df["model"].apply(get_model_family)
        dfs[label] = df

    # Find models in top N of any geometry
    top_models = set()
    for df in dfs.values():
        if metric in df.columns:
            top_models.update(df.nlargest(top_n, metric)["model"].tolist())

    pairs = list(itertools.combinations(labels, 2))
    n_pairs = len(pairs)

    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(1,
                             n_pairs,
                             figsize=(7 * n_pairs, 6),
                             facecolor="white",
                             squeeze=False)

    for idx, (lbl_x, lbl_y) in enumerate(pairs):
        axs = axes[0, idx]
        axs.set_facecolor("white")

        df_x = dfs[lbl_x]
        df_y = dfs[lbl_y]

        merged = df_x[["model", metric, "family"]].merge(
            df_y[["model", metric]],
            on="model",
            suffixes=("_x", "_y"),
        )
        merged = merged[merged["model"].isin(top_models)]

        col_x = f"{metric}_x"
        col_y = f"{metric}_y"

        for fam, grp in merged.groupby("family"):
            color = FAMILY_COLORS.get(fam, "grey")
            axs.scatter(grp[col_x],
                        grp[col_y],
                        color=color,
                        s=50,
                        alpha=0.7,
                        edgecolors="white",
                        linewidth=0.5,
                        zorder=3)
            for _, row in grp.iterrows():
                axs.annotate(shorten_model_name(row["model"]),
                             xy=(row[col_x], row[col_y]),
                             xytext=(5, 3),
                             textcoords="offset points",
                             fontsize=5,
                             alpha=0.7,
                             color=color)

        # Diagonal line
        all_vals = list(merged[col_x]) + list(merged[col_y])
        if all_vals:
            lo = min(all_vals)
            hi = max(all_vals)
            margin = (hi - lo) * 0.05
            axs.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                     "k--",
                     alpha=0.3,
                     linewidth=0.8,
                     zorder=1)
            axs.set_xlim(lo - margin, hi + margin)
            axs.set_ylim(lo - margin, hi + margin)

        axs.set_xlabel(f"{metric} ({lbl_x})", fontsize=10)
        axs.set_ylabel(f"{metric} ({lbl_y})", fontsize=10)
        axs.set_aspect("equal")
        general.style_ticks(axs)

    _add_family_legend(axes[0, -1], fontsize=7, loc="lower right")

    metric_label = metric.replace("_", " ").replace("r2", "R²")
    fig.suptitle(f"Paired comparison: {metric_label} across geometries",
                 fontsize=13,
                 y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    general.save_figure(fig, str(plot_dir), f"paired_scatter_{metric}",
                        save_svg)
