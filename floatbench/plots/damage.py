# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
"""Plots for visualizing tower damage profiles."""

from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt

from floatbench.colors import COLORS_DICT
from . import general


def plot_tower_damage_profiles_vs_reference(
    damage_profiles: Sequence[np.ndarray],
    ref_profile: np.ndarray,
    heights: np.ndarray,
    labels: Sequence[str],
    plot_dir: Optional[str] = None,
    filename: str = "tower_damage_profiles_vs_reference",
    top: int = 3,
    save_svg: bool = False,
    style: str = "fivethirtyeight",
    title: Optional[bool] = None,
    xlabel: str = "Weighted Fatigue Damage",
    train_profile: Optional[np.ndarray] = None,
) -> None:
    """Plot the top-k tower damage profiles compared to a reference profile.

    Selects the k profiles with the lowest mean relative error to the
    reference tower damage profile and plots them alongside the reference.
    The saved filename does not depend on k.

    Args:
        damage_profiles: Iterable of 1D arrays, each a tower damage profile.
        ref_profile: 1D array with the reference tower damage profile.
        heights: 1D array of tower heights corresponding to each damage point.
        labels: List of labels for each damage profile.
        plot_dir: Directory where the plot will be saved. Defaults to CWD.
        filename: Fixed base name of the saved file (no dependence on k).
        top: Number of profiles to select and plot.
        save_svg: Whether to also save the SVG version of the figure.
        style: Matplotlib style to use.
        title: Optional title for the plot.
        xlabel: Label for the x-axis.
        train_profile: Optional 1D array with the train-only damage profile.

    Raises:
        ValueError: If damage_profiles and labels have different lengths.
    """

    damage_profiles = np.array(damage_profiles)
    num_profiles = len(damage_profiles)
    top = min(top, num_profiles)

    if num_profiles != len(labels):
        raise ValueError(
            "damage_profiles and labels must have matching lengths.")

    # compute mean relative error per profile
    combo_errors = []
    for i in range(num_profiles):
        rel_error = np.abs(damage_profiles[i] -
                           ref_profile) / np.abs(ref_profile) * 100.0
        mean_rel_error = np.mean(rel_error)
        combo_errors.append(mean_rel_error)

    combo_errors = np.array(combo_errors)
    top_indices = np.argsort(combo_errors)[:top]

    # set global visual style
    plt.style.use(style)
    fig, axs = plt.subplots(figsize=(6, 8), facecolor="white")
    axs.set_facecolor("white")

    default_colors = [
        COLORS_DICT["blue_paper"],
        COLORS_DICT["red_paper"],
        COLORS_DICT["dark_red_paper"],
        COLORS_DICT["light_red_paper"],
    ]

    # reference profile (plotted first for legend order)
    axs.step(
        ref_profile,
        heights,
        where="mid",
        label="Reference Profile",
        linewidth=2,
        linestyle="-",
        color=COLORS_DICT["grey_paper"],
    )

    # train-only profile
    if train_profile is not None:
        train_rel_err = np.abs(train_profile -
                               ref_profile) / np.abs(ref_profile) * 100.0
        train_mre = np.mean(train_rel_err)
        axs.step(
            train_profile,
            heights,
            where="mid",
            label=f"Train Profile (MRE = {train_mre:.2f}%)",
            linewidth=2,
            linestyle="-",
            color=COLORS_DICT["red_paper"],
        )

    # plot top-k profiles
    for k_i, prof_idx in enumerate(top_indices):
        prof = damage_profiles[prof_idx]
        err = combo_errors[prof_idx]
        label = labels[prof_idx]
        color = default_colors[k_i % len(default_colors)]

        axs.step(
            prof,
            heights,
            where="mid",
            label=f"{label} (MRE = {err:.2f}%)",
            linewidth=2,
            color=color,
        )

    # axes + style
    general.style_axes(axs, xlabel, "Tower Height [m]", fontsize=9)

    # clean title (sem "closest")
    if title is not None:
        axs.set_title(title, fontsize=12, pad=10)
    else:
        axs.set_title(f"Top-{top} Tower Damage Profiles vs Reference",
                      fontsize=12,
                      pad=10)

    axs.set_ylim(bottom=0)
    axs.legend(fontsize=8, frameon=True, framealpha=0.5)
    plt.tight_layout()

    # save figure
    general.save_figure(fig, plot_dir, filename, save_svg)


def plot_error_profile(
    heights: np.ndarray,
    damage_mre: np.ndarray,
    del_mre: Optional[np.ndarray] = None,
    plot_dir: Optional[str] = None,
    filename: str = "error_profile",
    xlabel: str = "MRE [%]",
    save_svg: bool = False,
    style: str = "fivethirtyeight",
) -> None:
    """Plot MRE along the tower height.

    Displays a step plot with tower height on the y-axis and the
    mean relative error (%) on the x-axis, following the same visual
    style as the tower damage profile plots.

    Args:
        heights: Tower section heights (m).
        damage_mre: MRE (%) per section for damage.
        del_mre: Optional MRE (%) per section for DEL.
        plot_dir: Directory where the plot will be saved.
        filename: Base name of the saved file.
        xlabel: Label for the x-axis.
        save_svg: Whether to also save SVG.
        style: Matplotlib style to use.
    """
    plt.style.use(style)
    fig, axs = plt.subplots(figsize=(6, 8), facecolor="white")
    axs.set_facecolor("white")

    axs.step(
        damage_mre,
        heights,
        where="mid",
        label="Damage",
        linewidth=2,
        color=COLORS_DICT["red_paper"],
    )

    if del_mre is not None:
        axs.step(
            del_mre,
            heights,
            where="mid",
            label="DEL",
            linewidth=2,
            linestyle="--",
            color=COLORS_DICT["red_paper"],
        )

    general.style_axes(axs, xlabel, "Tower Height [m]")
    axs.set_ylim(bottom=0)
    axs.legend(fontsize=8, frameon=True, framealpha=0.5)
    plt.tight_layout()

    general.save_figure(fig, plot_dir, filename, save_svg)
