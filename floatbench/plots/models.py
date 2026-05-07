#pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""Plotting functions for model training."""

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from floatbench.colors import COLORS_DICT

from . import general


def plot_learning_curves_xgboost(
    eval_history: dict,
    plot_dir: str,
    metric: str = "rmse",
    train_key: str = "validation_0",
    valid_key: str = "validation_1",
    style: str = "fivethirtyeight",
    save_svg: bool = False,
) -> None:
    """Plot train/validation learning curves for a given metric.

    Args:
        eval_history: Dictionary from XGBoost `evals_result()`.
        plot_dir: Directory where the figure will be saved.
        metric: Metric name (e.g. 'rmse', 'mae').
        train_key: Key for the training set in eval_history.
        valid_key: Key for the validation set in eval_history.
        style: Matplotlib style.
        save_svg: If True, also save the SVG version.

    Raises:
        ValueError: If the specified keys or metric are not found in
            eval_history.
    """
    # Style setup
    plt.style.use(style)

    # Validations and conversions
    if train_key not in eval_history or valid_key not in eval_history:
        raise ValueError(f"Keys '{train_key}' and/or '{valid_key}' not "
                         "found in eval_history.")
    if metric not in eval_history[train_key]:
        raise ValueError(
            f"Metric '{metric}' not found in eval_history['{train_key}'].")

    train_metric = np.array(eval_history[train_key][metric])
    valid_metric = np.array(eval_history[valid_key][metric])

    fig, axs = plt.subplots(figsize=(6, 4), facecolor="white")
    general.style_axes(axs, "Boosting Round", metric.upper())

    axs.plot(
        train_metric,
        label=f"Train {metric}",
        color=COLORS_DICT.get("blue_paper", "tab:blue"),
        linewidth=1.5,
    )
    axs.plot(
        valid_metric,
        label=f"Valid {metric}",
        color=COLORS_DICT.get("red_paper", "tab:red"),
        linewidth=1.5,
    )

    axs.set_title(f"Learning Curves ({metric})", fontsize=12, pad=10)

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
    general.save_figure(fig, plot_dir, f"learning_curves_{metric}", save_svg)


def plot_learning_curves_mlp(
    train_losses: List[float],
    valid_losses: List[float],
    plot_dir: str,
    style: str = "fivethirtyeight",
    save_svg: bool = False,
) -> None:
    """Plot train/validation learning curves for MLP training.

    Args:
        train_losses: List of training losses per epoch.
        valid_losses: List of validation losses per epoch.
        plot_dir: Directory where the figure will be saved.
        style: Matplotlib style.
        save_svg: If True, also save the SVG version.
    """
    plt.style.use(style)

    fig, axs = plt.subplots(figsize=(6, 4), facecolor="white")
    general.style_axes(axs, "Epoch", "Loss")

    axs.plot(
        train_losses,
        label="Train loss",
        color=COLORS_DICT.get("blue_paper", "tab:blue"),
        linewidth=1.5,
    )
    axs.plot(
        valid_losses,
        label="Valid loss",
        color=COLORS_DICT.get("red_paper", "tab:red"),
        linewidth=1.5,
    )

    axs.set_title("Learning Curves (Loss)", fontsize=12, pad=10)

    legend = axs.legend(
        frameon=True,
        framealpha=0.5,
        fontsize=9,
        title_fontsize=10,
    )
    legend.get_frame().set_facecolor(COLORS_DICT["light_gray_paper"])
    legend.get_frame().set_linewidth(0)

    plt.tight_layout()

    general.save_figure(fig, plot_dir, "learning_curves_loss", save_svg)
