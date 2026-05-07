"""General plotting utilities for damage prediction module."""

import os

import matplotlib.pyplot as plt
from floatbench.colors import COLORS_DICT


def set_paper_style() -> None:
    """Set a compact, technical plotting style for paper figures."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS_DICT["light_gray_paper"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS_DICT["light_gray_paper"],
        "grid.linewidth": 0.55,
        "grid.linestyle": "-",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


def style_axes(axs: plt.Axes,
               xlabel: str,
               ylabel: str,
               fontsize: int = 8) -> None:
    """Apply consistent styling to the given axes.

    Args:
        axs: Matplotlib Axes object to style.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        fontsize: Font size for axis labels and ticks.
    """
    axs.set_xlabel(xlabel, fontsize=fontsize)
    axs.set_ylabel(ylabel, fontsize=fontsize)
    axs.grid(True,
             linestyle="-",
             color=COLORS_DICT["light_gray_paper"],
             linewidth=0.55,
             zorder=0)
    style_ticks(axs, fontsize=fontsize)
    for axis_obj in (axs.xaxis, axs.yaxis):
        axis_obj.get_offset_text().set_fontsize(fontsize - 1)


def style_ticks(axs: plt.Axes, fontsize: int = 7) -> None:
    """Apply consistent tick and spine styling to the given axes.

    Args:
        axs: Matplotlib Axes object to style.
        fontsize: Font size for tick labels.
    """
    axs.tick_params(axis="both",
                    which="major",
                    length=3,
                    width=0.8,
                    labelsize=fontsize,
                    color=COLORS_DICT["dark_gray_paper"])
    style_spines(axs)


def style_spines(axs: plt.Axes) -> None:
    """Apply consistent spine styling to the given axes.

    Args:
        axs: Matplotlib Axes object to style.
    """
    for spine in ["top", "right", "left", "bottom"]:
        axs.spines[spine].set_visible(True)
        axs.spines[spine].set_linewidth(0.8)
        axs.spines[spine].set_color(COLORS_DICT["light_gray_paper"])


def save_figure(fig: plt.Figure,
                plot_dir: str,
                filename: str,
                save_svg: bool = False) -> None:
    """Save a figure as PNG and optionally SVG, then close it.

    Args:
        fig: Matplotlib Figure to save.
        plot_dir: Directory where the figure will be saved.
        filename: Base name of the saved file (without extension).
        save_svg: Whether to also save the SVG version of the figure.
    """
    os.makedirs(plot_dir, exist_ok=True)
    png_path = os.path.join(plot_dir, f"{filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)

    if save_svg:
        svg_path = os.path.join(plot_dir, f"{filename}.svg")
        plt.savefig(svg_path,
                    bbox_inches="tight",
                    transparent=True,
                    format="svg")

    plt.close(fig)
