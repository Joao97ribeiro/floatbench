"""Reproduce Figure 4 from the FLOATBench paper.

Figure 4: Tower outer diameter, wall thickness, and FLOATBench lifetime
weighted damage along the tower height for the three released towers
REF, OPT1, and OPT2.

Usage:
    python examples/figure_geom_damage.py
    python examples/figure_geom_damage.py --output=fig4.png
    python examples/figure_geom_damage.py --data_root=data --hf=False

The script reads ``<data_root>/<tower>/data.csv`` for each of REF, OPT1,
and OPT2. With ``--hf=True`` it instead loads the dataset from
HuggingFace (DeCoDELab/FLOATBench).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags, logging

from floatbench.colors import COLORS_DICT

FLAGS = flags.FLAGS

flags.DEFINE_string("data_root", "data",
                    "Local directory containing ref/, opt1/, opt2/.")
flags.DEFINE_boolean(
    "hf", False, "If True, load from HuggingFace "
    "(DeCoDELab/FLOATBench) instead of --data_root.")
flags.DEFINE_string("output", "figure_geom_damage.png",
                    "Output path for the figure.")

TOWERS = ("ref", "opt1", "opt2")
TOWER_COLORS = {
    "ref": COLORS_DICT["dark_blue_paper"],
    "opt1": COLORS_DICT["red_paper"],
    "opt2": COLORS_DICT["brown_paper"],
}


def load_tower(name):
    """Return the full data.csv for a tower as a DataFrame."""
    if FLAGS.hf:
        from datasets import load_dataset  # pylint: disable=import-outside-toplevel
        ds = load_dataset("DeCoDELab/FLOATBench", name)
        return pd.concat(
            [ds["train"].to_pandas(), ds["test"].to_pandas()],
            ignore_index=True,
        )
    csv = Path(FLAGS.data_root) / name / "data.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} not found (use --hf=True or --data_root)")
    return pd.read_csv(csv)


def per_section_summary(df):
    """Aggregate per-section diameter, thickness, and lifetime damage.

    Lifetime damage is sum(damage * damage_weight) per section across
    all operating conditions and turbulence seeds.
    """
    df = df.copy()
    df["weighted_damage"] = df["damage"] * df["damage_weight"]
    grouped = df.groupby("section_id")
    summary = grouped.agg(
        height_m=("section_height_m", "first"),
        radius_m=("section_radius_m", "first"),
        thickness_m=("section_thickness_m", "first"),
        lifetime_damage=("weighted_damage", "sum"),
    ).reset_index()
    summary["diameter_m"] = 2.0 * summary["radius_m"]
    return summary.sort_values("section_id")


def plot_figure(summaries, out_path):
    """Plot 3-panel figure: diameter, thickness, lifetime damage."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=True)
    panels = (
        ("diameter_m", "Outer diameter (m)"),
        ("thickness_m", "Wall thickness (m)"),
        ("lifetime_damage", "Lifetime weighted damage"),
    )
    grid_color = COLORS_DICT["light_gray_paper"]
    spine_color = COLORS_DICT["dark_gray_paper"]
    for ax, (col, xlabel) in zip(axes, panels):
        for tower, df in summaries.items():
            ax.plot(df[col],
                    df["height_m"],
                    color=TOWER_COLORS[tower],
                    lw=2.0,
                    label=tower.upper())
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(True, color=grid_color, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(spine_color)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=spine_color, labelsize=10)
    axes[2].set_xscale("log")
    axes[0].set_ylabel("Tower height (m)", fontsize=11)
    axes[0].legend(loc="best",
                   frameon=False,
                   fontsize=11,
                   labelcolor=spine_color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="none", transparent=True)
    logging.info("Figure saved to %s", out_path)


def main(_):
    summaries = {}
    for tower in TOWERS:
        logging.info("Loading %s", tower)
        df = load_tower(tower)
        summaries[tower] = per_section_summary(df)
        logging.info("  %s: %d sections", tower, len(summaries[tower]))
    plot_figure(summaries, FLAGS.output)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
