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
    "ref": COLORS_DICT["dark_gray_paper"],
    "opt1": COLORS_DICT["blue_paper"],
    "opt2": COLORS_DICT["red_paper"],
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
    """Plot 3-panel figure: diameter, thickness, lifetime damage.

    Mirrors the paper Figure 4 aesthetic: step plots per section,
    thickness in mm, linear damage axis, REF=grey / OPT1=blue /
    OPT2=red, panel titles on top, shared legend at the bottom.
    """
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(1,
                             3,
                             figsize=(10.8, 4.0),
                             facecolor="white",
                             sharey=True,
                             gridspec_kw={"wspace": 0.18})
    ax_d, ax_t, ax_dmg = axes

    grid_color = COLORS_DICT["light_gray_paper"]
    tick_color = COLORS_DICT["dark_gray_paper"]

    for tower in TOWERS:
        df = summaries[tower]
        color = TOWER_COLORS[tower]
        ax_d.step(df["diameter_m"], df["height_m"], where="mid",
                  color=color, linewidth=1.5, label=tower.upper())
        ax_t.step(df["thickness_m"] * 1000.0, df["height_m"], where="mid",
                  color=color, linewidth=1.5, label=tower.upper())
        ax_dmg.step(df["lifetime_damage"], df["height_m"], where="mid",
                    color=color, linewidth=1.5, label=tower.upper())

    ax_d.set_xlabel("Outer Diameter (m)", fontsize=10, labelpad=8)
    ax_d.set_ylabel("Tower Height (m)", fontsize=10, labelpad=8)
    ax_d.set_title("Outer Diameter", fontsize=12)

    ax_t.set_xlabel("Wall Thickness (mm)", fontsize=10, labelpad=8)
    ax_t.set_title("Wall Thickness", fontsize=12)

    ax_dmg.set_xlabel("Lifetime weighted damage", fontsize=10, labelpad=8)
    ax_dmg.set_title("Damage Profile", fontsize=12)

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, linestyle="-", color=grid_color)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1)
            ax.spines[side].set_color(grid_color)
        ax.tick_params(axis="both",
                       which="major",
                       length=4,
                       width=1,
                       labelsize=10,
                       color=tick_color)

    handles, labels = ax_d.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.22)
    fig.legend(handles, labels, loc="lower center",
               ncol=len(TOWERS), bbox_to_anchor=(0.5, 0.0),
               fontsize=10, frameon=False)

    fig.savefig(out_path, dpi=300, transparent=False)
    plt.close(fig)
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
