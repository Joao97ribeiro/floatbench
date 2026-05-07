# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""Selectors for training and test splits based on wind and wave conditions."""

from __future__ import annotations

import os
import json
from typing import Iterable, List

from absl import logging

import numpy as np
import pandas as pd

from . import domain_groups
from .. import utils, plots


def select_train_ids(
        df: pd.DataFrame,
        combinations_wind: int = None,
        combinations_waves: int = None,
        combinations_seed: int = None,
        train_first_n_rows: int = None,
        train_size: float = None,
        train_size_seed: int = 42) -> tuple[list[str], dict[str, Iterable]]:
    """Select training ``sim_id`` values and record used states/pairs.

    Mirrors the logic of the original script while being concise and testable.

    Args:
        df: Long-format dataframe containing ``sim_id``, ``wind_seed``,
          ``wind_speed``, ``wave_hs`` and ``wave_tp``.
        combinations_wind: Preset index for wind speed selection.
        combinations_waves: Preset index for wave selection.
        combinations_seed: Number of seeds to include (1..6).
        train_first_n_rows: Select the first N rows as training data.
        train_size: Fraction of data to use for training (0-1).
        train_size_seed: Random seed for train_size split.

    Returns:
        Sorted list of training ``sim_id`` values.
    """
    # CASE 0: fraction-based random split
    if train_size is not None:
        unique_ids = df["sim_id"].astype(str).unique()
        n_train = int(len(unique_ids) * train_size)
        rng = np.random.default_rng(train_size_seed)
        selected = rng.choice(unique_ids, size=n_train, replace=False)
        return sorted(selected.tolist())

    # CASE 1: simple "first N rows" split
    if train_first_n_rows is not None:
        df_sel = df[0:train_first_n_rows].copy()
        return sorted(df_sel["sim_id"].astype(str).tolist())

    # CASE 2: original combination-based logic
    n_seeds = max(0, min(int(combinations_seed), 6))
    seeds_keep = set(range(1, n_seeds + 1))

    unique_ws = sorted(df["wind_speed"].unique())
    ws_idx = utils.maps.get_wind_wave_indices(selection=combinations_wind,
                                              mode="wind")
    ws_selected = [unique_ws[i] for i in ws_idx]

    df_pool = df[df["wind_seed_id"].isin(seeds_keep)].copy()

    train_ids: List[str] = []
    states_used: set[tuple] = set()
    wave_pairs_used: set[tuple] = set()

    for ws in ws_selected:
        df_ws = df_pool[df_pool["wind_speed"] == ws]
        unique_hs = sorted(df_ws["wave_hs"].unique())
        hs_sel = [
            unique_hs[i] for i in utils.maps.get_wind_wave_indices(
                selection=combinations_waves, mode="waves")
        ]

        for hs in hs_sel:
            df_ws_hs = df_ws[df_ws["wave_hs"] == hs]
            unique_tp = sorted(df_ws_hs["wave_tp"].unique())
            tp_sel = [
                unique_tp[i] for i in utils.maps.get_wind_wave_indices(
                    selection=combinations_waves, mode="waves")
            ]
            df_sel = df_ws_hs[df_ws_hs["wave_tp"].isin(tp_sel)]
            train_ids.extend(df_sel["sim_id"].astype(str).tolist())

            for _, row in df_sel.iterrows():
                states_used.add(
                    (row["wind_speed"], row["wave_hs"], row["wave_tp"]))
                wave_pairs_used.add((row["wave_hs"], row["wave_tp"]))

    return sorted(set(train_ids))


def split_train_test(
        df: pd.DataFrame,
        train_first_n_rows: str = None,
        combinations_wind: int = None,
        combinations_waves: int = None,
        combinations_seed: int = None,
        train_size: float = None,
        train_size_seed: int = 42,
        output_dir: str = None,
        save_svg: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test split from a dataframe.

    Args:
        df: Dataframe.
        train_first_n_rows: Select the first N rows as training data.
        combinations_wind: Preset index for wind speeds.
        combinations_waves: Preset index for waves (both Hs and Tp).
        combinations_seed: Number of seeds in training pool.
        train_size: Fraction of data to use for training (0-1).
        train_size_seed: Random seed for train_size split.
        output_dir: Directory to save outputs.
        save_svg: Whether to save plots as SVG in addition to PNG.

    Returns:
        (df_train, df_test) dataframes.
    """
    train_ids = select_train_ids(df, combinations_wind, combinations_waves,
                                 combinations_seed, train_first_n_rows,
                                 train_size, train_size_seed)

    df_ = df.copy()
    df_["sim_id"] = df_["sim_id"].astype(str)
    df_["is_train"] = df_["sim_id"].isin(train_ids)

    df_train = df_[df_["is_train"]]
    df_test = df_[~df_["is_train"]]

    n_all = len(df_)
    n_train = len(df_train)
    n_test = len(df_test)

    meta = {
        "train": {
            "num_samples":
                n_train,
            "%_samples":
                round(100 * n_train / n_all, 2),
            "mode":
                "train_size" if train_size is not None else "first_n_rows"
                if train_first_n_rows is not None else "combinations",
            "train_size":
                train_size if train_size is not None else None,
            "train_size_seed":
                train_size_seed if train_size is not None else None,
            "train_first_n_rows":
                train_first_n_rows,
            "combinations_wind":
                combinations_wind if train_size is None else None,
            "combinations_waves":
                combinations_waves if train_size is None else None,
            "combinations_seed":
                combinations_seed if train_size is None else None,
        },
        "test": {
            "num_samples": n_test,
            "%_samples": round(100 * n_test / n_all, 2),
        },
        "total_samples": n_all,
    }

    logging.info("Train/test split summary:")
    for split in ("train", "test"):
        info = meta[split]
        logging.info("  %s: %d samples (%.2f%%)", split, info["num_samples"],
                     info["%_samples"])
    logging.info("  total: %d samples", meta["total_samples"])

    if output_dir is not None:
        base_dir = os.path.join(output_dir, "train_test")
        os.makedirs(base_dir, exist_ok=True)

        df_train.to_csv(os.path.join(base_dir, "train_damage.csv"), index=False)
        df_test.to_csv(os.path.join(base_dir, "test_damage.csv"), index=False)

        json_path = os.path.join(base_dir, "split_metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        plot_dir = os.path.join(base_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plots.plot_train_test_subplots(
            df_train=df_train,
            df_test=df_test,
            pairs=[("wind_speed", "wave_hs"), ("wave_hs", "wave_tp")],
            axis_labels=[("Wind Speed (m/s)", "Wave Height (m)"),
                         (["Wave Height (m)", "Wave Period (s)"])],
            output_dir=plot_dir,
            filename="train_test",
            save_svg=save_svg,
            save_separate=True,
            save_combined=True,
            separate_suffix=("wind", "wave"),
            titles=("Wind Distribution: Train vs. Test",
                    "Wave Distribution: Train vs. Test"))
        plots.plot_train_test_subplots(
            df_train=df_train,
            df_test=df_test,
            pairs=[("mean_wind_speed", "std_wind_speed"),
                   ("wave_hs", "wave_tp")],
            axis_labels=[("Mean Wind Speed (m/s)", "Std Wind Speed (m/s)"),
                         (["Wave Height (m)", "Wave Period (s)"])],
            output_dir=plot_dir,
            filename="train_test_mean_std",
            save_svg=save_svg,
            save_separate=True,
            save_combined=True,
            separate_suffix=("wind", "wave"),
            titles=("Wind Distribution: Train vs. Test",
                    "Wave Distribution: Train vs. Test"))

    return df_train, df_test


def split_test_groups(df_train: pd.DataFrame,
                      df_test: pd.DataFrame,
                      interp_names: List[str],
                      interp_edges: List[float],
                      extrap_names: List[str] = None,
                      extrap_edges: List[float] = None,
                      output_dir: str = None,
                      save_svg: bool = False) -> pd.DataFrame:
    """
    Split test set into domain groups based on wind and wave conditions.

    Args:
        df_train: Training dataframe.
        df_test: Test dataframe.
        interp_names: Names for interpolation domain groups.
        interp_edges: Distance edges for interpolation groups.
        extrap_names: Names for extrapolation groups outside the
            alpha shape boundary.
        extrap_edges: Distance edges to subdivide extrapolation
            groups by k-NN distance.
        output_dir: Directory to save outputs.
        save_svg: Whether to save plots as SVG in addition to PNG.

    Returns:
        df_test_groups: DataFrame with domain group assignments.
    """
    if extrap_names is None:
        extrap_names = ["Extrapolate"]
    if extrap_edges is None:
        extrap_edges = []

    wind_cols = ["mean_wind_speed", "std_wind_speed"]
    wave_cols = ["wave_hs", "wave_tp"]

    wind_wave_grouper = domain_groups.WindWaveDomainGrouper(
        wind_cols=wind_cols,
        wave_cols=wave_cols,
        interp_edges=interp_edges,
        interp_names=interp_names,
        extrap_names=extrap_names,
        extrap_edges=extrap_edges,
        k=1,
        aggregate="min",
        scale_stat="mean",
        kind_scaler="standard")

    (df_test_groups, domain_thresholds_metadata,
     test_groups_metadata) = wind_wave_grouper.group(df_train=df_train,
                                                     df_test=df_test,
                                                     plot_dir=output_dir,
                                                     save_svg=save_svg)

    logging.info("Test groups split (wind+wave) summary:")
    groups = (test_groups_metadata.get("test_groups",
                                       {}).get("Windgroup_Wavegroup", {}))
    for group_name, info in groups.items():
        logging.info(
            "  %s: %d samples (%.2f%%)",
            group_name,
            info["num_samples"],
            info["%_samples"],
        )

    logging.info(
        "  total: %d samples",
        test_groups_metadata["total_test_samples"],
    )

    if output_dir is not None:
        wind_wave_grouper.crosstab(df_test_groups, output_dir)

        base_dir = os.path.join(output_dir, "train_test", "test_groups")
        os.makedirs(base_dir, exist_ok=True)

        df_test_groups.to_csv(os.path.join(base_dir, "test_groups_damage.csv"),
                              index=False)

        with open(os.path.join(base_dir, "domain_thresholds_metadata.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(domain_thresholds_metadata,
                      f,
                      indent=4,
                      ensure_ascii=False)

        with open(os.path.join(base_dir, "test_groups_metadata.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(test_groups_metadata, f, indent=4, ensure_ascii=False)

        plot_base_dir = os.path.join(output_dir, "train_test", "test_groups",
                                     "plots")

        # Get alpha shape boundary polygons for plotting
        wind_polygon, wave_polygon = wind_wave_grouper.boundary_polygons()

        # Build per-subplot group names including extrapolation labels
        wind_names = domain_groups.extend_group_names(
            wind_wave_grouper.wind_names, extrap_names)
        wave_names = domain_groups.extend_group_names(
            wind_wave_grouper.wave_names, extrap_names)

        plots.plot_train_test_subplots(
            df_train=df_train,
            df_test=df_test_groups,
            pairs=[("mean_wind_speed", "std_wind_speed"),
                   ("wave_hs", "wave_tp")],
            group_col=("wind_group", "wave_group"),
            axis_labels=[("Mean Wind Speed (m/s)", "Std Wind Speed (m/s)"),
                         (["Wave Height (m)", "Wave Period (s)"])],
            group_names=(wind_names, wave_names),
            output_dir=plot_base_dir,
            filename="train_testgroups",
            save_svg=save_svg,
            save_separate=True,
            save_combined=True,
            separate_suffix=("wind", "wave"),
            titles=("Wind Distribution: Train vs. Test Groups",
                    "Wave Distribution: Train vs. Test Groups"),
            domain_polygons=(wind_polygon, wave_polygon))

    return df_test_groups
