# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-positional-arguments
"""FLOATBench benchmark analysis pipeline.

Loads leaderboard CSVs from best and extreme AutoGluon presets,
merges them, and generates benchmark plots.
"""

import os

import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
from absl import app, flags, logging

from floatbench.plots import benchmark as bench_plots

FLAGS = flags.FLAGS

# ── Input directories ──────────────────────────────────────────────
flags.DEFINE_string(
    "base_dir", None, "Base directory containing leaderboard CSVs. "
    "Searches in base_dir/ and base_dir/best/model/ "
    "+ base_dir/extreme/model/.")

# ── Output ─────────────────────────────────────────────────────────
flags.DEFINE_string("output_dir", None, "Directory to save plots and CSVs.")
flags.DEFINE_boolean("save_svg", False, "Whether to save plots in SVG format.")

# ── Benchmark parameters ──────────────────────────────────────────
flags.DEFINE_integer("heatmap_top_n", 10, "Models shown in regime heatmaps.")
flags.DEFINE_integer("ranking_top_n", 5,
                     "Top N per context for the bump chart selection.")
flags.DEFINE_list(
    "detail_sections", ["section_1", "section_30"],
    "Sections for detailed analysis (default: base + top for asymmetry).")
flags.DEFINE_boolean(
    "bump_only_global", False,
    "If True, the bump chart shows only the Global Rel L² DEL context "
    "(useful for splits with no extrapolation regime to compare).")


def _add_mae_ratio_columns(frame: pd.DataFrame,
                           target: str = "damage") -> pd.DataFrame:
    """Add MAE-ratio columns normalised by each row's IT_IT baseline.

    For each column ``mae_{target}_<suffix>`` present in *frame*, a
    sibling ``mae_ratio_{target}_<suffix>`` column is added, equal to
    the original value divided by ``mae_{target}_IT_IT`` for the same
    row. A global ``mae_ratio_{target}`` is also added from
    ``mae_{target}`` when available. Operates in place and returns
    *frame*.
    """
    base_col = f"mae_{target}_IT_IT"
    if base_col not in frame.columns:
        return frame
    base = frame[base_col].replace(0, np.nan)
    prefix = f"mae_{target}_"
    ratio_prefix = f"mae_ratio_{target}_"
    for col in list(frame.columns):
        if col.startswith(prefix):
            frame[ratio_prefix + col[len(prefix):]] = frame[col] / base
    global_col = f"mae_{target}"
    if global_col in frame.columns:
        frame[f"mae_ratio_{target}"] = frame[global_col] / base
    return frame


def _original_name(model_name):
    """Strip _B/_E suffix to get original model name."""
    if model_name.endswith("_B") or model_name.endswith("_E"):
        return model_name[:-2]
    return model_name


def _find_in_bases(bases, model_name, rel_path, preset=None):
    """Find a file across multiple base directories.

    If *bases* is a ``{preset: dir}`` dict and *preset* is given, only
    that preset's dir is searched (so a model appearing in both presets
    resolves to the matching one). Falls back to scanning all bases for
    backwards compatibility when *bases* is a list.
    """
    real_name = _original_name(model_name)
    if isinstance(bases, dict):
        candidates = ([bases[preset]]
                      if preset and preset in bases else list(bases.values()))
    else:
        candidates = list(bases)
    for base in candidates:
        path = os.path.join(base, real_name, rel_path)
        if os.path.isfile(path):
            return path
    return None


def _resolve_leaderboard_path(directory, filename):
    """Returns the first existing path for a leaderboard CSV.

    New runs keep the extended leaderboards in
    ``<directory>/leaderboard_test_summaries/``; legacy runs kept
    them at the directory root. Native ``leaderboard_test.csv``
    lives at the root in both layouts.
    """
    candidates = [
        os.path.join(directory, "leaderboard_test_summaries", filename),
        os.path.join(directory, filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_leaderboard_csv(directory, filename):
    """Load a leaderboard CSV from a directory if it exists."""
    path = _resolve_leaderboard_path(directory, filename)
    if path is None:
        logging.warning("File not found: %s", os.path.join(directory, filename))
        return None
    return pd.read_csv(path)


def _copy_merged_summaries(dirs, output_dir):
    """Replicates every CSV from each preset's summaries folder.

    Walks ``<directory>/leaderboard_test_summaries/`` for each entry in
    *dirs*, reads every CSV found, tags it with a ``preset`` column and
    concatenates same-path files across presets. Writes the merged files
    to ``<output_dir>/leaderboard_test_summaries/`` preserving the
    subfolder layout (e.g. ``damage/``, ``del/``).
    """
    out_root = os.path.join(output_dir, "leaderboard_test_summaries")
    merged: dict[str, list[pd.DataFrame]] = {}
    for directory, preset in dirs:
        src_root = os.path.join(directory, "leaderboard_test_summaries")
        if not os.path.isdir(src_root):
            continue
        for root, _, files in os.walk(src_root):
            for fname in files:
                if not fname.endswith(".csv"):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, src_root)
                df = pd.read_csv(full)
                df["preset"] = preset
                merged.setdefault(rel, []).append(df)

    # Raw numeric rel_l2 per (model, preset), used to re-rank merged
    # summary tables with a cross-preset global rank.
    rel_l2_key: dict[str, dict[tuple[str, str], float]] = {}
    metrics_parts = merged.get("leaderboard_test_metrics.csv", [])
    if metrics_parts:
        mdf = pd.concat(metrics_parts, ignore_index=True)
        for tgt in ("damage", "del"):
            col = f"rel_l2_{tgt}"
            if col in mdf.columns:
                rel_l2_key[tgt] = {
                    (row["model"], row["preset"]): float(row[col])
                    for _, row in mdf[["model", "preset", col]].iterrows()
                }

    for rel, parts in merged.items():
        df = pd.concat(parts, ignore_index=True)
        key_col = next((c for c in ("model", "Model") if c in df.columns), None)
        # Recompute a global rank (1 = best) when a rank column exists.
        if "rank" in df.columns and key_col is not None:
            target = ("damage" if rel.startswith("damage/") else
                      "del" if rel.startswith("del/") else None)
            sort_map = rel_l2_key.get(target)
            if sort_map:
                df["_sort_key"] = df.apply(
                    lambda r: sort_map.get(
                        (r[key_col], r["preset"]), float("inf")),
                    axis=1,
                )
                df = df.sort_values("_sort_key").reset_index(drop=True)
                df["rank"] = range(1, len(df) + 1)
                df = df.drop(columns="_sort_key")
        lead = [c for c in ("rank", key_col, "preset") if c in df.columns]
        df = df[lead + [c for c in df.columns if c not in lead]]
        out_path = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
    logging.info("Merged summaries copied to %s (%d files)", out_root,
                 len(merged))


def _discover_dirs(base_dir):
    """Find directories containing leaderboard CSVs.

    Searches base_dir itself and base_dir/*/model/ subdirs.

    Args:
        base_dir: Root directory to search.

    Returns:
        List of (directory_path, preset_label) tuples.
    """
    dirs = []
    # Check base_dir itself
    if _resolve_leaderboard_path(base_dir,
                                 "leaderboard_test_metrics.csv") is not None:
        dirs.append((base_dir, "base"))

    # Check preset subdirs (best/model, extreme/model)
    for preset in ("best", "extreme"):
        sub = os.path.join(base_dir, preset, "model")
        if _resolve_leaderboard_path(
                sub, "leaderboard_test_metrics.csv") is not None:
            dirs.append((sub, preset))

    return dirs


_FAMILY_SHORT = {
    "LightGBM": "LGBM",
    "CatBoost": "CB",
    "XGBoost": "XGB",
    "TabM": "TabM",
    "RandomForest": "RF",
    "ExtraTrees": "ET",
    "Ensemble": "WE",
}


def _classify_neuralnet(name):
    """Split NeuralNet into NN-F (FastAI) and NN-T (Torch)."""
    if "NeuralNetFastAI" in name:
        return "NN-F"
    if "NeuralNetTorch" in name:
        return "NN-T"
    return None


def _build_model_pool_table(dirs, output_dir):
    """Build a pivot table of model counts per (preset, family).

    Reads ``leaderboard.csv`` from each discovered directory,
    classifies every fitted model by family, and writes a wide
    pivot table to ``<output_dir>/model_pool.csv`` matching the
    layout of Table~\\ref{tab:model_pool} in the paper.

    Args:
        dirs: List of (model_dir, preset_label) tuples.
        output_dir: Directory to write ``model_pool.csv`` into.
    """
    family_order = [
        "XGB", "LGBM", "CB", "NN-F", "NN-T", "TabM", "ET", "RF", "WE"
    ]
    rows = []
    for directory, preset in dirs:
        lb_path = os.path.join(directory, "leaderboard.csv")
        if not os.path.isfile(lb_path):
            logging.warning("model_pool: skipping %s (no leaderboard.csv)",
                            directory)
            continue
        df_lb = pd.read_csv(lb_path)
        for name in df_lb["model"]:
            family = _classify_neuralnet(name)
            if family is None:
                fam = bench_plots.get_model_family(name)
                family = _FAMILY_SHORT.get(fam) if fam else None
            if family is None:
                continue
            rows.append({"preset": preset, "family": family})

    if not rows:
        logging.warning("model_pool: no models found across leaderboards")
        return

    df = pd.DataFrame(rows)
    pivot = (df.groupby(["preset",
                         "family"]).size().unstack(fill_value=0).reindex(
                             columns=family_order, fill_value=0))
    pivot["Total"] = pivot.sum(axis=1)
    pivot.loc["Total"] = pivot.sum(axis=0)
    out_path = os.path.join(output_dir, "model_pool.csv")
    pivot.to_csv(out_path)
    logging.info("Model pool table saved to %s (%d rows)", out_path, len(pivot))


def _load_and_merge(base_dir):
    """Load and merge leaderboard data from discovered directories.

    Args:
        base_dir: Root directory to search for leaderboard CSVs.

    Returns:
        Tuple of (metrics_df, groups_df, sections_df, full_df).
    """
    dirs = _discover_dirs(base_dir)
    if not dirs:
        raise FileNotFoundError(f"No leaderboard CSVs found in {base_dir}")
    logging.info("Found leaderboard data in: %s",
                 ", ".join(f"{d} ({p})" for d, p in dirs))

    csv_names = [
        "leaderboard_test_metrics.csv",
        "leaderboard_test_groups.csv",
        "leaderboard_test_sections.csv",
        "leaderboard_test.csv",
    ]

    frames = {}
    for name in csv_names:
        parts = []
        for directory, preset in dirs:
            df = _load_leaderboard_csv(directory, name)
            if df is not None:
                df["preset"] = preset
                parts.append(df)
        if not parts:
            logging.warning("No data found for %s", name)
            frames[name] = pd.DataFrame()
            continue
        frames[name] = pd.concat(parts, ignore_index=True)

    # Metrics: sort by r2_damage desc; dedup on (model, preset) so the
    # same model name under different presets survives as two rows.
    all_metrics = frames[csv_names[0]]
    if not all_metrics.empty:
        all_metrics = all_metrics.sort_values("r2_damage", ascending=False)
        all_metrics = all_metrics.drop_duplicates(["model", "preset"],
                                                  keep="first").copy()

    # Groups
    all_groups = frames[csv_names[1]]
    if not all_groups.empty:
        all_groups = all_groups.drop_duplicates(["model", "preset"],
                                                keep="first")

    # Sections
    all_sections = frames[csv_names[2]]
    if not all_sections.empty:
        all_sections = all_sections.drop_duplicates(["model", "preset"],
                                                    keep="first")

    # Native leaderboard: merge timing info
    all_native = frames[csv_names[3]]
    if not all_native.empty:
        all_native = all_native.drop_duplicates(["model", "preset"],
                                                keep="first")

    # Full: metrics + native leaderboard columns
    all_full = all_metrics.copy()
    if not all_native.empty:
        native_merge = ["model", "preset"]
        for col in ("score_val", "eval_metric", "pred_time_val", "fit_time",
                    "pred_time_val_marginal", "fit_time_marginal",
                    "pred_time_test_marginal", "stack_level", "can_infer",
                    "fit_order"):
            if col in all_native.columns:
                native_merge.append(col)
        all_full = all_full.merge(all_native[native_merge],
                                  on=["model", "preset"],
                                  how="left")

    # Capacity columns already computed in damage/summary.csv.
    # Peak Memory may be absent (depends on AutoGluon extra_info).
    cap_parts = []
    wanted = ["Peak Memory (GB)", "Mean Latency (ms)"]
    for directory, preset in dirs:
        summary_path = os.path.join(directory, "leaderboard_test_summaries",
                                    "damage", "leaderboard_test_summary.csv")
        if not os.path.isfile(summary_path):
            continue
        df = pd.read_csv(summary_path)
        keep = ["Model"] + [c for c in wanted if c in df.columns]
        df = df[keep].rename(columns={"Model": "model"})
        df["preset"] = preset
        cap_parts.append(df)
    if cap_parts:
        cap_df = pd.concat(cap_parts, ignore_index=True)
        cap_df = cap_df.drop_duplicates(["model", "preset"], keep="first")
        all_full = all_full.merge(cap_df, on=["model", "preset"], how="left")

    # Add model family
    all_full["family"] = all_full["model"].apply(bench_plots.get_model_family)
    all_full = all_full[all_full["family"].notna()]

    return (all_metrics, all_groups, all_sections, all_full)


def main(_):
    """Run the FLOATBench benchmark analysis pipeline."""
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # ── 1. Load and merge leaderboard data ─────────────────────────
    logging.info("Loading leaderboard data from %s", FLAGS.base_dir)
    (all_metrics, all_groups, all_sections,
     all_full) = _load_and_merge(FLAGS.base_dir)
    logging.info("Merged %d models (%d with family info)", len(all_metrics),
                 len(all_full))

    # Add MAE-ratio columns (per-regime MAE normalised by each row's
    # IT_IT MAE) so that heatmaps/bars can compare degradation across
    # regimes without the denominator artefact that rel_l2/R² have when
    # ||y_regime|| or var(y_regime) differ across regimes.
    _add_mae_ratio_columns(all_groups, target="damage")
    if "mae_damage_IT_IT" in all_groups.columns:
        key_cols = ["model", "preset"
                   ] if "preset" in all_metrics.columns else ["model"]
        baseline = all_groups[key_cols + ["mae_damage_IT_IT"]]
        all_metrics = all_metrics.merge(baseline, on=key_cols, how="left")
        if "mae_damage" in all_metrics.columns:
            all_metrics["mae_ratio_damage"] = (
                all_metrics["mae_damage"] /
                all_metrics["mae_damage_IT_IT"].replace(0, np.nan))

    detail_sections = ([] if FLAGS.detail_sections == [] else
                       FLAGS.detail_sections)
    heatmap_top_n = FLAGS.heatmap_top_n
    ranking_top_n = FLAGS.ranking_top_n

    # ── 2. Generate benchmark plots ────────────────────────────────
    regime_dir = os.path.join(FLAGS.output_dir, "leaderboard", "regimes")
    extrapolation_dir = os.path.join(FLAGS.output_dir, "leaderboard",
                                     "extrapolation")
    comparison_dir = os.path.join(FLAGS.output_dir, "leaderboard", "comparison")
    ranking_dir = os.path.join(FLAGS.output_dir, "leaderboard", "ranking")
    top_models_dir = ranking_dir
    # Dirs created on demand below, not upfront.

    has_groups = (not all_groups.empty and any(
        c.startswith("rel_l2_damage_wind_") for c in all_groups.columns))

    if has_groups:
        # Regime heatmaps: MRE DEL (point-wise, surfaces the
        # cross-regime degradation trend EX > IP > IT that Rel L²
        # flattens via its Σy² denominator). Anchor ordering kept:
        # models picked by lowest rel_l2_del.
        logging.info("Generating heatmap: wave x wind groups...")
        bench_plots.plot_heatmap_groups(
            all_metrics,
            all_groups,
            plot_dir=regime_dir,
            top_n=heatmap_top_n,
            save_svg=FLAGS.save_svg,
            metric_prefix="mre_del",
            sort_col="rel_l2_del",
            sort_ascending=True,
            global_col="mre_del",
            cmap="custom_red_seq",
            vmin=0.0,
            vmax=100.0,
            title="MRE DEL (%)",
            fmt=".1f",
            filename="heatmap_groups_mre_del",
        )
        logging.info("Generating heatmap: 9 groups...")
        bench_plots.plot_heatmap_9groups(
            all_metrics,
            all_groups,
            plot_dir=regime_dir,
            top_n=heatmap_top_n,
            save_svg=FLAGS.save_svg,
            metric_prefix="mre_del",
            sort_col="rel_l2_del",
            sort_ascending=True,
            global_col="mre_del",
            cmap="custom_red_seq",
            vmin=0.0,
            vmax=100.0,
            title="MRE DEL (%)",
            fmt=".1f",
            filename="heatmap_9groups_mre_del",
        )

        logging.info("Generating global vs EX_EX scatter...")
        bench_plots.plot_scatter_global_vs_context(all_metrics,
                                                   all_groups,
                                                   plot_dir=extrapolation_dir,
                                                   save_svg=FLAGS.save_svg)

        logging.info("Generating family bar chart...")
        bench_plots.plot_bar_family_regime(all_metrics,
                                           all_groups,
                                           plot_dir=extrapolation_dir,
                                           save_svg=FLAGS.save_svg,
                                           metric="mre_del")
    else:
        logging.info("No wind/wave groups — skipping group plots.")

    has_timing = ("pred_time_test_marginal" in all_full.columns and
                  all_full["pred_time_test_marginal"].notna().any())
    if has_timing:
        bubble_df = all_full.dropna(subset=["pred_time_test_marginal"])
        logging.info("Generating bubble chart (rel_l2_del)...")
        bench_plots.plot_bubble_efficiency(bubble_df,
                                           plot_dir=comparison_dir,
                                           save_svg=FLAGS.save_svg,
                                           metric="rel_l2_del")
    else:
        logging.info("No timing data — skipping bubble chart.")

    logging.info("Generating bump chart...")
    bump_ctx_keys = bench_plots.plot_bump_chart_values(
        all_metrics,
        all_groups,
        all_sections,
        plot_dir=top_models_dir,
        top_n=ranking_top_n,
        save_svg=FLAGS.save_svg,
        only_global=FLAGS.bump_only_global)

    logging.info("Generating family distribution...")
    bench_plots.plot_family_distribution(all_metrics,
                                         plot_dir=comparison_dir,
                                         save_svg=FLAGS.save_svg)

    # ── 3. Replicate source summaries folder with preset column ───
    _copy_merged_summaries(_discover_dirs(FLAGS.base_dir), FLAGS.output_dir)

    # ── 3b. Model pool counts per (preset, family) ────────────────
    _build_model_pool_table(_discover_dirs(FLAGS.base_dir), FLAGS.output_dir)

    # Save a filtered copy of the DEL summary table next to each bump
    # chart, containing only the (model, preset) rows that appear in
    # that chart.
    dmg_summary_path = os.path.join(FLAGS.output_dir,
                                    "leaderboard_test_summaries", "damage",
                                    "leaderboard_test_summary.csv")
    if os.path.isfile(dmg_summary_path) and bump_ctx_keys:
        dmg_sum = pd.read_csv(dmg_summary_path)
        if "Model" in dmg_sum.columns and "preset" in dmg_sum.columns:
            reasons = bump_ctx_keys
            key_set = set(reasons.keys())
            mask = dmg_sum.apply(lambda r, ks=key_set:
                                 (r["Model"], r["preset"]) in ks,
                                 axis=1)
            sub = dmg_sum[mask].copy()
            sub["selected_by"] = sub.apply(
                lambda r, rs=reasons: "; ".join(
                    rs.get((r["Model"], r["preset"]), [])),
                axis=1,
            )
            out_csv = os.path.join(top_models_dir, "bump_chart.csv")
            sub.to_csv(out_csv, index=False)
            logging.info("Saved filtered summary to %s (%d rows)", out_csv,
                         int(mask.sum()))

    # ── 4. Plots that need test predictions ───────────────────────
    model_bases: dict = {}
    for preset in ("best", "extreme"):
        candidate = os.path.join(FLAGS.base_dir, preset, "model", "models")
        if os.path.isdir(candidate):
            model_bases[preset] = candidate
    candidate = os.path.join(FLAGS.base_dir, "models")
    if os.path.isdir(candidate):
        model_bases["base"] = candidate

    if not model_bases:
        logging.info("No model dirs — skipping prediction plots.")
        logging.info("Benchmark complete. Results in %s", FLAGS.output_dir)
        return

    # Tower profile candidates = models drawn in the contexts bump
    # chart, keyed by (model, preset). Same model name across two
    # presets gets disambiguated via a ``_B``/``_E`` display suffix so
    # downstream dicts/plots don't collide.
    profile_pairs = sorted(bump_ctx_keys.keys()) if bump_ctx_keys else []
    presets_per_model: dict = {}
    for mn, _p in profile_pairs:
        presets_per_model.setdefault(mn, set()).add(_p)

    def _display_key(model_name, preset):
        if len(presets_per_model.get(model_name, ())) > 1:
            suffix = "_B" if preset == "best" else "_E"
            return model_name + suffix
        return model_name

    profile_models = [_display_key(mn, p) for mn, p in profile_pairs]
    profile_presets = {_display_key(mn, p): p for mn, p in profile_pairs}
    profile_real_names = {_display_key(mn, p): mn for mn, p in profile_pairs}
    predictions = {}
    metrics_dict = {}
    missing = []

    for model_name in profile_models:
        preset = profile_presets[model_name]
        real = profile_real_names[model_name]
        pred_path = _find_in_bases(model_bases,
                                   model_name,
                                   "test/predictions.csv",
                                   preset=preset)
        if pred_path is None:
            missing.append(model_name)
            continue
        pred = pd.read_csv(pred_path)
        predictions[model_name] = pred["predicted_damage"].values
        meta_match = ((all_metrics["model"] == real) &
                      (all_metrics["preset"] == preset))
        row = all_metrics[meta_match]
        grp = all_groups[(all_groups["model"] == real) &
                         (all_groups["preset"] == preset)] if (
                             not all_groups.empty) else pd.DataFrame()
        sec = all_sections[(all_sections["model"] == real) &
                           (all_sections["preset"] == preset)] if (
                               not all_sections.empty) else pd.DataFrame()
        if len(row) > 0:
            metrics_dict[model_name] = {
                "r2":
                    row["r2_damage"].values[0],
                "rel_l2_damage":
                    row["rel_l2_damage"].values[0],
                "rel_l2_del": (row["rel_l2_del"].values[0] if "rel_l2_del"
                               in row.columns else float("inf")),
                "rel_l2_del_ex_ex": (grp["rel_l2_del_EX_EX"].values[0]
                                     if "rel_l2_del_EX_EX" in grp.columns and
                                     len(grp) > 0 else float("nan")),
                "rel_l2_del_section_1":
                    (sec["rel_l2_del_section_1"].values[0]
                     if "rel_l2_del_section_1" in sec.columns and len(sec) > 0
                     else float("nan")),
                "rel_l2_del_section_30":
                    (sec["rel_l2_del_section_30"].values[0]
                     if "rel_l2_del_section_30" in sec.columns and len(sec) > 0
                     else float("nan")),
            }

    # Save predictions report
    report_path = os.path.join(ranking_dir, "predictions_report.log")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark predictions report\n")
        f.write(f"# base_dir: {FLAGS.base_dir}\n")
        f.write(f"# profile_models: {len(profile_models)}\n\n")
        f.write("## Available predictions:\n")
        for m in profile_models:
            if m in predictions:
                f.write(f"  OK  {profile_real_names[m]} "
                        f"(preset={profile_presets[m]})\n")
        if missing:
            f.write("\n## Missing predictions:\n")
            for m in missing:
                f.write(f"  MISSING  {profile_real_names[m]} "
                        f"(preset={profile_presets[m]})\n")
            f.write("\n## To generate, run:\n")
            f.write("# python task_test_model.py \\\n")
            f.write("#   --predict_only=True \\\n")
            f.write("#   --model_name=<model> \\\n")
            f.write("#   --model_cfg_path="
                    "<base_dir>/<preset>/model/"
                    "autogluon_meta.json\n")
    logging.info("Report saved to %s", report_path)
    if missing:
        logging.warning("%d models missing predictions.", len(missing))

    # Order the available models by Rel L² DEL ascending (anchor
    # metric: stable cross-regime since DEL magnitudes don't explode
    # like damage). Panels read left-to-right from best to worst.
    available = [m for m in profile_models if m in predictions]
    available.sort(
        key=lambda m: metrics_dict.get(m, {}).get("rel_l2_del", float("inf")))

    # Two focused scatter grids (4×3 each) instead of one wide combined
    # plot: one restricted to EX_EX, another highlighting sections 1/30.
    if available:
        first_pred = pd.read_csv(
            _find_in_bases(model_bases,
                           available[0],
                           "test/predictions.csv",
                           preset=profile_presets[available[0]]))
        if "damage" in first_pred.columns:
            logging.info("Generating scatter grids (%d models)...",
                         len(available))
            bench_plots.plot_scatter_ex_ex_grid(first_pred,
                                                predictions,
                                                metrics_dict,
                                                available,
                                                plot_dir=top_models_dir,
                                                save_svg=FLAGS.save_svg,
                                                presets=profile_presets)
            bench_plots.plot_scatter_sections_grid(first_pred,
                                                   predictions,
                                                   metrics_dict,
                                                   available,
                                                   plot_dir=top_models_dir,
                                                   save_svg=FLAGS.save_svg,
                                                   presets=profile_presets)
        else:
            logging.info("Predictions in old format — skipping scatter.")
            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n## scatter grids not generated\n")
                f.write("# Predictions in old format (missing "
                        "damage, section_name, groups).\n")
                f.write("# Re-run task_test_model.py with "
                        "--predict_only=True for new format.\n")

    # Tower profiles
    logging.info("Generating tower profiles (%d models)...", len(available))
    bench_plots.plot_tower_profiles_comparison(available,
                                               model_bases,
                                               plot_dir=top_models_dir,
                                               top_n=len(available),
                                               save_svg=FLAGS.save_svg,
                                               presets=profile_presets)

    logging.info("Benchmark analysis complete. Results in %s", FLAGS.output_dir)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
