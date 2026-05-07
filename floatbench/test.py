# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-few-public-methods
"""AutoGluon damage predictor and evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path

from absl import logging

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from floatbench.base_predictor import (BaseDamagePredictor)


class AGDamagePredictor(BaseDamagePredictor):
    """AutoGluon predictor for damage prediction.

    Attributes:
        model_cfg_path: Path to the file with the saved model config.
        model_name: Name of the saved model.
        output_dir: Directory to save predictions and plots.
        report_del: Whether to report DEL metrics and plots in addition to
          DAMAGE.
        model_path: Path to the saved model directory.
        features: List of feature column names.
        target: Target column name.
        target_transform_to_del: Whether target is DEL-transformed.
        fine_tune_base_model_name: Name of fine-tune base model.
        fine_tune_base_model_path: Path to fine-tune base model.
        fine_tune_extra_features: Extra features from fine-tuning.
        fine_tune_base_model: Loaded fine-tune base model (or None).
        model: Loaded AutoGluon predictor.
    """

    def __init__(
        self,
        model_cfg_path: str | Path = None,
        model_name: str = None,
        output_dir: str | Path = None,
        report_del: bool = False,
    ) -> None:
        """Initializes the predictor.

        Args:
            model_cfg_path: Path to the file with the saved model config.
            model_name: Name of the saved model.
            output_dir: Directory to save predictions and plots.
            report_del: Whether to report DEL metrics and plots in addition
              to DAMAGE.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_cfg_path = model_cfg_path
        self.model_name = model_name

        model_meta = self._load_model_meta()

        self.model_path = model_meta["model_path"]
        self.features = model_meta["features"]
        self.target = model_meta["target"]
        self.target_transform_to_del = model_meta["target_transform_to_del"]

        self.fine_tune_base_model_name = model_meta["fine_tune_base_model_name"]
        self.fine_tune_base_model_path = model_meta["fine_tune_base_model_path"]
        self.fine_tune_extra_features = model_meta["extra_features"]

        self.model = self._load_model(self.model_path)

        if self.fine_tune_base_model_path is not None:

            self.fine_tune_base_model = self._load_model(
                self.fine_tune_base_model_path)

            if self.fine_tune_base_model_name is None:
                self.fine_tune_base_model_name = (
                    self.fine_tune_base_model.model_best)
        else:
            self.fine_tune_base_model = None

        if self.model_name is None:
            self.model_name = self.model.model_best

        self._save_artifacts(self.output_dir)

        self.report_del = report_del

        super().__init__(
            features=self.features,
            target=self.target,
            output_dir=output_dir,
            report_del=self.report_del,
        )

    def _load_model_meta(self) -> dict:
        """Loads the AutoGluon metadata from disk.

        Returns:
            Dictionary with model metadata.
        """
        with open(self.model_cfg_path, "r", encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)

        features = cfg["data"]["features"]
        target = cfg["data"]["target"]
        target_transform_to_del = cfg["data"]["target_transform_to_del"]

        model_path = cfg["model"]["predictor_path"]
        fine_tuning_cfg = cfg.get("fine_tuning")

        if fine_tuning_cfg is not None:
            fine_tune_base_model_path = fine_tuning_cfg["base_predictor_path"]
            fine_tune_base_model_name = fine_tuning_cfg["base_model_name"]
            extra_features = fine_tuning_cfg.get("extra_features", [])
        else:
            fine_tune_base_model_path = None
            fine_tune_base_model_name = None
            extra_features = []

        return {
            "features": features,
            "target": target,
            "target_transform_to_del": target_transform_to_del,
            "model_path": model_path,
            "fine_tune_base_model_path": fine_tune_base_model_path,
            "fine_tune_base_model_name": fine_tune_base_model_name,
            "extra_features": extra_features,
        }

    def _load_model(self, model_path: str | Path = None) -> TabularPredictor:
        """Loads the AutoGluon model from disk.

        Args:
            model_path: Path to the saved model.

        Returns:
            Loaded AutoGluon regressor.
        """
        model = TabularPredictor.load(str(model_path))

        return model

    def build_test_data(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Builds a DataFrame suitable for ``TabularPredictor.leaderboard``.

        Mirrors the construction used by ``AGDamageTrainer`` so that the
        schema (DEL-transformed target, fine-tune ``damage_sim`` column)
        matches what the model was trained on.

        Args:
            df_test: Test DataFrame with the feature columns and the
                raw-damage target column.

        Returns:
            DataFrame with ``self.features + [self.target]`` columns
            (plus ``damage_sim`` for fine-tuned models), ready to be
            passed to ``model.leaderboard(data=...)``.
        """
        test_data = df_test[self.features].copy()
        y_target = df_test[self.target].astype(float).values
        if self.target_transform_to_del:
            y_target = np.cbrt(y_target)
        test_data[self.target] = y_target

        if self.fine_tune_base_model is not None:
            x_t = test_data[self.features].copy()
            test_data["damage_sim"] = self.fine_tune_base_model.predict(
                x_t, model=self.fine_tune_base_model_name).values

        return test_data

    def predict(self, df: pd.DataFrame, save_csv: bool = True) -> np.ndarray:
        """Predicts damage for a new dataframe.

        Args:
            df: Dataframe with feature columns.
            save_csv: Whether to save predictions to a CSV file.

        Returns:
            Array with predicted damage values.
        """
        features_df = df[self.features].copy()

        if self.fine_tune_base_model is not None:

            y_pred = self.fine_tune_base_model.predict(
                features_df, model=self.fine_tune_base_model_name).values
            features_df[self.fine_tune_extra_features[0]] = y_pred

        y_pred = self.model.predict(features_df, model=self.model_name).values

        if self.target_transform_to_del:
            y_pred_damage = self._compute_damage(y_pred)
            y_pred_damage_del = y_pred
        else:
            y_pred_damage = y_pred
            y_pred_damage_del = None

        if save_csv:
            self._add_pred_csv(df, y_pred_damage, y_pred_damage_del)

        return y_pred_damage

    def _save_artifacts(self, out_dir: str | Path) -> None:
        """Saves model artifacts to the specified output directory.

        Args:
            out_dir: Directory to save the artifacts.
        """

        if out_dir is None:
            out_dir = self.output_dir

        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "model_cfg_path": str(self.model_cfg_path),
            "model_name": self.model_name,
        }

        meta_path = os.path.join(out_dir, "artifacts_meta.json")
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file, indent=4)

    def generate_leaderboard(
        self,
        df_test: pd.DataFrame,
        detail_sections: list[str] | None = None,
        bootstrap: bool = True,
        n_bootstrap: int = 2000,
        bootstrap_seed: int = 42,
        bootstrap_alpha: float = 0.05,
        max_models: int | None = None,
    ) -> None:
        """Generates extended leaderboard CSVs with per-group/section metrics.

        Iterates over every model stored in the AutoGluon predictor and
        computes R², RMSE, MAE, MRE and bias for both DEL and damage
        targets, broken down by wind/wave groups and selected tower
        sections.

        Results are saved progressively after each model so that partial
        leaderboards are available even if the process is interrupted.

        Three CSVs are written to the model directory:
          - ``leaderboard_test_metrics.csv``  – global metrics.
          - ``leaderboard_test_groups.csv``   – per wind/wave group.
          - ``leaderboard_test_sections.csv`` – per tower section.

        Args:
            df_test: Test DataFrame containing the feature columns, the
                target column, and optionally ``wind_group``,
                ``wave_group``, ``wind_wave_group`` and
                ``section_name`` columns.
            detail_sections: Optional list of section names for which
                per-section metrics are computed. Ignored when
                ``section_name`` is absent from *df_test*.
            bootstrap: If True, compute bootstrap CIs (percentile
                method) for the global leaderboard metrics.
            n_bootstrap: Number of bootstrap resamples.
            bootstrap_seed: RNG seed for reproducible bootstrap sampling.
            bootstrap_alpha: Significance level; CI is (1 - alpha).
                Default 0.05 -> 95% CI.
            max_models: Dev flag — if set, only evaluate the first N
                models (smoke-test the pipeline without the full pass).
        """

        model_names = self.model.model_names()
        if max_models is not None:
            model_names = model_names[:max_models]
        model_dir = os.path.dirname(self.model_cfg_path)
        out_dir = os.path.join(model_dir, "leaderboard_test_summaries")
        os.makedirs(out_dir, exist_ok=True)
        logging.info("Generating leaderboard for %d models.", len(model_names))

        # Remove stale CSVs to avoid duplicates on re-runs
        for csv_name in [
                "leaderboard_test_metrics.csv",
                "leaderboard_test_groups.csv",
                "leaderboard_test_sections.csv",
        ]:
            csv_path = os.path.join(out_dir, csv_name)
            if os.path.exists(csv_path):
                os.remove(csv_path)

        y_true = df_test[self.target].astype(float).values
        y_true_del = self._compute_del(y_true)

        has_section = "section_name" in df_test.columns
        has_groups = any(col in df_test.columns
                         for col in ("wind_group", "wave_group",
                                     "wind_wave_group"))
        rows_metrics = []
        rows_groups = []
        rows_sections = []
        # Scalar point estimates + percentile errors
        point_keys = (
            "r2",
            "mse",
            "rmse",
            "mae",
            "mre",
            "bias",
            "rel_l2",
            "max_err",
            "p50_abs",
            "p90_abs",
            "p95_abs",
            "p99_abs",
            "p50_rel",
            "p90_rel",
            "p95_rel",
            "p99_rel",
        )
        # Bootstrap summaries: each metric yields ci_lo/ci_hi/boot_mean
        # /boot_std suffixed keys.
        boot_base = ("r2", "mse", "mae", "mre", "rmse", "bias", "rel_l2",
                     "max_err")
        boot_keys = tuple(f"{m}_{s}" for m in boot_base
                          for s in ("ci_lo", "ci_hi", "boot_mean", "boot_std"))
        keep = point_keys + boot_keys

        total = len(model_names)
        for idx, name in enumerate(model_names, 1):
            logging.info("  [%d/%d] %s ...", idx, total, name)
            try:
                features_df = df_test[self.features].copy()
                y_pred_del = self.model.predict(features_df, model=name).values
                y_pred = self._compute_damage(y_pred_del)
            except (ValueError, KeyError) as exc:
                logging.warning("  Skip %s: %s", name, exc)
                continue

            logging.info("    metrics...")
            m_del = self._compute_error_metrics(y_true_del,
                                                y_pred_del,
                                                bootstrap=bootstrap,
                                                n_bootstrap=n_bootstrap,
                                                seed=bootstrap_seed,
                                                alpha=bootstrap_alpha)
            m_dmg = self._compute_error_metrics(y_true,
                                                y_pred,
                                                bootstrap=bootstrap,
                                                n_bootstrap=n_bootstrap,
                                                seed=bootstrap_seed,
                                                alpha=bootstrap_alpha)

            rows_metrics.append({
                "model": name,
                **{
                    f"{k}_del": v for k, v in m_del.items() if k in keep
                },
                **{
                    f"{k}_damage": v for k, v in m_dmg.items() if k in keep
                },
            })

            # Keys reported per group/section (point estimates only).
            subset_keys = ("r2", "mse", "rmse", "mae", "mre", "bias", "rel_l2",
                           "max_err")

            # Groups
            g_row = None
            if has_groups:
                logging.info("    groups...")
                g_row = {"model": name}
                for k in subset_keys:
                    g_row[f"{k}_damage"] = m_dmg[k]
                    g_row[f"{k}_del"] = m_del[k]
                for col, pfx in [("wind_group", "wind"), ("wave_group", "wave"),
                                 ("wind_wave_group", "")]:
                    if col not in df_test.columns:
                        continue
                    for grp in sorted(df_test[col].unique()):
                        mask = (df_test[col] == grp).values
                        if mask.sum() == 0:
                            continue
                        gm_del = self._compute_error_metrics(
                            y_true_del[mask], y_pred_del[mask])
                        gm_dmg = self._compute_error_metrics(
                            y_true[mask], y_pred[mask])
                        short = _shorten(grp)
                        col_pfx = (f"{pfx}_{short}" if pfx else short)
                        for k in subset_keys:
                            g_row[f"{k}_damage_{col_pfx}"] = gm_dmg[k]
                            g_row[f"{k}_del_{col_pfx}"] = gm_del[k]
                rows_groups.append(g_row)

            # Sections
            logging.info("    sections...")
            s_row = None
            if has_section and detail_sections:
                s_row = {"model": name}
                for k in subset_keys:
                    s_row[f"{k}_damage"] = m_dmg[k]
                    s_row[f"{k}_del"] = m_del[k]
                for sec in detail_sections:
                    mask = (df_test["section_name"] == sec).values
                    if mask.sum() == 0:
                        continue
                    sm_del = self._compute_error_metrics(
                        y_true_del[mask], y_pred_del[mask])
                    sm_dmg = self._compute_error_metrics(
                        y_true[mask], y_pred[mask])
                    for k in subset_keys:
                        s_row[f"{k}_damage_{sec}"] = sm_dmg[k]
                        s_row[f"{k}_del_{sec}"] = sm_del[k]
                rows_sections.append(s_row)

            # Append progressively after each model
            _append_leaderboard(out_dir, rows_metrics[-1], g_row, s_row)

        # Re-sort all CSVs by r2_damage at the end
        _sort_leaderboard(out_dir)
        logging.info("Leaderboard saved to %s (%d models)", out_dir,
                     len(rows_metrics))


def _append_leaderboard(
    out_dir: str | Path,
    metric_row: dict,
    group_row: dict,
    section_row: dict | None,
) -> None:
    """Appends a single model's results to the leaderboard CSVs.

    Each CSV is written in append mode so that partial results survive
    process interruptions. Headers are written only when the file does
    not yet exist.

    Args:
        out_dir: Directory where CSVs are saved.
        metric_row: Dict with global metric columns for one model.
        group_row: Dict with per-group metric columns for one model.
        section_row: Dict with per-section metric columns for one
            model, or ``None`` if sections are not being tracked.
    """
    for fname, row in [
        ("leaderboard_test_metrics.csv", metric_row),
        ("leaderboard_test_groups.csv", group_row),
        ("leaderboard_test_sections.csv", section_row),
    ]:
        if row is None:
            continue
        path = os.path.join(out_dir, fname)
        header = not os.path.exists(path)
        pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def _sort_leaderboard(out_dir: str | Path) -> None:
    """Re-sorts all leaderboard CSVs by R² damage descending.

    Called once after all models are processed to produce the final
    ordered leaderboards.

    Args:
        out_dir: Directory containing the leaderboard CSVs.
    """
    for fname in [
            "leaderboard_test_metrics.csv",
            "leaderboard_test_groups.csv",
            "leaderboard_test_sections.csv",
    ]:
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            continue
        df_lb = pd.read_csv(path)
        df_lb = df_lb.sort_values("r2_damage", ascending=False)
        df_lb.to_csv(path, index=False)


def _shorten(name: str) -> str:
    """Shortens a wind/wave group name for use in CSV column headers.

    Replaces verbose regime labels with two-letter abbreviations so that
    column names stay compact in leaderboard CSVs.

    Args:
        name: Full group name (e.g. ``"In-train_Extrapolate"``).

    Returns:
        Abbreviated name (e.g. ``"IT_EX"``).
    """
    return (name.replace("In-train", "IT").replace("Interpolate", "IP").replace(
        "High-Interpolate",
        "HI").replace("Extrapolate", "EX").replace("High-Extrapolate", "HE"))
