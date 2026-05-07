# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-statements
# pylint: disable=too-many-function-args
# pylint: disable=too-many-lines
# pylint: disable=too-many-branches
"""Base damage predictor class for model-agnostic evaluation and plotting."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src import colors as colors_mod
from floatbench import plots
from floatbench.utils import (bootstrap_regression_metrics, resolve_group_order)


class BaseDamagePredictor(ABC):
    """Model-agnostic damage predictor evaluator.
    
    Attributes:
        features: List of feature column names.
        target: Target column name.
        output_dir: Directory to save plots and metrics.
        report_del: Whether to report DEL metrics and plots in addition to
          DAMAGE.
    """

    def __init__(
        self,
        features: List[str],
        target: str,
        output_dir: str | Path,
        report_del: bool = False,
    ) -> None:
        """Initializes the predictor.
        
        Args:
            features: List of feature column names.
            target: Target column name.
            output_dir: Directory to save plots and metrics.
            report_del: Whether to report DEL metrics and plots in addition to
              DAMAGE.
        """

        self.features = list(features)
        self.target = str(target)
        self.output_dir = str(output_dir or "./outputs/damage_predictor")
        os.makedirs(self.output_dir, exist_ok=True)
        self.report_del = bool(report_del)

    @abstractmethod
    def predict(self, df: pd.DataFrame, save_csv: bool = True) -> np.ndarray:
        """Return predicted DAMAGE (not DEL).
        
        Args:
            df: DataFrame with features and (optionally) target column.
            save_csv: Whether to save predictions in CSV format.
        
        Returns:
            Array with predicted DAMAGE values.
        """

    @abstractmethod
    def _load_model(self, model_path: str | Path = None):
        """Loads the model.
        
        Args:
            model_path: Path to the directory with the saved model.
        """

    @abstractmethod
    def _load_model_meta(self):
        """Loads the model metadata."""

    def _compute_del(self, y_damage: np.ndarray) -> np.ndarray:
        """Computes DEL from DAMAGE using a cubic relationship.
        
        Args:
            y_damage: Array with DAMAGE values.
        
        Returns:
            Array with DEL values.
        """
        return np.cbrt(y_damage.astype(float))

    def _compute_damage(self, y_del: np.ndarray) -> np.ndarray:
        """Computes DAMAGE from DEL using a cubic relationship.
        
        Args:
            y_del: Array with DEL values.
            
        Returns:
            Array with DAMAGE values.
        """
        return y_del**3

    def _compute_error_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bootstrap: bool = False,
        n_bootstrap: int = 2000,
        seed: int = 42,
        alpha: float = 0.05,
    ) -> dict:
        """Compute pointwise errors and aggregate metrics.

        Args:
            y_true: Array with true target values.
            y_pred: Array with predicted target values.
            bootstrap: If True, compute bootstrap CIs (percentile
              method) plus bootstrap mean/std for r2, mse, mae, mre,
              rmse, bias, rel_l2, max_err.
            n_bootstrap: Number of bootstrap resamples (B=2000 follows
              the convention in common practice for stable 95% CIs).
            seed: RNG seed for reproducible bootstrap sampling.
            alpha: Significance level; CI is (1 - alpha). Default 0.05
              gives a 95% CI using the 2.5 / 97.5 percentiles.

        Returns:
            Dict with:
            - err, abs_err, rel_err
            - r2, mre, mae, mse, rmse, bias, rel_l2, max_err
            - p50_abs, p90_abs, p95_abs, p99_abs (percentiles of abs_err)
            - p50_rel, p90_rel, p95_rel, p99_rel (percentiles of rel_err)
            - <metric>_ci_lo, <metric>_ci_hi, <metric>_boot_mean,
              <metric>_boot_std (only if bootstrap=True)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        err = y_pred - y_true
        abs_err = np.abs(err)
        rel_err = abs_err / np.abs(y_true) * 100.0

        # Note: r2_score needs >= 2 samples
        if y_true.size >= 2:
            r2 = float(r2_score(y_true, y_pred))
        else:
            r2 = float("nan")
        mre = float(np.mean(rel_err))
        mae = float(np.mean(abs_err))
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        bias = float(np.mean(err))
        max_err = float(np.max(abs_err)) if abs_err.size else float("nan")
        norm_true = float(np.linalg.norm(y_true))
        rel_l2 = (float(np.linalg.norm(y_pred - y_true)) /
                  norm_true if norm_true > 0 else float("nan"))

        # Percentile errors
        if abs_err.size:
            p_abs = np.percentile(abs_err, [50, 90, 95, 99])
            p_rel = np.percentile(rel_err, [50, 90, 95, 99])
        else:
            p_abs = np.full(4, np.nan)
            p_rel = np.full(4, np.nan)

        result = {
            "err": err,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "r2": r2,
            "mre": mre,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "bias": bias,
            "rel_l2": rel_l2,
            "max_err": max_err,
            "p50_abs": float(p_abs[0]),
            "p90_abs": float(p_abs[1]),
            "p95_abs": float(p_abs[2]),
            "p99_abs": float(p_abs[3]),
            "p50_rel": float(p_rel[0]),
            "p90_rel": float(p_rel[1]),
            "p95_rel": float(p_rel[2]),
            "p99_rel": float(p_rel[3]),
        }

        if bootstrap and y_true.size >= 2:
            result.update(
                bootstrap_regression_metrics(y_true,
                                             y_pred,
                                             n_bootstrap=n_bootstrap,
                                             alpha=alpha,
                                             seed=seed))

        return result

    def _add_pred_csv(self,
                      df: pd.DataFrame,
                      y_pred_damage: np.ndarray,
                      y_pred_damage_del: np.ndarray = None) -> None:
        """Adds predictions to a CSV file.
        
        Args:
            df: DataFrame with a "sim_id" column.
            y_pred_damage: Array with predicted DAMAGE values.
            y_pred_damage_del: Optional array with predicted DEL values.
        """

        out = {
            "sim_id": df["sim_id"].values,
            "predicted_damage": y_pred_damage,
        }
        if self.target in df.columns:
            out["damage"] = df[self.target].astype(float).values
        if self.report_del:
            out["predicted_damage_del"] = y_pred_damage_del
            if self.target in df.columns:
                out["damage_del"] = self._compute_del(
                    df[self.target].astype(float).values)
        for col in ("section_name", "wind_group", "wave_group",
                    "wind_wave_group"):
            if col in df.columns:
                out[col] = df[col].values

        predictions_df = pd.DataFrame(out)
        predictions_csv_path = os.path.join(self.output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)

    def _color_by_variants(
        self,
        df: pd.DataFrame,
        color_by_cols: List[str] = None,
    ) -> List[dict]:
        """Builds a list of (color_by, color_by_label) pairs.

        Always starts with plain (no colour), then one entry per column
        in *color_by_cols* that exists in *df*.

        Args:
            df: DataFrame that may contain the requested columns.
            color_by_cols: Column names to colour scatter plots by.

        Returns:
            List of dicts with keys ``color_by`` and ``color_by_label``.
        """
        variants = [{"color_by": None, "color_by_label": None, "subdir": None}]
        for col in (color_by_cols or []):
            if col in df.columns:
                variants.append({
                    "color_by": df[col].values,
                    "color_by_label": col.replace("_", " ").title(),
                    "subdir": os.path.join("colored", f"by_{col}"),
                })
        return variants

    def evaluate(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray = None,
        save_svg: bool = False,
        color_by_cols: List[str] = None,
    ) -> None:
        """Predicts, computes R² and saves diagnostic plots.
        Args:
            df: DataFrame with features and (optionally) target column.
            y_pred: Optional pre-computed predictions.
            save_svg: Whether to save plots in SVG format.
            color_by_cols: Columns to color scatter plots by.
        """
        # ground truth
        y_true = df[self.target].astype(float).values

        # predict
        if y_pred is None:
            y_pred = self.predict(df)

        # color variants: plain first, then one per requested column
        color_variants = self._color_by_variants(df, color_by_cols)

        # plots
        plot_dir = os.path.join(self.output_dir, "y_true_vs_y_pred")
        os.makedirs(plot_dir, exist_ok=True)
        # damage
        plot_dir_damage = os.path.join(plot_dir, "damage")
        os.makedirs(plot_dir_damage, exist_ok=True)
        for cv in color_variants:
            cv_dir = (os.path.join(plot_dir_damage, cv["subdir"])
                      if cv["subdir"] else plot_dir_damage)
            os.makedirs(cv_dir, exist_ok=True)
            plots.plot_y_true_vs_y_pred(y_true=y_true,
                                        y_pred=y_pred,
                                        color_by=cv["color_by"],
                                        color_by_label=cv["color_by_label"],
                                        plot_dir=cv_dir,
                                        save_svg=save_svg)
            plots.plot_signed_error_vs_y_true(
                y_true=y_true,
                y_pred=y_pred,
                color_by=cv["color_by"],
                color_by_label=cv["color_by_label"],
                plot_dir=cv_dir,
                save_svg=save_svg)
        plots.plot_relative_error_hist(y_true=y_true,
                                       y_pred=y_pred,
                                       plot_dir=plot_dir_damage,
                                       save_svg=save_svg)
        # del
        if self.report_del:
            y_true_del = self._compute_del(y_true)
            y_pred_del = self._compute_del(y_pred)

            plot_dir_del = os.path.join(plot_dir, "DEL")
            os.makedirs(plot_dir_del, exist_ok=True)
            for cv in color_variants:
                cv_dir = (os.path.join(plot_dir_del, cv["subdir"])
                          if cv["subdir"] else plot_dir_del)
                os.makedirs(cv_dir, exist_ok=True)
                plots.plot_y_true_vs_y_pred(y_true=y_true_del,
                                            y_pred=y_pred_del,
                                            color_by=cv["color_by"],
                                            color_by_label=cv["color_by_label"],
                                            plot_dir=cv_dir,
                                            save_svg=save_svg,
                                            label="DEL")
                plots.plot_signed_error_vs_y_true(
                    y_true=y_true_del,
                    y_pred=y_pred_del,
                    color_by=cv["color_by"],
                    color_by_label=cv["color_by_label"],
                    plot_dir=cv_dir,
                    save_svg=save_svg,
                    label="DEL")
            plots.plot_relative_error_hist(y_true=y_true_del,
                                           y_pred=y_pred_del,
                                           plot_dir=plot_dir_del,
                                           save_svg=save_svg,
                                           label="DEL")

        # metrics summary
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # damage
        metrics_dict = self._compute_error_metrics(y_true, y_pred)
        summary = {
            "r2": metrics_dict["r2"],
            "mre": metrics_dict["mre"],
            "mae": metrics_dict["mae"],
            "mse": metrics_dict["mse"],
            "rmse": metrics_dict["rmse"],
            "bias": metrics_dict["bias"],
        }
        # del
        if self.report_del:
            metrics_dict_del = self._compute_error_metrics(
                y_true_del, y_pred_del)
            summary.update({
                "r2_del": metrics_dict_del["r2"],
                "mre_del": metrics_dict_del["mre"],
                "mae_del": metrics_dict_del["mae"],
                "mse_del": metrics_dict_del["mse"],
                "rmse_del": metrics_dict_del["rmse"],
                "bias_del": metrics_dict_del["bias"],
            })

        # metrics by sim_id
        predictions_df = pd.DataFrame({
            "sim_id": df["sim_id"].values,
            "true_damage": y_true,
            "predicted_damage": y_pred,
            "mre": metrics_dict["rel_err"],
            "mae": metrics_dict["abs_err"],
        })
        predictions_csv_path = os.path.join(metrics_dir, "by_simid.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)

        with open(os.path.join(metrics_dir, "summary.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    def evaluate_by_section(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray = None,
        save_svg: bool = False,
    ) -> None:
        """Predicts, computes R² and saves diagnostic plots.
        Args:
            df: DataFrame with features and (optionally) target column.
            y_pred: Optional pre-computed predictions.
            save_svg: Whether to save plots in SVG format.
        """
        sections = df["section_name"].values

        # ground truth
        y_true = df[self.target].astype(float).values

        # predict
        if y_pred is None:
            y_pred = self.predict(df)

        # plots
        plot_dir = os.path.join(self.output_dir, "y_true_vs_y_pred",
                                "by_section")
        os.makedirs(plot_dir, exist_ok=True)
        # damage
        plot_dir_damage = os.path.join(plot_dir, "damage")
        os.makedirs(plot_dir_damage, exist_ok=True)
        plots.plot_y_true_vs_y_pred_by_section(y_true=y_true,
                                               y_pred=y_pred,
                                               section_labels=sections,
                                               plot_dir=plot_dir_damage,
                                               save_svg=save_svg)
        plots.plot_signed_error_vs_y_true_by_section(y_true=y_true,
                                                     y_pred=y_pred,
                                                     section_labels=sections,
                                                     plot_dir=plot_dir_damage,
                                                     save_svg=save_svg)
        plots.plot_relative_error_hist_by_section(y_true=y_true,
                                                  y_pred=y_pred,
                                                  section_labels=sections,
                                                  plot_dir=plot_dir_damage,
                                                  save_svg=save_svg)
        # del
        if self.report_del:
            y_true_del = self._compute_del(y_true)
            y_pred_del = self._compute_del(y_pred)

            plot_dir_del = os.path.join(plot_dir, "DEL")
            os.makedirs(plot_dir_del, exist_ok=True)

            plots.plot_y_true_vs_y_pred_by_section(y_true=y_true_del,
                                                   y_pred=y_pred_del,
                                                   section_labels=sections,
                                                   plot_dir=plot_dir_del,
                                                   save_svg=save_svg,
                                                   label="DEL")
            plots.plot_signed_error_vs_y_true_by_section(
                y_true=y_true_del,
                y_pred=y_pred_del,
                section_labels=sections,
                plot_dir=plot_dir_del,
                save_svg=save_svg,
                label="DEL")
            plots.plot_relative_error_hist_by_section(y_true=y_true_del,
                                                      y_pred=y_pred_del,
                                                      section_labels=sections,
                                                      plot_dir=plot_dir_del,
                                                      save_svg=save_svg,
                                                      label="DEL")

        # summary metrics by section
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_dict = self._compute_error_metrics(y_true, y_pred)
        df_eval = pd.DataFrame({
            "section_name": sections,
            "true_damage": y_true,
            "predicted_damage": y_pred,
            "mre": metrics_dict["rel_err"],
            "mae": metrics_dict["abs_err"],
        })

        rows = []

        unique_sections = sorted(
            df_eval["section_name"].unique(),
            key=plots.sec_key,
        )
        for sec in unique_sections:
            g = df_eval[df_eval["section_name"] == sec]
            y_true_sec = g["true_damage"].values
            y_pred_sec = g["predicted_damage"].values

            if len(y_true_sec) == 0:
                continue

            # damage
            metrics_dict_sec = self._compute_error_metrics(
                y_true_sec, y_pred_sec)
            row = {
                "section_name": sec,
                "r2": metrics_dict_sec["r2"],
                "mre": metrics_dict_sec["mre"],
                "mae": metrics_dict_sec["mae"],
                "mse": metrics_dict_sec["mse"],
                "rmse": metrics_dict_sec["rmse"],
                "bias": metrics_dict_sec["bias"],
            }

            # del
            if self.report_del:
                y_true_sec_del = self._compute_del(y_true_sec)
                y_pred_sec_del = self._compute_del(y_pred_sec)

                metrics_dict_del_sec = self._compute_error_metrics(
                    y_true_sec_del, y_pred_sec_del)
                row.update({
                    "r2_del": metrics_dict_del_sec["r2"],
                    "mre_del": metrics_dict_del_sec["mre"],
                    "mae_del": metrics_dict_del_sec["mae"],
                    "mse_del": metrics_dict_del_sec["mse"],
                    "rmse_del": metrics_dict_del_sec["rmse"],
                    "bias_del": metrics_dict_del_sec["bias"],
                })

            row.update({"n_samples": len(g)})
            rows.append(row)

        metrics_by_section_df = pd.DataFrame(rows)
        metrics_by_section_path = os.path.join(metrics_dir, "by_section.csv")
        metrics_by_section_df.to_csv(metrics_by_section_path, index=False)

    def evaluate_by_section_group(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray = None,
        save_svg: bool = False,
        color_by_cols: List[str] = None,
    ) -> None:
        """Section-group scatter plots (e.g. sections 1-5, 6-10, ...).

        Always produces a plain (no-color) version. If *color_by_cols*
        is given, also produces one coloured version per column.

        Args:
            df: DataFrame with features and target column.
            y_pred: Optional pre-computed predictions.
            save_svg: Whether to save SVG plots.
            color_by_cols: Columns to color scatter plots by.
        """
        sections = df["section_name"].values
        y_true = df[self.target].astype(float).values

        if y_pred is None:
            y_pred = self.predict(df)

        color_variants = self._color_by_variants(df, color_by_cols)

        plot_dir = os.path.join(self.output_dir, "y_true_vs_y_pred",
                                "by_section_group")
        os.makedirs(plot_dir, exist_ok=True)

        # damage
        plot_dir_damage = os.path.join(plot_dir, "damage")
        os.makedirs(plot_dir_damage, exist_ok=True)
        for cv in color_variants:
            cv_dir = (os.path.join(plot_dir_damage, cv["subdir"])
                      if cv["subdir"] else plot_dir_damage)
            os.makedirs(cv_dir, exist_ok=True)
            plots.plot_y_true_vs_y_pred_by_section_group(
                y_true=y_true,
                y_pred=y_pred,
                section_labels=sections,
                plot_dir=cv_dir,
                save_svg=save_svg,
                color_by=cv["color_by"],
                color_by_label=cv["color_by_label"])

        # signed error vs variable (damage)
        for cv in color_variants:
            if cv["color_by"] is None:
                continue
            bias_dir = os.path.join(plot_dir_damage, cv["subdir"])
            os.makedirs(bias_dir, exist_ok=True)
            plots.plot_signed_error_vs_var_by_section_group(
                y_true=y_true,
                y_pred=y_pred,
                section_labels=sections,
                x_values=cv["color_by"],
                x_label=cv["color_by_label"],
                plot_dir=bias_dir,
                save_svg=save_svg)

        # del
        if self.report_del:
            y_true_del = self._compute_del(y_true)
            y_pred_del = self._compute_del(y_pred)

            plot_dir_del = os.path.join(plot_dir, "DEL")
            os.makedirs(plot_dir_del, exist_ok=True)
            for cv in color_variants:
                cv_dir = (os.path.join(plot_dir_del, cv["subdir"])
                          if cv["subdir"] else plot_dir_del)
                os.makedirs(cv_dir, exist_ok=True)
                plots.plot_y_true_vs_y_pred_by_section_group(
                    y_true=y_true_del,
                    y_pred=y_pred_del,
                    section_labels=sections,
                    plot_dir=cv_dir,
                    save_svg=save_svg,
                    label="DEL",
                    color_by=cv["color_by"],
                    color_by_label=cv["color_by_label"])

    def evaluate_by_section_selected(
        self,
        df: pd.DataFrame,
        sections: List[str],
        y_pred: np.ndarray = None,
        save_svg: bool = False,
        color_by_cols: List[str] = None,
    ) -> None:
        """Scatter plots for a hand-picked list of sections.

        Produces one subplot per section (ncols=3, wraps to next row).
        Always produces a plain version; if *color_by_cols* is given,
        also one coloured version per column.

        Args:
            df: DataFrame with features and target column.
            sections: List of section names to plot.
            y_pred: Optional pre-computed predictions.
            save_svg: Whether to save SVG plots.
            color_by_cols: Columns to color scatter plots by.
        """
        section_labels = df["section_name"].values
        y_true = df[self.target].astype(float).values

        if y_pred is None:
            y_pred = self.predict(df)

        # filter to selected sections
        mask = np.isin(section_labels, sections)
        y_true_sel = y_true[mask]
        y_pred_sel = y_pred[mask]
        sec_sel = section_labels[mask]

        color_variants = self._color_by_variants(df, color_by_cols)

        plot_dir = os.path.join(self.output_dir, "y_true_vs_y_pred",
                                "by_section_selected")
        os.makedirs(plot_dir, exist_ok=True)

        # damage
        plot_dir_damage = os.path.join(plot_dir, "damage")
        os.makedirs(plot_dir_damage, exist_ok=True)
        for cv in color_variants:
            cv_dir = (os.path.join(plot_dir_damage, cv["subdir"])
                      if cv["subdir"] else plot_dir_damage)
            os.makedirs(cv_dir, exist_ok=True)
            cb = (cv["color_by"][mask] if cv["color_by"] is not None else None)
            plots.plot_y_true_vs_y_pred_by_section(
                y_true=y_true_sel,
                y_pred=y_pred_sel,
                section_labels=sec_sel,
                plot_dir=cv_dir,
                save_svg=save_svg,
                ncols=3,
                color_by=cb,
                color_by_label=cv["color_by_label"])

        # signed error vs variable (damage)
        for cv in color_variants:
            if cv["color_by"] is None:
                continue
            bias_dir = os.path.join(plot_dir_damage, cv["subdir"])
            os.makedirs(bias_dir, exist_ok=True)
            cb = cv["color_by"][mask]
            plots.plot_signed_error_vs_var_by_section(
                y_true=y_true_sel,
                y_pred=y_pred_sel,
                section_labels=sec_sel,
                x_values=cb,
                x_label=cv["color_by_label"],
                plot_dir=bias_dir,
                save_svg=save_svg,
                ncols=3)

        # del
        if self.report_del:
            y_true_del = self._compute_del(y_true_sel)
            y_pred_del = self._compute_del(y_pred_sel)

            plot_dir_del = os.path.join(plot_dir, "DEL")
            os.makedirs(plot_dir_del, exist_ok=True)
            for cv in color_variants:
                cv_dir = (os.path.join(plot_dir_del, cv["subdir"])
                          if cv["subdir"] else plot_dir_del)
                os.makedirs(cv_dir, exist_ok=True)
                cb = (cv["color_by"][mask]
                      if cv["color_by"] is not None else None)
                plots.plot_y_true_vs_y_pred_by_section(
                    y_true=y_true_del,
                    y_pred=y_pred_del,
                    section_labels=sec_sel,
                    plot_dir=cv_dir,
                    save_svg=save_svg,
                    ncols=3,
                    label="DEL",
                    color_by=cb,
                    color_by_label=cv["color_by_label"])

    def evaluate_cumulative_damage_by_section(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray = None,
        save_svg: bool = False,
    ) -> None:
        """Predicts and saves cumulative damage plots by section.
        
        Args:
            df: DataFrame with features and (optionally) target column.
            y_pred: Optional pre-computed predictions.
            save_svg: Whether to save plots in SVG format.        
        """
        # ground truth
        y_true = df[self.target].astype(float).values

        # predict
        if y_pred is None:
            y_pred = self.predict(df)

        # plots
        plot_dir = os.path.join(self.output_dir, "cumulative_damage_by_section")
        os.makedirs(plot_dir, exist_ok=True)

        # damage
        plot_dir_damage = os.path.join(plot_dir, "damage")
        os.makedirs(plot_dir_damage, exist_ok=True)
        # cumulative damage mode by section
        plots.plot_cumulative_damage_by_section(
            y_true=y_true,
            y_pred=y_pred,
            section_labels=df["section_name"],
            plot_dir=plot_dir_damage,
            mode="cumulative",
            normalize_cumulative=True,
            save_svg=save_svg)
        # relative damage error mode by section
        plots.plot_cumulative_damage_by_section(
            y_true=y_true,
            y_pred=y_pred,
            section_labels=df["section_name"],
            plot_dir=plot_dir_damage,
            mode="error",
            error_mode="abs",
            save_svg=save_svg)

        # del
        if self.report_del:
            plot_dir_del = os.path.join(plot_dir, "DEL")
            os.makedirs(plot_dir_del, exist_ok=True)
            # cumulative DEL damage by section
            plots.plot_cumulative_damage_by_section(
                y_true=y_true,
                y_pred=y_pred,
                section_labels=df["section_name"],
                plot_dir=plot_dir_del,
                mode="cumulative",
                normalize_cumulative=True,
                damage_type="DEL",
                save_svg=save_svg)
            # relative DEL damage error mode by section
            plots.plot_cumulative_damage_by_section(
                y_true=y_true,
                y_pred=y_pred,
                section_labels=df["section_name"],
                plot_dir=plot_dir_del,
                mode="error",
                error_mode="abs",
                damage_type="DEL",
                save_svg=save_svg)

        # metrics
        metrics_dir = os.path.join(self.output_dir, "metrics",
                                   "cumulative_damage")
        os.makedirs(metrics_dir, exist_ok=True)

        metrics = []

        unique_sections = sorted(
            df["section_name"].unique(),
            key=plots.sec_key,
        )
        for sec in unique_sections:
            mask = df["section_name"] == sec
            y_true_sec = y_true[mask]
            y_pred_sec = y_pred[mask]

            # damage metrics
            cum_true_sec = np.cumsum(y_true_sec)
            cum_pred_sec = np.cumsum(y_pred_sec)

            if len(y_true_sec) == 0:
                continue

            metrics_dict_sec = self._compute_error_metrics(
                cum_true_sec, cum_pred_sec)
            final_abs_err = float(np.abs(cum_pred_sec[-1] - cum_true_sec[-1]))
            final_rel_err = float(final_abs_err / np.abs(cum_true_sec[-1]) *
                                  100.0)

            summary_sec = {
                "section_name": sec,
                "r2": metrics_dict_sec["r2"],
                "damage_final_mae": final_abs_err,
                "damage_final_mre": final_rel_err,
            }

            # del metrics
            if self.report_del:
                cum_true_true_del = self._compute_del(cum_true_sec)
                cum_true_pred_del = self._compute_del(cum_pred_sec)

                metrics_dict_del_sec = self._compute_error_metrics(
                    cum_true_true_del, cum_true_pred_del)
                final_abs_err_del = float(
                    np.abs(cum_true_pred_del[-1] - cum_true_true_del[-1]))
                final_rel_err_del = float(final_abs_err_del /
                                          abs(cum_true_true_del[-1]) * 100)
                summary_sec.update({
                    "r2_del": metrics_dict_del_sec["r2"],
                    "DEL_final_mae": final_abs_err_del,
                    "DEL_final_mre": final_rel_err_del,
                })
            metrics.append(summary_sec)

        metrics_df = pd.DataFrame(metrics)
        metrics_path = os.path.join(metrics_dir, "by_section.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # summary
        summary = {
            "damage": {
                "mean_final_mae": float(metrics_df["damage_final_mae"].mean()),
                "mean_final_mre": float(metrics_df["damage_final_mre"].mean()),
            }
        }
        if self.report_del:
            summary.update({
                "DEL": {
                    "mean_final_mae": float(metrics_df["DEL_final_mae"].mean()),
                    "mean_final_mre": float(metrics_df["DEL_final_mre"].mean()),
                }
            })

        with open(os.path.join(metrics_dir, "summary.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # tower profile of cumulative error
        heights = np.array([
            df.loc[df["section_name"] == sec, "section_height_m"].iloc[0]
            for sec in unique_sections
        ])
        del_mre = (metrics_df["DEL_final_mre"].values
                   if self.report_del else None)
        plots.plot_error_profile(
            heights=heights,
            damage_mre=metrics_df["damage_final_mre"].values,
            del_mre=del_mre,
            plot_dir=plot_dir,
            xlabel="Final Cumulative MRE [%]",
            save_svg=save_svg,
        )

    def evaluate_by_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        group_order: Optional[List[str]] = None,
        y_pred: np.ndarray = None,
        save_svg: bool = False,
        plot_global: bool = True,
        plot_split_by_group: bool = True,
        color_by_cols: List[str] = None,
    ) -> None:
        """Predicts and computes R² by group, saving plots.

        Args:
            df: DataFrame with features and (optionally) target column.
            group_col: Column name to group by.
            y_pred: Optional pre-computed predictions.
            group_order: Optional list defining the order of groups in plots.
            save_svg: Whether to save plots in SVG format.
            plot_global: Whether to plot global metrics across all groups.
            plot_split_by_group: Whether to plot metrics split by group.
            color_by_cols: Columns to color scatter plots by.
        """
        # ground truth
        y_true = df[self.target].astype(float).values

        # predict
        if y_pred is None:
            y_pred = self.predict(df)

        # color variants: plain first, then one per requested column
        color_variants = self._color_by_variants(df, color_by_cols)

        # groups
        groups = df[group_col].values

        plot_dir = os.path.join(self.output_dir, "y_true_vs_y_pred",
                                f"by_{group_col}")
        os.makedirs(plot_dir, exist_ok=True)

        # infer distance column from group column
        dist_col = group_col.replace("_group", "_dist")
        has_dist = dist_col in df.columns

        # damage
        plot_dir_damage = os.path.join(plot_dir, "damage")
        os.makedirs(plot_dir_damage, exist_ok=True)

        # plots – global
        if plot_global:
            for cv in color_variants:
                cv_dir = (os.path.join(plot_dir_damage, cv["subdir"])
                          if cv["subdir"] else plot_dir_damage)
                os.makedirs(cv_dir, exist_ok=True)
                plots.plot_y_true_vs_y_pred(y_true=y_true,
                                            y_pred=y_pred,
                                            groups=groups,
                                            group_col=group_col,
                                            group_order=group_order,
                                            color_by=cv["color_by"],
                                            color_by_label=cv["color_by_label"],
                                            plot_dir=cv_dir,
                                            save_svg=save_svg)
                plots.plot_signed_error_vs_y_true(
                    y_true=y_true,
                    y_pred=y_pred,
                    groups=groups,
                    group_col=group_col,
                    group_order=group_order,
                    color_by=cv["color_by"],
                    color_by_label=cv["color_by_label"],
                    plot_dir=cv_dir,
                    save_svg=save_svg)
            plots.plot_relative_error_hist(y_true=y_true,
                                           y_pred=y_pred,
                                           groups=groups,
                                           group_col=group_col,
                                           group_order=group_order,
                                           plot_dir=plot_dir_damage,
                                           save_svg=save_svg)
        # plots – split by group
        if plot_split_by_group:
            for cv in color_variants:
                cv_dir = (os.path.join(plot_dir_damage, cv["subdir"])
                          if cv["subdir"] else plot_dir_damage)
                os.makedirs(cv_dir, exist_ok=True)
                plots.plot_y_true_vs_y_pred(y_true=y_true,
                                            y_pred=y_pred,
                                            groups=groups,
                                            group_col=group_col,
                                            group_order=group_order,
                                            color_by=cv["color_by"],
                                            color_by_label=cv["color_by_label"],
                                            plot_dir=cv_dir,
                                            split_by_group=True,
                                            save_svg=save_svg)
                plots.plot_signed_error_vs_y_true(
                    y_true=y_true,
                    y_pred=y_pred,
                    groups=groups,
                    group_col=group_col,
                    group_order=group_order,
                    color_by=cv["color_by"],
                    color_by_label=cv["color_by_label"],
                    plot_dir=cv_dir,
                    split_by_group=True,
                    save_svg=save_svg)
            plots.plot_relative_error_hist(y_true=y_true,
                                           y_pred=y_pred,
                                           groups=groups,
                                           group_col=group_col,
                                           group_order=group_order,
                                           plot_dir=plot_dir_damage,
                                           split_by_group=True,
                                           save_svg=save_svg)

        # plots – signed error vs distance
        if has_dist:
            distances = df[dist_col].values
            dist_label = dist_col.replace("_", " ").title()
            dist_dir = os.path.join(plot_dir_damage, "dist")
            for cv in color_variants:
                cv_dir = (os.path.join(dist_dir, cv["subdir"])
                          if cv["subdir"] else dist_dir)
                os.makedirs(cv_dir, exist_ok=True)
                plots.plot_signed_error_vs_distance(
                    y_true=y_true,
                    y_pred=y_pred,
                    distances=distances,
                    groups=groups,
                    group_col=group_col,
                    group_order=group_order,
                    color_by=cv["color_by"],
                    color_by_label=cv["color_by_label"],
                    plot_dir=cv_dir,
                    save_svg=save_svg,
                    distance_label=dist_label)

        # del
        if self.report_del:
            y_true_del = self._compute_del(y_true)
            y_pred_del = self._compute_del(y_pred)

            plot_dir_del = os.path.join(plot_dir, "DEL")
            os.makedirs(plot_dir_del, exist_ok=True)

            # plots – global
            if plot_global:
                for cv in color_variants:
                    cv_dir = (os.path.join(plot_dir_del, cv["subdir"])
                              if cv["subdir"] else plot_dir_del)
                    os.makedirs(cv_dir, exist_ok=True)
                    plots.plot_y_true_vs_y_pred(
                        y_true=y_true_del,
                        y_pred=y_pred_del,
                        groups=groups,
                        group_col=group_col,
                        group_order=group_order,
                        color_by=cv["color_by"],
                        color_by_label=cv["color_by_label"],
                        plot_dir=cv_dir,
                        save_svg=save_svg,
                        label="DEL")
                    plots.plot_signed_error_vs_y_true(
                        y_true=y_true_del,
                        y_pred=y_pred_del,
                        groups=groups,
                        group_col=group_col,
                        group_order=group_order,
                        color_by=cv["color_by"],
                        color_by_label=cv["color_by_label"],
                        plot_dir=cv_dir,
                        save_svg=save_svg,
                        label="DEL")
                plots.plot_relative_error_hist(y_true=y_true_del,
                                               y_pred=y_pred_del,
                                               groups=groups,
                                               group_col=group_col,
                                               group_order=group_order,
                                               plot_dir=plot_dir_del,
                                               save_svg=save_svg,
                                               label="DEL")

            # plots – split by group
            if plot_split_by_group:
                for cv in color_variants:
                    cv_dir = (os.path.join(plot_dir_del, cv["subdir"])
                              if cv["subdir"] else plot_dir_del)
                    os.makedirs(cv_dir, exist_ok=True)
                    plots.plot_y_true_vs_y_pred(
                        y_true=y_true_del,
                        y_pred=y_pred_del,
                        groups=groups,
                        group_col=group_col,
                        group_order=group_order,
                        color_by=cv["color_by"],
                        color_by_label=cv["color_by_label"],
                        plot_dir=cv_dir,
                        split_by_group=True,
                        save_svg=save_svg,
                        label="DEL")
                    plots.plot_signed_error_vs_y_true(
                        y_true=y_true_del,
                        y_pred=y_pred_del,
                        groups=groups,
                        group_col=group_col,
                        group_order=group_order,
                        color_by=cv["color_by"],
                        color_by_label=cv["color_by_label"],
                        plot_dir=cv_dir,
                        split_by_group=True,
                        save_svg=save_svg,
                        label="DEL")
                plots.plot_relative_error_hist(y_true=y_true_del,
                                               y_pred=y_pred_del,
                                               groups=groups,
                                               group_col=group_col,
                                               group_order=group_order,
                                               plot_dir=plot_dir_del,
                                               split_by_group=True,
                                               save_svg=save_svg,
                                               label="DEL")

            # plots – signed error vs distance (DEL)
            if has_dist:
                dist_dir_del = os.path.join(plot_dir_del, "dist")
                for cv in color_variants:
                    cv_dir = (os.path.join(dist_dir_del, cv["subdir"])
                              if cv["subdir"] else dist_dir_del)
                    os.makedirs(cv_dir, exist_ok=True)
                    plots.plot_signed_error_vs_distance(
                        y_true=y_true_del,
                        y_pred=y_pred_del,
                        distances=distances,
                        groups=groups,
                        group_col=group_col,
                        group_order=group_order,
                        color_by=cv["color_by"],
                        color_by_label=cv["color_by_label"],
                        plot_dir=cv_dir,
                        save_svg=save_svg,
                        label="DEL",
                        distance_label=dist_label)

        # metrics summary by group
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_dict = self._compute_error_metrics(y_true, y_pred)
        df_eval = pd.DataFrame({
            group_col: groups,
            "true_damage": y_true,
            "predicted_damage": y_pred,
            "mre": metrics_dict["rel_err"],
            "mae": metrics_dict["abs_err"],
        })

        unique_groups = resolve_group_order(df_eval[group_col].unique(),
                                            group_order)

        rows = []
        for g in unique_groups:
            sub = df_eval[df_eval[group_col] == g]

            y_true_grp = sub["true_damage"].values
            y_pred_grp = sub["predicted_damage"].values

            if len(y_true_grp) == 0:
                continue

            # damage
            metrics_dict_grp = self._compute_error_metrics(
                y_true_grp, y_pred_grp)
            row = {
                "group": g,
                "r2": metrics_dict_grp["r2"],
                "mre": metrics_dict_grp["mre"],
                "mae": metrics_dict_grp["mae"],
                "mse": metrics_dict_grp["mse"],
                "rmse": metrics_dict_grp["rmse"],
                "bias": metrics_dict_grp["bias"],
            }

            # del
            if self.report_del:
                y_true_del_grp = self._compute_del(y_true_grp)
                y_pred_del_grp = self._compute_del(y_pred_grp)

                metrics_dict_del_grp = self._compute_error_metrics(
                    y_true_del_grp, y_pred_del_grp)
                row.update({
                    "r2_del": metrics_dict_del_grp["r2"],
                    "mre_del": metrics_dict_del_grp["mre"],
                    "mae_del": metrics_dict_del_grp["mae"],
                    "mse_del": metrics_dict_del_grp["mse"],
                    "rmse_del": metrics_dict_del_grp["rmse"],
                    "bias_del": metrics_dict_del_grp["bias"],
                })

            row.update({"n_samples": len(sub)})
            rows.append(row)

        metrics_by_group_df = pd.DataFrame(rows)
        metrics_by_group_path = os.path.join(
            metrics_dir,
            f"by_{group_col}.csv",
        )
        metrics_by_group_df.to_csv(metrics_by_group_path, index=False)

    def evaluate_by_section_within_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        group_order: Optional[List[str]] = None,
        y_pred: np.ndarray = None,
        save_svg: bool = False,
    ) -> None:
        """By-section plots for each domain group.

        Args:
            df: DataFrame with features and target column.
            group_col: Column defining groups (e.g. wind_group).
            group_order: Optional order of groups.
            y_pred: Optional pre-computed predictions.
            save_svg: Whether to save SVG plots.
        """
        y_true = df[self.target].astype(float).values

        if y_pred is None:
            y_pred = self.predict(df)

        sections = df["section_name"].values
        groups = df[group_col].values
        unique_groups = resolve_group_order(np.unique(groups), group_order)

        base_dir = os.path.join(self.output_dir, "y_true_vs_y_pred",
                                f"by_{group_col}")

        # group colors
        blue = colors_mod.COLORS_DICT["blue_paper"]
        grey = colors_mod.COLORS_DICT["grey_paper"]
        red = colors_mod.COLORS_DICT["red_paper"]
        base_colors = [blue, grey, red]
        if group_order is not None and len(group_order) == 9:
            base_colors = [
                blue,
                colors_mod.mix_colors(blue, grey),
                colors_mod.mix_colors(blue, red),
                colors_mod.mix_colors(grey, blue),
                grey,
                colors_mod.mix_colors(grey, red),
                colors_mod.mix_colors(red, blue),
                colors_mod.mix_colors(red, grey),
                red,
            ]

        for i, g in enumerate(unique_groups):
            mask = groups == g
            g_y_true = y_true[mask]
            g_y_pred = y_pred[mask]
            g_sections = sections[mask]
            g_color = base_colors[i % len(base_colors)]
            g_title = f"{g} ({group_col})"

            g_slug = str(g).lower().replace("-", "_")
            grp_dir = os.path.join(base_dir, "by_section", g_slug)

            # damage
            plot_dir_damage = os.path.join(grp_dir, "damage")
            os.makedirs(plot_dir_damage, exist_ok=True)
            plots.plot_y_true_vs_y_pred_by_section(y_true=g_y_true,
                                                   y_pred=g_y_pred,
                                                   section_labels=g_sections,
                                                   plot_dir=plot_dir_damage,
                                                   save_svg=save_svg,
                                                   color=g_color,
                                                   title_suffix=g_title)
            plots.plot_signed_error_vs_y_true_by_section(
                y_true=g_y_true,
                y_pred=g_y_pred,
                section_labels=g_sections,
                plot_dir=plot_dir_damage,
                save_svg=save_svg,
                color=g_color,
                title_suffix=g_title)
            plots.plot_relative_error_hist_by_section(y_true=g_y_true,
                                                      y_pred=g_y_pred,
                                                      section_labels=g_sections,
                                                      plot_dir=plot_dir_damage,
                                                      save_svg=save_svg,
                                                      color=g_color,
                                                      title_suffix=g_title)

            # del
            if self.report_del:
                g_y_true_del = self._compute_del(g_y_true)
                g_y_pred_del = self._compute_del(g_y_pred)

                plot_dir_del = os.path.join(grp_dir, "DEL")
                os.makedirs(plot_dir_del, exist_ok=True)
                plots.plot_y_true_vs_y_pred_by_section(
                    y_true=g_y_true_del,
                    y_pred=g_y_pred_del,
                    section_labels=g_sections,
                    plot_dir=plot_dir_del,
                    save_svg=save_svg,
                    label="DEL",
                    color=g_color,
                    title_suffix=g_title)
                plots.plot_signed_error_vs_y_true_by_section(
                    y_true=g_y_true_del,
                    y_pred=g_y_pred_del,
                    section_labels=g_sections,
                    plot_dir=plot_dir_del,
                    save_svg=save_svg,
                    label="DEL",
                    color=g_color,
                    title_suffix=g_title)
                plots.plot_relative_error_hist_by_section(
                    y_true=g_y_true_del,
                    y_pred=g_y_pred_del,
                    section_labels=g_sections,
                    plot_dir=plot_dir_del,
                    save_svg=save_svg,
                    label="DEL",
                    color=g_color,
                    title_suffix=g_title)

    def evaluate_tower_damage_profile_vs_reference(
        self,
        df_test: pd.DataFrame,
        df_train: pd.DataFrame,
        df_reference_damage_profile: pd.DataFrame,
        y_pred_test: np.ndarray = None,
        save_svg: bool = False,
    ) -> None:
        """Evaluates tower damage profiles against a reference profile.
        
        Args:
            df_test: DataFrame with test set predictions.
            df_train: DataFrame with training set predictions.
            df_reference_damage_profile: DataFrame with reference damage
              profile.
            y_pred_test: Optional pre-computed test predictions.
            save_svg: Whether to save plots in SVG format.
        """

        def section_key(name: str) -> int:
            """Extracts numeric section ID from section name.
            
            Args:
                name: Section name string (e.g., "section_1").
            
            Returns:
                Integer section ID.
            """
            return int(re.search(r"\d+", name).group())

        def profile_by_section(
            df: pd.DataFrame,
            damage_values: np.ndarray,
            section_col: str = "section_name",
            weight_col: str = "damage_weight",
        ) -> pd.DataFrame:
            """Computes weighted damage profile by section.
            
            Args:
                df: DataFrame with section and weight columns.
                damage_values: Array with damage values.
                section_col: Name of the section column.
                weight_col: Name of the weight column.
            
            Returns:
                DataFrame with columns [section_name, damage_total].
            """
            tmp = df[[section_col, weight_col]].copy()
            tmp["damage_weighted"] = damage_values.astype(
                float) * tmp[weight_col].astype(float).values

            out = (tmp.groupby(
                section_col,
                as_index=False).agg(damage_total=("damage_weighted", "sum")))

            out = (out.assign(section_id=out[section_col].apply(
                section_key)).sort_values("section_id").drop(
                    columns="section_id").reset_index(drop=True))
            return out

        # Inputs / predictions
        y_true_train = df_train[self.target].astype(float).values

        if y_pred_test is None:
            y_pred_test = self.predict(df_test)

        # Build section profiles
        train_damage_profile = profile_by_section(
            df=df_train,
            damage_values=y_true_train,
        )
        pred_damage_profile = profile_by_section(
            df=df_test,
            damage_values=y_pred_test,
        )

        merged_profiles = train_damage_profile.merge(
            pred_damage_profile,
            on="section_name",
            how="left",
            suffixes=("_train", "_test"),
        )
        merged_profiles["damage_total_test"] = (
            merged_profiles["damage_total_test"].fillna(0))

        # For sections with known damage (used as input features),
        # use real test damage instead of predictions or zero.
        known_sections = [
            f.replace("damage_", "")
            for f in self.features
            if f.startswith("damage_section_")
        ]
        for section in known_sections:
            mask_test = df_test["section_name"] == section
            if mask_test.any():
                real_total = (
                    df_test.loc[mask_test, self.target].astype(float).values *
                    df_test.loc[mask_test,
                                "damage_weight"].astype(float).values).sum()
            else:
                feat_col = f"damage_{section}"
                per_sim = df_test.drop_duplicates(subset="sim_id")
                real_total = (
                    per_sim[feat_col].astype(float).values *
                    per_sim["damage_weight"].astype(float).values).sum()
            sec_idx = merged_profiles["section_name"] == section
            merged_profiles.loc[sec_idx, "damage_total_test"] = real_total

        true_damage = (merged_profiles["damage_total_train"].values +
                       merged_profiles["damage_total_test"].values)

        # Reference profile
        heights = df_reference_damage_profile["mean z [m]"].values
        ref_damage = df_reference_damage_profile["damage"].values

        plot_dir = os.path.join(self.output_dir,
                                "tower_damage_profiles_vs_reference")
        os.makedirs(plot_dir, exist_ok=True)

        # Renormalize train profile per wind bin.
        # damage_weight was computed for the full (train+test) domain.
        # Each wind bin may have a different train/test split ratio, so
        # we scale per bin: weight_renorm = weight * (N_total / N_train).
        wind_col = "wind_speed_mid"
        df_train[wind_col] = pd.to_numeric(df_train[wind_col], errors="coerce")
        df_test[wind_col] = pd.to_numeric(df_test[wind_col], errors="coerce")
        all_sims = pd.concat([
            df_train[["sim_id", wind_col]].drop_duplicates("sim_id"),
            df_test[["sim_id", wind_col]].drop_duplicates("sim_id"),
        ])
        n_total_per_bin = (all_sims.groupby(wind_col).size().rename("n_total"))
        train_sims = df_train[["sim_id", wind_col]].drop_duplicates("sim_id")
        n_train_per_bin = (
            train_sims.groupby(wind_col).size().rename("n_train"))
        bin_scale = (n_total_per_bin / n_train_per_bin).rename("bin_scale")

        df_train_renorm = df_train.merge(bin_scale,
                                         left_on=wind_col,
                                         right_index=True,
                                         how="left")
        df_train_renorm["damage_weight_renorm"] = (
            df_train_renorm["damage_weight"].astype(float) *
            df_train_renorm["bin_scale"])

        train_damage_profile_renorm = profile_by_section(
            df=df_train_renorm,
            damage_values=y_true_train,
            weight_col="damage_weight_renorm",
        )

        # Plot: DAMAGE
        train_damage = train_damage_profile_renorm["damage_total"].values
        plot_dir_damage = os.path.join(plot_dir, "damage")
        os.makedirs(plot_dir_damage, exist_ok=True)
        plots.plot_tower_damage_profiles_vs_reference(
            damage_profiles=[true_damage],
            ref_profile=ref_damage,
            heights=heights,
            labels=["Train + Predicted Test Profile"],
            plot_dir=plot_dir_damage,
            title="Tower Damage Profile vs Reference",
            save_svg=save_svg,
            train_profile=train_damage)

        if self.report_del:
            # Plot: DEL
            plot_dir_del = os.path.join(plot_dir, "DEL")
            os.makedirs(plot_dir_del, exist_ok=True)
            ref_del = self._compute_del(ref_damage)
            true_del = self._compute_del(true_damage)
            train_del = self._compute_del(train_damage)
            plots.plot_tower_damage_profiles_vs_reference(
                damage_profiles=[true_del],
                ref_profile=ref_del,
                heights=heights,
                labels=["Train + Predicted Test Profile"],
                plot_dir=plot_dir_del,
                filename="tower_del_profiles_vs_reference",
                title="Tower DEL Profile vs Reference",
                xlabel="Weighted Fatigue DEL",
                save_svg=save_svg,
                train_profile=train_del)

        # metrics summary
        metrics_dict = self._compute_error_metrics(ref_damage, true_damage)
        df_eval = pd.DataFrame({
            "section_name": merged_profiles["section_name"].values,
            "height": heights,
            "reference_damage": ref_damage,
            "predicted_damage": true_damage,
            "mae": metrics_dict["abs_err"],
            "mre": metrics_dict["rel_err"]
        })

        # del
        if self.report_del:
            metrics_dict_del = self._compute_error_metrics(ref_del, true_del)
            df_eval["reference_DEL"] = ref_del
            df_eval["predicted_DEL"] = true_del
            df_eval["mae_DEL"] = metrics_dict_del["abs_err"]
            df_eval["mre_DEL"] = metrics_dict_del["rel_err"]

        with open(os.path.join(plot_dir, "damage_profile_comparison.csv"),
                  "w",
                  encoding="utf-8") as f:
            df_eval.to_csv(f, index=False)

        # tower error profile
        del_mre = df_eval["mre_DEL"].values if self.report_del else None
        plots.plot_error_profile(
            heights=heights,
            damage_mre=df_eval["mre"].values,
            del_mre=del_mre,
            plot_dir=plot_dir,
            save_svg=save_svg,
        )
