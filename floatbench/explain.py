# pylint: disable=too-many-arguments
# pylint: disable=duplicate-code
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=no-member
"""AutoGluon explainer for damage prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import shap

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from scipy import stats

from floatbench import plots


class AutogluonWrapper:
    """Wrapper to adapt AutoGluon TabularPredictor for SHAP compatibility.

    This class ensures that input data passed to SHAP explainers is converted
    to a pandas DataFrame with the correct feature names before calling
    AutoGluon's prediction API.

    Attributes:
        model: Fitted AutoGluon TabularPredictor.
        features: List of feature names in correct column order.
        model_name: Name of the AutoGluon model to explain.
    """

    def __init__(
        self,
        model: TabularPredictor,
        features: List[str],
        model_name: Optional[str] = None,
    ):
        """Initializes the AutogluonWrapper.

        Args:
            model: Fitted AutoGluon TabularPredictor.
            features: List of feature names in correct column order.
            model_name: Name of the AutoGluon model to explain.
        """
        self.model = model
        self.features = features
        self.model_name = model_name

    def predict(
            self,
            input_data: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Generates predictions using the wrapped AutoGluon model.

        Args:
            input_data: Input features.

        Returns:
            Model predictions as returned by AutoGluon.
        """
        if isinstance(input_data, pd.Series):
            input_data = input_data.values.reshape(1, -1)

        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame(input_data, columns=self.features)

        return self.model.predict(input_data, model=self.model_name)


class AGDamageExplainer:
    """AutoGluon explainer for damage prediction.

    Attributes:
        model_cfg_path: Path to the file with the saved model config.
        output_dir: Directory to save outputs (plots, CSVs, configs).
        model_name: Specific AutoGluon model to explain (optional).
        model_path: Path to the saved AutoGluon predictor.
        features: List of feature column names.
        target: Target column name.
        target_transform_to_del: Whether target is DEL-transformed.
        model: Loaded AutoGluon predictor.
        fi_num_shuffle_sets: Number of shuffle sets for permutation importance.
        fi_subsample_size: Number of rows subsampled for permutation importance.
        shap_kmeans_clusters: Number of k-means clusters for SHAP background.
        shap_num_explain_samples: Number of validation samples to explain.
        shap_num_model_evals: Number of model evaluations for SHAP.
        shap_random_state: Random seed for SHAP sampling.
    """

    def __init__(
        self,
        model_cfg_path: str | Path,
        output_dir: str | Path,
        model_name: str | None = None,
        fi_num_shuffle_sets: int = 5,
        fi_subsample_size: int = 5000,
        shap_kmeans_clusters: int = 20,
        shap_num_explain_samples: int = 500,
        shap_num_model_evals: int = 50,
        shap_random_state: int = 0,
    ) -> None:
        """Initializes the explainer.

        Args:
            model_cfg_path: Path to the file with the saved model config.
            output_dir: Directory to save outputs (plots, CSVs, configs).
            model_name: Specific AutoGluon model to explain (optional).
            fi_num_shuffle_sets: Number of shuffle sets for permutation
              importance.
            fi_subsample_size: Number of rows subsampled for permutation
              importance.
            shap_kmeans_clusters: Number of k-means clusters for SHAP
              background.
            shap_num_explain_samples: Number of validation samples to
              explain.
            shap_num_model_evals: Number of model evaluations for SHAP.
            shap_random_state: Random seed for SHAP sampling.
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

        self.model = self._load_model()

        # Feature importance config
        self.fi_num_shuffle_sets = fi_num_shuffle_sets
        self.fi_subsample_size = fi_subsample_size

        # SHAP config
        self.shap_kmeans_clusters = shap_kmeans_clusters
        self.shap_num_explain_samples = shap_num_explain_samples
        self.shap_num_model_evals = shap_num_model_evals
        self.shap_random_state = shap_random_state

        self._save_artifacts(self.output_dir)

    def _load_model_meta(self) -> dict:
        """Loads the AutoGluon metadata from disk.

        Returns:
            Dictionary with model metadata.
        """
        with open(self.model_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        features = cfg["data"]["features"]
        target = cfg["data"]["target"]
        target_transform_to_del = cfg["data"]["target_transform_to_del"]
        model_path = cfg["model"]["predictor_path"]

        return {
            "features": features,
            "target": target,
            "target_transform_to_del": target_transform_to_del,
            "model_path": model_path,
        }

    def _get_x_y(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extracts x, y from a dataframe.
        
        Args:
            df: Input dataframe.
        
        Returns:
            Tuple of (x, y).
        """

        x = df[self.features].copy().values
        y = df[self.target].astype(float).values

        if self.target_transform_to_del:
            y = np.cbrt(y)

        return x, y

    def _load_model(self) -> TabularPredictor:
        """Loads the AutoGluon model from disk.
        
        Returns:
            Loaded AutoGluon regressor.
        """
        model = TabularPredictor.load(str(self.model_path))

        return model

    def explain(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        save_svg: bool = True,
    ) -> None:
        """Trains the model on given train/valid splits.

        Args:
            df_train: Training dataframe.
            df_test: Test dataframe.
            save_svg: Whether to save plots in SVG format.
        """
        # train data
        x_train, y_train = self._get_x_y(df_train)
        xy_train = np.concatenate([x_train, y_train.reshape(-1, 1)], axis=1)
        train_data = pd.DataFrame(xy_train,
                                  columns=self.features + [self.target])

        # valid data
        x_test, y_test = self._get_x_y(df_test)
        xy_test = np.concatenate([x_test, y_test.reshape(-1, 1)], axis=1)
        test_data = pd.DataFrame(xy_test, columns=self.features + [self.target])

        # feature correlation plot
        fc_dir = os.path.join(self.output_dir, "feature_correlation")
        os.makedirs(fc_dir, exist_ok=True)
        plots.plot_feature_correlation(
            train_data,
            self.features,
            plot_dir=fc_dir,
            save_svg=save_svg,
        )

        # feature importance
        self.compute_feature_importance(test_data)

        # shap
        mean_abs_shap = self.compute_shap(x_train, x_test, save_svg=save_svg)

        # feature analysis summary
        self.compute_feature_analysis(
            x_data=x_test,
            y_data=y_test,
            mean_abs_shap=mean_abs_shap,
        )

    def compute_feature_importance(self, test_data: pd.DataFrame) -> None:
        """
        Computes feature importance using AutoGluon's built-in method.
        
        Args:
            test_data: Validation dataframe.
        """

        fi_dir = os.path.join(self.output_dir, "feature_importance")
        os.makedirs(fi_dir, exist_ok=True)

        feature_importance_pf = self.model.feature_importance(
            data=test_data,
            model=self.model_name,
            num_shuffle_sets=self.fi_num_shuffle_sets,
            subsample_size=self.fi_subsample_size)

        fi_path = os.path.join(fi_dir, "feature_importance.csv")
        feature_importance_pf.reset_index().to_csv(fi_path, index=False)

        config = {
            "num_shuffle_sets": self.fi_num_shuffle_sets,
            "subsample_size": self.fi_subsample_size,
        }
        config_path = os.path.join(fi_dir, "feature_importance_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def compute_shap(self,
                     x_train: np.ndarray,
                     x_test: np.ndarray,
                     save_svg: bool = True) -> np.ndarray:
        """Computes and plots SHAP values for feature importance.

        Args:
            x_train: Training features.
            x_test: Validation features.
            save_svg: Whether to save plots in SVG format.

        Returns:
            Mean absolute SHAP values per feature.
        """

        shape_dir = os.path.join(self.output_dir, "shap")
        os.makedirs(shape_dir, exist_ok=True)

        # SHAP wrapper
        ag_wrapper = AutogluonWrapper(self.model, self.features,
                                      self.model_name)

        # background and explanation data
        x_train_bg = shap.kmeans(x_train, self.shap_kmeans_clusters)
        x_test_explain = TabularDataset(x_test).sample(
            self.shap_num_explain_samples, random_state=self.shap_random_state)
        x_test_explain.columns = self.features

        # SHAP values
        explainer = shap.KernelExplainer(ag_wrapper.predict, x_train_bg)
        shap_values = explainer.shap_values(x_test_explain,
                                            nsamples=self.shap_num_model_evals)

        # Global importance (mean |SHAP|)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        imp_df = (pd.DataFrame({
            "feature": self.features,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))

        # save to csv
        imp_path = os.path.join(shape_dir, "feature_importance_shap.csv")
        imp_df.to_csv(imp_path, index=False)

        # bar plot horizontal
        plots.plot_shap_importance_bar(mean_abs_shap=mean_abs_shap,
                                       feature_names=self.features,
                                       plot_dir=shape_dir,
                                       save_svg=save_svg)

        # beeswarm
        plots.plot_shap_beeswarm(shap_values=shap_values,
                                 feature_names=self.features,
                                 x_val_scaled=x_test_explain,
                                 plot_dir=shape_dir,
                                 save_svg=save_svg)

        # config
        config = {
            "kmeans_clusters": self.shap_kmeans_clusters,
            "num_explain_samples": self.shap_num_explain_samples,
            "num_model_evals": self.shap_num_model_evals,
            "random_state": self.shap_random_state,
        }
        config_path = os.path.join(shape_dir, "shap_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        return mean_abs_shap

    def compute_feature_analysis(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        mean_abs_shap: np.ndarray,
    ) -> None:
        """Computes feature analysis summary and redundancy pairs.

        Produces two CSVs:
        - feature_analysis_summary.csv: per-feature correlations,
          SHAP importance, and permutation importance.
        - feature_redundancy_pairs.csv: feature pairs with |r| > 0.85.

        Args:
            x_data: Feature array (n_samples, n_features).
            y_data: Target array (n_samples,).
            mean_abs_shap: Mean absolute SHAP values per feature.
        """
        fa_dir = os.path.join(self.output_dir, "feature_analysis")
        os.makedirs(fa_dir, exist_ok=True)

        n_features = len(self.features)

        # Correlations with target
        pearson_corr = np.full(n_features, np.nan)
        spearman_corr = np.full(n_features, np.nan)
        if y_data is not None:
            for i in range(n_features):
                pearson_corr[i] = np.corrcoef(x_data[:, i], y_data)[0, 1]
                spearman_corr[i] = stats.spearmanr(x_data[:, i],
                                                   y_data).statistic

        # Permutation importance from already-computed CSV
        perm_importance = np.full(n_features, np.nan)
        fi_path = os.path.join(self.output_dir, "feature_importance",
                               "feature_importance.csv")
        if os.path.exists(fi_path):
            fi_df = pd.read_csv(fi_path)
            for i, feat in enumerate(self.features):
                row = fi_df[fi_df["index"] == feat]
                if not row.empty:
                    perm_importance[i] = row["importance"].values[0]

        # Summary CSV
        summary_df = pd.DataFrame({
            "feature": self.features,
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
            "mean_abs_shap": mean_abs_shap,
            "permutation_importance": perm_importance,
        })
        summary_df = summary_df.sort_values(
            "mean_abs_shap", ascending=False).reset_index(drop=True)
        summary_df.to_csv(
            os.path.join(fa_dir, "feature_analysis_summary.csv"),
            index=False,
        )

        # Redundancy pairs (|Pearson r| > 0.85)
        corr_matrix = np.corrcoef(x_data, rowvar=False)
        pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                r_val = corr_matrix[i, j]
                if abs(r_val) > 0.85:
                    pairs.append({
                        "feature_1": self.features[i],
                        "feature_2": self.features[j],
                        "pearson_corr": r_val,
                    })
        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            pairs_df["abs_corr"] = pairs_df["pearson_corr"].abs()
            pairs_df = pairs_df.sort_values(
                "abs_corr",
                ascending=False).drop(columns="abs_corr").reset_index(drop=True)
        pairs_df.to_csv(
            os.path.join(fa_dir, "feature_redundancy_pairs.csv"),
            index=False,
        )

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

        with open(os.path.join(out_dir, "artifacts_meta.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
