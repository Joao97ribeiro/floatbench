# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=no-member
# pylint: disable=too-many-function-args
# pylint: disable=too-many-statements
# pylint: disable=duplicate-code
""""AutoGluon trainer for damage prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.model_selection import train_test_split

from floatbench.utils import natural_key


class AGDamageTrainer:
    """Train AutoGluon for damage prediction.

    Attributes:
        features: List of feature column names.
        target: Target column name.
        output_dir: Directory to save model and plots.
        model_params: Hyperparameters for AutoGluon model.
        random_state: Random seed for reproducibility.
        subsample_size: Subsample size for training (optional).
        subsample_random_state: Random state for subsampling.
        target_transform_to_del: Whether to transform target to del.
        fine_tune_base_model_path: Path to base model for fine-tuning.
        fine_tune_base_model_name: Name of base model for fine-tuning.
        fine_tune_base_model: Loaded base model for fine-tuning.
        model: AutoGluon TabularPredictor instance.
        train_info: Training history info.
    """

    def __init__(
        self,
        features: List[str],
        target: str = "damage",
        output_dir: str | Path = None,
        model_params: Dict[str, Any] | None = None,
        random_state: int = 42,
        subsample_size: Optional[int] = None,
        subsample_random_state: int = 0,
        target_transform_to_del: bool = False,
        fine_tune_base_model_path: str = None,
        fine_tune_base_model_name: Optional[str] = None,
    ) -> None:
        """Initializes the trainer.

        Args:
            features: List of feature column names.
            target: Target column name.
            output_dir: Directory to save model and plots.
            model_params: AutoGluon fit config (optional).
            random_state: Random seed for reproducibility.
            subsample_size: Subsample size for training (optional).
            subsample_random_state: Random state for subsampling.
            target_transform_to_del: Whether to transform target to del.
            fine_tune_base_model_path: Path to base model for fine-tuning.
            fine_tune_base_model_name: Name of base model for fine-tuning.
        """
        self.features = features
        self.target = target
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.random_state = random_state
        self.fine_tune_base_model_path = fine_tune_base_model_path
        self.fine_tune_base_model_name = fine_tune_base_model_name

        self.model_params = model_params or {}
        self.model = None
        self.fine_tune_base_model = None

        if self.fine_tune_base_model_path is not None:
            self.fine_tune_base_model = TabularPredictor.load(
                self.fine_tune_base_model_path)

            if self.fine_tune_base_model_name is None:
                self.fine_tune_base_model_name = (
                    self.fine_tune_base_model.model_best)

        self.subsample_size = subsample_size
        self.subsample_random_state = subsample_random_state

        self.target_transform_to_del = target_transform_to_del
        self.train_info = None

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

    def _build_model(self, model_dir: str) -> TabularPredictor:
        """Creates the AutoGluon predictor.

        Args:
            model_dir: Directory to save the model.

        Returns:
            TabularPredictor instance.
        """
        eval_metric = self.model_params.get("eval_metric",
                                            "mean_absolute_error")
        predictor = TabularPredictor(label=self.target,
                                     path=model_dir,
                                     eval_metric=eval_metric,
                                     problem_type='regression')
        return predictor

    def train(
        self,
        df_train: pd.DataFrame,
        df_valid: Optional[pd.DataFrame] = None,
        valid_size: Optional[float] = None,
        df_test: Optional[pd.DataFrame] = None,
    ) -> None:
        """Trains the model on given train/valid splits.

        Args:
            df_train: Training dataframe.
            df_valid: Validation dataframe (optional).
            valid_size: Proportion of data for validation if
              df_valid is None (split by sim_id).
            df_test: Test dataframe for leaderboard evaluation
              (optional, independent of tuning data).
        """
        if df_valid is None and valid_size is not None:
            df = df_train.copy()
            unique_sims = df["sim_id"].unique()

            train_sims, valid_sims = train_test_split(
                unique_sims,
                test_size=valid_size,
                random_state=self.random_state)

            df_train = df[df["sim_id"].isin(train_sims)]
            df_valid = df[df["sim_id"].isin(valid_sims)]

            sections_train = sorted(df_train["section_name"].unique().tolist(),
                                    key=natural_key)
            sections_valid = sorted(df_valid["section_name"].unique().tolist(),
                                    key=natural_key)
            self.train_info = {
                "split": {
                    "method": "train_test_split",
                    "valid_size": float(valid_size),
                    "random_state": int(self.random_state),
                },
                "counts": {
                    "n_train": int(len(df_train)),
                    "n_valid": int(len(df_valid)),
                },
                "sections_names": {
                    "train": sections_train,
                    "valid": sections_valid,
                    "counts": {
                        "n_train": int(len(sections_train)),
                        "n_valid": int(len(sections_valid)),
                    }
                }
            }

        x_train, y_train = self._get_x_y(df_train)
        xy_train = np.concatenate([x_train, y_train.reshape(-1, 1)], axis=1)
        train_data = pd.DataFrame(xy_train,
                                  columns=self.features + [self.target])

        # optional subsample
        if self.subsample_size is not None and self.subsample_size < len(
                train_data):
            train_data = TabularDataset(train_data)
            train_data = train_data.sample(
                n=self.subsample_size,
                random_state=self.subsample_random_state,
            )

        # model
        if self.fine_tune_base_model is None:
            model_dir = os.path.join(self.output_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            self.model = self._build_model(model_dir)

        else:
            model_dir = os.path.join(self.output_dir, "model_ft")
            os.makedirs(model_dir, exist_ok=True)

            x = train_data[self.features].copy()
            y_pred = self.fine_tune_base_model.predict(
                x, model=self.fine_tune_base_model_name).values
            train_data['damage_sim'] = y_pred

            self.model = self._build_model(model_dir)

        # build tuning_data from df_valid (split or explicit)
        tuning_data = None
        if df_valid is not None:
            x_valid, y_valid = self._get_x_y(df_valid)
            xy_valid = np.concatenate([x_valid, y_valid.reshape(-1, 1)], axis=1)
            tuning_data = pd.DataFrame(xy_valid,
                                       columns=self.features + [self.target])

            if self.fine_tune_base_model is not None:
                x_v = tuning_data[self.features].copy()
                y_pred_v = self.fine_tune_base_model.predict(
                    x_v, model=self.fine_tune_base_model_name).values
                tuning_data['damage_sim'] = y_pred_v

        presets = self.model_params.get("presets", "best")
        time_limit = self.model_params.get("time_limit", 3600 * 4)
        hyperparameters = self.model_params.get("hyperparameters", "zeroshot")
        num_cpus = self.model_params.get("num_cpus", 24)
        num_gpus = self.model_params.get("num_gpus", 1)
        num_bag_folds = self.model_params.get("num_bag_folds", 8)
        num_gpus_per_fold = self.model_params.get("num_gpus_per_fold", num_gpus)
        num_cpus_per_fold = self.model_params.get("num_cpus_per_fold",
                                                  num_cpus // 2)
        self.model.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            use_bag_holdout=tuning_data is not None,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=hyperparameters,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            num_bag_folds=num_bag_folds,
            ag_args_ensemble={
                "ag_args_fit": {
                    "num_cpus": num_cpus,
                    "num_gpus": num_gpus
                }
            },
            ag_args_fit={
                "num_cpus": num_cpus_per_fold,
                "num_gpus": num_gpus_per_fold
            },
        )

        # save artifacts
        self._save_artifacts(model_dir)

        # build test_data for leaderboard evaluation
        test_data = None
        if df_test is not None:
            x_test, y_test = self._get_x_y(df_test)
            xy_test = np.concatenate([x_test, y_test.reshape(-1, 1)], axis=1)
            test_data = pd.DataFrame(xy_test,
                                     columns=self.features + [self.target])

            if self.fine_tune_base_model is not None:
                x_t = test_data[self.features].copy()
                y_pred_t = self.fine_tune_base_model.predict(
                    x_t, model=self.fine_tune_base_model_name).values
                test_data['damage_sim'] = y_pred_t

        # save reports
        self._save_training_reports(model_dir,
                                    valid_data=tuning_data,
                                    test_data=test_data)

    def _save_training_reports(
        self,
        out_dir: str,
        valid_data: pd.DataFrame = None,
        test_data: pd.DataFrame = None,
    ) -> None:
        """Saves leaderboard and fit summary.

        Args:
            out_dir: Output directory.
            valid_data: Validation/tuning data for leaderboard (optional).
            test_data: Test data for leaderboard (optional).
        """
        extra_metrics = ['r2', 'mae', 'mse', 'rmse', 'mape']

        leaderboard = self.model.leaderboard(silent=True)
        leaderboard.to_csv(os.path.join(out_dir, "leaderboard.csv"),
                           index=False)

        if valid_data is not None:
            lb_valid = self.model.leaderboard(data=valid_data,
                                              extra_metrics=extra_metrics,
                                              extra_info=True)
            lb_valid.to_csv(os.path.join(out_dir, "leaderboard_valid.csv"),
                            index=False)

        if test_data is not None:
            lb_test = self.model.leaderboard(data=test_data,
                                             extra_metrics=extra_metrics,
                                             extra_info=True)
            lb_test.to_csv(os.path.join(out_dir, "leaderboard_test.csv"),
                           index=False)

        summary = self.model.fit_summary(verbosity=0)
        with open(os.path.join(out_dir, "fit_summary.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        best_model = self.model.model_best
        with open(os.path.join(out_dir, "best_model.txt"),
                  "w",
                  encoding="utf-8") as f:
            f.write(str(best_model))

    def _save_artifacts(
        self,
        out_dir: str,
    ) -> None:
        """Saves AutoGluon artifacts to disk.

        Args:
            out_dir: Output directory.
        """
        if out_dir is None:
            out_dir = self.output_dir

        assert self.model is not None

        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "task": "damage_prediction",
            "data": {
                "features": list(self.features),
                "target": self.target,
                "target_transform_to_del": bool(self.target_transform_to_del),
            },
            "reproducibility": {
                "random_state": int(self.random_state),
            },
            "model": {
                "best_model": self.model.model_best,
                "predictor_path": self.model.path,
                "model_params": self.model_params,
            }
        }

        if self.train_info is not None:
            meta["train"] = self.train_info

        if self.fine_tune_base_model is not None:
            meta["fine_tuning"] = {
                "base_predictor_path": self.fine_tune_base_model_path,
                "base_model_name": self.fine_tune_base_model_name,
                "extra_features": ["damage_sim"],
            }

        with open(os.path.join(out_dir, "autogluon_meta.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
