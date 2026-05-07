# pylint: disable=too-many-branches
# pylint: disable=duplicate-code
"""Train an Autogluon for damage prediction using hyperparameters."""

from absl import app
from absl import logging
from absl import flags

import pandas as pd

import floatbench as model
from floatbench import data

FLAGS = flags.FLAGS

# Train data flags
flags.DEFINE_list("train_csv", None,
                  "Comma-separated paths to training CSV(s).")
flags.DEFINE_string("subsample_size", None, "Subsample size for training.")
flags.DEFINE_integer("subsample_random_state", None,
                     "Random state for subsampling.")

# Valid data flags
flags.DEFINE_string("valid_csv", None,
                    "Directory containing the validation dataset.")
flags.DEFINE_string("valid_size", "",
                    "Proportion of data for validation (split by sim_id).")
flags.DEFINE_integer("random_state", None, "Random seed for train/valid split.")

# Section filter flags
flags.DEFINE_string("section_col", "section_name", "Column with section id.")
flags.DEFINE_list("train_sections", [], "Sections to keep in training data.")
flags.DEFINE_list("valid_sections", [], "Sections to keep in validation data.")

# Test data flags
flags.DEFINE_list("test_csv", [], "Comma-separated paths to test CSV(s).")

# Features and target flags
flags.DEFINE_list("features", None, "List of feature column names.")
flags.DEFINE_boolean("target_transform_to_del", None,
                     "Whether to transform target to del.")

# Fine tuning flags
flags.DEFINE_string("fine_tune_base_model_path", None,
                    "Directory containing the base model for fine-tuning.")
flags.DEFINE_string("fine_tune_base_model_name", None,
                    "Name of the base model for fine-tuning.")

# Autogluon hyperparameters
flags.DEFINE_string("eval_metric", None, "Evaluation metric for training.")
flags.DEFINE_string("presets", None, "Preset configurations for Autogluon.")
flags.DEFINE_integer("time_limit", None, "Time limit for training in seconds.")
flags.DEFINE_string("hyperparameters", None,
                    "Hyperparameters for Autogluon model.")
flags.DEFINE_integer("num_cpus", None, "Number of cpus to use for training.")
flags.DEFINE_integer("num_gpus", None, "Number of gpus to use for training.")
flags.DEFINE_integer("num_bag_folds", None, "Number of bagging folds.")
flags.DEFINE_integer("num_cpus_per_fold", None, "Number of cpus per fold.")
flags.DEFINE_integer("num_gpus_per_fold", None, "Number of gpus per fold.")

# Output flags
flags.DEFINE_string("output_dir", None, "Directory to save model and plots.")


def main(_) -> None:
    """Run the full training pipeline using the provided configuration."""

    if FLAGS.subsample_size == "":
        subsample_size = None
    else:
        subsample_size = int(FLAGS.subsample_size)

    if FLAGS.valid_size == "":
        valid_size = None
    else:
        valid_size = float(FLAGS.valid_size)

    if FLAGS.fine_tune_base_model_path == "":
        fine_tune_base_model_path = None
    else:
        fine_tune_base_model_path = FLAGS.fine_tune_base_model_path

    if FLAGS.fine_tune_base_model_name == "":
        fine_tune_base_model_name = None
    else:
        fine_tune_base_model_name = FLAGS.fine_tune_base_model_name

    # 1. Load and filter data
    train_paths = [p for p in FLAGS.train_csv if p]
    df_train = pd.concat(
        [
            data.ensure_section_name(pd.read_csv(p, low_memory=False))
            for p in train_paths
        ],
        ignore_index=True,
    )
    if len(FLAGS.train_sections) > 0:
        df_train = data.filter_sections(df_train, FLAGS.section_col,
                                        FLAGS.train_sections)

    df_valid = None
    if FLAGS.valid_csv != "":
        df_valid = data.ensure_section_name(
            pd.read_csv(FLAGS.valid_csv, low_memory=False))
        if len(FLAGS.valid_sections) > 0:
            df_valid = data.filter_sections(df_valid, FLAGS.section_col,
                                            FLAGS.valid_sections)

    test_paths = [p for p in FLAGS.test_csv if p]
    df_test = None
    if test_paths:
        df_test = pd.concat(
            [
                data.ensure_section_name(pd.read_csv(p, low_memory=False))
                for p in test_paths
            ],
            ignore_index=True,
        )

    # 2. Setup Parameters
    model_params = {
        "eval_metric": FLAGS.eval_metric,
        "presets": FLAGS.presets,
        "time_limit": FLAGS.time_limit,
        "hyperparameters": FLAGS.hyperparameters,
        "num_cpus": FLAGS.num_cpus,
        "num_gpus": FLAGS.num_gpus,
        "num_bag_folds": FLAGS.num_bag_folds,
        "num_gpus_per_fold": FLAGS.num_gpus_per_fold,
        "num_cpus_per_fold": FLAGS.num_cpus_per_fold,
    }

    # 3. Final Model Training
    logging.info("Starting training Autogluon model...")
    trainer = model.AGDamageTrainer(
        features=FLAGS.features,
        target="damage",
        output_dir=FLAGS.output_dir,
        model_params=model_params,
        random_state=FLAGS.random_state,
        subsample_size=subsample_size,
        subsample_random_state=FLAGS.subsample_random_state,
        target_transform_to_del=FLAGS.target_transform_to_del,
        fine_tune_base_model_path=fine_tune_base_model_path,
        fine_tune_base_model_name=fine_tune_base_model_name,
    )
    if fine_tune_base_model_path is not None:
        logging.info(
            f"Fine-tuning from base model at {fine_tune_base_model_path}.")
    if fine_tune_base_model_name is not None:
        logging.info(
            f"Fine-tuning from base model named {fine_tune_base_model_name}.")
    trainer.train(df_train, df_valid, valid_size=valid_size, df_test=df_test)
    logging.info("Model training completed.")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
