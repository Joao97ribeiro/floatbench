# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
"""Testing pipeline for damage prediction using AutoGluon model."""

from absl import app
from absl import logging
from absl import flags

import pandas as pd

import floatbench as model
from floatbench import data

FLAGS = flags.FLAGS

# Test data flags
flags.DEFINE_string("test_csv", None, "Directory containing the test dataset.")
flags.DEFINE_boolean("test_by_time", None, "Whether to test by time.")
flags.DEFINE_boolean("test_by_regime", None,
                     "Whether to evaluate by wind/wave regime labels.")
flags.DEFINE_boolean("test_by_regime_by_section", None,
                     "Whether to evaluate by section within each regime.")
flags.DEFINE_boolean(
    "report_del", True,
    "Whether to report DEL metrics and plots in addition to DAMAGE.")

# Section filter flags
flags.DEFINE_string("section_col", "section_name", "Column with section id.")
flags.DEFINE_list("test_sections", [], "Sections to keep in test data.")

# Model trained flags
flags.DEFINE_string("model_cfg_path", None,
                    "Path to the model configuration file.")
flags.DEFINE_string("model_name", None, "Name of the trained model.")

# Damage Profile Data
flags.DEFINE_boolean("test_damage_profile", False,
                     "Whether to evaluate tower damage profile vs reference.")
flags.DEFINE_string("train_csv", None,
                    "Directory containing the train dataset.")
flags.DEFINE_string("reference_damage_profile", None,
                    "Path to reference damage profile.")

# Output flags
flags.DEFINE_string("output_dir", None, "Directory to save model and plots.")
flags.DEFINE_boolean("save_svg", None, "Whether to save plots in SVG format.")
flags.DEFINE_list(
    "color_by_cols", [],
    "Columns to color scatter plots by (e.g., section_id,std_mean_wind).")
flags.DEFINE_boolean("predict_only", False,
                     "Only save predictions CSV, skip all plots.")


def _run_damage_profile(
    save_svg: bool,
    predictor: model.AGDamagePredictor,
    df_test: pd.DataFrame,
    predictions,
    train_csv: str,
    reference_damage_profile: str,
) -> None:
    """Runs tower damage profile vs reference.

    Args:
        save_svg: Whether to save plots in SVG format.
        predictor: Initialized AGDamagePredictor.
        df_test: Test DataFrame.
        predictions: Predicted damage array from ``predictor.predict``.
        train_csv: Path to train CSV.
        reference_damage_profile: Path to reference profile CSV.
    """
    df_train = data.ensure_wind_wave_group(
        data.ensure_section_name(pd.read_csv(train_csv, low_memory=False)))
    df_ref = pd.read_csv(reference_damage_profile)
    logging.info("Generating tower damage profile...")
    predictor.evaluate_tower_damage_profile_vs_reference(
        df_train=df_train,
        df_test=df_test,
        df_reference_damage_profile=df_ref,
        y_pred_test=predictions,
        save_svg=save_svg)


def main(_) -> None:
    """Run the full testing pipeline."""

    if FLAGS.model_name == "":
        model_name = None
    else:
        model_name = FLAGS.model_name

    if FLAGS.train_csv == "":
        train_csv = None
    else:
        train_csv = FLAGS.train_csv

    if FLAGS.reference_damage_profile == "":
        reference_damage_profile = None
    else:
        reference_damage_profile = FLAGS.reference_damage_profile

    # Load data
    df_test = data.ensure_wind_wave_group(
        data.ensure_section_name(pd.read_csv(FLAGS.test_csv, low_memory=False)))
    if len(FLAGS.test_sections) > 0:
        df_test = data.filter_sections(df_test, FLAGS.section_col,
                                       FLAGS.test_sections)

    logging.info("Starting model testing...")
    predictor = model.AGDamagePredictor(model_cfg_path=FLAGS.model_cfg_path,
                                        model_name=model_name,
                                        output_dir=FLAGS.output_dir,
                                        report_del=FLAGS.report_del)

    logging.info("Making predictions on test set...")
    predictions = predictor.predict(df_test)

    run_profile = (FLAGS.test_damage_profile and train_csv is not None and
                   reference_damage_profile is not None)

    if FLAGS.predict_only:
        logging.info("Predict-only mode — skipping plots.")
        if run_profile:
            _run_damage_profile(
                save_svg=False,
                predictor=predictor,
                df_test=df_test,
                predictions=predictions,
                train_csv=train_csv,
                reference_damage_profile=reference_damage_profile)
        logging.info("Model testing completed.")
        return

    logging.info("Evaluating model performance...")
    predictor.evaluate(df=df_test,
                       y_pred=predictions,
                       save_svg=FLAGS.save_svg,
                       color_by_cols=FLAGS.color_by_cols)

    logging.info("Evaluating model performance by section...")
    predictor.evaluate_by_section(df=df_test,
                                  y_pred=predictions,
                                  save_svg=FLAGS.save_svg)

    if FLAGS.test_by_regime:
        logging.info("Evaluating model performance by regime...")
        names = ["In-train", "Interpolate", "Extrapolate"]
        group_cols = ["wind_group", "wave_group", "wind_wave_group"]
        group_orders = [
            names, names, [f"{w}_{v}" for w in names for v in names]
        ]
        plot_global = [True, True, False]
        for group_col, group_order, global_plot in zip(group_cols, group_orders,
                                                       plot_global):
            logging.info("Evaluating by regime: %s...", group_col)
            predictor.evaluate_by_group(df=df_test,
                                        group_col=group_col,
                                        group_order=group_order,
                                        y_pred=predictions,
                                        save_svg=FLAGS.save_svg,
                                        plot_global=global_plot,
                                        color_by_cols=FLAGS.color_by_cols)
            if FLAGS.test_by_regime_by_section:
                logging.info("Evaluating by section within regime %s...",
                             group_col)
                predictor.evaluate_by_section_within_group(
                    df=df_test,
                    group_col=group_col,
                    group_order=group_order,
                    y_pred=predictions,
                    save_svg=FLAGS.save_svg)

    if FLAGS.test_by_time:
        logging.info("Evaluating model performance by time...")
        predictor.evaluate_cumulative_damage_by_section(df=df_test,
                                                        y_pred=predictions,
                                                        save_svg=FLAGS.save_svg)

    if run_profile and not FLAGS.test_by_time:
        _run_damage_profile(save_svg=FLAGS.save_svg,
                            predictor=predictor,
                            df_test=df_test,
                            predictions=predictions,
                            train_csv=train_csv,
                            reference_damage_profile=reference_damage_profile)
    logging.info("Model testing completed.")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
