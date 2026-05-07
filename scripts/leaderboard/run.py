"""Generate extended leaderboard with per-group and per-section metrics."""

import os
import pandas as pd
from absl import app, flags, logging

import floatbench as model
from floatbench import data
from floatbench.utils import (format_full_tables, format_paper_tables,
                              format_percentile_tables, format_regime_tables,
                              format_section_tables)

FLAGS = flags.FLAGS

# Test Data
flags.DEFINE_string("test_csv", None, "Path to test CSV with groups.")

# Model Trained
flags.DEFINE_string("model_cfg_path", None, "Path to autogluon_meta.json.")

# Sections for detailed metrics
flags.DEFINE_list(
    "detail_sections", ["section_1", "section_30"],
    "Sections for detailed metrics (default: base + top for asymmetry).")

# Bootstrap uncertainty quantification
flags.DEFINE_boolean("bootstrap", True,
                     "Compute bootstrap CIs for global metrics.")
flags.DEFINE_integer("n_bootstrap", 2000, "Number of bootstrap resamples.")
flags.DEFINE_integer("bootstrap_seed", 42, "Bootstrap RNG seed.")
flags.DEFINE_float("bootstrap_alpha", 0.05,
                   "Significance level; CI is (1 - alpha). 0.05 -> 95% CI.")

# Regenerate AutoGluon leaderboard_test.csv
flags.DEFINE_boolean(
    "regenerate_test_leaderboard", False,
    "Regenerate native AutoGluon leaderboard_test.csv. Set to True "
    "to overwrite the one produced by the training task.")

# Dev: limit number of models for smoke-tests
flags.DEFINE_integer(
    "max_models", None,
    "If set, only evaluate the first N models (dev smoke-test).")


def main(_):
    """Run the leaderboard generation pipeline."""
    logging.info("Loading test data from %s", FLAGS.test_csv)
    df_test = data.ensure_wind_wave_group(
        data.ensure_section_name(pd.read_csv(FLAGS.test_csv, low_memory=False)))

    detail_sections = (None if FLAGS.detail_sections == [] else
                       FLAGS.detail_sections)

    logging.info("Initializing predictor with model config from %s",
                 FLAGS.model_cfg_path)
    model_dir = os.path.dirname(FLAGS.model_cfg_path)
    predictor = model.AGDamagePredictor(model_cfg_path=FLAGS.model_cfg_path,
                                        model_name=None,
                                        output_dir=model_dir,
                                        report_del=True)

    logging.info("Generating leaderboard with detailed sections: %s",
                 detail_sections)
    predictor.generate_leaderboard(df_test=df_test,
                                   detail_sections=detail_sections,
                                   bootstrap=FLAGS.bootstrap,
                                   n_bootstrap=FLAGS.n_bootstrap,
                                   bootstrap_seed=FLAGS.bootstrap_seed,
                                   bootstrap_alpha=FLAGS.bootstrap_alpha,
                                   max_models=FLAGS.max_models)

    # Paper-style summary tables collected in the same subfolder as
    # the extended leaderboards (leaderboard_test_summaries/).
    if FLAGS.bootstrap:
        native_csv = os.path.join(model_dir, "leaderboard_test.csv")
        summaries_dir = os.path.join(model_dir, "leaderboard_test_summaries")
        metrics_csv = os.path.join(summaries_dir,
                                   "leaderboard_test_metrics.csv")
        os.makedirs(summaries_dir, exist_ok=True)

        # FLOATBench reports DEL only (damage is the raw label).
        targets = ("del",)
        logging.info("Generating ranked uncertainty table...")
        format_paper_tables(
            leaderboard_csv=metrics_csv,
            out_dir=summaries_dir,
            native_leaderboard_csv=native_csv,
            n_test_samples=len(df_test),
            targets=targets,
        )
        logging.info("Generating percentile error-distribution table...")
        format_percentile_tables(
            leaderboard_csv=metrics_csv,
            out_dir=summaries_dir,
            targets=targets,
        )
        logging.info("Generating full summary table (mean ± std [lo, hi])...")
        format_full_tables(
            leaderboard_csv=metrics_csv,
            out_dir=summaries_dir,
            native_leaderboard_csv=native_csv,
            n_test_samples=len(df_test),
            alpha=FLAGS.bootstrap_alpha,
            targets=targets,
        )
        groups_csv = os.path.join(summaries_dir, "leaderboard_test_groups.csv")
        if os.path.isfile(groups_csv):
            logging.info("Generating per-regime Rel_L2 table...")
            format_regime_tables(
                groups_csv=groups_csv,
                out_dir=summaries_dir,
                targets=targets,
                metric="rel_l2",
                sort_by_regime="EX_EX",
                ascending=True,
            )
        sections_csv = os.path.join(summaries_dir,
                                    "leaderboard_test_sections.csv")
        if os.path.isfile(sections_csv):
            logging.info("Generating per-section Rel_L2 table...")
            format_section_tables(
                sections_csv=sections_csv,
                out_dir=summaries_dir,
                targets=targets,
                metric="rel_l2",
                sort_by_section="section_1",
                ratio_sections=None,
                ascending=True,
            )

    # Native AutoGluon leaderboard with extra_info
    if FLAGS.regenerate_test_leaderboard:
        logging.info("Generating native leaderboard with extra_info...")
        test_data = predictor.build_test_data(df_test)
        lb = predictor.model.leaderboard(
            data=test_data,
            extra_metrics=['r2', 'mae', 'mse', 'rmse', 'mape'],
            extra_info=True,
            silent=True,
        )
        lb.to_csv(os.path.join(model_dir, "leaderboard_test.csv"), index=False)
    else:
        logging.info("Skipping native leaderboard "
                     "(--regenerate_test_leaderboard=False).")

    logging.info("Leaderboard generation complete. Results saved to %s",
                 model_dir)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
