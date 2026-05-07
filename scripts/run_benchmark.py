"""Run the full FLOATBench AutoGluon benchmark.

Modes:
    --experiment=within  : within-tower (E2). Trains on the tower's own
        train set, evaluates on its test set. Set --tower to one of
        ref / opt1 / opt2, or to ``all`` to iterate over all three.
    --experiment=cross   : cross-tower (E3). Trains on the union of
        two towers and evaluates on the held-out third. Set --held_out
        to one of ref / opt1 / opt2, or to ``all`` to iterate over the
        three folds.

Stages (per run):
    1. train (best preset, hyperparameters=zeroshot)
    2. train (extreme preset, hyperparameters=zeroshot_2025_tabfm)
    3. leaderboard (bootstrap CI tables) for each preset
    4. benchmark (cross-preset merge: heatmaps, bump, family, model_pool)

Usage:
    # Within-tower E2 on ref only
    python scripts/run_benchmark.py --experiment=within --tower=ref

    # Within-tower E2 on all three towers
    python scripts/run_benchmark.py --experiment=within --tower=all

    # Cross-tower E3, opt2 held out (train on ref + opt1)
    python scripts/run_benchmark.py --experiment=cross --held_out=opt2

    # All 3 cross-tower folds
    python scripts/run_benchmark.py --experiment=cross --held_out=all

    # Smoke test (120s per training)
    python scripts/run_benchmark.py --experiment=within --tower=ref \\
        --time_limit=120
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from absl import app, flags, logging

FLAGS = flags.FLAGS

TOWERS = ("ref", "opt1", "opt2")

flags.DEFINE_enum("experiment", "within", ["within", "cross", "all"],
                  "Experiment type: within-tower (E2), cross-tower (E3), or "
                  "all (3 within + 3 cross folds in sequence).")
flags.DEFINE_enum("tower", None, list(TOWERS) + ["all"],
                  "Tower for within-tower mode.")
flags.DEFINE_enum("held_out", None, list(TOWERS) + ["all"],
                  "Held-out tower for cross-tower mode.")
flags.DEFINE_string("output_root", None,
                    "Root directory for outputs (default: outputs/<tower>).")
flags.DEFINE_integer("time_limit", 14400,
                     "AutoGluon time limit per training in seconds.")
flags.DEFINE_string("data_root", "data",
                    "Root directory for the dataset (default: data).")
flags.DEFINE_boolean(
    "skip_train", False, "Skip the training stages and reuse existing trained "
    "models in <output_root>/{best,extreme}/model.")
flags.DEFINE_boolean("skip_leaderboard", False,
                     "Skip the bootstrap-leaderboard stages.")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _run(cmd, stage_name):
    """Run a subprocess command, raising on failure."""
    logging.info("=== %s ===", stage_name)
    logging.info("$ %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed (exit {result.returncode}, {elapsed:.0f}s)")
    logging.info("=== %s done (%.0fs) ===", stage_name, elapsed)


def _run_train(out_root, train_csv, test_csv, preset, hp, time_limit):
    out = out_root / preset
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "train" / "run.py"),
        f"--flagfile={SCRIPTS / 'train' / 'config.cfg'}",
        f"--train_csv={train_csv}",
        f"--test_csv={test_csv}",
        f"--presets={preset}",
        f"--hyperparameters={hp}",
        f"--time_limit={time_limit}",
        f"--output_dir={out}",
    ]
    _run(cmd, f"train {preset}")


def _run_leaderboard(out_root, test_csv, preset):
    cfg = out_root / preset / "model" / "autogluon_meta.json"
    cmd = [
        sys.executable,
        str(SCRIPTS / "leaderboard" / "run.py"),
        f"--flagfile={SCRIPTS / 'leaderboard' / 'config.cfg'}",
        f"--test_csv={test_csv}",
        f"--model_cfg_path={cfg}",
    ]
    _run(cmd, f"leaderboard {preset}")


def _run_benchmark(out_root):
    out = out_root / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "benchmark" / "run.py"),
        f"--flagfile={SCRIPTS / 'benchmark' / 'config.cfg'}",
        f"--base_dir={out_root}",
        f"--output_dir={out}",
    ]
    _run(cmd, "benchmark")


def _build_cross_train_csv(held_out, data_root, out_root):
    """Concatenate train CSVs of the two non-held-out towers."""
    sources = [t for t in TOWERS if t != held_out]
    paths = [data_root / t / "train_damage.csv" for t in sources]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"train_{'_'.join(sources)}.csv"
    df.to_csv(out_csv, index=False)
    logging.info("Cross-tower train CSV: %s rows from %s -> %s", len(df),
                 ", ".join(sources), out_csv)
    return out_csv


def _resolve_runs():
    """Expand FLAGS.experiment + tower/held_out into a list of runs.

    Returns:
        List of (run_label, train_csv, test_csv, output_root) tuples.
    """
    data_root = REPO_ROOT / FLAGS.data_root
    runs = []

    do_within = FLAGS.experiment in ("within", "all")
    do_cross = FLAGS.experiment in ("cross", "all")

    if do_within:
        towers = (TOWERS if FLAGS.experiment == "all" or FLAGS.tower == "all"
                  else (FLAGS.tower, ))
        if towers == (None, ):
            raise ValueError("--tower is required for --experiment=within")
        for t in towers:
            out_root = Path(FLAGS.output_root
                            or REPO_ROOT / "outputs" / "within" / t)
            runs.append((t, data_root / t / "train_damage.csv",
                         data_root / t / "test_damage.csv", out_root))

    if do_cross:
        held = (TOWERS if FLAGS.experiment == "all" or FLAGS.held_out == "all"
                else (FLAGS.held_out, ))
        if held == (None, ):
            raise ValueError("--held_out is required for --experiment=cross")
        for h in held:
            label = f"cross_{h}"
            out_root = Path(FLAGS.output_root
                            or REPO_ROOT / "outputs" / "cross" / h)
            train_csv = _build_cross_train_csv(h, data_root, out_root)
            runs.append((label, train_csv,
                         data_root / h / "test_damage.csv", out_root))
    return runs


def main(_):
    runs = _resolve_runs()
    presets = (("best", "zeroshot"), ("extreme", "zeroshot_2025_tabfm"))

    for label, train_csv, test_csv, out_root in runs:
        logging.info("######## RUN %s ########", label)
        if not train_csv.is_file() or not test_csv.is_file():
            logging.error("Missing CSV(s): %s, %s", train_csv, test_csv)
            sys.exit(1)
        out_root.mkdir(parents=True, exist_ok=True)
        if not FLAGS.skip_train:
            for preset, hp in presets:
                _run_train(out_root, train_csv, test_csv, preset, hp,
                           FLAGS.time_limit)
        if not FLAGS.skip_leaderboard:
            for preset, _ in presets:
                _run_leaderboard(out_root, test_csv, preset)
        _run_benchmark(out_root)
        logging.info("######## DONE %s -> %s ########", label, out_root)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
