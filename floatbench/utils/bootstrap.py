# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
"""Bootstrap CIs for regression metrics and paper-style leaderboards.

Exports two pieces that work together:

- :func:`bootstrap_regression_metrics` — nonparametric percentile-
  method bootstrap CIs + mean + std for 8 regression metrics
  (Algorithm 1 of CarBench, Elrefaie et al., 2025, arXiv:2512.07847).
- :func:`format_paper_table` / :func:`format_paper_tables` — consume
  the bootstrap ``_boot_mean`` / ``_boot_std`` columns written into
  ``leaderboard_test_metrics.csv`` and emit a CarBench Table 1-style
  ranked CSV (``mean ± std``, ISO GUM rounding).

Given a test set D_test = {(y_i, y_pred_i)}_{i=1}^N and a set of
metrics {f_m}, the bootstrap implementation follows:

    1. Initialize Theta_m = [] for each metric m
    2. for b = 1 to B:
    3.     Sample N indices with replacement: I^(b) ~ {0, ..., N-1}
    4.     Compute every metric on the resample in one pass:
           theta_m^(b) = f_m({(y_i, y_pred_i) : i in I^(b)})
    5.     Append each theta_m^(b) to its Theta_m
    6. end for
    7. For each metric m:
    8.     theta_bar_m    = (1/B) * sum_{b=1}^B theta_m^(b)
    9.     sigma_boot_m   = sqrt( (1/(B-1)) * sum_b (theta_m^(b) -
                                                   theta_bar_m)^2 )
    10.    CI_{1-alpha, m} = percentile_linear(
                                Theta_m, [alpha/2, 1-alpha/2])
    11. return {theta_bar_m, sigma_boot_m, CI_{1-alpha, m}} for all m

Notes on how this relates to Algorithm 1 / eq. (1) of CarBench
(Elrefaie et al., 2025, arXiv:2512.07847):

- Step 10 uses ``numpy.percentile`` with linear interpolation between
  adjacent order statistics instead of the integer indexing
  ``Theta[alpha/2 * B]`` shown in the paper's pseudocode. Linear
  interpolation is the statistical default (scipy, R type 7,
  statsmodels) and gives a lower-bias/lower-variance estimate of the
  quantile for continuous metric distributions; an explicit
  ``np.sort`` is therefore not needed. At B = 2000 the two methods
  differ by < 0.1% of the CI width.
- Step 9 uses ``ddof=1`` to match equation (1) of the paper (divide
  by B - 1, not B).
- The paper's Algorithm 1 handles a single metric; we fold the loop
  over metrics into the same bootstrap pass (step 4) to reuse the
  residual / abs-error / norm intermediates.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from absl import logging
from sklearn.metrics import r2_score

# Metric columns: cell = "mean ± std" (mean -> <key>_boot_mean_<target>,
# std -> <key>_boot_std_<target> in the extended leaderboard).
DEFAULT_PAPER_METRICS: Dict[str, str] = {
    "mse": "MSE",
    "mae": "MAE",
    "rmse": "RMSE",
    "mre": "MRE (%)",
    "r2": "R2",
    "rel_l2": "Rel_L2",
    "max_err": "MaxErr",
}

# Non-metric columns (capacity block + trailing Training Time) pulled
# from the native AutoGluon leaderboard. Label -> source column name.
# Transforms happen in format_paper_table:
#   Peak Memory (GB):   memory_size_w_ancestors (bytes) / 1e9
#   Mean Latency (ms):  pred_time_test (s)  / n_test_samples * 1000
#   Throughput (sps):   n_test_samples / pred_time_test (s)
#   Training Time (s):  fit_time (s)  passthrough
DEFAULT_CAPACITY_COLS: Dict[str, str] = {
    "Peak Memory (GB)": "memory_size_w_ancestors",
    "Mean Latency (ms)": "pred_time_test",
    "Throughput (sps)": "pred_time_test",
    "Training Time (s)": "fit_time",
}

# Percentile columns for the error distribution table (CarBench T4
# analogue). Median Rel. Error is the percentile of the relative
# error (dimensionless, %); the rest are percentiles of the absolute
# error in target units.
DEFAULT_PERCENTILE_COLS: Dict[str, str] = {
    "p50_rel": "Median Rel. Error (%)",
    "p50_abs": "P50 Abs Error",
    "p90_abs": "P90 Abs Error",
    "p95_abs": "P95 Abs Error",
    "p99_abs": "P99 Abs Error",
}


def bootstrap_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI + mean + std for 8 regression metrics.

    Metrics bootstrapped: r2, mse, mae, mre, rmse, bias, rel_l2,
    max_err. All 8 are computed in a single pass through each
    resample to share the ``y_pred - y_true`` intermediates.

    Args:
        y_true: Ground-truth values, shape (N,).
        y_pred: Predicted values, shape (N,).
        n_bootstrap: Number of bootstrap resamples B.
        alpha: Significance level; CI is (1 - alpha). 0.05 -> 95% CI.
        seed: RNG seed for reproducible resampling.

    Returns:
        Dict with keys ``<metric>_ci_lo``, ``<metric>_ci_hi``,
        ``<metric>_boot_mean``, ``<metric>_boot_std`` for every
        metric listed above (32 keys total).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.size

    # Step numbers below refer to the algorithm listed in this
    # module's docstring.

    # Step 1: initialize one accumulator array per metric (Theta_m).
    keys = ("r2", "mse", "mae", "mre", "rmse", "bias", "rel_l2", "max_err")
    boot = {k: np.empty(n_bootstrap) for k in keys}

    rng = np.random.default_rng(seed)

    # Steps 2-6: run B bootstrap iterations.
    for i in range(n_bootstrap):
        # Step 3: sample N indices with replacement from {0, ..., N-1}.
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]

        # Step 4: compute every metric on this resample in one pass;
        # shared intermediates (residual, abs/rel error, norm) are reused.
        e_b = yp - yt
        ae_b = np.abs(e_b)
        re_b = ae_b / np.abs(yt) * 100.0
        mse_b = float((e_b**2).mean())
        nt_b = float(np.linalg.norm(yt))

        # Step 5: append theta_m^(b) to each metric's accumulator.
        boot["r2"][i] = r2_score(yt, yp)
        boot["mse"][i] = mse_b
        boot["mae"][i] = ae_b.mean()
        boot["mre"][i] = re_b.mean()
        boot["rmse"][i] = np.sqrt(mse_b)
        boot["bias"][i] = e_b.mean()
        boot["rel_l2"][i] = (np.linalg.norm(e_b) / nt_b if nt_b > 0 else np.nan)
        boot["max_err"][i] = ae_b.max()

    # Step 10 bounds: percentile-method CI via linear-interpolation
    # quantiles (np.percentile sorts internally).
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    result: dict = {}
    for k in keys:
        # Step 8: theta_bar_m.
        result[f"{k}_boot_mean"] = float(np.mean(boot[k]))
        # Step 9: sigma_boot_m (eq. (1): ddof=1 -> 1/(B-1)).
        result[f"{k}_boot_std"] = float(np.std(boot[k], ddof=1))
        # Step 10: CI_{1-alpha, m}.
        lo, hi = np.percentile(boot[k], [lo_pct, hi_pct])
        result[f"{k}_ci_lo"] = float(lo)
        result[f"{k}_ci_hi"] = float(hi)

    return result


def _format_scientific(value: float, n_sig: int = 3) -> str:
    """Formats a single value, using scientific notation if tiny/huge.

    Args:
        value: Number to format.
        n_sig: Significant figures to keep.

    Returns:
        Fixed-point string when ``|value|`` is in [1e-3, 1e4], otherwise
        scientific notation (e.g. ``"1.37e-07"``). ``"nan"`` if
        non-finite.
    """
    if not math.isfinite(value):
        return "nan"
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.{n_sig - 1}e}"
    return f"{value:.{n_sig}g}"


def _iso_gum_format(mean: float, std: float, n_sig: int = 2) -> str:
    """Formats ``mean ± std`` with uncertainty to ``n_sig`` sig figs.

    Both values are rounded to the same decimal position so that
    trailing digits align as ISO GUM recommends. Values whose
    magnitude falls outside [1e-3, 1e4] are rendered in ISO GUM
    scientific notation with a shared exponent, e.g.
    ``"(1.370 ± 0.057)e-07"``.

    Args:
        mean: Central value.
        std: Standard deviation / standard error.
        n_sig: Number of significant figures to keep on the
          uncertainty (default 2).

    Returns:
        Formatted string. Returns ``"nan"`` if *mean* is non-finite.
    """
    if not math.isfinite(mean):
        return "nan"
    if not math.isfinite(std) or std <= 0.0:
        # No uncertainty information available — just print the mean.
        return f"{mean:g}"

    # Scientific notation when the magnitude makes fixed-point ugly.
    if mean != 0 and (abs(mean) < 1e-3 or abs(mean) >= 1e4):
        exp_base = math.floor(math.log10(abs(mean)))
        scale = 10.0**exp_base
        mean_sc = mean / scale
        std_sc = std / scale
        std_exp = math.floor(math.log10(abs(std_sc))) if std_sc > 0 else 0
        decimals = max(0, n_sig - 1 - std_exp)
        return (f"({mean_sc:.{decimals}f} ± {std_sc:.{decimals}f})"
                f"e{exp_base:+03d}")

    exponent = math.floor(math.log10(abs(std)))
    decimals = max(0, n_sig - 1 - exponent)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_paper_table(
    leaderboard_csv: str | Path,
    output_csv: str | Path,
    target: str = "damage",
    sort_by: str = "rel_l2",
    ascending: bool | None = None,
    metrics: Dict[str, str] | None = None,
    native_leaderboard_csv: str | Path | None = None,
    n_test_samples: int | None = None,
) -> pd.DataFrame:
    """Produces a ranked paper-style table from an extended leaderboard.

    Column order mirrors a CarBench-style benchmark row: model first,
    then capacity block (Peak Memory, Mean Latency, Throughput), then
    metric cells formatted as ``mean ± std``.

    Args:
        leaderboard_csv: Path to ``leaderboard_test_metrics.csv`` from
          ``AGDamagePredictor.generate_leaderboard``.
        output_csv: Where to write the formatted table.
        target: Target suffix (``"damage"`` or ``"del"``) to pull
          metrics from.
        sort_by: Metric key (no suffix) to sort the ranking by.
        ascending: Sort direction. When ``None`` (default), ``r2`` is
          descending and every other metric is ascending.
        metrics: Mapping of metric key (no suffix) to header label.
          Defaults to ``DEFAULT_PAPER_METRICS``.
        native_leaderboard_csv: Optional path to the AutoGluon native
          ``leaderboard_test.csv``. When provided, ``Peak Memory
          (GB)`` is pulled from ``memory_size_w_ancestors`` (converted
          from bytes), and ``Mean Latency (ms)`` / ``Throughput
          (sps)`` are derived from ``pred_time_test`` if
          ``n_test_samples`` is also given.
        n_test_samples: Number of test samples. Required to derive
          latency / throughput columns.

    Returns:
        The formatted DataFrame (also written to ``output_csv``).
    """
    if metrics is None:
        metrics = DEFAULT_PAPER_METRICS

    df_lb = pd.read_csv(leaderboard_csv)

    if ascending is None:
        ascending = sort_by != "r2"

    sort_col = f"{sort_by}_boot_mean_{target}"
    if sort_col not in df_lb.columns:
        # Fall back to the point estimate when the bootstrap column is
        # missing (e.g. leaderboards generated with bootstrap=False).
        sort_col = f"{sort_by}_{target}"
    if sort_col not in df_lb.columns:
        raise ValueError(
            f"Column {sort_col!r} not in leaderboard; cannot sort by "
            f"{sort_by!r}. Available: {list(df_lb.columns)}")
    df_lb = df_lb.sort_values(sort_col,
                              ascending=ascending).reset_index(drop=True)

    out = pd.DataFrame({
        "rank": range(1,
                      len(df_lb) + 1),
        "Model": df_lb["model"].values,
    })

    df_native = None
    if native_leaderboard_csv is not None and os.path.exists(
            native_leaderboard_csv):
        df_native = pd.read_csv(native_leaderboard_csv).set_index("model")

    # Capacity block (Peak Memory / Mean Latency / Throughput /
    # Training Time) pulled from the native AutoGluon leaderboard when
    # available. Training Time sits with the other timing columns so
    # cost-vs-accuracy reads top-to-bottom.
    if df_native is not None:
        if "memory_size_w_ancestors" in df_native.columns:
            out["Peak Memory (GB)"] = (
                df_native["memory_size_w_ancestors"].reindex(
                    out["Model"]).values / 1e9)
        if (n_test_samples is not None and n_test_samples > 0 and
                "pred_time_test" in df_native.columns):
            pred_t = df_native["pred_time_test"].reindex(out["Model"]).values
            out["Mean Latency (ms)"] = pred_t / n_test_samples * 1000.0
            out["Throughput (sps)"] = n_test_samples / pred_t
        if "fit_time" in df_native.columns:
            out["Training Time (s)"] = (df_native["fit_time"].reindex(
                out["Model"]).values)

    # Metric block: one "mean ± std" cell per metric.
    for key, label in metrics.items():
        mean_col = f"{key}_boot_mean_{target}"
        std_col = f"{key}_boot_std_{target}"
        if mean_col in df_lb.columns:
            std_series = df_lb[std_col]
        elif f"{key}_{target}" in df_lb.columns:
            # Leaderboard was produced without bootstrap — use the
            # point estimate with std=0 so the cell falls back to the
            # plain mean.
            mean_col = f"{key}_{target}"
            std_series = pd.Series([0.0] * len(df_lb))
        else:
            # Metric absent from this leaderboard — skip the column.
            continue
        out[label] = [
            _iso_gum_format(m, s) for m, s in zip(df_lb[mean_col], std_series)
        ]

    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _format_ci(lo: float, hi: float, n_sig: int = 3) -> str:
    """Formats a confidence interval as ``[lo, hi]`` with sensible units.

    Uses scientific notation when either bound falls outside the
    ``[1e-3, 1e4]`` window.

    Args:
        lo: Lower bound.
        hi: Upper bound.
        n_sig: Significant figures per bound.

    Returns:
        Formatted string, e.g. ``"[0.94428, 0.94614]"`` or
        ``"[6.30e-14, 6.52e-14]"``.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "[nan, nan]"
    return f"[{_format_scientific(lo, n_sig)}, {_format_scientific(hi, n_sig)}]"


def _ci_suffix(alpha: float) -> str:
    """Returns the ``ci<pct>`` filename suffix for a given alpha."""
    return f"ci{int(round((1.0 - alpha) * 100))}"


def format_full_table(
    leaderboard_csv: str | Path,
    output_csv: str | Path,
    target: str = "damage",
    sort_by: str = "rel_l2",
    ascending: bool | None = None,
    metrics: Dict[str, str] | None = None,
    native_leaderboard_csv: str | Path | None = None,
    n_test_samples: int | None = None,
) -> pd.DataFrame:
    """Produces a single "full" summary table with both ``mean ± std``
    and the bootstrap CI bounds in each metric cell.

    Use when a paper or appendix table should let the reader verify
    CI overlap without cross-referencing a second file. Layout is the
    same as :func:`format_paper_table` (capacity block + metrics +
    Training Time); only the metric cells differ, now formatted as
    ``"mean ± std [lo, hi]"``.

    Args:
        leaderboard_csv: Path to ``leaderboard_test_metrics.csv``.
        output_csv: Where to write the formatted table.
        target: Target suffix (``"damage"`` or ``"del"``).
        sort_by: Metric key to sort by.
        ascending: Sort direction. ``None`` -> descending for ``r2``,
          ascending otherwise.
        metrics: Metric key -> output label. Defaults to
          :data:`DEFAULT_PAPER_METRICS`.
        native_leaderboard_csv: Optional path to the native
          AutoGluon leaderboard for the capacity block.
        n_test_samples: Number of test samples, needed to derive
          Mean Latency / Throughput from ``pred_time_test``.

    Returns:
        The formatted DataFrame (also written to ``output_csv``).
    """
    if metrics is None:
        metrics = DEFAULT_PAPER_METRICS

    df_lb = pd.read_csv(leaderboard_csv)

    if ascending is None:
        ascending = sort_by != "r2"

    sort_col = f"{sort_by}_boot_mean_{target}"
    if sort_col not in df_lb.columns:
        sort_col = f"{sort_by}_{target}"
    if sort_col not in df_lb.columns:
        raise ValueError(
            f"Column {sort_col!r} not in leaderboard; cannot sort by "
            f"{sort_by!r}. Available: {list(df_lb.columns)}")
    df_lb = df_lb.sort_values(sort_col,
                              ascending=ascending).reset_index(drop=True)

    out = pd.DataFrame({
        "rank": range(1,
                      len(df_lb) + 1),
        "Model": df_lb["model"].values,
    })

    df_native = None
    if native_leaderboard_csv is not None and os.path.exists(
            native_leaderboard_csv):
        df_native = pd.read_csv(native_leaderboard_csv).set_index("model")

    if df_native is not None:
        mem_src = DEFAULT_CAPACITY_COLS["Peak Memory (GB)"]
        if mem_src in df_native.columns:
            out["Peak Memory (GB)"] = (
                df_native[mem_src].reindex(out["Model"]).values / 1e9)
        lat_src = DEFAULT_CAPACITY_COLS["Mean Latency (ms)"]
        if (n_test_samples is not None and n_test_samples > 0 and
                lat_src in df_native.columns):
            pred_t = df_native[lat_src].reindex(out["Model"]).values
            out["Mean Latency (ms)"] = pred_t / n_test_samples * 1000.0
            out["Throughput (sps)"] = n_test_samples / pred_t
        if "fit_time" in df_native.columns:
            out["Training Time (s)"] = (df_native["fit_time"].reindex(
                out["Model"]).values)

    for key, label in metrics.items():
        mean_col = f"{key}_boot_mean_{target}"
        std_col = f"{key}_boot_std_{target}"
        lo_col = f"{key}_ci_lo_{target}"
        hi_col = f"{key}_ci_hi_{target}"
        have_boot = mean_col in df_lb.columns
        have_ci = lo_col in df_lb.columns and hi_col in df_lb.columns
        if not have_boot and f"{key}_{target}" not in df_lb.columns:
            continue
        cells = []
        for i in range(len(df_lb)):
            if have_boot:
                mean_v = df_lb[mean_col].iloc[i]
                std_v = df_lb[std_col].iloc[i]
                cell = _iso_gum_format(mean_v, std_v)
            else:
                cell = _iso_gum_format(df_lb[f"{key}_{target}"].iloc[i], 0.0)
            if have_ci:
                cell += " " + _format_ci(df_lb[lo_col].iloc[i],
                                         df_lb[hi_col].iloc[i])
            cells.append(cell)
        out[label] = cells

    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def format_full_tables(
    leaderboard_csv: str | Path,
    out_dir: str | Path,
    targets: Iterable[str] = ("damage", "del"),
    alpha: float = 0.05,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Writes one "full" summary per target, with CI level in the name.

    Filenames follow ``leaderboard_test_summary_<ci95>_<target>.csv``
    where ``ci95`` encodes the ``(1-alpha) * 100`` confidence level,
    so different alphas coexist without overwriting.

    Args:
        leaderboard_csv: Extended leaderboard CSV.
        out_dir: Directory to write the outputs into.
        targets: Target suffixes to iterate.
        alpha: Significance level used to compute the CIs in
          ``leaderboard_csv``. Only affects the filename suffix.
        **kwargs: Forwarded to :func:`format_full_table`.

    Returns:
        Mapping of target suffix to the formatted DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    suffix = _ci_suffix(alpha)
    for target in targets:
        target_dir = Path(out_dir) / target
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"leaderboard_test_summary_{suffix}.csv"
        out[target] = format_full_table(leaderboard_csv,
                                        path,
                                        target=target,
                                        **kwargs)
    return out


def format_ci_table(
    leaderboard_csv: str | Path,
    output_csv: str | Path,
    target: str = "damage",
    sort_by: str = "rel_l2",
    ascending: bool | None = None,
    metrics: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Produces a ranked table of bootstrap CI bounds per metric.

    Complementary to :func:`format_paper_table`: same ranking, but
    each metric cell holds the percentile ``[ci_lo, ci_hi]`` bounds
    instead of ``mean ± std``. Intended for significance discussions
    in the paper ("models A and B show overlapping 95% CIs...").

    Args:
        leaderboard_csv: Path to ``leaderboard_test_metrics.csv``.
        output_csv: Where to write the formatted table.
        target: Target suffix (``"damage"`` or ``"del"``).
        sort_by: Metric key (no suffix) to sort by.
        ascending: Sort direction. ``None`` -> descending for ``r2``,
          ascending otherwise.
        metrics: Metric key -> output label. Defaults to
          :data:`DEFAULT_PAPER_METRICS`.

    Returns:
        The formatted DataFrame (also written to ``output_csv``).
    """
    if metrics is None:
        metrics = DEFAULT_PAPER_METRICS

    df_lb = pd.read_csv(leaderboard_csv)

    if ascending is None:
        ascending = sort_by != "r2"

    sort_col = f"{sort_by}_boot_mean_{target}"
    if sort_col not in df_lb.columns:
        sort_col = f"{sort_by}_{target}"
    if sort_col not in df_lb.columns:
        raise ValueError(
            f"Column {sort_col!r} not in leaderboard; cannot sort by "
            f"{sort_by!r}. Available: {list(df_lb.columns)}")
    df_lb = df_lb.sort_values(sort_col,
                              ascending=ascending).reset_index(drop=True)

    out = pd.DataFrame({
        "rank": range(1,
                      len(df_lb) + 1),
        "Model": df_lb["model"].values,
    })
    for key, label in metrics.items():
        lo_col = f"{key}_ci_lo_{target}"
        hi_col = f"{key}_ci_hi_{target}"
        if lo_col not in df_lb.columns or hi_col not in df_lb.columns:
            continue
        out[f"{label} [lo, hi]"] = [
            _format_ci(lo, hi) for lo, hi in zip(df_lb[lo_col], df_lb[hi_col])
        ]

    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def format_ci_tables(
    leaderboard_csv: str | Path,
    out_dir: str | Path,
    targets: Iterable[str] = ("damage", "del"),
    alpha: float = 0.05,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Writes one CI-bounds table per target, with CI level in name.

    Filenames follow
    ``leaderboard_test_summary_bounds_<ci95>_<target>.csv`` where
    ``ci95`` encodes the ``(1-alpha) * 100`` confidence level.

    Args:
        leaderboard_csv: Extended leaderboard CSV.
        out_dir: Directory to write the outputs into.
        targets: Target suffixes to iterate.
        alpha: Significance level used to compute the CIs. Only
          affects the filename suffix.
        **kwargs: Forwarded to :func:`format_ci_table`.

    Returns:
        Mapping of target suffix to the formatted DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    suffix = _ci_suffix(alpha)
    for target in targets:
        target_dir = Path(out_dir) / target
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"leaderboard_test_bounds_{suffix}.csv"
        out[target] = format_ci_table(leaderboard_csv,
                                      path,
                                      target=target,
                                      **kwargs)
    return out


def format_regime_table(
    groups_csv: str | Path,
    output_csv: str | Path,
    target: str = "damage",
    metric: str = "rel_l2",
    sort_by_regime: str = "EX_EX",
    ascending: bool = True,
) -> pd.DataFrame:
    """Produces a table with a metric broken down by regime.

    Reads the per-group leaderboard and extracts one metric
    (default ``rel_l2``) across every wind / wave / wind×wave regime
    for a given target. Sort defaults to the ``EX_EX`` wind×wave
    extrapolate column, which surfaces models that break down in
    joint-extrapolation at the top when ``ascending=True``.

    Args:
        groups_csv: Path to ``leaderboard_test_groups.csv``.
        output_csv: Where to write the formatted table.
        target: Target suffix (``"damage"`` or ``"del"``).
        metric: Metric key (no suffix). Must have both a global
          ``<metric>_<target>`` column and per-regime
          ``<metric>_<target>_<regime>`` columns.
        sort_by_regime: Regime suffix to sort by (e.g. ``"EX_EX"``,
          ``"wind_EX"``, ``"wave_IP"``).
        ascending: Sort direction.

    Returns:
        The formatted DataFrame (also written to ``output_csv``).
    """
    df = pd.read_csv(groups_csv)
    prefix = f"{metric}_{target}_"
    global_col = f"{metric}_{target}"
    regime_cols = [c for c in df.columns if c.startswith(prefix)]

    sort_col = f"{prefix}{sort_by_regime}"
    if sort_col not in df.columns:
        raise ValueError(
            f"Column {sort_col!r} not in groups CSV; cannot sort by "
            f"regime {sort_by_regime!r}. Available regime cols: "
            f"{regime_cols}")
    df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    out = pd.DataFrame({
        "rank": range(1,
                      len(df) + 1),
        "Model": df["model"].values,
    })
    metric_label = metric.upper() if metric != "rel_l2" else "Rel_L2"
    if global_col in df.columns:
        out[f"{metric_label} Global"] = df[global_col].values
    for c in regime_cols:
        regime = c[len(prefix):]
        out[f"{metric_label} {regime}"] = df[c].values

    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def format_regime_tables(
    groups_csv: str | Path,
    out_dir: str | Path,
    targets: Iterable[str] = ("damage", "del"),
    metric: str = "rel_l2",
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Writes one regime-breakdown table per target.

    Filenames: ``<target>/leaderboard_test_regime_<metric>.csv``.

    Args:
        groups_csv: Per-group leaderboard CSV.
        out_dir: Directory to write the outputs into.
        targets: Target suffixes to iterate.
        metric: Metric key to project across regimes.
        **kwargs: Forwarded to :func:`format_regime_table`.

    Returns:
        Mapping of target suffix to the formatted DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    for target in targets:
        target_dir = Path(out_dir) / target
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"leaderboard_test_regime_{metric}.csv"
        try:
            out[target] = format_regime_table(groups_csv,
                                              path,
                                              target=target,
                                              metric=metric,
                                              **kwargs)
        except ValueError as exc:
            # Target lacks the per-regime columns (legacy groups CSVs
            # only carry a subset of metrics). Skip gracefully.
            out[target] = None
            logging.warning("Skipping regime table for %s: %s", target, exc)
    return out


def format_section_tables(
    sections_csv: str | Path,
    out_dir: str | Path,
    targets: Iterable[str] = ("damage", "del"),
    metric: str = "rel_l2",
    sort_by_section: str | None = None,
    ratio_sections: tuple = ("section_30", "section_1"),
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Writes one per-section breakdown table per target.

    Thin wrapper around :func:`format_regime_table` that reuses the
    same logic against ``leaderboard_test_sections.csv``. When
    ``ratio_sections`` is given and both sections are present in the
    output, appends a ``"<metric> ratio (<num>/<denom>)"`` column so
    asymmetric models (e.g. good at base / bad at top) surface
    without reading across many columns.

    Filenames: ``<target>/leaderboard_test_section_<metric>.csv``.

    Args:
        sections_csv: Per-section leaderboard CSV.
        out_dir: Directory to write the outputs into.
        targets: Target suffixes to iterate.
        metric: Metric key to project across sections.
        sort_by_section: Section name to sort on (e.g. ``section_1``).
        ratio_sections: ``(numerator, denominator)`` pair for the
          asymmetry column. Pass ``None`` to skip.
        **kwargs: Forwarded to :func:`format_regime_table`.

    Returns:
        Mapping of target suffix to the formatted DataFrame.
    """
    ascending = kwargs.pop("ascending", True)
    out: Dict[str, pd.DataFrame] = {}
    for target in targets:
        target_dir = Path(out_dir) / target
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"leaderboard_test_section_{metric}.csv"
        seed_sort = sort_by_section or "section_1"
        try:
            df = format_regime_table(sections_csv,
                                     path,
                                     target=target,
                                     metric=metric,
                                     sort_by_regime=seed_sort,
                                     ascending=ascending,
                                     **kwargs)
        except ValueError as exc:
            out[target] = None
            logging.warning("Skipping section table for %s: %s", target, exc)
            continue

        metric_label = ("Rel_L2" if metric == "rel_l2" else metric.upper())

        # Default: sort by Rel_L2 Global (not the mean of sections —
        # they are near-identical) when no section was requested.
        if sort_by_section is None:
            global_col = f"{metric_label} Global"
            if global_col in df.columns:
                df = df.sort_values(global_col,
                                    ascending=ascending).reset_index(drop=True)
                df["rank"] = range(1, len(df) + 1)

        # Asymmetry diagnostic: (s_num - s_denom) / Global. Signed,
        # dimensionless, normalized by the overall error. 0 -> uniform;
        # positive -> top worse than base; negative -> base worse
        # (typical for damage prediction).
        global_col = f"{metric_label} Global"
        if ratio_sections is not None and global_col in df.columns:
            num, denom = ratio_sections
            num_col = next((c for c in df.columns if c.endswith(f" {num}")),
                           None)
            denom_col = next((c for c in df.columns if c.endswith(f" {denom}")),
                             None)
            if num_col and denom_col:
                asym_label = f"{metric_label} ({num}-{denom})/Global"
                df[asym_label] = (df[num_col] - df[denom_col]) / df[global_col]

        df.to_csv(path, index=False)
        out[target] = df
    return out


def format_percentile_table(
    leaderboard_csv: str | Path,
    output_csv: str | Path,
    target: str = "damage",
    ascending: bool = True,
    percentile_cols: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Produces a percentile error-distribution table.

    Content mirrors CarBench Table 4: one row per model with the
    Median Rel. Error plus percentiles of the absolute error (P50,
    P90, P95, P99). Default sort is best-to-worst by Median Rel.
    Error so rank 1 is the top performer — the usual NeurIPS
    leaderboard convention. Pass ``ascending=False`` to replicate
    CarBench's worst-to-best ordering.

    Args:
        leaderboard_csv: Path to ``leaderboard_test_metrics.csv``.
        output_csv: Where to write the formatted table.
        target: Target suffix (``"damage"`` or ``"del"``).
        ascending: If True (default), lowest Median Rel. Error goes
          first (rank 1 = best).
        percentile_cols: Mapping of source column key to output label.
          Defaults to :data:`DEFAULT_PERCENTILE_COLS`.

    Returns:
        The formatted DataFrame (also written to ``output_csv``).
    """
    if percentile_cols is None:
        percentile_cols = DEFAULT_PERCENTILE_COLS

    df_lb = pd.read_csv(leaderboard_csv)

    sort_col = f"p50_rel_{target}"
    if sort_col not in df_lb.columns:
        raise ValueError(
            f"Column {sort_col!r} not in leaderboard; cannot sort. "
            f"Available: {list(df_lb.columns)}")
    df_lb = df_lb.sort_values(sort_col,
                              ascending=ascending).reset_index(drop=True)

    out = pd.DataFrame({
        "rank": range(1,
                      len(df_lb) + 1),
        "Model": df_lb["model"].values,
    })
    for key, label in percentile_cols.items():
        src = f"{key}_{target}"
        if src not in df_lb.columns:
            continue
        # Median Rel. Error stays in % (dimensionless, readable range).
        # Absolute-error percentiles get scientific notation when tiny.
        if key.endswith("_rel"):
            out[label] = df_lb[src].values
        else:
            out[label] = [_format_scientific(v) for v in df_lb[src].values]

    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def format_percentile_tables(
    leaderboard_csv: str | Path,
    out_dir: str | Path,
    targets: Iterable[str] = ("damage", "del"),
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Writes one percentile error table per target.

    Args:
        leaderboard_csv: Extended leaderboard CSV.
        out_dir: Directory to write
          ``leaderboard_test_percentiles_<target>.csv`` into.
        targets: Iterable of target suffixes.
        **kwargs: Forwarded to :func:`format_percentile_table`.

    Returns:
        Mapping of target suffix to the formatted DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    for target in targets:
        target_dir = Path(out_dir) / target
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "leaderboard_test_percentiles.csv"
        out[target] = format_percentile_table(leaderboard_csv,
                                              path,
                                              target=target,
                                              **kwargs)
    return out


def format_paper_tables(
    leaderboard_csv: str | Path,
    out_dir: str | Path,
    targets: Iterable[str] = ("damage", "del"),
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Writes one paper-style table per target next to the leaderboard.

    Args:
        leaderboard_csv: Extended leaderboard CSV.
        out_dir: Directory to write
          ``leaderboard_test_summary_<target>.csv`` into.
        targets: Iterable of target suffixes to iterate (``"damage"``,
          ``"del"``).
        **kwargs: Forwarded to :func:`format_paper_table`.

    Returns:
        Mapping of target suffix to the formatted DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    for target in targets:
        target_dir = Path(out_dir) / target
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "leaderboard_test_summary.csv"
        out[target] = format_paper_table(leaderboard_csv,
                                         path,
                                         target=target,
                                         **kwargs)
    return out
