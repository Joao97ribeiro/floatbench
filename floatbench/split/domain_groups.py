# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-positional-arguments
"""Domain grouping (wind & wave) via k-NN distance thresholds."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, List, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .. import plots
from .domain_boundary import AlphaShape


class WindWaveDomainGrouper:
    """Groups wind and wave feature spaces by k-NN distance from training data.

    This grouper learns absolute distance thresholds from the training set's
    internal spacing and then assigns each test point to a domain region
    (e.g., "in-train", "interpolate", "extrapolate") according to its
    distance to the nearest training samples.

    Attributes:
        wind_cols: Column names defining the wind feature space.
        wave_cols: Column names defining the wave feature space.
        k: Number of neighbors used for k-NN distance calculation.
        aggregate: Distance aggregation method ('min', 'kth', 'mean').
        scale_stat: Statistic used to define the characteristic train
          spacing ('median', 'mean').
        wind_edges: Distance thresholds (shared with wave_edges).
        wave_edges: Distance thresholds (shared with wind_edges).
        wind_names: Ordered interpolation group names (shared).
        wave_names: Ordered interpolation group names (shared).
        extrap_names: Ordered extrapolation group names.
        extrap_edges: Distance thresholds for extrapolation sub-groups.
        boundary_alpha: Alpha parameter for the alpha shape boundary.
        kind_scaler: Type of scaler to apply to the feature spaces.

    Notes:
      - The thresholds (edges) are derived once from the train-train distances
          and remain fixed when labeling test data.
      - The resulting groups represent different extrapolation levels relative
          to the training domain.
    """

    def __init__(
        self,
        wind_cols: Tuple[str, str] = ("mean_wind_speed", "std_wind_speed"),
        wave_cols: Tuple[str, str] = ("wave_hs", "wave_tp"),
        k: int = 1,
        aggregate: Literal["min", "kth", "mean"] = "min",
        scale_stat: Literal["median", "mean"] = "median",
        interp_edges: Optional[Sequence[float]] = None,
        interp_names: Optional[Sequence[str]] = None,
        extrap_names: Optional[Sequence[str]] = None,
        extrap_edges: Optional[Sequence[float]] = None,
        kind_scaler: Literal["standard", "minmax", "robust"] = "standard",
        boundary_alpha: float = 0.1,
    ) -> None:
        """Initializes the domain grouper.

        Args:
            wind_cols: Column names (x, y) for the wind feature space.
            wave_cols: Column names (x, y) for the wave feature space.
            k: Number of neighbors for k-NN distance.
            aggregate: Distance aggregation ('min', 'kth', 'mean').
            scale_stat: Statistic for train spacing ('median', 'mean').
            interp_edges: Distance thresholds for interpolation groups
                (shared by wind and wave).
            interp_names: Ordered group names for interpolation
                (shared by wind and wave).
            extrap_names: Names for extrapolation groups outside the
                alpha shape boundary. Defaults to ['Extrapolate'].
            extrap_edges: Distance thresholds to subdivide
                extrapolation groups by k-NN distance (d_norm).
                If empty, all outside points get extrap_names[0].
            kind_scaler: Scaler type ('standard', 'minmax', 'robust').
            boundary_alpha: Alpha parameter for the alpha shape boundary.
                Smaller values produce tighter (more concave) hulls.
        """

        self.wind_cols: List[str] = list(wind_cols)
        self.wave_cols: List[str] = list(wave_cols)

        self.k = k
        self.aggregate = aggregate
        self.scale_stat = scale_stat

        # Shared interpolation edges and names for wind and wave
        edges = (list(interp_edges)
                 if interp_edges is not None else [0.25, 1.0, 1.5, 2.0])
        names = (list(interp_names) if interp_names is not None else [
            "In-train", "Interpolate", "Low-extrapolate", "Extrapolate",
            "High-extrapolate"
        ])
        self.wind_edges: List[float] = edges
        self.wave_edges: List[float] = edges
        self.wind_names: List[str] = names
        self.wave_names: List[str] = names

        self.extrap_names: List[str] = (list(extrap_names) if extrap_names
                                        is not None else ["Extrapolate"])
        self.extrap_edges: List[float] = (list(extrap_edges)
                                          if extrap_edges else [])

        self.boundary_alpha = boundary_alpha

        # state
        self._wind_scale: Optional[float] = None
        self._wave_scale: Optional[float] = None
        self._wind_summary: Optional[Dict[str, Any]] = None
        self._wave_summary: Optional[Dict[str, Any]] = None
        self._wind_shape: Optional[AlphaShape] = None
        self._wave_shape: Optional[AlphaShape] = None
        self._fitted: bool = False

        self.kind_scaler = kind_scaler

        if len(self.wind_names) != len(self.wind_edges) + 1:
            raise ValueError("len(interp_names) must be len(interp_edges)+1")
        if any(m <= 0 for m in self.wind_edges):
            raise ValueError("All multipliers must be positive.")

    def _learn_thresholds(self,
                          df_train_scaled: pd.DataFrame,
                          plot_dir: str = None,
                          save_svg: bool = False) -> "WindWaveDomainGrouper":
        """Fit edges from train spacing only.

        Args:
            df_train_scaled: DataFrame with scaled training data.
            plot_dir: Directory to save diagnostic plots (if None, no plots are
              saved).
            save_svg: Whether to save plots as SVG in addition to PNG.

        Returns:
            Self.
        """

        # wind
        xtr_wind = df_train_scaled[self.wind_cols].to_numpy(float)
        xtr_wind_unique = np.unique(xtr_wind, axis=0)
        sp_wind = _nn_train_spacing(xtr_wind_unique, dedup=True)
        self._wind_scale = _compute_scale_from_spacing(sp_wind, self.scale_stat)
        self._wind_summary = _spacing_summary(sp_wind)

        # wave
        xtr_wave = df_train_scaled[self.wave_cols].to_numpy(float)
        xtr_wave_unique = np.unique(xtr_wave, axis=0)
        sp_waves = _nn_train_spacing(xtr_wave_unique, dedup=True)
        self._wave_scale = _compute_scale_from_spacing(sp_waves,
                                                       self.scale_stat)
        self._wave_summary = _spacing_summary(sp_waves)

        if plot_dir is not None:
            plot_base_dir = os.path.join(plot_dir, "train_test", "test_groups",
                                         "plots", "dist")
            os.makedirs(os.path.dirname(plot_base_dir), exist_ok=True)

            # wind
            plots.plot_distance_histogram(
                values=sp_wind,
                space="wind",
                plot_dir=plot_base_dir,
                show_mean=True,
                save_svg=save_svg,
                train_reference=True,
                title=f"Normalized ({self.kind_scaler}) Distance Histogram")

            # wave
            plots.plot_distance_histogram(
                values=sp_waves,
                space="wave",
                plot_dir=plot_base_dir,
                show_mean=True,
                save_svg=save_svg,
                train_reference=True,
                title=f"Normalized ({self.kind_scaler}) Distance Histogram")

        self._fitted = True

        return self

    def assign_groups(
        self,
        df_train_scaled: pd.DataFrame,
        df_test_scaled: pd.DataFrame,
        plot_dir: str = None,
        save_svg: bool = False,
        df_train_orig: pd.DataFrame = None,
        df_test_orig: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Label test/val points.

        Args:
            df_train_scaled: DataFrame with scaled training data.
            df_test_scaled: DataFrame with scaled test/validation data.
            plot_dir: Directory to save diagnostic plots (if None, no plots are
              saved).
            save_svg: Whether to save plots as SVG in addition to PNG.
            df_train_orig: DataFrame with original (unscaled) training data
              for convex hull check.
            df_test_orig: DataFrame with original (unscaled) test data
              for convex hull check.

        Returns:
            Tuple of (labeled test DataFrame, metadata dictionary).
        """
        self._check_fitted()

        # wind
        xtr_wind = df_train_scaled[self.wind_cols].to_numpy(float)
        xtr_wind_unique = np.unique(xtr_wind, axis=0)
        xte_wind = df_test_scaled[self.wind_cols].to_numpy(float)
        d_wind = _knn_dist(xtr_wind_unique,
                           xte_wind,
                           k=self.k,
                           aggregate=self.aggregate)
        d_wind_norm = d_wind / self._wind_scale
        lab_wind = _labels_from_edges(d_wind_norm, self.wind_edges,
                                      self.wind_names)

        lab_wind, self._wind_shape = self._apply_alpha_override(
            lab_wind, d_wind_norm, self.wind_edges, self._wind_scale,
            xtr_wind_unique, xte_wind, self.wind_cols, df_train_orig,
            df_test_orig)

        # wave
        xtr_wave = df_train_scaled[self.wave_cols].to_numpy(float)
        xtr_wave_unique = np.unique(xtr_wave, axis=0)
        xte_wave = df_test_scaled[self.wave_cols].to_numpy(float)
        d_wave = _knn_dist(xtr_wave_unique,
                           xte_wave,
                           k=self.k,
                           aggregate=self.aggregate)

        d_wave_norm = d_wave / self._wave_scale
        lab_wave = _labels_from_edges(d_wave_norm, self.wave_edges,
                                      self.wave_names)

        lab_wave, self._wave_shape = self._apply_alpha_override(
            lab_wave, d_wave_norm, self.wave_edges, self._wave_scale,
            xtr_wave_unique, xte_wave, self.wave_cols, df_train_orig,
            df_test_orig)

        df_groups_test = df_test_scaled.copy()
        df_groups_test["wind_dist"] = d_wind_norm
        df_groups_test["wind_group"] = lab_wind
        df_groups_test["wave_dist"] = d_wave_norm
        df_groups_test["wave_group"] = lab_wave
        df_groups_test["wind_wave_group"] = df_groups_test["wind_group"].astype(
            str) + "_" + df_groups_test["wave_group"].astype(str)

        meta = {
            "k": int(self.k),
            "aggregate": str(self.aggregate),
            "kind_scaler": str(self.kind_scaler),
            "wind": {
                "cols": self.wind_cols,
                "edges": self.wind_edges,
                "names": self.wind_names,
                "train_spacing_summary": {
                    k:
                        float(v) if isinstance(v,
                                               (np.floating, np.integer)) else v
                    for k, v in self._wind_summary.items()
                },
            },
            "wave": {
                "cols": self.wave_cols,
                "edges": self.wave_edges,
                "names": self.wave_names,
                "train_spacing_summary": {
                    k:
                        float(v) if isinstance(v,
                                               (np.floating, np.integer)) else v
                    for k, v in self._wave_summary.items()
                },
            },
        }

        if plot_dir is not None:
            plot_base_dir = os.path.join(plot_dir, "train_test", "test_groups",
                                         "plots")
            os.makedirs(os.path.dirname(plot_base_dir), exist_ok=True)

            plots.plot_train_test_subplots(
                df_train=df_train_scaled,
                df_test=df_groups_test,
                pairs=[
                    self.wind_cols,
                    self.wave_cols,
                ],
                group_col=("wind_group", "wave_group"),
                axis_labels=[("Normalized Mean Wind Speed (m/s)",
                              "Normalized Std Wind Speed (m/s)"),
                             ([
                                 "Normalized Wave Height (m)",
                                 "Normalized Wave Period (s)"
                             ])],
                group_names=(extend_group_names(self.wind_names,
                                                self.extrap_names),
                             extend_group_names(self.wave_names,
                                                self.extrap_names)),
                output_dir=plot_base_dir,
                filename="train_testgroups_norm",
                save_svg=save_svg,
                save_separate=True,
                save_combined=True,
                separate_suffix=("wind", "wave"),
                titles=("Wind Distribution: Train vs. Test Groups",
                        "Wave Distribution: Train vs. Test Groups"))

            plot_dist_dir = os.path.join(plot_dir, "train_test", "test_groups",
                                         "plots", "dist")
            os.makedirs(os.path.dirname(plot_dist_dir), exist_ok=True)

            plots.plot_distance_histogram(
                values=d_wind_norm,
                space="wind",
                edges=self.wind_edges,
                group_names=self.wind_names,
                plot_dir=plot_dist_dir,
                save_svg=save_svg,
                title=(f"Normalized ({self.kind_scaler}/{self.scale_stat}) "
                       "Distance Histogram"))
            plots.plot_distance_histogram(
                values=d_wave_norm,
                space="wave",
                edges=self.wave_edges,
                group_names=self.wave_names,
                plot_dir=plot_dist_dir,
                save_svg=save_svg,
                title=(f"Normalized ({self.kind_scaler}/{self.scale_stat}) "
                       "Distance Histogram"))

        return df_groups_test, meta

    def group(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        plot_dir: str = None,
        save_svg: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Learn thresholds and assign groups.

        Args:
            df_train: DataFrame with training data.
            df_test: DataFrame with test/validation data.
            plot_dir: Directory to save diagnostic plots (if None, no plots are
              saved).
            save_svg: Whether to save plots as SVG in addition to PNG.

        Returns:
            Tuple of (labeled test DataFrame, domain thresholds metadata,
              test groups metadata).
        """
        scaler = _make_scaler(kind=self.kind_scaler)
        features = list(self.wind_cols) + list(self.wave_cols)
        scaler.fit(df_train[features])

        df_train_scaled = df_train.copy()
        df_test_scaled = df_test.copy()

        df_train_scaled[features] = scaler.transform(df_train[features])
        df_test_scaled[features] = scaler.transform(df_test[features])

        out, domain_thresholds_metadata = self._learn_thresholds(
            df_train_scaled, plot_dir,
            save_svg).assign_groups(df_train_scaled, df_test_scaled, plot_dir,
                                    save_svg, df_train, df_test)

        # copy wind/wave distances and groups to original df_test
        df_out = df_test.copy()
        df_out["wind_dist"] = out["wind_dist"]
        df_out["wind_group"] = out["wind_group"]
        df_out["wave_dist"] = out["wave_dist"]
        df_out["wave_group"] = out["wave_group"]
        df_out["wind_wave_group"] = out["wind_wave_group"]

        # test groups summary
        test_groups_metadata = {"test_groups": {}}

        # define group types
        test_groups_metadata["test_groups"]["group_types"] = [
            "Windgroup_Wavegroup",
            "Windgroup",
            "Wavegroup",
        ]

        # 1) Combined wind + wave groups
        test_groups_metadata["test_groups"]["Windgroup_Wavegroup"] = {}

        for group in df_out["wind_wave_group"].unique():
            n_samples = int((df_out["wind_wave_group"] == group).sum())
            pct_samples = round((n_samples / len(df_out)) * 100, 2)

            test_groups_metadata["test_groups"]["Windgroup_Wavegroup"][
                group] = {
                    "num_samples": n_samples,
                    "%_samples": pct_samples,
                }

        # 2) Wind groups only
        test_groups_metadata["test_groups"]["Windgroup"] = {}

        for group in df_out["wind_group"].unique():
            n_samples = int((df_out["wind_group"] == group).sum())
            pct_samples = round((n_samples / len(df_out)) * 100, 2)

            test_groups_metadata["test_groups"]["Windgroup"][group] = {
                "num_samples": n_samples,
                "%_samples": pct_samples,
            }

        # 3) Wave groups only
        test_groups_metadata["test_groups"]["Wavegroup"] = {}

        for group in df_out["wave_group"].unique():
            n_samples = int((df_out["wave_group"] == group).sum())
            pct_samples = round((n_samples / len(df_out)) * 100, 2)

            test_groups_metadata["test_groups"]["Wavegroup"][group] = {
                "num_samples": n_samples,
                "%_samples": pct_samples,
            }

        # total
        test_groups_metadata["total_test_samples"] = len(df_out)

        return df_out, domain_thresholds_metadata, test_groups_metadata

    def crosstab(
        self,
        df_domains: pd.DataFrame,
        csv_dir: str = None,
        add_totals: bool = True,
        save_percentages: bool = True,
    ) -> pd.DataFrame:
        """Contingency table wind_group × wave_group.

        Args:
            df_domains: DataFrame output from .assign_groups() or .group().
            csv_dir: Directory to save CSV file (if None, no file is saved).
            add_totals: Whether to add row/column totals.
            save_percentages: Whether to also save percentages in addition to
              counts.

        Returns:
            Crosstab DataFrame.
        """

        ct = pd.crosstab(df_domains["wind_group"],
                         df_domains["wave_group"]).sort_index()
        all_wind = extend_group_names(self.wind_names, self.extrap_names)
        all_wave = extend_group_names(self.wave_names, self.extrap_names)
        ct = ct.reindex(index=all_wind, columns=all_wave).fillna(0).astype(int)

        if add_totals:
            ct["Total (wind_group)"] = ct.sum(axis=1)
            ct.loc["Total (wave_group)"] = ct.sum(axis=0)

        if save_percentages:
            total = ct.loc["Total (wave_group)", "Total (wind_group)"]
            ct_pct = (ct / total * 100).round(2)

        if csv_dir is not None:
            save_path = os.path.join(csv_dir, "train_test", "test_groups",
                                     "crosstab_wind_wave.csv")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ct.to_csv(save_path, index_label="wind_group\\wave_group")

            if save_percentages:
                ct_pct.to_csv(save_path.replace(".csv", "_pct.csv"),
                              index_label="wind_group\\wave_group")

        return ct

    def boundary_polygons(self) -> Tuple[Optional[list], Optional[list]]:
        """Return the wind and wave alpha shape boundary polygons.

        Returns:
            Tuple of (wind_polygon, wave_polygon). Each is a list of
            (x, y) tuples forming the boundary, or None if not fitted.
        """
        wind_poly = (self._wind_shape.boundary_polygon()
                     if self._wind_shape is not None else None)
        wave_poly = (self._wave_shape.boundary_polygon()
                     if self._wave_shape is not None else None)
        return wind_poly, wave_poly

    def _apply_alpha_override(
        self,
        labels: np.ndarray,
        d_norm: np.ndarray,
        edges: List[float],
        scale: float,
        xtr_scaled: np.ndarray,
        xte_scaled: np.ndarray,
        cols: List[str],
        df_train_orig: Optional[pd.DataFrame],
        df_test_orig: Optional[pd.DataFrame],
    ) -> Tuple[np.ndarray, AlphaShape]:
        """Override labels for points outside the alpha shape boundary.

        Points beyond the boundary + offset are labelled using
        ``self.extrap_names`` and ``self.extrap_edges``.

        Args:
            labels: Current group labels array.
            d_norm: Normalized distances.
            edges: Distance threshold edges.
            scale: Scale factor for the feature space.
            xtr_scaled: Unique scaled training points.
            xte_scaled: Scaled test points.
            cols: Column names for the feature space.
            df_train_orig: Original (unscaled) training DataFrame.
            df_test_orig: Original (unscaled) test DataFrame.

        Returns:
            Tuple of (updated labels, fitted AlphaShape).
        """
        # Offset = In-train threshold converted to original space.
        # Points outside the alpha shape but closer than this to the
        # boundary are kept as interpolation (not extrapolation).
        boundary_offset = edges[0] * scale

        # Use original (unscaled) points for alpha shape if available,
        # so the boundary follows the real data shape (not distorted
        # by the scaler).
        if df_train_orig is not None and df_test_orig is not None:
            train_pts = np.unique(df_train_orig[cols].to_numpy(float), axis=0)
            test_pts = df_test_orig[cols].to_numpy(float)
        else:
            train_pts = xtr_scaled
            test_pts = xte_scaled

        # Fit alpha shape and check which test points are outside
        shape = AlphaShape(alpha=self.boundary_alpha).fit(train_pts)
        is_outside, dist_to_boundary = shape.query(test_pts)

        # Only mark as extrapolation if outside AND far enough from
        # the boundary (beyond the offset)
        is_extrapolation = is_outside & (dist_to_boundary >= boundary_offset)

        # Assign extrapolation labels
        labels[is_extrapolation] = self.extrap_names[0]

        # If multiple extrap levels defined, subdivide by k-NN distance
        for i, edge_threshold in enumerate(self.extrap_edges):
            if i + 1 < len(self.extrap_names):
                is_far = is_extrapolation & (d_norm >= edge_threshold)
                labels[is_far] = self.extrap_names[i + 1]

        return labels, shape

    def _check_fitted(self) -> None:
        """Raise if not yet learned thresholds."""
        if not self._fitted:
            raise RuntimeError(
                "Call ._learn_thresholds() before assigning groups.")


def _nn_train_spacing(xtr: np.ndarray, dedup: bool = True) -> np.ndarray:
    """Compute nearest-neighbor spacing within training data.

    Args:
        xtr: Training data points (N x D).
        dedup: Whether to remove duplicate points before spacing calc.

    Returns:
        Array of nearest-neighbor distances (N,).
    """

    if dedup:

        xtr = np.unique(xtr, axis=0)
    if len(xtr) <= 1:

        return np.array([0.0])
    nn = NearestNeighbors(n_neighbors=min(2, len(xtr))).fit(xtr)
    d, _ = nn.kneighbors(xtr)

    return d[:, 1]


def _compute_scale_from_spacing(dist: np.ndarray, use: str = "median") -> float:
    """Compute a scale factor from the train spacing distribution.

    Args:
        dist: Array of nearest-neighbor distances within training data.
        use: Statistic to compute scale ('median' or 'mean').

    Returns:
        Scale factor (float).
    """

    if use == "median":
        scale = float(np.median(dist))
    elif use == "mean":
        scale = float(np.mean(dist))
    else:
        raise ValueError("scale_stat must be 'median' or 'mean'")
    if scale < 1e-12:
        scale = float(np.quantile(dist, 0.9)) or 1e-3

    return scale


def _knn_dist(xtr: np.ndarray,
              xte: np.ndarray,
              k: int = 1,
              aggregate: str = "min") -> np.ndarray:
    """Compute k-NN distances from test to train points.

    Args:
        xtr: Training data points (N x D).
        xte: Test data points (M x D).
        k: Number of neighbors for k-NN distance.
        aggregate: Aggregation method ('min', 'kth', or 'mean').

    Returns:
        Array of distances (M,).
    """
    k_eff = max(1, min(k, len(xtr)))

    nn = NearestNeighbors(n_neighbors=k_eff).fit(xtr)
    d, _ = nn.kneighbors(xte, n_neighbors=k_eff)
    if aggregate == "min":
        return d[:, 0]
    if aggregate == "kth":
        return d[:, -1]
    if aggregate == "mean":
        return d.mean(axis=1)
    raise ValueError("aggregate must be one of {'min','kth','mean'}")


def _labels_from_edges(dist: np.ndarray, edges: np.ndarray,
                       names: Tuple[str, ...]) -> np.ndarray:
    """Assign labels based on distance edges.

    Args:
        dist: Array of distances (M,).
        edges: Sorted array of distance edges.
        names: Ordered group names (len = len(edges)+1).

    Returns:
        Array of labels (M,).
    """
    bins = np.digitize(dist, edges, right=True)
    return np.array([names[i] for i in bins], dtype=object)


def _spacing_summary(spacing: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics of spacing array.

    Args:
        spacing: Nearest-neighbor distances within training data.

    Returns:
        Summary dictionary.
    """
    return {
        "n": int(len(spacing)),
        "median": float(np.median(spacing)),
        "mean": float(np.mean(spacing)),
        "p25": float(np.quantile(spacing, 0.25)),
        "p90": float(np.quantile(spacing, 0.90)),
    }


def extend_group_names(
    names: Sequence[str],
    extrap_names: Sequence[str],
) -> List[str]:
    """Append extrapolation names if not already present.

    Args:
        names: Original interpolation group name sequence.
        extrap_names: Extrapolation group names to append.

    Returns:
        Extended list with extrapolation names appended.
    """
    extended = list(names)
    for ext in extrap_names:
        if ext not in extended:
            extended.append(ext)
    return extended


def _make_scaler(kind: str) -> Optional[Any]:
    """Create a scaler instance based on the specified kind.

    Args:
        kind: Type of scaler ('standard', 'minmax', 'robust', or 'none').

    Returns:
        An instance of the specified scaler or None.
    """
    kind = (kind or "standard").lower()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "robust":
        return RobustScaler()
    if kind in {"none", "identity", "no"}:
        return None
    raise ValueError(
        "scaler must be one of {'standard','minmax','robust','none'}")
