"""
Initializes the plots module.
"""

from .domains import (
    plot_distance_histogram,
    plot_train_test_subplots,
    plot_feature_correlation,
)
from .damage import (
    plot_tower_damage_profiles_vs_reference,
    plot_error_profile,
)
from .shap import plot_shap_importance_bar, plot_shap_beeswarm
from .metrics import (
    plot_y_true_vs_y_pred_by_section,
    plot_y_true_vs_y_pred_by_section_group,
    plot_relative_error_hist_by_section,
    plot_signed_error_vs_y_true_by_section,
    plot_signed_error_vs_var_by_section,
    plot_signed_error_vs_var_by_section_group,
    plot_cumulative_damage_by_section,
    plot_signed_error_vs_y_true,
    plot_y_true_vs_y_pred,
    plot_relative_error_hist,
    plot_signed_error_vs_distance,
    sec_key,
)
from .models import plot_learning_curves_xgboost, plot_learning_curves_mlp
from .general import style_axes, style_ticks, style_spines, save_figure
from . import benchmark
