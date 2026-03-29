"""Global and local model explanations.

Two levels of explanation are provided:

1. **Global feature importance** — which features drive the model overall?
   For logistic regression this is the absolute coefficient; for tree-based
   models it is the Gini / entropy importance.

2. **Local (case-level) explanation** — for a single student, which features
   pushed the prediction toward or away from the at-risk class?

   - For logistic regression we compute *feature-value × coefficient*
     contributions.  This is an exact decomposition of the linear score,
     but it does not account for feature interactions.
   - For tree ensembles we use a simple single-feature perturbation
     heuristic (zero-out each feature and measure the probability shift).
     This is **not** equivalent to SHAP values: it ignores feature
     correlations, is not additive, and provides no theoretical
     guarantees.  It is included as a lightweight, dependency-free
     indicator of the dominant factors for a given prediction.
     For rigorous local explanations, SHAP should be integrated as a
     future extension.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _get_feature_names(trained_pipeline: Pipeline) -> list[str]:
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    raw_names = preprocessor.get_feature_names_out()
    return [name.replace("num__", "").replace("cat__", "") for name in raw_names]


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------

def global_feature_importance(
    trained_pipeline: Pipeline,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return a DataFrame of the top-n most important features.

    Works for any estimator that exposes ``coef_`` (linear models) or
    ``feature_importances_`` (tree ensembles).
    """
    model = trained_pipeline.named_steps["model"]
    feature_names = _get_feature_names(trained_pipeline)

    if hasattr(model, "coef_"):
        raw_importance = np.abs(model.coef_[0])
        method = "abs_coefficient"
    elif hasattr(model, "feature_importances_"):
        raw_importance = model.feature_importances_
        method = "gini_importance"
    else:
        raise ValueError(
            f"Model type {type(model).__name__} does not expose feature importance."
        )

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": raw_importance, "method": method})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info("Global importance (top 5): %s", list(importance["feature"].head()))
    return importance


# ---------------------------------------------------------------------------
# Local explanation — linear models
# ---------------------------------------------------------------------------

def _explain_linear(
    model: Any,
    transformed_row: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    """Feature-value × coefficient decomposition for logistic regression."""

    contributions = transformed_row * model.coef_[0]
    df = pd.DataFrame({"feature": feature_names, "contribution": contributions})
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).head(top_n)

    return [
        {
            "feature": row["feature"],
            "contribution": round(float(row["contribution"]), 4),
            "direction": "risk" if row["contribution"] > 0 else "protective",
        }
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Local explanation — tree ensembles
# ---------------------------------------------------------------------------

def _explain_forest(
    model: RandomForestClassifier,
    transformed_row: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    """Approximate local importance for a random forest via perturbation.

    For each feature, we set its (standardised) value to zero and measure
    the change in predicted probability.  A large drop means the feature
    was pushing the prediction toward the positive class; a large increase
    means it was protective.

    Caveats
    -------
    * This is a simple heuristic, **not** a Shapley-value computation.
    * It does not account for feature correlations or interactions.
    * It is not additive: the individual contributions do not sum to the
      total prediction shift.
    * Zero may not be the true "neutral" value for every feature after
      standardisation.

    The method is included because it requires no extra dependency and
    gives a useful first-order indication.  For production-grade local
    explanations, SHAP (or a similar method with theoretical backing)
    should be used instead.
    """
    base_prob = float(model.predict_proba(transformed_row.reshape(1, -1))[0, 1])

    importances: list[float] = []
    for idx in range(len(feature_names)):
        perturbed = transformed_row.copy()
        perturbed[idx] = 0.0  # approximate "neutral" after standard scaling
        perturbed_prob = float(model.predict_proba(perturbed.reshape(1, -1))[0, 1])
        importances.append(base_prob - perturbed_prob)

    df = pd.DataFrame({
        "feature": feature_names,
        "contribution": importances,
    })
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).head(top_n)

    return [
        {
            "feature": row["feature"],
            "contribution": round(float(row["contribution"]), 4),
            "direction": "risk" if row["contribution"] > 0 else "protective",
        }
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def explain_single_case(
    trained_pipeline: Pipeline,
    raw_case: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Explain the prediction for **one** student.

    Parameters
    ----------
    trained_pipeline:
        A fitted sklearn Pipeline whose last step is the estimator.
    raw_case:
        A single-row DataFrame with the same columns as the training data
        (before preprocessing).
    top_n:
        Number of top contributing features to return.

    Returns
    -------
    A list of dicts, each containing ``feature``, ``contribution`` (signed
    float) and ``direction`` ("risk" or "protective").
    """
    model = trained_pipeline.named_steps["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    feature_names = _get_feature_names(trained_pipeline)

    transformed = preprocessor.transform(raw_case)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_row = transformed[0]

    if hasattr(model, "coef_"):
        explanation = _explain_linear(model, transformed_row, feature_names, top_n)
    elif isinstance(model, RandomForestClassifier):
        explanation = _explain_forest(model, transformed_row, feature_names, top_n)
    else:
        logger.warning("No local explanation method for %s.", type(model).__name__)
        return []

    logger.info(
        "Local explanation for sample — top factor: %s (%.3f)",
        explanation[0]["feature"] if explanation else "n/a",
        explanation[0]["contribution"] if explanation else 0.0,
    )
    return explanation
