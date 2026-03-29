"""Model training, cross-validated evaluation and baseline comparison.

Design decisions
----------------
* **Majority-class baseline** is included so results tables always start with
  a trivial reference point.  This makes the value of ML immediately visible.
* **Stratified K-Fold cross-validation** is run *in addition to* the held-out
  test set evaluation.  CV scores expose variance and make the comparison more
  convincing for a small dataset (n < 500).
* Both models use ``class_weight="balanced"`` because the at-risk class is the
  minority and recall on that class matters most.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

N_CV_FOLDS = 5
CV_SCORING = {
    "accuracy": "accuracy",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


@dataclass
class TrainArtifacts:
    """Everything downstream stages need: fitted models, splits and scores."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    probabilities: dict[str, pd.Series]
    predictions: dict[str, pd.Series]
    feature_names_after_preprocessing: list[str] = field(default_factory=list)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create a column transformer that scales numerics and one-hot-encodes
    categoricals.  Returns the transformer and the two column lists so callers
    can inspect them."""

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def _build_candidates(
    preprocessor: ColumnTransformer,
    random_state: int,
) -> dict[str, Pipeline]:
    """Return an ordered dict of (name → sklearn Pipeline) to evaluate."""

    return {
        "majority_baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }


def _evaluate_on_test(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict[str, float]:
    """Return mean CV scores across ``N_CV_FOLDS`` stratified folds."""

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=random_state)
    try:
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=CV_SCORING,
            error_score="raise",
        )
    except Exception:
        logger.warning("CV failed for pipeline — returning zeros.")
        return {k: 0.0 for k in CV_SCORING}

    return {
        metric: float(np.mean(scores[f"test_{metric}"]))
        for metric in CV_SCORING
    }


def train_and_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainArtifacts:
    """Split, preprocess, train, cross-validate and evaluate all candidates.

    The returned ``TrainArtifacts`` contains both held-out test metrics and
    cross-validation metrics so downstream reporting can show both.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Train/test split: %d train, %d test (%.0f%% test, stratified).",
        len(X_train), len(X_test), test_size * 100,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    candidates = _build_candidates(preprocessor, random_state)

    test_rows: list[dict] = []
    cv_rows: list[dict] = []
    probabilities: dict[str, pd.Series] = {}
    predictions: dict[str, pd.Series] = {}

    for name, pipeline in candidates.items():
        logger.info("Training %-25s ...", name)
        pipeline.fit(X_train, y_train)

        y_prob = pd.Series(
            pipeline.predict_proba(X_test)[:, 1],
            index=X_test.index,
        )
        y_pred = pd.Series(pipeline.predict(X_test), index=X_test.index)

        test_scores = _evaluate_on_test(y_test, y_pred, y_prob)
        test_rows.append({"model": name, **test_scores})
        probabilities[name] = y_prob
        predictions[name] = y_pred

        cv_scores = _run_cross_validation(pipeline, X_train, y_train, random_state)
        cv_rows.append({"model": name, **{f"cv_{k}": v for k, v in cv_scores.items()}})

        logger.info(
            "  test  recall=%.3f  f1=%.3f  auc=%.3f",
            test_scores["recall"], test_scores["f1"], test_scores["roc_auc"],
        )
        logger.info(
            "  cv    recall=%.3f  f1=%.3f  auc=%.3f",
            cv_scores["recall"], cv_scores["f1"], cv_scores["roc_auc"],
        )

    metrics = (
        pd.DataFrame(test_rows)
        .sort_values(by=["recall", "f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    cv_metrics = pd.DataFrame(cv_rows)

    fitted_preprocessor = candidates["logistic_regression"].named_steps["preprocessor"]
    try:
        raw_names = fitted_preprocessor.get_feature_names_out()
        feature_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
    except Exception:
        feature_names = []

    return TrainArtifacts(
        models=candidates,
        metrics=metrics,
        cv_metrics=cv_metrics,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        probabilities=probabilities,
        predictions=predictions,
        feature_names_after_preprocessing=feature_names,
    )
