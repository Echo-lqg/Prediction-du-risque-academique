"""Data loading, validation and label construction for student risk prediction.

This module handles the transition from raw CSV to a clean (X, y) pair ready
for modelling.  It deliberately keeps preprocessing minimal — scaling and
encoding are deferred to the modelling pipeline so that the same preprocessor
is always fitted on training data only.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INTERIM_GRADE_COLUMNS = ("G1", "G2")

EXPECTED_COLUMNS_STUDENT_MAT = {
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1", "G2", "G3",
}


@dataclass
class DatasetSummary:
    """Lightweight summary printed after loading so the user can sanity-check
    the data before training starts."""

    n_rows: int
    n_features: int
    n_at_risk: int
    n_not_at_risk: int
    risk_ratio: float
    missing_cells: int
    numeric_features: list[str]
    categorical_features: list[str]

    def log(self) -> None:
        logger.info("Dataset summary")
        logger.info("  rows            : %d", self.n_rows)
        logger.info("  features        : %d", self.n_features)
        logger.info("  at-risk (y=1)   : %d (%.1f%%)", self.n_at_risk, self.risk_ratio * 100)
        logger.info("  not-at-risk     : %d", self.n_not_at_risk)
        logger.info("  missing cells   : %d", self.missing_cells)
        logger.info("  numeric cols    : %d", len(self.numeric_features))
        logger.info("  categorical cols: %d", len(self.categorical_features))


def _detect_delimiter(file_path: Path) -> str:
    """Sniff the first 2 KiB of the file to decide between `;` and `,`.

    The UCI student-performance CSVs use `;` while most other sources use `,`.
    """
    sample = file_path.read_text(encoding="utf-8", errors="ignore")[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except csv.Error:
        return ","


def _validate_columns(dataset: pd.DataFrame, target_column: str) -> None:
    """Raise early if the target is missing or the schema looks wrong."""
    if target_column not in dataset.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {sorted(dataset.columns.tolist())}"
        )

    overlap = EXPECTED_COLUMNS_STUDENT_MAT & set(dataset.columns)
    if len(overlap) < len(EXPECTED_COLUMNS_STUDENT_MAT) * 0.5:
        logger.warning(
            "Only %d/%d expected columns found — the CSV may not be the "
            "standard UCI student-performance file.",
            len(overlap),
            len(EXPECTED_COLUMNS_STUDENT_MAT),
        )


def _build_label(series: pd.Series, passing_grade: int) -> pd.Series:
    """Convert a numeric grade column into a binary risk label.

    A student is labelled *at-risk* (1) when their grade is strictly below
    ``passing_grade``.  This mirrors the convention used in the original
    Cortez & Silva (2008) study.
    """
    return (series < passing_grade).astype(int)


def _summarise(X: pd.DataFrame, y: pd.Series) -> DatasetSummary:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    n_at_risk = int(y.sum())

    return DatasetSummary(
        n_rows=len(X),
        n_features=X.shape[1],
        n_at_risk=n_at_risk,
        n_not_at_risk=len(y) - n_at_risk,
        risk_ratio=n_at_risk / len(y) if len(y) > 0 else 0.0,
        missing_cells=int(X.isna().sum().sum()),
        numeric_features=numeric,
        categorical_features=categorical,
    )


def load_student_dataset(
    data_path: str | Path,
    target_column: str = "G3",
    passing_grade: int = 10,
    include_interim_grades: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, DatasetSummary]:
    """Load a student-performance CSV and return (raw, X, y, summary).

    Parameters
    ----------
    data_path:
        Path to a CSV file (`;` or `,` delimited).
    target_column:
        The grade column used to derive the risk label.
    passing_grade:
        Threshold below which a student is considered at-risk.
    include_interim_grades:
        If *False* (default), ``G1`` and ``G2`` are dropped so that the
        model cannot cheat by seeing highly-correlated future grades.

    Returns
    -------
    raw_dataset:
        The unmodified DataFrame as read from disk.
    X:
        Feature matrix with the target (and optionally G1/G2) removed.
    y:
        Binary risk labels (1 = at-risk).
    summary:
        A ``DatasetSummary`` with key statistics for logging.
    """
    file_path = Path(data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    delimiter = _detect_delimiter(file_path)
    logger.info("Reading %s (delimiter='%s')", file_path.name, delimiter)
    dataset = pd.read_csv(file_path, sep=delimiter)

    _validate_columns(dataset, target_column)

    if not np.issubdtype(dataset[target_column].dtype, np.number):
        raise TypeError(
            f"Target column '{target_column}' is not numeric "
            f"(dtype={dataset[target_column].dtype})."
        )

    y = _build_label(dataset[target_column], passing_grade)

    columns_to_drop = [target_column]
    if not include_interim_grades:
        for col in INTERIM_GRADE_COLUMNS:
            if col in dataset.columns and col != target_column:
                columns_to_drop.append(col)
        logger.info(
            "Dropping interim grade columns %s for early-intervention mode.",
            [c for c in columns_to_drop if c != target_column],
        )

    X = dataset.drop(columns=columns_to_drop)
    summary = _summarise(X, y)
    summary.log()

    return dataset, X, y, summary
