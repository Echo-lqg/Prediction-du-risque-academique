"""Post-hoc error analysis: confusion matrices, failure cases and threshold
sensitivity.

This module answers three questions that every responsible ML project should
address:

1. **Where does the model fail?** — confusion matrix and per-class metrics.
2. **Who is affected?** — individual false-negative and false-positive cases
   with their feature profiles.
3. **How sensitive is the model to the decision threshold?** — a sweep from
   0.1 to 0.9 showing recall/precision/f1 at each threshold, so the user
   can make an informed choice about the operating point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score

logger = logging.getLogger(__name__)


@dataclass
class ErrorReport:
    confusion: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    false_negatives: pd.DataFrame
    false_positives: pd.DataFrame
    threshold_sweep: pd.DataFrame
    summary_text: str


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def _confusion_dict(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _class_distribution(y_train: pd.Series, y_test: pd.Series) -> dict[str, dict[str, int]]:
    def _counts(s: pd.Series) -> dict[str, int]:
        return {
            "not_at_risk": int((s == 0).sum()),
            "at_risk": int((s == 1).sum()),
            "total": len(s),
        }
    return {"train": _counts(y_train), "test": _counts(y_test)}


# ---------------------------------------------------------------------------
# Error case extraction
# ---------------------------------------------------------------------------

def _extract_error_cases(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
    error_type: str,
) -> pd.DataFrame:
    if error_type == "false_negative":
        mask = (y_true == 1) & (y_pred == 0)
    elif error_type == "false_positive":
        mask = (y_true == 0) & (y_pred == 1)
    else:
        raise ValueError(f"Unknown error type: {error_type}")

    cases = X_test.loc[mask].copy()
    cases["true_label"] = y_true.loc[mask]
    cases["predicted_label"] = y_pred.loc[mask]
    cases["risk_score"] = y_prob.loc[mask]
    return cases


# ---------------------------------------------------------------------------
# Threshold sensitivity sweep
# ---------------------------------------------------------------------------

def _threshold_sweep(
    y_true: pd.Series,
    y_prob: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate recall, precision and F1 at multiple decision thresholds.

    This helps answer the question: *"If we lower the threshold to catch
    more at-risk students, how many more false alarms do we get?"*
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    rows: list[dict[str, float]] = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            "threshold": t,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "f1": round(f1, 3),
            "predicted_positive": tp + fp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown formatting helpers
# ---------------------------------------------------------------------------

def _format_confusion_text(cm: dict[str, int], model_name: str) -> str:
    tp, fn, fp, tn = cm["true_positives"], cm["false_negatives"], cm["false_positives"], cm["true_negatives"]
    total = tp + fn + fp + tn
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return "\n".join([
        f"### Matrice de confusion : {model_name}",
        "",
        "```text",
        "                    Prédit",
        "              Non-risque   À risque",
        f"Réel Non   {tn:>8}      {fp:>8}",
        f"Réel Risque{fn:>8}      {tp:>8}",
        "```",
        "",
        f"- Échantillons de test : {total}",
        f"- Faux négatifs (étudiants à risque manqués) : {fn}",
        f"- Faux positifs (fausses alertes) : {fp}",
        f"- Rappel (recall) : {recall:.3f}",
        f"- Précision : {precision:.3f}",
    ])


def _format_error_patterns(
    false_negatives: pd.DataFrame,
    false_positives: pd.DataFrame,
    feature_columns: list[str],
) -> str:
    sections: list[str] = ["### Profil des erreurs", ""]

    fr_labels = {
        "False negatives": ("Faux négatifs", "étudiants à risque manqués"),
        "False positives": ("Faux positifs", "fausses alertes"),
    }
    for en_label, df in [("False negatives", false_negatives), ("False positives", false_positives)]:
        fr_name, fr_desc = fr_labels[en_label]
        if len(df) == 0:
            sections.append(f"Aucun {fr_name.lower()} trouvé.")
            continue
        sections.append(f"**{fr_name}** ({len(df)} {fr_desc}) :")
        sections.append("")
        for col in feature_columns:
            if col in df.columns and df[col].dtype in (np.float64, np.int64, float, int):
                sections.append(f"- Médiane `{col}` : {df[col].median():.1f}")
        sections.append("")

    return "\n".join(sections)


def _format_threshold_table(sweep: pd.DataFrame) -> str:
    lines = [
        "### Sensibilité au seuil de décision",
        "",
        "| seuil | rappel | précision | f1 | prédits+ | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in sweep.iterrows():
        lines.append(
            f"| {row['threshold']:.2f} "
            f"| {row['recall']:.3f} "
            f"| {row['precision']:.3f} "
            f"| {row['f1']:.3f} "
            f"| {int(row['predicted_positive'])} "
            f"| {int(row['false_positives'])} "
            f"| {int(row['false_negatives'])} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_error_analysis(
    model_name: str,
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
    y_train: pd.Series,
    highlight_features: list[str] | None = None,
) -> ErrorReport:
    """Run the full error analysis for one model and return an ``ErrorReport``.

    The report includes:
    * confusion matrix counts,
    * class distribution in train / test,
    * individual false-negative and false-positive DataFrames,
    * a threshold sweep DataFrame,
    * a Markdown summary combining all of the above.
    """
    cm = _confusion_dict(y_true, y_pred)
    dist = _class_distribution(y_train, y_true)

    fn_cases = _extract_error_cases(X_test, y_true, y_pred, y_prob, "false_negative")
    fp_cases = _extract_error_cases(X_test, y_true, y_pred, y_prob, "false_positive")
    sweep = _threshold_sweep(y_true, y_prob)

    if highlight_features is None:
        highlight_features = ["failures", "absences", "studytime", "age", "goout"]

    cm_text = _format_confusion_text(cm, model_name)
    pattern_text = _format_error_patterns(fn_cases, fp_cases, highlight_features)
    threshold_text = _format_threshold_table(sweep)

    summary_text = "\n\n".join([
        "# Analyse des erreurs",
        f"Modèle : `{model_name}`",
        cm_text,
        pattern_text,
        threshold_text,
    ])

    logger.info(
        "Error analysis [%s]: FN=%d, FP=%d, best-F1 threshold=%.2f",
        model_name,
        cm["false_negatives"],
        cm["false_positives"],
        sweep.loc[sweep["f1"].idxmax(), "threshold"] if len(sweep) > 0 else 0.5,
    )

    return ErrorReport(
        confusion=cm,
        class_distribution=dist,
        false_negatives=fn_cases,
        false_positives=fp_cases,
        threshold_sweep=sweep,
        summary_text=summary_text,
    )
