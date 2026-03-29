"""Rule-based recommendation engine.

This module implements the *hybrid reasoning* layer of the project: it takes
the ML model's risk score together with the student's raw features and
generates a short list of personalised, actionable recommendations.

Design rationale
----------------
* Rules are **explicit and auditable** — each one maps a concrete condition
  to a concrete suggestion.  This is intentional: in an educational context,
  opaque "black-box" advice would not be acceptable.
* Rules are intentionally kept **simple and few** (~10).  The goal is not to
  replace an educational professional but to surface the most relevant
  starting points for a support conversation.
* Each recommendation carries a ``priority`` (high / medium / low) so that
  the output is ordered by urgency.
* At most 3 recommendations are returned.  Returning too many dilutes the
  signal and makes the output less actionable.

Extending the rules
-------------------
To add a new rule, append a ``_check_*`` function and register it in
``_ALL_CHECKS``.  Each check receives the student row, the risk score and
the threshold, and returns either ``None`` (rule not triggered) or a
recommendation dict.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

Recommendation = dict[str, Any]

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Helper accessors — safe extraction from a student Series
# ---------------------------------------------------------------------------

def _num(student: pd.Series, column: str, default: float = 0.0) -> float:
    value = student.get(column, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cat(student: pd.Series, column: str, default: str = "") -> str:
    value = student.get(column, default)
    return str(value).strip().lower()


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------

def _check_high_absences(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if risk_score >= threshold and _num(student, "absences") >= 10:
        return {
            "rule": "high_absences",
            "reason": "Un nombre élevé d'absences est associé à un risque académique.",
            "action": "Fixer un objectif hebdomadaire de présence et surveiller les absences dès le début.",
            "priority": "high",
        }
    return None


def _check_low_studytime(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if risk_score >= threshold and _num(student, "studytime") <= 1:
        return {
            "rule": "low_studytime",
            "reason": "Un temps d'étude faible peut limiter l'assimilation des contenus de cours.",
            "action": "Mettre en place une routine d'étude fixe avec des séances courtes et régulières.",
            "priority": "high",
        }
    return None


def _check_past_failures(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    failures = _num(student, "failures")
    if failures >= 1:
        return {
            "rule": "past_failures",
            "reason": (
                f"L'étudiant a {int(failures)} échec(s) antérieur(s), "
                "ce qui suggère des difficultés d'apprentissage persistantes."
            ),
            "action": "Proposer une remédiation ciblée sur les prérequis et des suivis réguliers.",
            "priority": "high",
        }
    return None


def _check_no_school_support(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if _cat(student, "schoolsup") == "no" and risk_score >= threshold:
        return {
            "rule": "no_school_support",
            "reason": "L'étudiant ne bénéficie actuellement d'aucun soutien scolaire.",
            "action": "Recommander du tutorat, des permanences ou des séances de soutien académique structurées.",
            "priority": "medium",
        }
    return None


def _check_no_family_support(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if _cat(student, "famsup") == "no" and risk_score >= threshold:
        return {
            "rule": "no_family_support",
            "reason": "Un soutien familial limité peut réduire la stabilité d'étude en dehors de l'école.",
            "action": "Encourager le mentorat, les groupes d'étude entre pairs ou les espaces d'étude encadrés.",
            "priority": "medium",
        }
    return None


def _check_high_goout_low_study(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if _num(student, "goout") >= 4 and _num(student, "studytime") <= 2:
        return {
            "rule": "goout_study_imbalance",
            "reason": "Une activité sociale élevée combinée à un faible temps d'étude peut affecter la régularité.",
            "action": "Aider l'étudiant à construire un planning hebdomadaire équilibrant études et loisirs.",
            "priority": "medium",
        }
    return None


def _check_long_travel(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if _num(student, "traveltime") >= 3:
        return {
            "rule": "long_travel",
            "reason": "Un temps de trajet long peut réduire la disponibilité pour étudier et augmenter la fatigue.",
            "action": "Proposer des séances d'étude plus courtes dans la journée et planifier autour des contraintes de transport.",
            "priority": "low",
        }
    return None


def _check_no_internet(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if _cat(student, "internet") == "no":
        return {
            "rule": "no_internet",
            "reason": "Un accès limité à Internet peut réduire l'accès aux ressources pédagogiques.",
            "action": "Fournir des supports hors ligne ou identifier des points d'accès numériques sur le campus.",
            "priority": "low",
        }
    return None


def _check_no_higher_aspiration(
    student: pd.Series, risk_score: float, threshold: float,
) -> Recommendation | None:
    if _cat(student, "higher") == "no" and risk_score >= threshold:
        return {
            "rule": "no_higher_aspiration",
            "reason": "L'étudiant n'aspire pas à poursuivre des études supérieures, ce qui peut réduire son engagement.",
            "action": "Explorer les objectifs académiques et professionnels pour identifier des sources potentielles de motivation.",
            "priority": "medium",
        }
    return None


_ALL_CHECKS = [
    _check_high_absences,
    _check_low_studytime,
    _check_past_failures,
    _check_no_school_support,
    _check_no_family_support,
    _check_high_goout_low_study,
    _check_long_travel,
    _check_no_internet,
    _check_no_higher_aspiration,
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_recommendations(
    student: pd.Series,
    risk_score: float,
    risk_threshold: float = 0.5,
    max_recommendations: int = 3,
) -> list[Recommendation]:
    """Evaluate all rules and return the top recommendations.

    Parameters
    ----------
    student:
        A single row (pd.Series) from the feature matrix.
    risk_score:
        Predicted probability of being at-risk (output of the ML model).
    risk_threshold:
        Score above which the student is considered at-risk for rule purposes.
    max_recommendations:
        Maximum number of recommendations to return.

    Returns
    -------
    A list of recommendation dicts sorted by priority, each containing
    ``rule``, ``reason``, ``action`` and ``priority``.
    """
    triggered: list[Recommendation] = []

    for check_fn in _ALL_CHECKS:
        result = check_fn(student, risk_score, risk_threshold)
        if result is not None:
            triggered.append(result)
            logger.debug("Rule triggered: %s", result["rule"])

    if not triggered:
        triggered.append({
            "rule": "fallback",
            "reason": "Le modèle indique un certain risque mais aucun facteur dominant n'a été identifié.",
            "action": "Surveiller les résultats de près et planifier un suivi académique rapide.",
            "priority": "medium" if risk_score >= risk_threshold else "low",
        })

    triggered.sort(key=lambda r: PRIORITY_ORDER[r["priority"]])
    selected = triggered[:max_recommendations]

    logger.info(
        "Recommendations (%d triggered, %d returned): %s",
        len(triggered),
        len(selected),
        [r["rule"] for r in selected],
    )
    return selected
