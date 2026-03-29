"""End-to-end pipeline: load → train → explain → recommend → analyse → save.

This module is the single orchestrator that downstream callers (``main.py``)
invoke.  It deliberately keeps *no global state*: every run reads its
configuration from function arguments so experiments are fully reproducible.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.data_loader import load_student_dataset
from src.error_analysis import run_error_analysis
from src.explainability import explain_single_case, global_feature_importance
from src.modeling import train_and_compare_models
from src.recommender import generate_recommendations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_serializable(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain Python types."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_to_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Markdown table builder (no extra dependency needed)
# ---------------------------------------------------------------------------

def _metrics_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(
        "---:" if col != columns[0] else "---" for col in columns
    ) + "|"
    body_lines: list[str] = []
    for row in rows:
        cells: list[str] = []
        for col in columns:
            val = row.get(col, "")
            cells.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body_lines])


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str,
    output_dir: str = "results",
    target_column: str = "G3",
    passing_grade: int = 10,
    include_interim_grades: bool = False,
    random_state: int = 42,
) -> None:
    """Execute the full experiment and write all artefacts to *output_dir*.

    Stages
    ------
    1. Load and validate the dataset.
    2. Train majority-class baseline, logistic regression and random forest.
    3. Cross-validate all models on the training set.
    4. Compute global feature importance for the best model.
    5. Pick the highest-risk test sample as a demo case and explain it.
    6. Generate rule-based recommendations for that case.
    7. Run error analysis (confusion matrix, failure cases, threshold sweep)
       for every model.
    8. Write everything to ``output_dir``.
    """
    t0 = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Étape 1/7  Chargement des données")
    logger.info("=" * 60)

    raw_dataset, X, y, data_summary = load_student_dataset(
        data_path=data_path,
        target_column=target_column,
        passing_grade=passing_grade,
        include_interim_grades=include_interim_grades,
    )

    # ------------------------------------------------------------------
    # 2–3. Train + cross-validate
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Étape 2/7  Entraînement et validation croisée des modèles")
    logger.info("=" * 60)

    artifacts = train_and_compare_models(X=X, y=y, random_state=random_state)

    metrics = artifacts.metrics
    metrics.to_csv(output_path / "metrics.csv", index=False)
    _write_json(output_path / "metrics.json", metrics.to_dict(orient="records"))

    cv_metrics = artifacts.cv_metrics
    cv_metrics.to_csv(output_path / "cv_metrics.csv", index=False)
    _write_json(output_path / "cv_metrics.json", cv_metrics.to_dict(orient="records"))

    # ------------------------------------------------------------------
    # 4. Global feature importance
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Étape 3/7  Calcul de l'importance globale des variables")
    logger.info("=" * 60)

    ml_model_names = [n for n in artifacts.models if n != "majority_baseline"]
    best_model_name = metrics.loc[
        metrics["model"].isin(ml_model_names)
    ].iloc[0]["model"]
    best_model = artifacts.models[best_model_name]

    importance = global_feature_importance(best_model)
    importance.to_csv(output_path / "global_feature_importance.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Demo case explanation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Étape 4/7  Explication d'un cas à haut risque")
    logger.info("=" * 60)

    best_probs = artifacts.probabilities[best_model_name].sort_values(ascending=False)
    sample_idx = best_probs.index[0]
    sample_case = artifacts.X_test.loc[[sample_idx]]
    sample_prob = float(best_probs.loc[sample_idx])
    sample_label = int(artifacts.predictions[best_model_name].loc[sample_idx])

    explanations: dict[str, list[dict]] = {}
    for mname in ml_model_names:
        explanations[mname] = explain_single_case(artifacts.models[mname], sample_case)

    # ------------------------------------------------------------------
    # 6. Recommendations
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Étape 5/7  Génération des recommandations à base de règles")
    logger.info("=" * 60)

    recommendations = generate_recommendations(sample_case.iloc[0], sample_prob)

    sample_payload = {
        "sample_index": int(sample_idx) if isinstance(sample_idx, int) else str(sample_idx),
        "predicted_risk_score": round(sample_prob, 4),
        "predicted_label": sample_label,
        "selected_model": best_model_name,
        "features": sample_case.iloc[0].to_dict(),
        "explanations": explanations,
    }
    _write_json(output_path / "sample_case.json", sample_payload)
    _write_json(output_path / "sample_recommendations.json", recommendations)

    # ------------------------------------------------------------------
    # 7. Error analysis per model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Étape 6/7  Analyse des erreurs")
    logger.info("=" * 60)

    error_sections: list[str] = []
    for model_name in artifacts.models:
        report = run_error_analysis(
            model_name=model_name,
            X_test=artifacts.X_test,
            y_true=artifacts.y_test,
            y_pred=artifacts.predictions[model_name],
            y_prob=artifacts.probabilities[model_name],
            y_train=artifacts.y_train,
        )
        _write_json(output_path / f"confusion_{model_name}.json", report.confusion)
        _write_json(output_path / "class_distribution.json", report.class_distribution)

        report.threshold_sweep.to_csv(
            output_path / f"threshold_sweep_{model_name}.csv", index=False,
        )

        if len(report.false_negatives) > 0:
            report.false_negatives.to_csv(
                output_path / f"false_negatives_{model_name}.csv", index=False,
            )
        if len(report.false_positives) > 0:
            report.false_positives.to_csv(
                output_path / f"false_positives_{model_name}.csv", index=False,
            )
        (output_path / f"error_analysis_{model_name}.md").write_text(
            report.summary_text, encoding="utf-8",
        )
        error_sections.append(report.summary_text)

    # ------------------------------------------------------------------
    # 8. Summary report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Étape 7/7  Rédaction du rapport de synthèse")
    logger.info("=" * 60)

    test_table = _metrics_to_markdown(
        metrics.to_dict(orient="records"),
        ["model", "accuracy", "recall", "f1", "roc_auc"],
    )
    cv_table = _metrics_to_markdown(
        cv_metrics.to_dict(orient="records"),
        ["model", "cv_accuracy", "cv_recall", "cv_f1", "cv_roc_auc"],
    )
    error_block = "\n\n---\n\n".join(error_sections)
    elapsed = time.perf_counter() - t0

    interim_fr = "oui" if include_interim_grades else "non"
    summary = f"""# Résumé de l'expérience

## Jeu de données

- Fichier source : `{data_path}`
- Nombre d'enregistrements : {len(raw_dataset)}
- Variables utilisées : {X.shape[1]}
- Colonne cible : `{target_column}`
- Seuil de passage : {passing_grade}
- Notes intermédiaires incluses : {interim_fr}
- Équilibre des classes : {data_summary.n_at_risk} à risque ({data_summary.risk_ratio:.1%}) / {data_summary.n_not_at_risk} non à risque

## Métriques sur le jeu de test réservé

{test_table}

## Métriques de validation croisée (5 plis stratifiés)

{cv_table}

## Cas de démonstration

- Modèle utilisé : `{best_model_name}`
- Score de risque prédit : {sample_prob:.3f}
- Étiquette prédite : {sample_label}

{error_block}

## Temps d'exécution

Pipeline terminé en {elapsed:.1f} secondes.

## Pistes d'amélioration

- Comparer avec une baseline uniquement à base de règles
- Enrichir les recommandations avec des règles issues du domaine éducatif
- Analyser l'équité du modèle par sous-groupes démographiques
"""

    (output_path / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("Tous les résultats ont été écrits dans %s (%.1fs au total).", output_path, elapsed)
