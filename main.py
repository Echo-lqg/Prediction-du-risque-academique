"""Command-line entry point for the student risk prediction pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from src.pipeline import run_pipeline


def _configure_logging() -> None:
    import io

    handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prédiction de la réussite étudiante avec recommandations explicables.",
    )
    parser.add_argument( 
        "--data", required=True,
        help="Chemin vers le fichier CSV du jeu de données.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Répertoire de sortie des résultats (défaut : results).",
    )
    parser.add_argument(
        "--target-column", default="G3",
        help="Nom de la colonne cible (défaut : G3).",
    )
    parser.add_argument(
        "--passing-grade", type=int, default=10,
        help="Seuil en dessous duquel un étudiant est considéré à risque (défaut : 10).",
    )
    parser.add_argument(  
        "--include-interim-grades", action="store_true",
        help="Inclure G1 et G2 comme variables prédictives (défaut : exclus).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Graine aléatoire pour la reproductibilité (défaut : 42).",
    )
    return parser.parse_args()
  

def main() -> None:
    _configure_logging()
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        target_column=args.target_column,
        passing_grade=args.passing_grade,
        include_interim_grades=args.include_interim_grades,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
