# Prédiction de la Réussite Étudiante et Recommandations Explicables

Ce projet prédit si un étudiant est en risque d'échec académique à partir de données comportementales et de performance, puis génère des recommandations personnalisées et explicables via un système de règles.

Il est conçu comme un projet compact et opérationnel, avec un accent sur :

- l'apprentissage automatique sur données tabulaires pour la prédiction de risque
- l'explicabilité par inspection du modèle
- le raisonnement hybride par recommandations à base de règles
- une démarche expérimentale rigoureuse avec baselines, métriques et analyse d'erreurs

## Définition du problème

L'objectif est d'identifier les étudiants susceptibles d'échouer suffisamment tôt pour permettre une intervention.

### Pourquoi c'est important

Un système d'aide à la décision éducatif ne doit pas seulement prédire un risque, mais aussi expliquer pourquoi un étudiant est signalé et proposer des pistes d'action concrètes. Ce projet aborde la tâche comme un problème d'aide à la décision centré sur l'humain, et non comme un simple benchmark de classification.

### Entrées

- variables démographiques et contextuelles de l'étudiant
- variables comportementales : absences, temps d'étude, indicateurs de soutien
- variables de performance académique : notes antérieures ou intermédiaires

### Sorties

- une probabilité de risque pour chaque étudiant
- une étiquette binaire (à risque / non à risque)
- des signaux d'explication globaux et locaux
- des recommandations personnalisées générées par des règles explicites

### Métriques d'évaluation

- `Recall` : prioritaire car manquer un étudiant à risque a un coût élevé
- `F1-score` : équilibre entre précision et rappel
- `ROC-AUC` : qualité du classement global
- `Accuracy` : incluse pour référence

## Structure du projet

```text
student-success-prediction/
├── data/
│   └── README.md
├── results/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── error_analysis.py
│   ├── explainability.py
│   ├── modeling.py
│   ├── pipeline.py
│   └── recommender.py
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Jeu de données

Le pipeline utilise par défaut le jeu de données UCI Student Performance. Il attend un fichier CSV tel que `student-mat.csv` avec la colonne `G3` comme note finale. L'étiquette de risque est définie comme suit :

```text
at_risk = 1 si G3 < 10, sinon 0
```

Pour rendre la prédiction plus utile en intervention précoce, le pipeline exclut `G1` et `G2` par défaut. On peut les inclure via un argument en ligne de commande afin de comparer les performances avec et sans notes intermédiaires.

Les instructions de téléchargement sont dans [`data/README.md`](data/README.md).

## Méthodes

### 1. Baselines et comparaison de modèles

Le projet compare trois modèles :

- `DummyClassifier` (classe majoritaire) : baseline triviale qui prédit toujours « non à risque »
- `LogisticRegression` : baseline ML interprétable
- `RandomForestClassifier` : modèle non linéaire plus puissant

L'inclusion de la baseline majoritaire rend immédiatement visible l'apport du ML. Les deux modèles ML utilisent `class_weight="balanced"` pour compenser le déséquilibre de classes (environ 33 % à risque vs 67 % non à risque).

Tous les modèles sont évalués sur un jeu de test réservé **et** par validation croisée stratifiée à 5 plis sur le jeu d'entraînement. Cette double évaluation permet d'observer à la fois la performance absolue et la variance, ce qui est important sur un petit jeu de données (n=395).

### 2. Explicabilité

Le module d'explicabilité fournit deux niveaux d'analyse :

- **Importance globale des variables** : coefficients absolus pour la régression logistique, importance Gini pour la forêt aléatoire.
- **Explication locale (au niveau d'un cas)** : pour la régression logistique, la contribution de chaque variable est calculée comme `valeur_normalisée × coefficient`, ce qui est une décomposition exacte du score linéaire. Pour la forêt aléatoire, une méthode approchée par perturbation est utilisée : chaque variable est remise à zéro individuellement et le changement de probabilité prédite est mesuré.

**Limites de l'approche locale** : ces explications ne sont pas des valeurs de Shapley (SHAP). La décomposition linéaire est exacte pour la régression logistique mais ne tient pas compte des interactions. La méthode par perturbation pour la forêt aléatoire est une heuristique rapide qui ne fournit pas de garanties d'additivité ni de cohérence théorique. Elle donne une indication utile des facteurs dominants, mais ne remplace pas une analyse SHAP complète. L'intégration de SHAP est listée comme extension possible.

### 3. Système de recommandation hybride

Le module de recommandation (`src/recommender.py`) combine :

- le score de risque prédit par le modèle ML
- les valeurs brutes des variables de l'étudiant
- les signaux d'explication

Il évalue 9 règles explicites (ex. absences élevées, échecs passés, absence de soutien scolaire) et associe les conditions déclenchées à des suggestions d'intervention concrètes. Chaque règle est implémentée comme une fonction indépendante pour faciliter les tests et l'auditabilité. Au maximum 3 recommandations sont retournées, triées par priorité.

### 4. Analyse d'erreurs et sensibilité au seuil

Pour chaque modèle, le pipeline produit automatiquement :

- une matrice de confusion avec les comptes de faux négatifs et faux positifs
- les cas d'erreur individuels (sauvegardés en CSV) pour inspection post-hoc
- un profil médian des étudiants manqués, pour identifier les angles morts du modèle
- un balayage de seuils de 0.10 à 0.90 montrant l'évolution du rappel, de la précision et du F1 en fonction du seuil de décision

## Résultats actuels

Le dépôt contient deux séries d'expériences sur `student-mat.csv`.

### Scénario A : intervention précoce (sans notes intermédiaires)

Configuration la plus réaliste pour l'intervention car `G1` et `G2` sont exclus.

**Jeu de test réservé**

| Modèle | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline majoritaire | 0.671 | 0.000 | 0.000 | 0.500 |
| Régression logistique | 0.696 | 0.692 | 0.600 | 0.727 |
| Forêt aléatoire | 0.734 | 0.385 | 0.488 | 0.693 |

**Validation croisée à 5 plis (entraînement)**

| Modèle | CV Accuracy | CV Recall | CV F1 | CV AUC |
|---|---:|---:|---:|---:|
| Baseline majoritaire | 0.671 | 0.000 | 0.000 | 0.500 |
| Régression logistique | 0.614 | 0.540 | 0.478 | 0.631 |
| Forêt aléatoire | 0.680 | 0.309 | 0.389 | 0.650 |

Constats principaux :

- La baseline majoritaire atteint 67 % d'accuracy en prédisant toujours « non à risque », mais son rappel est nul — elle ne détecte aucun étudiant en difficulté. Cela montre pourquoi l'accuracy seule est trompeuse pour cette tâche.
- La régression logistique a un rappel nettement supérieur à la forêt aléatoire (0.692 vs 0.385). Pour un système d'alerte précoce où manquer un cas réel est coûteux, elle est le meilleur choix.
- La validation croisée confirme que l'avantage en rappel de la régression logistique est stable et ne résulte pas d'un découpage particulier.

### Scénario B : avec notes intermédiaires

Cette configuration inclut `G1` et `G2`, ce qui facilite la prédiction mais réduit l'utilité pour l'intervention précoce.

**Jeu de test réservé**

| Modèle | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline majoritaire | 0.671 | 0.000 | 0.000 | 0.500 |
| Forêt aléatoire | 0.911 | 0.885 | 0.868 | 0.965 |
| Régression logistique | 0.886 | 0.846 | 0.830 | 0.975 |

Constats principaux :

- La performance augmente fortement avec les notes intermédiaires, mais la valeur d'intervention diminue car ces notes ne sont disponibles qu'en fin de semestre.
- Cette comparaison est méthodologiquement importante : une meilleure performance prédictive ne signifie pas toujours une plus grande valeur pratique.

### Exemple de recommandation

Pour un cas à haut risque, le moteur de règles a généré des recommandations liées à :

- la remédiation ciblée après des échecs répétés
- le soutien académique structuré
- l'accès alternatif aux ressources d'apprentissage

### Analyse d'erreurs (Scénario A)

Chaque exécution produit automatiquement des matrices de confusion et des rapports de cas d'erreur.

**Régression logistique**

```text
                 Prédit
              Non-risque   À risque
Réel Non          37           16
Réel Risque        8           18
```

- Recall : 0.692 / Precision : 0.529
- 8 étudiants à risque ont été manqués (faux négatifs)
- 16 étudiants non à risque ont été signalés à tort (faux positifs)
- Les étudiants manqués avaient un nombre médian d'échecs de 0 et d'absences de 3 — peu de signaux d'alerte visibles dans les variables disponibles.

**Forêt aléatoire**

```text
                 Prédit
              Non-risque   À risque
Réel Non          48            5
Réel Risque       16           10
```

- Recall : 0.385 / Precision : 0.667
- 16 étudiants à risque ont été manqués, soit près de la moitié du groupe positif.
- Le modèle est plus conservateur : moins de fausses alertes, mais au prix de nombreux cas réels manqués.

**Observations clés**

- Les deux modèles représentent des compromis différents : la régression logistique favorise la sensibilité, la forêt aléatoire favorise la spécificité.
- Pour un système d'alerte précoce, manquer un étudiant à risque est généralement plus coûteux qu'une fausse alerte. Cela rend la régression logistique plus appropriée dans ce contexte.
- Les faux négatifs des deux modèles partagent un profil : peu d'échecs passés et des absences modérées. Ces étudiants sont peut-être à risque pour des raisons mal captées par les variables disponibles (motivation, circonstances externes, etc.).

## Comment exécuter

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Télécharger le jeu de données

Placer le fichier CSV dans `data/raw/`, par exemple :

```text
data/raw/student-mat.csv
```

### 3. Lancer le pipeline

```bash
python main.py --data data/raw/student-mat.csv
```

Options :

```bash
python main.py --data data/raw/student-mat.csv --include-interim-grades
python main.py --data data/raw/student-mat.csv --passing-grade 12
python main.py --data data/raw/student-mat.csv --output-dir results/run_01
```

## Fichiers produits

Chaque exécution génère environ 23 fichiers :

- `metrics.json` / `metrics.csv` : métriques sur le jeu de test pour tous les modèles
- `cv_metrics.json` / `cv_metrics.csv` : métriques de validation croisée à 5 plis
- `global_feature_importance.csv` : variables les plus importantes pour le meilleur modèle
- `sample_case.json` : un exemple à haut risque avec les explications des deux modèles ML
- `sample_recommendations.json` : recommandations à base de règles pour cet étudiant
- `confusion_<modèle>.json` : matrice de confusion par modèle
- `error_analysis_<modèle>.md` : rapport d'erreur narratif par modèle
- `threshold_sweep_<modèle>.csv` : recall/precision/F1 à 17 seuils différents
- `false_negatives_<modèle>.csv` / `false_positives_<modèle>.csv` : cas d'erreur individuels
- `class_distribution.json` : équilibre des classes train/test
- `summary.md` : résumé complet de l'expérience

## Extensions possibles

- Analyse d'équité par sous-groupes démographiques (sexe, type d'adresse)
- Courbes d'apprentissage pour évaluer si davantage de données aideraient
- Intégration de SHAP pour des explications locales avec garanties théoriques

## Limites

- Les jeux de données éducatifs publics sont de taille réduite et simplifiés.
- Les recommandations à base de règles sont heuristiques et ne remplacent pas un professionnel de l'éducation.
- La qualité de la prédiction dépend fortement de la disponibilité des variables et du moment de l'observation.
- L'inclusion des notes intermédiaires améliore la performance mais réduit la valeur d'intervention.
- Les explications locales fournies par ce projet sont des approximations : elles ne sont pas des valeurs de Shapley et ne possèdent pas de garanties d'additivité. Elles servent d'indication sur les facteurs dominants, pas de preuve formelle.

## Ce que ce projet démontre

Ce projet combine :

- apprentissage automatique
- explicabilité (avec ses limites clairement identifiées)
- raisonnement hybride
- aide à la décision orientée utilisateur

## Auteur

**LIU Qiange**


