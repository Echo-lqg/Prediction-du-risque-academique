# Préparation des données

Ce projet utilise le jeu de données UCI Student Performance.

## Jeu de données recommandé

- Dépôt UCI : <https://archive.ics.uci.edu/ml/datasets/student%20performance>
- Page alternative : <https://archive-beta.ics.uci.edu/dataset/320/student+performance>

## Fichiers

L'archive UCI contient les fichiers suivants :

- `student-mat.csv` — données des élèves en mathématiques (395 enregistrements)
- `student-por.csv` — données des élèves en portugais (649 enregistrements)
- `student.txt` — description originale des variables du jeu de données UCI
- `student-merge.R` — script R d'origine montrant comment identifier les 382 élèves présents dans les deux fichiers

Ce projet utilise principalement `student-mat.csv` pour la version actuelle. Le fichier doit être placé dans :

```text
data/raw/student-mat.csv
```

## Variable cible

Le pipeline utilise par défaut :

- colonne cible : `G3`
- seuil de passage : `10`
- étiquette de risque : `1` si `G3 < 10`, sinon `0`

## Remarque sur les notes intermédiaires

Le jeu de données contient `G1`, `G2` et `G3`.

- `G1` : note du premier trimestre
- `G2` : note du deuxième trimestre
- `G3` : note finale

`G1` et `G2` sont fortement corrélées avec `G3`, ce qui est attendu puisqu'il s'agit de notes successives. Le pipeline les exclut par défaut afin de simuler un scénario d'intervention précoce où seules les variables comportementales et contextuelles sont disponibles. Ce choix rend la tâche plus difficile mais plus réaliste.
