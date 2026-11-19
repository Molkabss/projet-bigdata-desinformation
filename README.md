# Détection de la Désinformation — Projet Big Data Avancé

Pipeline Big Data (PIG → HIVE → Spark MLlib) enrichi par l'IA pour détecter si un article de presse est vrai ou faux.

## Sujet

Classification binaire (Fake / Real) d'articles de presse, avec un pipeline 100% open source (Apache) :
aucun modèle développé sur mesure — uniquement des algorithmes existants de Spark MLlib.

## Dataset

**Fake and Real News Dataset** (Kaggle)
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

- ~44 000 articles, 2 fichiers (`Fake.csv` / `True.csv`)
- Colonnes : `title`, `text`, `subject`, `date`
- À télécharger manuellement depuis Kaggle et placer dans `data/` (non versionné, voir `.gitignore`)

## Architecture

```
Fake.csv + True.csv
  → prepare_dataset.py : fusion + ajout id/label → news.tsv
  → PIG   : nettoyage, suppression des biais (marqueurs "(Reuters)"), feature engineering
  → HIVE  : tables d'agrégats = features pour le modèle
  → SPARK MLlib : TF-IDF + classification (Random Forest)
  → Résultat : score de fiabilité par article
```

## Structure du repo

```
pig/        prepare_dataset.py (fusion CSV→TSV), clean_features.pig (nettoyage, feature engineering)
hive/       scripts HiveQL (création de tables, agrégats)
spark/      script PySpark (NLP, entraînement, évaluation)
results/    métriques, statistiques, exports
slides/     présentation finale (export)
data/       dataset brut (non versionné)
```

## Répartition des tâches

| Membre | Responsabilité |
|---|---|
| A | Dataset + PIG (préparation, nettoyage, feature engineering) |
| B | HIVE + Spark MLlib (tables, modélisation, évaluation) |
| A & B | Environnement, chargement HDFS initial, relecture finale |

## Lancer le pipeline

1. `python pig/prepare_dataset.py` (fusionne Fake.csv/True.csv → data/news.tsv)
2. `hdfs dfs -put data/news.tsv /user/cloudera/fakenews/raw/news.tsv`
3. `pig -x mapreduce -f pig/clean_features.pig`
4. `hive -f hive/create_tables.hql`
5. `spark-submit spark/train_model.py`

## Résultats obtenus

- Dataset nettoyé : 38 639 articles (sur 44 898 bruts)
- Modèle Random Forest (Spark MLlib) : **98.52% accuracy**
- Les articles Fake contiennent en moyenne 12x plus de points d'exclamation que les articles Real

## Extension (mentionnée, non implémentée)

Kafka (ingestion temps réel), Flume (collecte de logs de partage), Sqoop (import d'un historique de modération).