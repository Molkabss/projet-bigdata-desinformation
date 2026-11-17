\# Détection de la Désinformation — Projet Big Data Avancé



Pipeline Big Data (PIG → HIVE → Spark MLlib) enrichi par l'IA pour détecter si un article de presse est vrai ou faux.



\## Sujet



Classification binaire (Fake / Real) d'articles de presse, avec un pipeline 100% open source (Apache) :

aucun modèle développé sur mesure — uniquement des algorithmes existants de Spark MLlib.



\## Dataset



\*\*Fake and Real News Dataset\*\* (Kaggle)

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset



\- \~44 000 articles, 2 fichiers (`Fake.csv` / `True.csv`)

\- Colonnes : `title`, `text`, `subject`, `date`

\- À télécharger manuellement depuis Kaggle et placer dans `data/` (non versionné, voir `.gitignore`)



\## Architecture



```

Articles bruts (CSV → HDFS)

&#x20; → PIG   : nettoyage + feature engineering texte

&#x20; → HIVE  : tables d'agrégats = features pour le modèle

&#x20; → SPARK MLlib : TF-IDF + classification (Random Forest / Naive Bayes)

&#x20; → Résultat : score de fiabilité par article

```



\## Structure du repo



```

pig/        scripts PIG (nettoyage, feature engineering)

hive/       scripts HiveQL (création de tables, agrégats)

spark/      script(s)/notebook PySpark (NLP, entraînement, évaluation)

results/    métriques, matrice de confusion, exports

slides/     présentation finale (export)

data/       dataset brut (non versionné)

```



\## Répartition des tâches



| Membre | Responsabilité |

|---|---|

| A | Dataset + PIG (nettoyage, feature engineering) |

| B | HIVE + Spark MLlib (tables, modélisation, évaluation) |

| A \& B | Environnement, chargement HDFS initial, relecture finale |



\## Lancer le pipeline



1\. `hdfs dfs -put data/news.csv /user/data/`

2\. `pig pig/clean\_features.pig`

3\. `hive -f hive/create\_tables.hql`

4\. `spark-submit spark/train\_model.py` (ou notebook `spark/train\_model.ipynb`)



\## Extension (mentionnée, non implémentée)



Kafka (ingestion temps réel), Flume (collecte de logs de partage), Sqoop (import d'un historique de modération).

