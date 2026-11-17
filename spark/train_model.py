# Databricks notebook source
from pyspark.sql.functions import lit

fake_path = "/Volumes/workspace/default/desinformation_data/Fake.csv"
true_path = "/Volumes/workspace/default/desinformation_data/True.csv"

df_fake = spark.read.csv(fake_path, header=True, inferSchema=True, multiLine=True, escape='"')
df_true = spark.read.csv(true_path, header=True, inferSchema=True, multiLine=True, escape='"')

df_fake = df_fake.withColumn("label", lit(0))
df_true = df_true.withColumn("label", lit(1))

print("Fake:", df_fake.count())
print("True:", df_true.count())
df_fake.show(3)


# COMMAND ----------

# Fusion des deux datasets
df = df_fake.unionByName(df_true)

# Nettoyage de base : suppression des lignes sans texte, doublons
df = df.dropna(subset=["text"])
df = df.dropDuplicates(["text"])

print("Total après nettoyage :", df.count())
df.groupBy("label").count().show()

# COMMAND ----------

from pyspark.sql.functions import col, length, upper, size, split

# Features "PIG-like" : indicateurs simples à partir du texte
df = df.withColumn("text_length", length(col("text")))
df = df.withColumn("title_length", length(col("title")))
df = df.withColumn("exclamation_count", size(split(col("text"), "!")) - 1)
df = df.withColumn("word_count", size(split(col("text"), " ")))

df.select("text_length", "title_length", "exclamation_count", "word_count", "label").show(5)

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler

# 1. Tokenisation du texte
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_tok = tokenizer.transform(df)

# 2. Suppression des mots vides (the, a, is...)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tok)

# 3. TF-IDF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df_tf = hashingTF.transform(df_filtered)

idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

# 4. Assemblage : TF-IDF + features "PIG-like"
assembler = VectorAssembler(
    inputCols=["tfidf_features", "text_length", "title_length", "exclamation_count", "word_count"],
    outputCol="features"
)
df_final = assembler.transform(df_tfidf)

df_final.select("features", "label").show(3, truncate=80)


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Split train/test (80/20)
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

print("Train:", train_df.count())
print("Test:", test_df.count())

# Entraînement (algorithme existant de Spark MLlib, aucun modèle custom)
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, seed=42)
rf_model = rf.fit(train_df)

# Prédictions
predictions = rf_model.transform(test_df)
predictions.select("label", "prediction", "probability").show(5)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

accuracy = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)

print(f"Accuracy  : {accuracy:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"Précision : {precision:.4f}")
print(f"Rappel    : {recall:.4f}")

# Matrice de confusion
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()


# COMMAND ----------

import json

metrics = {
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "train_size": train_df.count(),
    "test_size": test_df.count(),
    "confusion_matrix": {
        "fake_predicted_fake": 3286,
        "fake_predicted_real": 109,
        "real_predicted_fake": 3,
        "real_predicted_real": 4184
    }
}

metrics_path = "/Volumes/workspace/default/desinformation_data/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Sauvegardé :", metrics_path)
print(json.dumps(metrics, indent=2))

# COMMAND ----------

# Export du dataset nettoyé + features, pour utilisation par HIVE
output_path = "/Volumes/workspace/default/desinformation_data/news_cleaned"

df.select("title", "text", "subject", "date", "label", "text_length", "title_length", "exclamation_count", "word_count") \
  .write.mode("overwrite").option("header", True).csv(output_path)

print("Exporté vers :", output_path)

# COMMAND ----------

# Création d'une table "HIVE-like" à partir du CSV exporté
df_cleaned = spark.read.csv(output_path, header=True, inferSchema=True)

df_cleaned.write.mode("overwrite").saveAsTable("workspace.default.news_cleaned")

print("Table créée : workspace.default.news_cleaned")
spark.sql("SELECT * FROM workspace.default.news_cleaned LIMIT 5").show()

# COMMAND ----------

# 1. Taux de fake news par catégorie (subject)
print("=== Répartition Fake/Real par catégorie ===")
spark.sql("""
    SELECT subject, label, COUNT(*) as count
    FROM workspace.default.news_cleaned
    GROUP BY subject, label
    ORDER BY subject, label
""").show(20)

# 2. Moyennes par label — ces stats deviennent des features de référence
print("=== Moyennes par label (fake=0, real=1) ===")
spark.sql("""
    SELECT 
        label,
        AVG(text_length) as avg_text_length,
        AVG(title_length) as avg_title_length,
        AVG(exclamation_count) as avg_exclamation,
        AVG(word_count) as avg_word_count,
        COUNT(*) as total
    FROM workspace.default.news_cleaned
    GROUP BY label
""").show()

# 3. Score moyen de ponctuation excessive par catégorie
print("=== Exclamation moyenne par catégorie ===")
spark.sql("""
    SELECT subject, AVG(exclamation_count) as avg_exclamation, COUNT(*) as total
    FROM workspace.default.news_cleaned
    GROUP BY subject
    ORDER BY avg_exclamation DESC
""").show()