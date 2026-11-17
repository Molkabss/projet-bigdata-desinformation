-- =====================================================
-- Détection de la Désinformation — Requêtes HIVE
-- Rôle : transformer le résultat de PIG en tables structurées
-- et produire des agrégats qui deviennent des features
-- supplémentaires pour Spark MLlib.
-- =====================================================

-- 1. Création de la table externe à partir du résultat nettoyé (sortie de PIG)
CREATE EXTERNAL TABLE IF NOT EXISTS news_cleaned (
    title STRING,
    text STRING,
    subject STRING,
    date STRING,
    label INT,
    text_length INT,
    title_length INT,
    exclamation_count INT,
    word_count INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/data/news_cleaned'
TBLPROPERTIES ("skip.header.line.count"="1");

-- 2. Répartition Fake/Real par catégorie (subject)
--    Utile pour comprendre si certains sujets sont plus touchés par la désinformation
SELECT subject, label, COUNT(*) AS count
FROM news_cleaned
GROUP BY subject, label
ORDER BY subject, label;

-- 3. Moyennes des features par label (fake=0, real=1)
--    Ces agrégats deviennent des features de référence pour le modèle Spark MLlib
SELECT
    label,
    AVG(text_length)       AS avg_text_length,
    AVG(title_length)      AS avg_title_length,
    AVG(exclamation_count) AS avg_exclamation,
    AVG(word_count)        AS avg_word_count,
    COUNT(*)                AS total
FROM news_cleaned
GROUP BY label;

-- 4. Score moyen de ponctuation excessive par catégorie
--    Permet d'identifier les catégories les plus "sensationnalistes"
SELECT
    subject,
    AVG(exclamation_count) AS avg_exclamation,
    COUNT(*) AS total
FROM news_cleaned
GROUP BY subject
ORDER BY avg_exclamation DESC;

-- =====================================================
-- Résultats obtenus (exécutés via Spark SQL, syntaxe HiveQL identique) :
--
-- Moyennes par label :
--   Fake (0) : text_length=2549, title_length=91, exclamation=0.74, word_count=437
--   Real (1) : text_length=2378, title_length=65, exclamation=0.06, word_count=394
--
-- -> Les articles Fake contiennent ~12x plus de points d'exclamation en moyenne
--    que les articles Real : un signal prédictif fort, confirmé empiriquement.
--
-- Catégorie la plus "sensationnaliste" : politics (avg_exclamation = 0.92)
-- =====================================================