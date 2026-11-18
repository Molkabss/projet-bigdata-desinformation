-- =====================================================================
-- clean_features.pig  --  Tache A : nettoyage + feature engineering
-- Projet : detection de la desinformation (Fake and Real News, Kaggle)
-- Compatible Pig 0.9 / 0.12 (Cloudera QuickStart) et versions plus recentes
--
-- Usage :
--   pig -x mapreduce -f clean_features.pig
--   pig -x mapreduce -param INPUT=... -param OUTPUT=... -param STATS=... -f clean_features.pig
-- =====================================================================

SET job.name 'fakenews_clean_features';
SET default_parallel 1;

%default INPUT  '/user/cloudera/fakenews/raw/news.tsv'
%default OUTPUT '/user/cloudera/fakenews/clean'
%default STATS  '/user/cloudera/fakenews/stats'

rmf $OUTPUT;
rmf $STATS;

-- 1. CHARGEMENT  (id, title, text, subject, date, label : Fake=0, Real=1)
raw = LOAD '$INPUT' USING PigStorage('\t')
      AS (id:int, title:chararray, text:chararray, subject:chararray,
          date:chararray, label:int);

-- 2. NETTOYAGE : lignes vides / nulles
valid = FILTER raw BY id IS NOT NULL AND label IS NOT NULL
                  AND text IS NOT NULL AND title IS NOT NULL
                  AND SIZE(TRIM(text)) > 0 AND SIZE(TRIM(title)) > 0;

-- 3. NETTOYAGE : doublons (meme texte) -> on garde l'article d'id minimal
by_text = GROUP valid BY text;
dedup = FOREACH by_text {
    ordered = ORDER valid BY id ASC;
    first_row = LIMIT ordered 1;
    GENERATE FLATTEN(first_row)
      AS (id:int, title:chararray, text:chararray, subject:chararray,
          date:chararray, label:int);
};

-- 4. ANTI-FUITE : suppression des marqueurs de source
no_dateline = FOREACH dedup GENERATE id, label, subject, date, title,
    REPLACE(text, '^[A-Z][A-Z .,/-]{0,60}\\(Reuters\\)\\s*[^A-Za-z0-9\\s]?\\s*', '') AS text;

no_source = FOREACH no_dateline GENERATE id, label, subject, date,
    REPLACE(title, '(?i)reuters', ' ') AS title,
    REPLACE(
      REPLACE(
        REPLACE(text, '(?i)\\(reuters\\)\\s*[^A-Za-z0-9\\s]?', ' '),
        '(?i)reuters', ' '),
      '(?i)(pic\\.twitter\\.com/\\S+|featured image|21st century wire|getty images)', ' ')
    AS text;

tidy = FOREACH no_source GENERATE id, label, subject, date,
    TRIM(REPLACE(title, '\\s+', ' ')) AS title,
    TRIM(REPLACE(text,  '\\s+', ' ')) AS text;

good = FILTER tidy BY SIZE(text) > 0 AND SIZE(title) > 0;

-- 5. FEATURE ENGINEERING (etape 1 : mesures brutes + mots)
base = FOREACH good GENERATE id, label, subject, date, title, text,
    SIZE(text)                              AS text_len,
    SIZE(title)                             AS title_len,
    SIZE(REPLACE(text,  '[^A-Z]', ''))      AS upper_cnt,
    SIZE(REPLACE(title, '[^A-Z]', ''))      AS title_upper_cnt,
    SIZE(REPLACE(text,  '[^!]', ''))        AS excl_cnt,
    SIZE(REPLACE(text,  '[^?]', ''))        AS ques_cnt,
    ((text MATCHES '.*[!?]{2,}.*') ? 1 : 0) AS multi_punct,
    TOKENIZE(REPLACE(LOWER(text),  '[^a-z0-9\\s]', ' ')) AS words,
    TOKENIZE(REPLACE(LOWER(title), '[^a-z0-9\\s]', ' ')) AS title_words;

-- 6. FEATURE ENGINEERING (etape 2 : ratios + mots sensationnalistes)
feat = FOREACH base {
    sens_t = FILTER words BY $0 MATCHES '^(breaking|shocking|shocked|unbelievable|outrageous|bombshell|exposed|scandal|disgusting|insane|stunning|incredible|horrifying|terrifying|hoax|conspiracy|wow|urgent|epic|destroys|destroyed|slams|slammed|busted|humiliated|meltdown|liar|crooked|traitor|treason|disaster|nightmare|hilarious|unreal|outrage|savage|brutal|crushed)$';
    sens_h = FILTER title_words BY $0 MATCHES '^(breaking|shocking|shocked|unbelievable|outrageous|bombshell|exposed|scandal|disgusting|insane|stunning|incredible|horrifying|terrifying|hoax|conspiracy|wow|urgent|epic|destroys|destroyed|slams|slammed|busted|humiliated|meltdown|liar|crooked|traitor|treason|disaster|nightmare|hilarious|unreal|outrage|savage|brutal|crushed)$';
    GENERATE id, label, subject, date, title, text,
        text_len, title_len,
        SIZE(words) AS word_count,
        upper_cnt,
        (double)upper_cnt / (double)text_len        AS upper_ratio,
        (double)title_upper_cnt / (double)title_len AS title_upper_ratio,
        excl_cnt, ques_cnt,
        (double)(excl_cnt + ques_cnt) / (double)text_len AS punct_ratio,
        multi_punct,
        COUNT_STAR(sens_t) AS sens_text,
        COUNT_STAR(sens_h) AS sens_title;
};

features_out = FOREACH feat GENERATE
    id, label, subject,
    (int)REGEX_EXTRACT(date, '(\\d{4})', 1)              AS pub_year,
    LOWER(REGEX_EXTRACT(date, '^\\s*([A-Za-z]{3})', 1))  AS pub_month,
    text_len, title_len, word_count,
    upper_cnt, upper_ratio, title_upper_ratio,
    excl_cnt, ques_cnt, punct_ratio, multi_punct,
    sens_text, sens_title,
    (word_count > 0 ? (double)sens_text * 100.0 / (double)word_count : 0.0) AS sens_density,
    title, text;

-- 7. STATISTIQUES AVANT / APRES (pour le rapport)
g_raw   = GROUP raw ALL;
g_valid = GROUP valid ALL;
g_dedup = GROUP dedup ALL;
g_final = GROUP features_out ALL;
g_lab   = GROUP features_out BY label;

c_raw   = FOREACH g_raw   GENERATE 'raw_loaded'      AS stage, COUNT_STAR(raw)          AS n;
c_valid = FOREACH g_valid GENERATE 'after_non_empty' AS stage, COUNT_STAR(valid)        AS n;
c_dedup = FOREACH g_dedup GENERATE 'after_dedup'     AS stage, COUNT_STAR(dedup)        AS n;
c_final = FOREACH g_final GENERATE 'final'           AS stage, COUNT_STAR(features_out) AS n;
c_lab   = FOREACH g_lab   GENERATE CONCAT('final_label_', (chararray)group) AS stage,
                                   COUNT_STAR(features_out) AS n;

stats = UNION c_raw, c_valid, c_dedup, c_final, c_lab;

-- 8. EXPORT vers HDFS (lu ensuite par HIVE : table externe sur $OUTPUT)
STORE features_out INTO '$OUTPUT' USING PigStorage('\t');
STORE stats        INTO '$STATS'  USING PigStorage('\t');