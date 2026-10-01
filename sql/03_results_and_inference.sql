
USE DATABASE PHISHING_DB;
USE SCHEMA PHISHING_SCHEMA;
USE WAREHOUSE PHISHING_ML_WH;

SELECT
    MODEL_NAME,
    ROUND(ACCURACY * 100, 2)        AS "Accuracy %",
    ROUND(PRECISION_SCORE * 100, 2) AS "Precision %",
    ROUND(RECALL_SCORE * 100, 2)    AS "Recall %",
    ROUND(F1_SCORE * 100, 2)        AS "F1 Score %",
    ROUND(AUC_ROC, 4)               AS "AUC-ROC",
    TRAINED_AT
FROM MODEL_METRICS
ORDER BY F1_SCORE DESC;

SELECT
    CLASSIFICATION_TYPE,
    COUNT           AS "Count",
    ROUND(COUNT * 100.0 / SUM(COUNT) OVER (), 2) AS "% of Total"
FROM CONFUSION_MATRIX
ORDER BY ACTUAL_LABEL DESC, PREDICTED_LABEL DESC;

SELECT
    p.EMAIL_ID,
    r.SENDER,
    r.SUBJECT,
    p.ACTUAL_LABEL,
    p.PREDICTED_LABEL,
    ROUND(p.PREDICTION_PROBA, 4) AS PHISHING_PROB,
    CASE
        WHEN p.ACTUAL_LABEL = 1 AND p.PREDICTED_LABEL = 0 THEN '⚠️  False Negative (Missed Phishing!)'
        WHEN p.ACTUAL_LABEL = 0 AND p.PREDICTED_LABEL = 1 THEN '🔔 False Positive (Wrongly Flagged)'
    END AS ERROR_TYPE
FROM MODEL_PREDICTIONS p
JOIN RAW_EMAILS r ON p.EMAIL_ID = r.EMAIL_ID
WHERE p.IS_CORRECT = FALSE
  AND p.SPLIT_TYPE = 'TEST'
ORDER BY p.ACTUAL_LABEL DESC;

SELECT
    e.EMAIL_ID,
    r.SENDER,
    r.SUBJECT,
    e.LABEL AS ACTUAL_LABEL,
    CASE WHEN e.LABEL = 1 THEN 'Phishing' ELSE 'Legitimate' END AS ACTUAL_TYPE,
    ROUND(predict_phishing(
        e.HAS_URL, e.URL_COUNT, e.HAS_IP_URL, e.URL_LENGTH,
        e.HAS_URGENT_WORDS, e.HTML_TAGS_COUNT, e.NUM_SPECIAL_CHARS,
        e.WORD_COUNT, e.SUBJECT_LENGTH, e.SENDER_NAME_MISMATCH,
        e.HAS_ATTACHMENT, e.REPLY_TO_DIFFERENT, e.SPF_PASS, e.DKIM_PASS,
        e.AUTH_BOTH_FAILED, e.URL_DENSITY, e.SPECIAL_CHAR_RATIO,
        e.HIGH_RISK_COMBO, e.HAS_LONG_URL, e.SHORT_BODY
    ), 4) AS PHISHING_PROB,
    CASE
        WHEN predict_phishing(
            e.HAS_URL, e.URL_COUNT, e.HAS_IP_URL, e.URL_LENGTH,
            e.HAS_URGENT_WORDS, e.HTML_TAGS_COUNT, e.NUM_SPECIAL_CHARS,
            e.WORD_COUNT, e.SUBJECT_LENGTH, e.SENDER_NAME_MISMATCH,
            e.HAS_ATTACHMENT, e.REPLY_TO_DIFFERENT, e.SPF_PASS, e.DKIM_PASS,
            e.AUTH_BOTH_FAILED, e.URL_DENSITY, e.SPECIAL_CHAR_RATIO,
            e.HIGH_RISK_COMBO, e.HAS_LONG_URL, e.SHORT_BODY
        ) >= 0.5 THEN '🚨 PHISHING'
        ELSE '✅ LEGITIMATE'
    END AS PREDICTION
FROM ENGINEERED_FEATURES e
JOIN RAW_EMAILS r ON e.EMAIL_ID = r.EMAIL_ID
ORDER BY PHISHING_PROB DESC;

SELECT predict_phishing(
    1,      -- has_url
    3,      -- url_count
    1,      -- has_ip_url
    110,    -- url_length
    1,      -- has_urgent_words
    8,      -- html_tags_count
    16,     -- num_special_chars
    35,     -- word_count
    48,     -- subject_length
    1,      -- sender_name_mismatch
    0,      -- has_attachment
    1,      -- reply_to_different
    0,      -- spf_pass
    0,      -- dkim_pass
    1,      -- auth_both_failed
    0.086,  -- url_density
    0.457,  -- special_char_ratio
    1,      -- high_risk_combo
    1,      -- has_long_url
    0       -- short_body
) AS PHISHING_PROBABILITY;

SELECT
    CASE
        WHEN PHISHING_PROB >= 0.8 THEN '🔴 High Risk (>=80%)'
        WHEN PHISHING_PROB >= 0.5 THEN '🟠 Medium Risk (50-79%)'
        WHEN PHISHING_PROB >= 0.3 THEN '🟡 Low Risk (30-49%)'
        ELSE '🟢 Safe (<30%)'
    END AS RISK_LEVEL,
    COUNT(*) AS EMAIL_COUNT,
    ROUND(AVG(ACTUAL_LABEL) * 100, 1) AS ACTUAL_PHISHING_RATE_PCT
FROM (
    SELECT
        e.LABEL AS ACTUAL_LABEL,
        predict_phishing(
            e.HAS_URL, e.URL_COUNT, e.HAS_IP_URL, e.URL_LENGTH,
            e.HAS_URGENT_WORDS, e.HTML_TAGS_COUNT, e.NUM_SPECIAL_CHARS,
            e.WORD_COUNT, e.SUBJECT_LENGTH, e.SENDER_NAME_MISMATCH,
            e.HAS_ATTACHMENT, e.REPLY_TO_DIFFERENT, e.SPF_PASS, e.DKIM_PASS,
            e.AUTH_BOTH_FAILED, e.URL_DENSITY, e.SPECIAL_CHAR_RATIO,
            e.HIGH_RISK_COMBO, e.HAS_LONG_URL, e.SHORT_BODY
        ) AS PHISHING_PROB
    FROM ENGINEERED_FEATURES e
) sub
GROUP BY RISK_LEVEL
ORDER BY MIN(PHISHING_PROB) DESC;

SELECT
    ROUND(
        SUM(CASE WHEN PREDICTED_LABEL = 1 AND ACTUAL_LABEL = 1 THEN 1 ELSE 0 END) * 100.0
        / NULLIF(SUM(CASE WHEN ACTUAL_LABEL = 1 THEN 1 ELSE 0 END), 0)
    , 2) AS "Phishing Detection Rate %",

    ROUND(
        SUM(CASE WHEN PREDICTED_LABEL = 0 AND ACTUAL_LABEL = 0 THEN 1 ELSE 0 END) * 100.0
        / NULLIF(SUM(CASE WHEN ACTUAL_LABEL = 0 THEN 1 ELSE 0 END), 0)
    , 2) AS "Legitimate Accuracy %",

    SUM(CASE WHEN PREDICTED_LABEL = 1 AND ACTUAL_LABEL = 0 THEN 1 ELSE 0 END)
        AS "False Positives (Legit flagged as Phishing)",

    SUM(CASE WHEN PREDICTED_LABEL = 0 AND ACTUAL_LABEL = 1 THEN 1 ELSE 0 END)
        AS "False Negatives (Phishing missed)"
FROM MODEL_PREDICTIONS
WHERE SPLIT_TYPE = 'TEST';

SELECT feature_name, importance_score
FROM VALUES
    ('auth_both_failed',   0.1842),
    ('has_urgent_words',   0.1435),
    ('sender_name_mismatch', 0.1312),
    ('high_risk_combo',    0.1188),
    ('spf_pass',           0.0965),
    ('dkim_pass',          0.0887),
    ('has_ip_url',         0.0754),
    ('url_length',         0.0612),
    ('reply_to_different', 0.0487),
    ('url_density',        0.0348),
    ('url_count',          0.0170)
AS t(feature_name, importance_score)
ORDER BY importance_score DESC;

CREATE OR REPLACE VIEW PHISHING_DETECTION_REPORT AS
SELECT
    e.EMAIL_ID,
    r.SENDER,
    r.SENDER_DOMAIN,
    r.SUBJECT,
    CASE WHEN e.LABEL = 1 THEN 'Phishing' ELSE 'Legitimate' END AS ACTUAL_LABEL,
    e.HAS_URGENT_WORDS,
    e.HAS_IP_URL,
    e.SENDER_NAME_MISMATCH,
    e.AUTH_BOTH_FAILED,
    e.HIGH_RISK_COMBO,
    e.SPF_PASS,
    e.DKIM_PASS,
    ROUND(predict_phishing(
        e.HAS_URL, e.URL_COUNT, e.HAS_IP_URL, e.URL_LENGTH,
        e.HAS_URGENT_WORDS, e.HTML_TAGS_COUNT, e.NUM_SPECIAL_CHARS,
        e.WORD_COUNT, e.SUBJECT_LENGTH, e.SENDER_NAME_MISMATCH,
        e.HAS_ATTACHMENT, e.REPLY_TO_DIFFERENT, e.SPF_PASS, e.DKIM_PASS,
        e.AUTH_BOTH_FAILED, e.URL_DENSITY, e.SPECIAL_CHAR_RATIO,
        e.HIGH_RISK_COMBO, e.HAS_LONG_URL, e.SHORT_BODY
    ), 4) AS RISK_SCORE,
    CASE
        WHEN predict_phishing(
            e.HAS_URL, e.URL_COUNT, e.HAS_IP_URL, e.URL_LENGTH,
            e.HAS_URGENT_WORDS, e.HTML_TAGS_COUNT, e.NUM_SPECIAL_CHARS,
            e.WORD_COUNT, e.SUBJECT_LENGTH, e.SENDER_NAME_MISMATCH,
            e.HAS_ATTACHMENT, e.REPLY_TO_DIFFERENT, e.SPF_PASS, e.DKIM_PASS,
            e.AUTH_BOTH_FAILED, e.URL_DENSITY, e.SPECIAL_CHAR_RATIO,
            e.HIGH_RISK_COMBO, e.HAS_LONG_URL, e.SHORT_BODY
        ) >= 0.8 THEN 'HIGH RISK'
        WHEN predict_phishing(
            e.HAS_URL, e.URL_COUNT, e.HAS_IP_URL, e.URL_LENGTH,
            e.HAS_URGENT_WORDS, e.HTML_TAGS_COUNT, e.NUM_SPECIAL_CHARS,
            e.WORD_COUNT, e.SUBJECT_LENGTH, e.SENDER_NAME_MISMATCH,
            e.HAS_ATTACHMENT, e.REPLY_TO_DIFFERENT, e.SPF_PASS, e.DKIM_PASS,
            e.AUTH_BOTH_FAILED, e.URL_DENSITY, e.SPECIAL_CHAR_RATIO,
            e.HIGH_RISK_COMBO, e.HAS_LONG_URL, e.SHORT_BODY
        ) >= 0.5 THEN 'MEDIUM RISK'
        ELSE 'SAFE'
    END AS RISK_CATEGORY
FROM ENGINEERED_FEATURES e
JOIN RAW_EMAILS r ON e.EMAIL_ID = r.EMAIL_ID;

SELECT * FROM PHISHING_DETECTION_REPORT
ORDER BY RISK_SCORE DESC
LIMIT 10;