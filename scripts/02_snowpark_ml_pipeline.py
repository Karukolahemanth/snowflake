
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

import joblib
import os

CONNECTION_PARAMETERS = {
    "account":   "YOUR_ACCOUNT_IDENTIFIER",  # e.g. "abc123.us-east-1"
    "user":      "YOUR_USERNAME",
    "password":  "YOUR_PASSWORD",
    "role":      "SYSADMIN",                 # or ACCOUNTADMIN for trial
    "warehouse": "PHISHING_ML_WH",
    "database":  "PHISHING_DB",
    "schema":    "PHISHING_SCHEMA"
}

print("=" * 60)
print("  PHISHING EMAIL DETECTION - ML PIPELINE (SNOWPARK)")
print("=" * 60)

session = Session.builder.configs(CONNECTION_PARAMETERS).create()
print(f"\n✅ Connected to Snowflake!")
print(f"   Account  : {session.get_current_account()}")
print(f"   User     : {session.get_current_user()}")
print(f"   Warehouse: {session.get_current_warehouse()}")
print(f"   Database : {session.get_current_database()}")
print(f"   Schema   : {session.get_current_schema()}\n")

print("─" * 60)
print("📊 SECTION 2: LOADING DATA")
print("─" * 60)

snow_df = session.table("ENGINEERED_FEATURES")

print(f"Total records in ENGINEERED_FEATURES: {snow_df.count()}")
print("\nSchema:")
snow_df.printSchema()

df = snow_df.to_pandas()

df.columns = df.columns.str.lower()

print(f"\n✅ Data loaded into Pandas DataFrame: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"Phishing rate: {df['label'].mean()*100:.1f}%")

print("\n" + "─" * 60)
print("🔍 SECTION 3: EXPLORATORY DATA ANALYSIS")
print("─" * 60)

print("\n📈 Descriptive Statistics:")
print(df.describe().round(3))

print("\n📊 Feature Averages by Class:")
feature_cols = [
    'has_url', 'url_count', 'has_ip_url', 'url_length',
    'has_urgent_words', 'html_tags_count', 'num_special_chars',
    'word_count', 'subject_length', 'sender_name_mismatch',
    'reply_to_different', 'spf_pass', 'dkim_pass',
    'auth_both_failed', 'url_density', 'special_char_ratio',
    'high_risk_combo', 'has_long_url', 'short_body'
]

comparison = df.groupby('label')[feature_cols].mean().round(3)
comparison.index = ['Legitimate (0)', 'Phishing (1)']
print(comparison.T.to_string())

print(f"\n🔍 Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅ No missing values found!")
else:
    print(missing[missing > 0])

print("\n📊 Feature Correlation with Label (Top 10):")
correlations = df[feature_cols + ['label']].corr()['label'].drop('label')
print(correlations.sort_values(ascending=False).head(10).round(4))

print("\n" + "─" * 60)
print("⚙️  SECTION 4: FEATURE ENGINEERING")
print("─" * 60)

FEATURES = [
    'has_url', 'url_count', 'has_ip_url', 'url_length',
    'has_urgent_words', 'html_tags_count', 'num_special_chars',
    'word_count', 'subject_length', 'sender_name_mismatch',
    'has_attachment', 'reply_to_different', 'spf_pass', 'dkim_pass',
    'auth_both_failed', 'url_density', 'special_char_ratio',
    'high_risk_combo', 'has_long_url', 'short_body'
]

X = df[FEATURES]
y = df['label']

print(f"✅ Feature matrix shape : {X.shape}")
print(f"✅ Target vector shape  : {y.shape}")
print(f"✅ Features used ({len(FEATURES)}): {FEATURES}")

print("\n" + "─" * 60)
print("✂️  SECTION 5: TRAIN / TEST SPLIT")
print("─" * 60)

train_mask = df['email_id'].apply(lambda x: x % 5 != 0)
test_mask  = df['email_id'].apply(lambda x: x % 5 == 0)

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

print(f"Training set : {X_train.shape[0]} samples "
      f"({y_train.sum()} phishing, {(y_train==0).sum()} legitimate)")
print(f"Test set     : {X_test.shape[0]} samples "
      f"({y_test.sum()} phishing, {(y_test==0).sum()} legitimate)")

print("\n" + "─" * 60)
print("🤖 SECTION 6: MODEL TRAINING")
print("─" * 60)

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
    ]),
    "Decision Tree": Pipeline([
        ('clf', DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=2))
    ]),
    "Random Forest": Pipeline([
        ('clf', RandomForestClassifier(
            n_estimators=100, random_state=42,
            max_depth=8, min_samples_leaf=2, n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ('clf', GradientBoostingClassifier(
            n_estimators=100, random_state=42,
            learning_rate=0.1, max_depth=5
        ))
    ])
}

results = {}

for model_name, pipeline in models.items():
    print(f"\n  Training: {model_name}...")

    pipeline.fit(X_train, y_train)

    y_pred       = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_pred_proba)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')

    results[model_name] = {
        'pipeline':    pipeline,
        'y_pred':      y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy':    acc,
        'precision':   prec,
        'recall':      rec,
        'f1':          f1,
        'auc_roc':     auc,
        'cv_f1_mean':  cv_scores.mean(),
        'cv_f1_std':   cv_scores.std()
    }

    print(f"    Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"    CV F1    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n" + "─" * 60)
print("📊 SECTION 7: MODEL COMPARISON")
print("─" * 60)

print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>10} {'F1 Score':>10} {'AUC-ROC':>10}")
print("─" * 80)

for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['auc_roc']:>10.4f}")

best_model_name = max(results, key=lambda k: results[k]['f1'])
best_result     = results[best_model_name]
best_pipeline   = best_result['pipeline']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1 Score : {best_result['f1']:.4f}")
print(f"   AUC-ROC  : {best_result['auc_roc']:.4f}")

print(f"\n📋 Classification Report ({best_model_name}):")
print(classification_report(
    y_test, best_result['y_pred'],
    target_names=['Legitimate', 'Phishing']
))

cm = confusion_matrix(y_test, best_result['y_pred'])
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]} | FP={cm[0,1]}")
print(f"  FN={cm[1,0]} | TP={cm[1,1]}")

print("\n" + "─" * 60)
print("🌟 SECTION 8: FEATURE IMPORTANCE")
print("─" * 60)

rf_model = results['Random Forest']['pipeline'].named_steps['clf']
importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
for i, (feat, imp) in enumerate(importances_sorted.head(10).items(), 1):
    bar = "█" * int(imp * 100)
    print(f"  {i:2}. {feat:<25} {imp:.4f} {bar}")

print("\n" + "─" * 60)
print("💾 SECTION 9: SAVING MODEL TO SNOWFLAKE STAGE")
print("─" * 60)

model_filename = "phishing_detector_model.pkl"
local_model_path = os.path.join(os.getcwd(), model_filename)

joblib.dump(best_pipeline, local_model_path)
print(f"✅ Model saved locally: {local_model_path}")

session.sql("""
    CREATE STAGE IF NOT EXISTS PHISHING_MODEL_STAGE
    COMMENT = 'Stage to store trained ML model files'
""").collect()

session.file.put(
    local_path=local_model_path,
    stage_location="@PHISHING_MODEL_STAGE",
    overwrite=True
)
print(f"✅ Model uploaded to: @PHISHING_MODEL_STAGE/{model_filename}")

stage_files = session.sql("LIST @PHISHING_MODEL_STAGE").collect()
print(f"\nFiles in @PHISHING_MODEL_STAGE:")
for f in stage_files:
    print(f"  → {f['name']}  ({f['size']} bytes)")

print("\n" + "─" * 60)
print("📤 SECTION 10: WRITING RESULTS TO SNOWFLAKE")
print("─" * 60)

all_predictions = []
for model_name, res in results.items():
    test_ids = df[test_mask]['email_id'].values
    for i, email_id in enumerate(test_ids):
        all_predictions.append({
            'EMAIL_ID':         int(email_id),
            'ACTUAL_LABEL':     int(y_test.iloc[i]),
            'PREDICTED_LABEL':  int(res['y_pred'][i]),
            'PREDICTION_PROBA': float(res['y_pred_proba'][i]),
            'IS_CORRECT':       bool(y_test.iloc[i] == res['y_pred'][i]),
            'SPLIT_TYPE':       'TEST'
        })

pred_df = pd.DataFrame(all_predictions)

session.sql("TRUNCATE TABLE IF EXISTS MODEL_PREDICTIONS").collect()
snow_pred_df = session.create_dataframe(pred_df)
snow_pred_df.write.mode("append").save_as_table("MODEL_PREDICTIONS")
print(f"✅ Predictions written: {len(pred_df)} rows to MODEL_PREDICTIONS")

metrics_rows = []
for model_name, res in results.items():
    metrics_rows.append({
        'MODEL_NAME':      model_name,
        'ACCURACY':        float(res['accuracy']),
        'PRECISION_SCORE': float(res['precision']),
        'RECALL_SCORE':    float(res['recall']),
        'F1_SCORE':        float(res['f1']),
        'AUC_ROC':         float(res['auc_roc']),
        'TRAINED_AT':      pd.Timestamp.now()
    })

metrics_df = pd.DataFrame(metrics_rows)
session.sql("TRUNCATE TABLE IF EXISTS MODEL_METRICS").collect()
snow_metrics_df = session.create_dataframe(metrics_df)
snow_metrics_df.write.mode("append").save_as_table("MODEL_METRICS")
print(f"✅ Metrics written: {len(metrics_df)} models to MODEL_METRICS")

print("\n📊 Model Metrics in Snowflake:")
session.sql("""
    SELECT MODEL_NAME,
           ROUND(ACCURACY*100,2) AS ACCURACY_PCT,
           ROUND(F1_SCORE*100,2) AS F1_PCT,
           ROUND(AUC_ROC,4) AS AUC_ROC
    FROM MODEL_METRICS
    ORDER BY F1_SCORE DESC
""").show()

print("\n📊 Confusion Matrix from Snowflake:")
session.table("CONFUSION_MATRIX").show()

print("\n" + "─" * 60)
print("🚀 SECTION 11: REGISTERING PREDICTION UDF")
print("─" * 60)

@session.udf.register(
    name="predict_phishing",
    is_permanent=True,
    stage_location="@PHISHING_MODEL_STAGE",
    packages=["scikit-learn", "joblib", "pandas"],
    replace=True,
    return_type="float",
    input_types=["float"] * len(FEATURES),
    comment="UDF to predict phishing probability for an email"
)
def predict_phishing(*feature_values):
    """
    Returns phishing probability (0.0 to 1.0) for a given email.
    Input: feature values in same order as FEATURES list.
    Output: probability score (>0.5 = phishing)
    """
    import joblib
    import pandas as pd
    import sys
    import os

    model_path = os.path.join(sys._xoptions.get("snowflake_import_directory", ""),
                              "phishing_detector_model.pkl")
    model = joblib.load(model_path)

    feature_names = [
        'has_url', 'url_count', 'has_ip_url', 'url_length',
        'has_urgent_words', 'html_tags_count', 'num_special_chars',
        'word_count', 'subject_length', 'sender_name_mismatch',
        'has_attachment', 'reply_to_different', 'spf_pass', 'dkim_pass',
        'auth_both_failed', 'url_density', 'special_char_ratio',
        'high_risk_combo', 'has_long_url', 'short_body'
    ]

    features_df = pd.DataFrame([list(feature_values)], columns=feature_names)
    proba = model.predict_proba(features_df)[0][1]
    return float(proba)

print("✅ UDF 'predict_phishing' registered in Snowflake!")

print("\n🧪 Testing UDF with a phishing email example:")
session.sql("""
    SELECT
        EMAIL_ID,
        LABEL,
        CASE WHEN LABEL = 1 THEN 'Phishing' ELSE 'Legitimate' END AS ACTUAL,
        ROUND(predict_phishing(
            HAS_URL, URL_COUNT, HAS_IP_URL, URL_LENGTH,
            HAS_URGENT_WORDS, HTML_TAGS_COUNT, NUM_SPECIAL_CHARS,
            WORD_COUNT, SUBJECT_LENGTH, SENDER_NAME_MISMATCH,
            HAS_ATTACHMENT, REPLY_TO_DIFFERENT, SPF_PASS, DKIM_PASS,
            AUTH_BOTH_FAILED, URL_DENSITY, SPECIAL_CHAR_RATIO,
            HIGH_RISK_COMBO, HAS_LONG_URL, SHORT_BODY
        ), 4) AS PHISHING_PROBABILITY,
        CASE
            WHEN predict_phishing(
                HAS_URL, URL_COUNT, HAS_IP_URL, URL_LENGTH,
                HAS_URGENT_WORDS, HTML_TAGS_COUNT, NUM_SPECIAL_CHARS,
                WORD_COUNT, SUBJECT_LENGTH, SENDER_NAME_MISMATCH,
                HAS_ATTACHMENT, REPLY_TO_DIFFERENT, SPF_PASS, DKIM_PASS,
                AUTH_BOTH_FAILED, URL_DENSITY, SPECIAL_CHAR_RATIO,
                HIGH_RISK_COMBO, HAS_LONG_URL, SHORT_BODY
            ) >= 0.5 THEN 'PHISHING'
            ELSE 'LEGITIMATE'
        END AS PREDICTION
    FROM ENGINEERED_FEATURES
    LIMIT 10
""").show()

print("\n" + "=" * 60)
print("🎉 PIPELINE COMPLETE - SUMMARY")
print("=" * 60)
print(f"\n✅ Data loaded          : {len(df)} emails")
print(f"✅ Features engineered  : {len(FEATURES)} features")
print(f"✅ Models trained       : {len(models)}")
print(f"🏆 Best model          : {best_model_name}")
print(f"   Accuracy            : {best_result['accuracy']*100:.2f}%")
print(f"   Precision           : {best_result['precision']*100:.2f}%")
print(f"   Recall              : {best_result['recall']*100:.2f}%")
print(f"   F1 Score            : {best_result['f1']*100:.2f}%")
print(f"   AUC-ROC             : {best_result['auc_roc']:.4f}")
print(f"\n✅ Model saved to      : @PHISHING_MODEL_STAGE")
print(f"✅ Predictions saved   : MODEL_PREDICTIONS table")
print(f"✅ Metrics saved       : MODEL_METRICS table")
print(f"✅ UDF deployed        : predict_phishing()")
print(f"\n{'=' * 60}")

session.close()
print("\n✅ Snowflake session closed.")