import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import joblib
import snowflake.connector

ACCOUNT  = "QVFJDDX-OA60979"
USER     = "HARSHA2501"
PASSWORD = "Karukolahemanth@1234"
BASE_DIR = r"C:\Users\DELL 7410\Downloads\kaggle_spam_data"
MODEL_PATH = r"C:\Users\DELL 7410\OneDrive\Desktop\snowflake\data\best_model.pkl"

print("=" * 60)
print("  Phishing / Spam Email Detector - ML Pipeline")
print("  Source: Local Kaggle CSVs -> Results -> Snowflake")
print("=" * 60)

print("\nStep 1: Loading Kaggle datasets from local disk...")
frames = []
configs = [
    ("Enron.csv",         ["subject", "body", "label"]),
    ("CEAS_08.csv",       ["sender", "subject", "body", "label"]),
    ("SpamAssasin.csv",   ["sender", "subject", "body", "label"]),
    ("Nazario.csv",       ["sender", "subject", "body", "label"]),
    ("Nigerian_Fraud.csv",["sender", "subject", "body", "label"]),
]

for fname, cols in configs:
    fpath = os.path.join(BASE_DIR, fname)
    try:
        header = open(fpath, encoding="utf-8").readline()
        available = [c for c in cols if c in header]
        df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip", usecols=available)
        df["source"] = fname.replace(".csv", "")
        if "subject" not in df.columns: df["subject"] = ""
        if "body"    not in df.columns: df["body"]    = ""
        df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        frames.append(df[["text", "label", "source"]])
        lc = df["label"].value_counts().to_dict()
        print(f"  {fname}: {len(df):,} rows | spam={lc.get(1,0):,} legit={lc.get(0,0):,}")
    except Exception as e:
        print(f"  {fname}: skipped ({e})")

combined = pd.concat(frames, ignore_index=True)
combined = combined[combined["text"].str.len() > 5].reset_index(drop=True)

total = len(combined)
spam  = combined["label"].sum()
legit = total - spam
print(f"\n  Total loaded: {total:,} | Spam: {spam:,} | Legit: {legit:,}")

print("\nStep 2: Cleaning text...")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

combined["text"] = combined["text"].apply(clean_text)

X = combined["text"]
y = combined["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

TFIDF = TfidfVectorizer(
    max_features  = 30_000,
    ngram_range   = (1, 2),
    sublinear_tf  = True,
    min_df        = 2,
    strip_accents = "unicode",
    token_pattern = r"\b[a-zA-Z]\w+\b",
)

MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15, class_weight="balanced", random_state=42,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42,
    ),
}

print("\nStep 3: Training 4 models...")
print("-" * 60)
results = []

for name, clf in MODELS.items():
    print(f"\n  [{name}] Training...", end="", flush=True)
    pipe = Pipeline([("tfidf", TFIDF), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f" done!")
    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    Precision: {prec*100:.2f}%")
    print(f"    Recall   : {rec*100:.2f}%")
    print(f"    F1 Score : {f1*100:.2f}%")
    print(f"    AUC-ROC  : {auc:.4f}")

    results.append({
        "name": name, "pipeline": pipe,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc_roc": auc, "cm": cm,
    })

best = max(results, key=lambda x: x["f1"])
print(f"\n{'='*60}")
print(f"  Best Model : {best['name']}")
print(f"  F1 Score   : {best['f1']*100:.2f}%")
print(f"  AUC-ROC    : {best['auc_roc']:.4f}")
print(f"{'='*60}")

print(f"\nStep 4: Saving best model...")
joblib.dump(best["pipeline"], MODEL_PATH)
print(f"  Saved: {MODEL_PATH}")

print("\nStep 5: Connecting to Snowflake to write results...")
conn = snowflake.connector.connect(
    account   = ACCOUNT,
    user      = USER,
    password  = PASSWORD,
    warehouse = "PHISHING_ML_WH",
    database  = "PHISHING_DB",
    schema    = "PHISHING_SCHEMA",
)
cs = conn.cursor()
print("  Connected!")

print("\nStep 6: Writing MODEL_METRICS to Snowflake...")
cs.execute("""
    CREATE OR REPLACE TABLE MODEL_METRICS (
        MODEL_NAME      VARCHAR(100),
        ACCURACY        FLOAT,
        PRECISION_SCORE FLOAT,
        RECALL_SCORE    FLOAT,
        F1_SCORE        FLOAT,
        AUC_ROC         FLOAT,
        IS_BEST         BOOLEAN
    )
""")
for r in results:
    is_best = str(r["name"] == best["name"]).upper()
    cs.execute(f"""
        INSERT INTO MODEL_METRICS VALUES (
            '{r["name"]}', {r["accuracy"]:.6f}, {r["precision"]:.6f},
            {r["recall"]:.6f}, {r["f1"]:.6f}, {r["auc_roc"]:.6f}, {is_best}
        )
    """)
print("  MODEL_METRICS saved.")

print("\nStep 7: Writing sample predictions to Snowflake...")
cs.execute("""
    CREATE OR REPLACE TABLE MODEL_PREDICTIONS (
        TRUE_LABEL INTEGER,
        PREDICTED  INTEGER,
        SPAM_PROB  FLOAT
    )
""")
y_pred_best  = best["pipeline"].predict(X_test)
y_proba_best = best["pipeline"].predict_proba(X_test)[:, 1]

pred_df = pd.DataFrame({
    "TRUE_LABEL": y_test.values[:2000],
    "PREDICTED":  y_pred_best[:2000],
    "SPAM_PROB":  y_proba_best[:2000],
})
for _, row in pred_df.iterrows():
    cs.execute(f"INSERT INTO MODEL_PREDICTIONS VALUES ({int(row.TRUE_LABEL)}, {int(row.PREDICTED)}, {row.SPAM_PROB:.6f})")
print(f"  {len(pred_df):,} predictions saved.")

print("\nStep 8: Creating CONFUSION_MATRIX view...")
cm = best["cm"]
tn, fp, fn, tp = cm.ravel()
cs.execute(f"""
    CREATE OR REPLACE VIEW CONFUSION_MATRIX AS
    SELECT
        {tp}   AS TRUE_POSITIVES,
        {tn}   AS TRUE_NEGATIVES,
        {fp}   AS FALSE_POSITIVES,
        {fn}   AS FALSE_NEGATIVES,
        ROUND({tp/(tp+fn)*100}, 2) AS RECALL_PCT,
        ROUND({tp/(tp+fp)*100}, 2) AS PRECISION_PCT
""")
print("  CONFUSION_MATRIX view created.")

cs.close()
conn.close()

print("\n" + "="*60)
print("  PIPELINE COMPLETE!")
print("="*60)
print(f"  Best Model      : {best['name']}")
print(f"  Accuracy        : {best['accuracy']*100:.2f}%")
print(f"  Precision       : {best['precision']*100:.2f}%")
print(f"  Recall          : {best['recall']*100:.2f}%")
print(f"  F1 Score        : {best['f1']*100:.2f}%")
print(f"  AUC-ROC         : {best['auc_roc']:.4f}")
print(f"  Training Emails : {len(X_train):,}")
print(f"  Test Emails     : {len(X_test):,}")
print(f"  True Positives  : {tp:,}  (spam caught)")
print(f"  False Negatives : {fn:,}  (spam missed)")
print("="*60)
print("\nSnowflake tables updated:")
print("  PHISHING_DB.PHISHING_SCHEMA.MODEL_METRICS")
print("  PHISHING_DB.PHISHING_SCHEMA.MODEL_PREDICTIONS")
print("  PHISHING_DB.PHISHING_SCHEMA.CONFUSION_MATRIX (view)")
print("\nAll done!")