import pandas as pd
import os

BASE = r"C:\Users\DELL 7410\Downloads\kaggle_spam_data"
OUT  = r"C:\Users\DELL 7410\OneDrive\Desktop\snowflake\data\kaggle_emails.csv"

frames = []

configs = [
    ("Enron.csv",         ["subject", "body", "label"]),
    ("CEAS_08.csv",       ["sender", "subject", "body", "label", "urls"]),
    ("SpamAssasin.csv",   ["sender", "subject", "body", "label", "urls"]),
    ("Nazario.csv",       ["sender", "subject", "body", "label", "urls"]),
    ("Nigerian_Fraud.csv",["sender", "subject", "body", "label", "urls"]),
]

for fname, desired_cols in configs:
    fpath = os.path.join(BASE, fname)
    try:
        df_raw = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip", nrows=5)
        available = [c for c in desired_cols if c in df_raw.columns]
        df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip", usecols=available)
        df["source"] = fname.replace(".csv", "")
        if "sender" not in df.columns:
            df["sender"] = ""
        if "urls" not in df.columns:
            df["urls"] = 0
        df = df[["sender", "subject", "body", "label", "urls", "source"]]
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        lc = df["label"].value_counts().to_dict()
        print(f"{fname}: {df.shape} | spam={lc.get(1,0)} legit={lc.get(0,0)}")
        frames.append(df)
    except Exception as e:
        print(f"{fname}: ERROR - {e}")

combined = pd.concat(frames, ignore_index=True)
combined = combined.dropna(subset=["body"])
combined = combined[combined["body"].str.strip().str.len() > 5]
combined["body"]    = combined["body"].str.replace("\n", " ").str.replace("\r", " ").str[:2000]
combined["subject"] = combined["subject"].fillna("").str[:300]
combined["sender"]  = combined["sender"].fillna("").str[:200]
combined["urls"]    = pd.to_numeric(combined["urls"], errors="coerce").fillna(0).astype(int)
combined["email_id"] = range(1, len(combined) + 1)

combined = combined[["email_id", "sender", "subject", "body", "urls", "label", "source"]]
combined.to_csv(OUT, index=False, encoding="utf-8")

total = len(combined)
spam  = combined["label"].sum()
legit = total - spam
print(f"\nFinal dataset: {total:,} rows | spam={spam:,} | legit={legit:,}")
print(f"Saved to: {OUT}")
print(f"File size: {os.path.getsize(OUT)/1024/1024:.1f} MB")
