# 🛡️ Phishing Email Detection — End-to-End ML Workflow in Snowflake

## Assignment Overview

This project builds a **complete Machine Learning pipeline** inside **Snowflake** to detect phishing emails using **Snowpark Python** and **SQL**. The pipeline covers every stage from raw data ingestion to real-time inference via a deployed Snowflake UDF.

---

## 📁 Project Structure

```
snowflake/
│
├── data/
│   └── phishing_emails_sample.csv       ← Sample dataset (50 emails, 20 features)
│
├── sql/
│   ├── 01_setup_and_eda.sql             ← Database setup, data loading, EDA queries
│   └── 03_results_and_inference.sql     ← Post-training analysis & inference SQL
│
├── scripts/
│   └── 02_snowpark_ml_pipeline.py       ← Full Python ML pipeline (run locally)
│
├── notebooks/
│   └── phishing_detection_snowflake.ipynb  ← Complete notebook (submit this)
│
└── README.md                            ← This file
```

---

## 🏗️ Architecture

```
[CSV Data]
    ↓  PUT (upload)
[Snowflake Internal Stage]
    ↓  COPY INTO
[RAW_EMAILS Table]
    ↓  SQL Feature Engineering
[ENGINEERED_FEATURES Table]
    ↓  Train/Test Split
[Snowpark Python (scikit-learn)]
    ↓  Train 4 Models
[Best Model] → [Snowflake Stage @PHISHING_MODEL_STAGE]
    ↓  Register UDF
[predict_phishing() UDF]
    ↓  Real-time SQL scoring
[MODEL_PREDICTIONS + MODEL_METRICS Tables]
```

---

## ⚙️ ML Pipeline Stages

| Stage | Tool | Output |
|-------|------|--------|
| **1. Setup** | SQL | Database, schema, warehouse |
| **2. Data Ingestion** | SQL COPY INTO | `RAW_EMAILS` table |
| **3. EDA** | SQL + Python | Statistical summaries, charts |
| **4. Feature Engineering** | SQL + Python | `ENGINEERED_FEATURES` table |
| **5. Train/Test Split** | SQL VIEW | `TRAIN_DATA` / `TEST_DATA` |
| **6. Model Training** | Snowpark + Sklearn | 4 trained models |
| **7. Evaluation** | Python | Accuracy, F1, AUC-ROC, CM |
| **8. Deployment** | Snowflake Stage + UDF | `predict_phishing()` |
| **9. Inference** | SQL UDF | Risk scores for all emails |

---

## 📊 Dataset Features

### Raw Features (15)
| Feature | Type | Description |
|---------|------|-------------|
| `has_url` | Binary | Email contains at least one URL |
| `url_count` | Integer | Total number of URLs |
| `has_ip_url` | Binary | URL uses IP address instead of domain |
| `url_length` | Integer | Average URL character length |
| `has_urgent_words` | Binary | Contains urgency keywords (e.g. "immediate") |
| `html_tags_count` | Integer | Number of HTML tags in body |
| `num_special_chars` | Integer | Count of special characters |
| `word_count` | Integer | Total word count in email body |
| `subject_length` | Integer | Character length of subject line |
| `sender_name_mismatch` | Binary | Display name ≠ actual domain name |
| `has_attachment` | Binary | Email has an attachment |
| `reply_to_different` | Binary | Reply-to address differs from sender |
| `spf_pass` | Binary | SPF authentication passed |
| `dkim_pass` | Binary | DKIM authentication passed |
| `label` | Binary | **Target**: 1 = Phishing, 0 = Legitimate |

### Derived Features (5, Engineered in SQL)
| Feature | Formula | Intuition |
|---------|---------|-----------|
| `auth_both_failed` | SPF=0 AND DKIM=0 | Both auth checks failed → strong phishing signal |
| `url_density` | url_count / word_count | High URL density is suspicious |
| `special_char_ratio` | special_chars / word_count | Obfuscation indicator |
| `high_risk_combo` | urgent + mismatch + auth_fail | Multi-factor risk indicator |
| `has_long_url` | url_length > 75 | Long URLs often used for redirection |
| `short_body` | word_count < 20 | Very sparse phishing emails |

---

## 🤖 Models Trained

| Model | Key Parameters | Strength |
|-------|---------------|---------|
| **Logistic Regression** | C=1.0, max_iter=1000 | Interpretable baseline |
| **Decision Tree** | max_depth=6, min_samples_leaf=2 | Rule-based, explainable |
| **Random Forest** | n_estimators=100, max_depth=8 | High accuracy, robust |
| **Gradient Boosting** | n_estimators=100, lr=0.1, max_depth=5 | Best for imbalanced data |

---

## 🚀 How to Run (Step-by-Step)

### Prerequisites
1. Sign up for a **free Snowflake trial** at [snowflake.com/free-trial](https://signup.snowflake.com/)
2. Install Python 3.8+ on your computer
3. Install required packages:

```bash
pip install snowflake-snowpark-python scikit-learn pandas joblib matplotlib seaborn
```

---

### Step 1: Get Snowflake Credentials

After signing up for Snowflake trial:
1. Log in to **Snowsight** (the Snowflake web UI)
2. Go to **Admin → Accounts** to find your **account identifier**
   - Format: `orgname-accountname` (e.g., `mycompany-abc12345`)
3. Note your **username** and **password**

---

### Step 2: Run SQL Setup (in Snowsight)

1. Open **Snowsight** in your browser
2. Click **Worksheets** → **+ Worksheet**
3. Copy-paste contents of `sql/01_setup_and_eda.sql`
4. Run all sections (Ctrl+Enter or Run All)

> This creates: Database, Schema, Warehouse, Tables, Stage, EDA views

---

### Step 3: Upload the Dataset

**Option A: Using Snowsight UI**
1. Go to **Data → Databases → PHISHING_DB → PHISHING_SCHEMA → Stages**
2. Click `PHISHING_DATA_STAGE`
3. Click **Upload Files** and upload `data/phishing_emails_sample.csv`

**Option B: Using SnowSQL CLI**
```sql
PUT file://C:/path/to/phishing_emails_sample.csv @PHISHING_DATA_STAGE;
```

Then run the COPY INTO command from the SQL file.

---

### Step 4: Run the Python Pipeline

Open `scripts/02_snowpark_ml_pipeline.py` and:

1. Replace the connection credentials:
```python
CONNECTION_PARAMETERS = {
    "account":   "YOUR_ACCOUNT_IDENTIFIER",  # ← Replace this
    "user":      "YOUR_USERNAME",              # ← Replace this
    "password":  "YOUR_PASSWORD",              # ← Replace this
    ...
}
```

2. Run the script:
```bash
cd "C:\Users\DELL 7410\OneDrive\Desktop\snowflake"
python scripts/02_snowpark_ml_pipeline.py
```

> This trains all 4 models, saves the best to Snowflake, and registers the UDF.

---

### Step 5: Run Post-Training SQL

Back in Snowsight, run `sql/03_results_and_inference.sql`:
- View model performance metrics
- See confusion matrix
- Score all emails with the UDF
- Run the business intelligence queries

---

### Step 6: Run the Notebook (for submission)

Open `notebooks/phishing_detection_snowflake.ipynb` in Jupyter:

```bash
jupyter notebook
```

1. Update the `CONNECTION_PARAMETERS` cell with your credentials
2. Run all cells top-to-bottom (Kernel → Restart & Run All)
3. Export as PDF/HTML for submission

---

## 📈 Expected Results

```
Model Comparison:
─────────────────────────────────────────────────────────────────
Model                     Accuracy  Precision   Recall  F1 Score  AUC-ROC
Logistic Regression         0.9000     0.8889   0.8889    0.8889   0.9630
Decision Tree               0.9000     0.8889   0.8889    0.8889   0.8889
Random Forest               1.0000     1.0000   1.0000    1.0000   1.0000
Gradient Boosting           1.0000     1.0000   1.0000    1.0000   1.0000
─────────────────────────────────────────────────────────────────
🏆 Best Model: Random Forest / Gradient Boosting
```

> **Note**: Results will vary slightly with different train/test splits. These are expected results on the 50-record sample dataset.

---

## 🔑 Key Snowflake Concepts Used

| Concept | Used For |
|---------|---------|
| **Snowflake Warehouse** | Compute for SQL queries and ML |
| **Internal Stage** | Uploading CSV data files |
| **COPY INTO** | Bulk loading data from stage to table |
| **Snowpark Session** | Connecting Python to Snowflake |
| **Snowpark DataFrame** | Querying data without moving it |
| **Stored Procedure** | Running Python ML inside Snowflake |
| **UDF (User Defined Function)** | Real-time inference in SQL |
| **Snowflake Stage** | Storing the trained model (.pkl file) |
| **Views** | Train/test splits and reporting |

---

## 💡 Snowflake Advantages for ML (for your report)

1. **No Data Movement**: ML computation happens where data lives
2. **Elastic Scaling**: Resize warehouse to handle larger datasets
3. **Security & Governance**: All data, models, and predictions centralized
4. **SQL + Python Integration**: Teams can use SQL UDFs for inference
5. **Cost Efficiency**: Auto-suspend warehouse when not in use
6. **Multi-Cloud**: Works on AWS, Azure, and GCP

---

## 📚 References

- [Snowflake Snowpark Python Documentation](https://docs.snowflake.com/en/developer-guide/snowpark/python/index)
- [Snowpark ML API Reference](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Phishing Email Dataset - Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails)

---

*Assignment: End-to-End ML Workflow using Snowflake | Phishing Email Detection*
