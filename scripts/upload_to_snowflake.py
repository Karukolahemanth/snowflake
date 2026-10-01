import snowflake.connector
import os

ACCOUNT  = "YOUR_ACCOUNT_IDENTIFIER"
USER     = "HARSHA2501"
PASSWORD = "YOUR_PASSWORD"

CSV_PATH = r"C:\Users\DELL 7410\OneDrive\Desktop\snowflake\data\kaggle_emails.csv"

print("Connecting to Snowflake...")
conn = snowflake.connector.connect(
    account   = ACCOUNT,
    user      = USER,
    password  = PASSWORD,
    warehouse = "PHISHING_ML_WH",
    database  = "PHISHING_DB",
    schema    = "PHISHING_SCHEMA",
)
cs = conn.cursor()

print("Uploading CSV to stage (this may take 1-2 minutes for 78MB)...")
cs.execute(f"PUT file://{CSV_PATH} @KAGGLE_EMAIL_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE")
print("Upload done!")

print("Loading data into KAGGLE_EMAILS table...")
cs.execute("""
    COPY INTO KAGGLE_EMAILS
    FROM @KAGGLE_EMAIL_STAGE
    FILE_FORMAT = (
        TYPE                         = 'CSV'
        FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        SKIP_HEADER                  = 1
        NULL_IF                      = ('NULL','null','')
        EMPTY_FIELD_AS_NULL          = TRUE
    )
    ON_ERROR = 'CONTINUE'
""")
print("Copy done!")

cs.execute("SELECT COUNT(*), SUM(LABEL), COUNT(*)-SUM(LABEL) FROM KAGGLE_EMAILS")
total, spam, legit = cs.fetchone()
print(f"\nSnowflake table loaded!")
print(f"  Total : {total:,}")
print(f"  Spam  : {spam:,}")
print(f"  Legit : {legit:,}")

cs.close()
conn.close()
print("\nDone! Data is live in Snowflake.")
