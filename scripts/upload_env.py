import snowflake.connector

conn = snowflake.connector.connect(
    account  = "QVFJDDX-OA60979",
    user     = "HARSHA2501",
    password = "Karukolahemanth@1234",
    warehouse= "COMPUTE_WH",
    database = "PHISHING_DB",
    schema   = "PHISHING_SCHEMA",
)
cs = conn.cursor()

env_path = "C:/Users/DELL 7410/OneDrive/Desktop/snowflake/dashboard/environment.yml"
stage    = "PHISHING_DB.PHISHING_SCHEMA.STREAMLIT_STAGE"

cs.execute(f"PUT 'file://{env_path}' @{stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
print("environment.yml uploaded!")

cs.execute(f"LIST @{stage}")
for row in cs.fetchall():
    print(" ", row[0])

cs.close()
conn.close()
print("Done!")
