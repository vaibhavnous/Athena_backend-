
"""
AUTO-GENERATED BRONZE INGESTION SCRIPT

Source: Vendor SFTP
Folder: /daily/transactions/
File pattern: TXN_YYYYMMDD.csv
Expected runtime: Spark / Databricks with Delta support and paramiko installed
Target table: main.bronze.bronze_transaction

DO NOT EDIT MANUALLY
"""

import hashlib
import os
from pathlib import Path

import paramiko
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, lit

spark = SparkSession.builder.getOrCreate()

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
except Exception:
    print("Could not create schema 'bronze' in the current catalog")

RUN_ID = os.getenv("ATHENA_RUN_ID", "BRONZE_POC_RUN_001")
SFTP_HOST = os.environ["SFTP_HOST"]
SFTP_PORT = int(os.getenv("SFTP_PORT", "22"))
SFTP_USERNAME = os.environ["SFTP_USERNAME"]
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_PRIVATE_KEY_PATH = os.getenv("SFTP_PRIVATE_KEY_PATH")

REMOTE_FOLDER = "/daily/transactions/"
FILE_PATTERN = "TXN_YYYYMMDD.csv"
FILE_NAME = FILE_PATTERN.replace("YYYYMMDD", os.getenv("INGESTION_DATE", ""))
if not FILE_NAME or "YYYYMMDD" in FILE_NAME:
    from datetime import datetime
    FILE_NAME = FILE_PATTERN.replace("YYYYMMDD", datetime.utcnow().strftime("%Y%m%d"))

REMOTE_PATH = REMOTE_FOLDER.rstrip("/") + "/" + FILE_NAME
LOCAL_DIR = Path(os.getenv("SFTP_LOCAL_DIR", f"/dbfs/tmp/athena_sftp/{RUN_ID}"))
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_PATH = LOCAL_DIR / FILE_NAME
SPARK_CSV_PATH = str(LOCAL_PATH)
if SPARK_CSV_PATH.startswith("/dbfs/"):
    SPARK_CSV_PATH = "dbfs:/" + SPARK_CSV_PATH[len("/dbfs/"):]

TARGET_TABLE = "bronze.bronze_transaction"
MANDATORY_COLUMNS = ['transaction_id', 'transaction_date', 'amount']
EXPECTED_ROW_COUNT = None
EXPECTED_CHECKSUM = os.getenv("EXPECTED_CHECKSUM", None)
CHECKSUM_ALGORITHM = "sha256"

def _open_sftp():
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    if SFTP_PRIVATE_KEY_PATH:
        key = paramiko.RSAKey.from_private_key_file(SFTP_PRIVATE_KEY_PATH)
        transport.connect(username=SFTP_USERNAME, pkey=key)
    else:
        if not SFTP_PASSWORD:
            raise ValueError("Set SFTP_PASSWORD or SFTP_PRIVATE_KEY_PATH.")
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    return transport, paramiko.SFTPClient.from_transport(transport)

def _file_checksum(path):
    digest = hashlib.new(CHECKSUM_ALGORITHM)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

transport, sftp = _open_sftp()
try:
    sftp.get(REMOTE_PATH, str(LOCAL_PATH))
finally:
    sftp.close()
    transport.close()

actual_checksum = _file_checksum(LOCAL_PATH)
if EXPECTED_CHECKSUM and actual_checksum.lower() != EXPECTED_CHECKSUM.lower():
    raise ValueError(
        f"Checksum mismatch for {REMOTE_PATH}: expected={EXPECTED_CHECKSUM}, actual={actual_checksum}"
    )

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(SPARK_CSV_PATH)
)

normalized_columns = [column.strip().lower() for column in df.columns]
df = df.toDF(*normalized_columns)

missing_columns = [column.lower() for column in MANDATORY_COLUMNS if column.lower() not in df.columns]
if missing_columns:
    raise ValueError(f"Missing mandatory columns in {REMOTE_PATH}: {missing_columns}")

actual_row_count = df.count()
if EXPECTED_ROW_COUNT is not None and actual_row_count != EXPECTED_ROW_COUNT:
    raise ValueError(
        f"Row count mismatch for {REMOTE_PATH}: expected={EXPECTED_ROW_COUNT}, actual={actual_row_count}"
    )

df = (
    df
    .withColumn("run_id", lit(RUN_ID))
    .withColumn("ingestion_timestamp", current_timestamp())
    .withColumn("source_system", lit("Vendor SFTP"))
    .withColumn("source_folder", lit(REMOTE_FOLDER))
    .withColumn("source_file", input_file_name())
    .withColumn("source_checksum", lit(actual_checksum))
    .withColumn("source_row_count", lit(actual_row_count))
)

(
    df.write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(TARGET_TABLE)
)

print(f"SUCCESS: SFTP Bronze ingestion completed for {TARGET_TABLE} from {REMOTE_PATH}")
