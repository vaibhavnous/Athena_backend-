
"""
AUTO-GENERATED SILVER TRANSFORMATION SCRIPT

Source table: bronze.bronze_claim_information
Target table: silver.silver_claim_information
Expected runtime: Spark / Databricks with Delta support

POC rule: generated bronze scripts are treated as proof that bronze tables exist.
Runtime checks below still fail clearly if the Databricks table is missing.

DO NOT EDIT MANUALLY
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit, trim, when

spark = SparkSession.builder.getOrCreate()

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
except Exception:
    print("Could not create schema 'silver' in the current catalog")

RUN_ID = "811cb5c1-7a1f-40c2-8598-1dc30db84790"
SOURCE_TABLE = "bronze.bronze_claim_information"
TARGET_TABLE = "silver.silver_claim_information"
TEMP_VIEW = "silver_src_claim_information"

EXPECTED_COLUMNS = ['claimid', 'lossdate', 'inserteddate', 'claimtranstype', 'updatenum', 'policytransid', 'claimclosedate', 'claimsettlementmodename', 'claimlosscoveragename', 'courtid', 'courtname', 'caseid', 'claimcausetypename', 'claimnaturesoflossid', 'claimnaturesoflossname']
STRING_COLUMNS = ['claimtranstype', 'policytransid', 'claimsettlementmodename', 'claimlosscoveragename', 'courtname', 'caseid', 'claimcausetypename', 'claimnaturesoflossname']
PII_COLUMNS = ['claimnaturesoflossname']
KEY_COLUMNS = []
CAST_RULES = {'claimid': 'bigint', 'lossdate': 'timestamp', 'inserteddate': 'timestamp', 'updatenum': 'int', 'claimclosedate': 'date', 'courtid': 'bigint', 'claimnaturesoflossid': 'bigint'}
COLUMN_ALIASES = {}

if not spark.catalog.tableExists(SOURCE_TABLE):
    raise ValueError(f"Missing bronze source table: {SOURCE_TABLE}")

df = spark.table(SOURCE_TABLE)

if df.limit(1).count() == 0:
    raise ValueError(f"Bronze source table has no rows: {SOURCE_TABLE}")

available_columns = set(df.columns)
for old_name, new_name in COLUMN_ALIASES.items():
    if old_name in available_columns and new_name not in available_columns:
        df = df.withColumnRenamed(old_name, new_name)

available_columns = set(df.columns)
metadata_columns = [
    name for name in ["run_id", "ingestion_timestamp", "source_system", "source_table"]
    if name in available_columns
]

def compact_name(name):
    return str(name).lower().replace("_", "")

available_by_compact = {
    compact_name(name): name
    for name in df.columns
}

if EXPECTED_COLUMNS:
    select_expressions = []
    missing_columns = []
    for expected_name in EXPECTED_COLUMNS:
        actual_name = available_by_compact.get(compact_name(expected_name))
        if actual_name:
            select_expressions.append(col(actual_name).alias(expected_name))
        else:
            missing_columns.append(expected_name)
else:
    select_expressions = [
        col(name)
        for name in df.columns
        if name not in metadata_columns
    ]
    missing_columns = []

if not select_expressions:
    raise ValueError(
        f"No expected business columns found in {SOURCE_TABLE}. "
        f"Available columns: {df.columns}"
    )

metadata_expressions = [col(name) for name in metadata_columns]
df = df.select(*select_expressions, *metadata_expressions)

if missing_columns:
    print(f"WARNING: Missing expected columns in {SOURCE_TABLE}: {missing_columns}")

for column_name in STRING_COLUMNS:
    if column_name in df.columns:
        df = df.withColumn(
            column_name,
            when(trim(col(column_name)) == "", None).otherwise(trim(col(column_name)))
        )

for column_name, target_type in CAST_RULES.items():
    if column_name in df.columns:
        df = df.withColumn(column_name, col(column_name).cast(target_type))

for column_name in PII_COLUMNS:
    if column_name in df.columns:
        df = df.withColumn(column_name, col(column_name).cast("string"))

dedup_keys = [column_name for column_name in KEY_COLUMNS if column_name in df.columns]
if dedup_keys:
    df = df.dropDuplicates(dedup_keys)
else:
    df = df.dropDuplicates()

df = (
    df
    .withColumn("silver_run_id", lit(RUN_ID))
    .withColumn("silver_processed_timestamp", current_timestamp())
)

df.createOrReplaceTempView(TEMP_VIEW)

create_table_sql = (
    f"CREATE TABLE IF NOT EXISTS {TARGET_TABLE} "
    f"USING DELTA "
    f"AS SELECT * FROM {TEMP_VIEW} WHERE 1 = 0"
)
spark.sql(create_table_sql)

(
    df.write
    .format("delta")
    .mode("append")
    .saveAsTable(TARGET_TABLE)
)

print(f"SUCCESS: Silver transformation completed for {TARGET_TABLE}")
