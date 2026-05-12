
"""
AUTO-GENERATED SILVER TRANSFORMATION SCRIPT

Source table: bronze.bronze_policy_transactions
Target table: silver.silver_policy_transactions
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

RUN_ID = "87675dc9-16bd-4fb5-b0a4-a5ec5196ef28"
SOURCE_TABLE = "bronze.bronze_policy_transactions"
TARGET_TABLE = "silver.silver_policy_transactions"
TEMP_VIEW = "silver_src_policy_transactions"

EXPECTED_COLUMNS = ['reference_id', 'policy_id', 'begin_date', 'end_date', 'policy_issued_date', 'policy_transaction_id', 'policy_transaction_type', 'branch_office_id', 'branch_office_name', 'client_full_name', 'product_id', 'product_name', 'product_group_name', 'segment_name', 'business_division_name', 'agent_id', 'agent_name', 'agent_category_id', 'agen_t_category_name', 'agent_sub_category_name', 'channel_id', 'channel_name', 'channel_group_id', 'channel_group_name', 'segment_division_name', 'mode_of_issuance_id', 'mode_of_issuance_name', 'is_aadhaar_attached', 'is_pan_attached', 'premium', 'commission', 'expenses', 'service_tax', 'stamp_duty', 'risk_sum_insured', 'no_of_employees', 'no_o_lives']
STRING_COLUMNS = ['policy_id', 'policy_transaction_id', 'policy_transaction_type', 'branch_office_id', 'client_full_name', 'product_name', 'product_group_name', 'segment_name', 'business_division_name', 'agent_name', 'agent_category_id', 'agen_t_category_name', 'agent_sub_category_name', 'channel_name', 'channel_group_name', 'segment_division_name', 'mode_of_issuance_id', 'mode_of_issuance_name', 'is_aadhaar_attached', 'is_pan_attached', 'premium', 'commission', 'expenses', 'service_tax', 'stamp_duty', 'risk_sum_insured', 'no_of_employees', 'no_o_lives']
PII_COLUMNS = ['is_aadhaar_attached', 'is_pan_attached']
KEY_COLUMNS = ['reference_id', 'policy_id', 'policy_transaction_id', 'branch_office_id', 'product_id', 'agent_id', 'agent_category_id', 'channel_id', 'channel_group_id', 'mode_of_issuance_id']
CAST_RULES = {'reference_id': 'int', 'begin_date': 'timestamp', 'end_date': 'timestamp', 'policy_issued_date': 'timestamp', 'branch_office_name': 'int', 'product_id': 'int', 'agent_id': 'double', 'channel_id': 'int', 'channel_group_id': 'int'}
COLUMN_ALIASES = {'rererence_id': 'reference_id'}

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
