
"""
AUTO-GENERATED BRONZE INGESTION SCRIPT

Source: insurance.dbo.claim_payment_indemnity
Expected runtime: Spark / Databricks with Delta support
Target table: main.bronze.bronze_claim_payment_indemnity

DO NOT EDIT MANUALLY
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit

spark = SparkSession.builder.getOrCreate()

# ------------------------------------------------------------------------------
# Databricks catalog/schema setup
# ------------------------------------------------------------------------------

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
except Exception:
    print("Could not create schema 'bronze' in the current catalog")

RUN_ID = "BRONZE_POC_RUN_001"
SOURCE_JDBC_URL = "jdbc:sqlserver://dataedge.database.windows.net:1433;databaseName=insurance;encrypt=true;trustServerCertificate=false;user=sqladmin;password=Dataedge@213"

TARGET_TABLE = "bronze.bronze_claim_payment_indemnity"
TEMP_VIEW = "bronze_src_claim_payment_indemnity"
CAST_RULES = {'paymentid': 'bigint', 'updatenum': 'int', 'paiddate': 'timestamp', 'paidamount': 'decimal(18,6)', 'servicetax': 'decimal(18,6)', 'claimid': 'bigint', 'payeeid': 'bigint', 'serviceproviderid': 'bigint', 'garageid': 'bigint', 'garagetypeid': 'bigint', 'hospitalid': 'bigint'}
DATE_COLUMN_HINTS = ("date", "_dt", "timestamp", "created_at", "updated_at", "modified_at")
RECREATE_TARGET_ON_SCHEMA_CONFLICT = True

df = (
    spark.read.format("jdbc")
    .option("url", SOURCE_JDBC_URL)
    .option("dbtable", "dbo.claim_payment_indemnity")
    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")
    .load()
)

if not df.schema or not df.schema.fields:
    raise ValueError("Source read returned an empty schema for insurance.dbo.claim_payment_indemnity.")

normalized_columns = []
seen_columns = {}
for original_name in df.columns:
    normalized_name = original_name.lower()
    if normalized_name in seen_columns:
        seen_columns[normalized_name] += 1
        normalized_name = f"{normalized_name}_{seen_columns[normalized_name]}"
    else:
        seen_columns[normalized_name] = 0
    normalized_columns.append(col(original_name).alias(normalized_name))

df = df.select(*normalized_columns)

for column_name, target_type in CAST_RULES.items():
    if column_name in df.columns:
        df = df.withColumn(column_name, col(column_name).cast(target_type))

for column_name in df.columns:
    lower_name = column_name.lower()
    if column_name in CAST_RULES:
        continue
    if any(hint in lower_name for hint in DATE_COLUMN_HINTS):
        df = df.withColumn(column_name, col(column_name).cast("timestamp"))

df = (
    df
    .withColumn("run_id", lit(RUN_ID))
    .withColumn("ingestion_timestamp", current_timestamp())
    .withColumn("source_system", lit("insurance"))
    .withColumn("source_table", lit("claim_payment_indemnity"))
)

df.createOrReplaceTempView(TEMP_VIEW)

if spark.catalog.tableExists(TARGET_TABLE):
    target_schema = {
        field.name.lower(): field.dataType.simpleString().lower()
        for field in spark.table(TARGET_TABLE).schema.fields
    }
    incoming_schema = {
        field.name.lower(): field.dataType.simpleString().lower()
        for field in df.schema.fields
    }
    schema_conflicts = [
        (name, target_schema[name], incoming_type)
        for name, incoming_type in incoming_schema.items()
        if name in target_schema and target_schema[name] != incoming_type
    ]

    if schema_conflicts:
        conflict_text = ", ".join(
            f"{name}: target={target_type}, incoming={incoming_type}"
            for name, target_type, incoming_type in schema_conflicts
        )
        if RECREATE_TARGET_ON_SCHEMA_CONFLICT:
            print(f"Recreating {TARGET_TABLE} due to schema conflicts: {conflict_text}")
            spark.sql(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
        else:
            raise ValueError(f"Schema conflicts detected for {TARGET_TABLE}: {conflict_text}")

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
    .option("mergeSchema", "true")
    .saveAsTable(TARGET_TABLE)
)

print(f"SUCCESS: Bronze ingestion completed for {TARGET_TABLE}")
