from utilis.db import get_connection, config
import pyodbc


TARGET_COLUMNS = {
    "brd_run_registry": {
        "registry_id": "NVARCHAR(100) NULL",
        "run_id": "NVARCHAR(100) NULL",
        "brd_fingerprint": "NVARCHAR(64) NULL",
        "brd_filename": "NVARCHAR(255) NULL",
        "brd_token_estimate": "INT NULL",
        "pipeline_status": "NVARCHAR(50) NULL",
        "certified_kpi_count": "INT NULL DEFAULT 0",
        "created_at": "DATETIME2 NULL",
        "completed_at": "DATETIME2 NULL",
    },
    "kpi_memory": {
        "memory_id": "NVARCHAR(100) NULL",
        "brd_fingerprint": "NVARCHAR(64) NULL",
        "kpi_name": "NVARCHAR(200) NULL",
        "gate1_decision": "NVARCHAR(20) NULL",
        "domain": "NVARCHAR(100) NULL",
        "stored_at": "DATETIME2 NULL",
    },
    "ai_store": {
        "artifact_id": "NVARCHAR(100) NULL",
        "run_id": "NVARCHAR(100) NULL",
        "fingerprint": "NVARCHAR(64) NULL",
        "stage": "NVARCHAR(50) NULL",
        "artifact_type": "NVARCHAR(100) NULL",
        "payload": "NVARCHAR(MAX) NULL",
        "schema_version": "NVARCHAR(50) NULL",
        "prompt_version": "NVARCHAR(50) NULL",
        "faithfulness_status": "NVARCHAR(20) NULL",
        "faithfulness_warn_count": "INT NULL",
        "retry_count": "INT NULL",
        "token_count": "INT NULL",
        "input_tokens": "INT NULL",
        "output_tokens": "INT NULL",
        "cost_usd": "FLOAT NULL",
        "created_at": "DATETIME2 NULL",
        "run_id_partition": "NVARCHAR(100) NULL",
    },
    "tables_metadata": {
        "table_name": "NVARCHAR(255) NULL",
        "schema_name": "NVARCHAR(255) NULL",
        "table_type": "NVARCHAR(50) NULL",
    },
    "columns_metadata": {
        "table_name": "NVARCHAR(255) NULL",
        "column_name": "NVARCHAR(255) NULL",
        "data_type": "NVARCHAR(100) NULL",
        "is_nullable": "NVARCHAR(10) NULL",
        "ordinal_position": "INT NULL",
    },
    "foreign_keys": {
        "table_name": "NVARCHAR(255) NULL",
        "column_name": "NVARCHAR(255) NULL",
        "ref_table": "NVARCHAR(255) NULL",
        "ref_column": "NVARCHAR(255) NULL",
    },
}


def table_exists(cursor, schema_name: str, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = ?
          AND TABLE_NAME = ?
        """,
        (schema_name, table_name),
    )
    return cursor.fetchone() is not None


def get_existing_columns(cursor, schema_name: str, table_name: str) -> set[str]:
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ?
          AND TABLE_NAME = ?
        """,
        (schema_name, table_name),
    )
    return {row.COLUMN_NAME.lower() for row in cursor.fetchall()}


def ensure_metadata_tables(schema_name: str) -> None:
    """
    Create Athena metadata tables if they do not exist.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        table_ddls = {
            "tables_metadata": f"""
                CREATE TABLE [{schema_name}].[tables_metadata] (
                    table_name NVARCHAR(255) NOT NULL,
                    schema_name NVARCHAR(255) NOT NULL,
                    table_type NVARCHAR(50) NOT NULL,
                    created_at DATETIME2 DEFAULT GETUTCDATE()
                )
            """,
            "columns_metadata": f"""
                CREATE TABLE [{schema_name}].[columns_metadata] (
                    table_name NVARCHAR(255) NOT NULL,
                    column_name NVARCHAR(255) NOT NULL,
                    data_type NVARCHAR(100) NOT NULL,
                    is_nullable NVARCHAR(10) NOT NULL,
                    ordinal_position INT NOT NULL,
                    created_at DATETIME2 DEFAULT GETUTCDATE()
                )
            """,
            "foreign_keys": f"""
                CREATE TABLE [{schema_name}].[foreign_keys] (
                    table_name NVARCHAR(255) NOT NULL,
                    column_name NVARCHAR(255) NOT NULL,
                    ref_table NVARCHAR(255) NOT NULL,
                    ref_column NVARCHAR(255) NOT NULL,
                    created_at DATETIME2 DEFAULT GETUTCDATE()
                )
            """,
        }

        for table_name, ddl in table_ddls.items():
            if table_exists(cursor, schema_name, table_name):
                print(f"[{schema_name}].[{table_name}] exists")
            else:
                print(f"Creating [{schema_name}].[{table_name}]")
                cursor.execute(ddl)

        conn.commit()
        print("✅ Metadata tables ensured")

    finally:
        conn.close()


def add_missing_columns() -> None:
    db_conf = config["azure_sql"]
    schema_name = db_conf.get("pipeline_schema", "metadata")

    host = db_conf.get("host", "unknown-host")
    database = db_conf.get("pipeline_database", "unknown-db")

    print(f"Connecting to {database} on {host} (schema={schema_name})")

    ensure_metadata_tables(schema_name)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        for table_name, expected_columns in TARGET_COLUMNS.items():
            print(f"\nChecking [{schema_name}].[{table_name}]")

            if not table_exists(cursor, schema_name, table_name):
                print("  ⚠️  Table does not exist — skipping")
                continue

            existing = get_existing_columns(cursor, schema_name, table_name)

            for column_name, column_type in expected_columns.items():
                if column_name.lower() not in existing:
                    print(f"  ➕ Adding {column_name}")
                    cursor.execute(
                        f"ALTER TABLE [{schema_name}].[{table_name}] "
                        f"ADD [{column_name}] {column_type}"
                    )
                else:
                    print(f"  ✅ Exists {column_name}")

        conn.commit()
        print("\n✅ All columns verified successfully")

    finally:
        conn.close()


if __name__ == "__main__":
    add_missing_columns()