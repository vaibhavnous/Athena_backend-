import os
import json
import hashlib
import pyodbc
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utilis.logger import logger

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

config = {
    "azure_sql": {
        # 🔹 COMMON SERVER CONFIG
        "host": os.getenv("AZURE_SQL_HOST", "dataedge.database.windows.net"),
        "port": int(os.getenv("AZURE_SQL_PORT", "1433")),
        "driver": os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server"),

        # 🔥 PIPELINE DB (YOUR SYSTEM DB)
        "pipeline_database": os.getenv("AZURE_SQL_PIPELINE_DATABASE", "AdventureWorks2019"),
        "pipeline_schema": os.getenv("AZURE_SQL_PIPELINE_SCHEMA", "metadata"),  # ✅ YOUR SCHEMA
        "schema_name": os.getenv("AZURE_SQL_PIPELINE_SCHEMA", "metadata"),  # backward-compat alias

        # 🔥 CLIENT DB (FAKE CLIENT)
        "source_database": os.getenv("AZURE_SQL_SOURCE_DATABASE", "insurance"),
        "source_schema": os.getenv("AZURE_SQL_SOURCE_SCHEMA", "dbo"),  # ✅ CLIENT SCHEMA

        # 🔹 AUTH
        "username": os.getenv("AZURE_SQL_USERNAME", "sqladmin"),
        "password": os.getenv("AZURE_SQL_PASSWORD", "Dataedge@213"),

        # 🔹 SOURCE DB HOST (FIXED)
        "source_host": os.getenv("AZURE_SQL_SOURCE_HOST", "dataedge.database.windows.net"),
        "source_username": os.getenv("AZURE_SQL_SOURCE_USERNAME", "sqladmin"),
        "source_password": os.getenv("AZURE_SQL_SOURCE_PASSWORD", "Dataedge@213"),
    }
}

FINGERPRINT_MAX_LEN = 64


# ─────────────────────────────────────────────────────────────
# SOURCE DB NORMALIZATION
# ─────────────────────────────────────────────────────────────

def _normalize_source_db(database_name: Optional[str]) -> str:
    db = database_name or config["azure_sql"]["source_database"]
    return db or "insurance"


# ─────────────────────────────────────────────────────────────
# CONNECTION BUILDER
# ─────────────────────────────────────────────────────────────

def _build_connection_string(host, port, database_name, username, password, driver):
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER=tcp:{host},{port};"
        f"DATABASE={database_name};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )


# ─────────────────────────────────────────────────────────────
# PIPELINE DB CONNECTION (WRITE)
# ─────────────────────────────────────────────────────────────

def get_pipeline_connection() -> pyodbc.Connection:
    db_conf = config["azure_sql"]

    conn_str = _build_connection_string(
        db_conf["host"],
        db_conf["port"],
        db_conf["pipeline_database"],
        db_conf["username"],
        db_conf["password"],
        db_conf["driver"],
    )

    return pyodbc.connect(conn_str)


# ─────────────────────────────────────────────────────────────
# CLIENT DB CONNECTION (READ ONLY)
# ─────────────────────────────────────────────────────────────

def get_client_connection(database_name: Optional[str] = None) -> pyodbc.Connection:
    db_conf = config["azure_sql"]

    db = _normalize_source_db(database_name)

    conn_str = _build_connection_string(
        db_conf["source_host"],
        db_conf["port"],
        db,
        db_conf["source_username"],
        db_conf["source_password"],
        db_conf["driver"],
    )

    return pyodbc.connect(conn_str)


# ─────────────────────────────────────────────────────────────
# CLIENT QUERY EXECUTOR
# ─────────────────────────────────────────────────────────────

def execute_source_sql(
    database_name: Optional[str],
    query: str,
    params: tuple = ()
) -> List[Any]:

    conn = None
    try:
        db = _normalize_source_db(database_name)

        conn = get_client_connection(db)
        cursor = conn.cursor()

        # Keep INFORMATION_SCHEMA scans within the configured source schema
        # without producing a second WHERE clause.
        query_to_run = query
        source_schema = config["azure_sql"].get("source_schema", "dbo")
        query_lower = query.lower()
        if "information_schema.tables" in query_lower:
            schema_filter = f" t.TABLE_SCHEMA = '{source_schema}'"
            if " where " in query_lower:
                query_to_run += f" AND{schema_filter}"
            else:
                query_to_run += f" WHERE{schema_filter}"

        cursor.execute(query_to_run, params)
        return cursor.fetchall()

    except Exception as exc:
        logger.warning(
            "Source DB query failed for %s: %s",
            database_name,
            exc,
        )
        return []

    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────
# PIPELINE DB WRITER (AI STORE)
# ─────────────────────────────────────────────────────────────

def ai_store_db_writer(
    run_id: str,
    stage: str,
    artifact_type: str,
    payload: Dict[str, Any],
    schema_version: str,
    prompt_version: str,
    faithfulness_status: str,
    faithfulness_warn_count: int = 0,
    retry_count: int = 0,
    token_count: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    fingerprint: Optional[str] = None,
) -> None:

    db_conf = config["azure_sql"]
    schema = db_conf["pipeline_schema"]  # 🔥 metadata schema

    conn = get_pipeline_connection()

    try:
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        # Extract fingerprint from payload if not provided explicitly
        _fingerprint = fingerprint or payload.get("fingerprint") or run_id
        cost_usd = payload.get("cost_usd")

        cursor.execute(
            f"""
            SELECT COUNT(1)
            FROM [{schema}].[ai_store]
            WHERE fingerprint = ?
            """,
            (_fingerprint,),
        )
        record_exists = cursor.fetchone()[0] > 0

        if record_exists:
            cursor.execute(
                f"""
                UPDATE [{schema}].[ai_store]
                SET
                    run_id = ?,
                    stage = ?,
                    artifact_type = ?,
                    payload = ?,
                    schema_version = ?,
                    prompt_version = ?,
                    faithfulness_status = ?,
                    faithfulness_warn_count = ?,
                    retry_count = ?,
                    token_count = ?,
                    input_tokens = ?,
                    output_tokens = ?,
                    cost_usd = ?,
                    stored_at = ?
                WHERE fingerprint = ?
                """,
                run_id,
                stage,
                artifact_type,
                json.dumps(payload),
                schema_version,
                prompt_version,
                faithfulness_status,
                faithfulness_warn_count,
                retry_count,
                token_count,
                input_tokens,
                output_tokens,
                cost_usd,
                now,
                _fingerprint,
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO [{schema}].[ai_store]
                (run_id, fingerprint, stage, artifact_type, payload, schema_version, prompt_version,
                 faithfulness_status, faithfulness_warn_count, retry_count, token_count, input_tokens,
                 output_tokens, cost_usd, stored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                run_id,
                _fingerprint,
                stage,
                artifact_type,
                json.dumps(payload),
                schema_version,
                prompt_version,
                faithfulness_status,
                faithfulness_warn_count,
                retry_count,
                token_count,
                input_tokens,
                output_tokens,
                cost_usd,
                now,
            )

        conn.commit()

    except Exception as e:
        logger.error(f"ai_store write failed: {e}")
        raise

    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# HITL QUEUE HELPERS
# ─────────────────────────────────────────────────────────────

def insert_hitl_queue_items(run_id: str, kpis: List[Dict[str, Any]], gate_number: int = 1) -> None:
    """Insert extracted KPIs into the HITL review queue."""
    db_conf = config["azure_sql"]
    schema = db_conf["pipeline_schema"]

    conn = get_pipeline_connection()
    try:
        cursor = conn.cursor()
        for idx, kpi in enumerate(kpis):
            item_id = f"{run_id}:{gate_number}:{idx}"
            cursor.execute(
                f"""
                INSERT INTO [{schema}].[hitl_review_queue]
                (item_id, run_id, gate_number, gate_status, original_content, queued_at)
                VALUES (?, ?, ?, ?, ?, GETUTCDATE())
                """,
                item_id,
                run_id,
                gate_number,
                "PENDING",
                json.dumps(kpi),
            )
        conn.commit()
        logger.info("Inserted %d HITL queue items for run_id=%s", len(kpis), run_id)
    except Exception as e:
        logger.error("HITL queue insert failed: %s", e)
        raise
    finally:
        conn.close()


def get_pending_items(run_id: str, gate: int) -> List[Dict[str, Any]]:
    """Return pending HITL items for a run and gate."""
    db_conf = config["azure_sql"]
    schema = db_conf["pipeline_schema"]

    conn = get_pipeline_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT item_id, original_content
            FROM [{schema}].[hitl_review_queue]
            WHERE run_id = ? AND gate_number = ? AND gate_status = 'PENDING'
            ORDER BY queued_at
            """,
            run_id,
            gate,
        )
        rows = cursor.fetchall()
        return [
            {
                "item_id": row.item_id,
                "kpi": json.loads(row.original_content) if row.original_content else {},
            }
            for row in rows
        ]
    finally:
        conn.close()


def update_hitl_item(
    item_id: str,
    status: str,
    edited_content: Optional[str] = None,
    *,
    rejection_reason: Optional[str] = None,
) -> None:
    """Update a HITL item status and store edits / rejection reason."""
    db_conf = config["azure_sql"]
    schema = db_conf["pipeline_schema"]

    conn = get_pipeline_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE [{schema}].[hitl_review_queue]
            SET gate_status = ?,
                edited_content = ?,
                rejection_reason = ?,
                decided_at = GETUTCDATE()
            WHERE item_id = ?
            """,
            status,
            edited_content,
            rejection_reason,
            item_id,
        )
        conn.commit()
    except Exception as e:
        logger.error("HITL item update failed for %s: %s", item_id, e)
        raise
    finally:
        conn.close()


def get_completed_items(run_id: str, gate: int) -> List[Dict[str, Any]]:
    """Return approved/completed HITL items for a run and gate."""
    db_conf = config["azure_sql"]
    schema = db_conf["pipeline_schema"]

    conn = get_pipeline_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT item_id, original_content, edited_content, gate_status, rejection_reason
            FROM [{schema}].[hitl_review_queue]
            WHERE run_id = ? AND gate_number = ? AND gate_status IN ('APPROVED', 'EDITED')
            ORDER BY decided_at
            """,
            run_id,
            gate,
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            kpi_raw = row.edited_content if row.edited_content else row.original_content
            result.append({
                "item_id": row.item_id,
                "kpi": json.loads(kpi_raw) if kpi_raw else {},
                "status": row.gate_status,
                "rejection_reason": row.rejection_reason,
            })
        return result
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def normalize_fingerprint(value: str) -> str:
    value = str(value)
    if len(value) <= FINGERPRINT_MAX_LEN:
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# Alias for backward compatibility — many modules import get_connection
get_connection = get_pipeline_connection
