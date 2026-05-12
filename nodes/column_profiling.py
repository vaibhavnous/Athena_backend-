"""
Deterministic column profiling node for LangGraph.

Implements NB08-style column profiling using:
- PASS 1: Full-table SQL pushdown statistics
- PASS 2: Sampled distribution statistics (MEASURE only)

Caching intentionally NOT implemented.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Literal, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from state import Stage01State
from utilis.db import ai_store_db_writer, get_client_connection, save_checkpoint_state, timed_stage
from utilis.logger import logger


# --------------------------------------------------------------------------------------
# Constants & Types
# --------------------------------------------------------------------------------------

ProfileTier = Literal[
    "ID",
    "AUDIT",
    "FLAG",
    "DATE",
    "MEASURE",
    "DIMENSION",
    "DEFAULT",
    "HIGH_CARD_TEXT",
]

NUMERIC_TYPES = {
    "bigint",
    "decimal",
    "float",
    "int",
    "money",
    "numeric",
    "real",
    "smallint",
    "smallmoney",
    "tinyint",
}

DATE_TYPES = {
    "date",
    "datetime",
    "datetime2",
    "datetimeoffset",
    "smalldatetime",
    "time",
}

TEXT_TYPES = {
    "char",
    "nchar",
    "nvarchar",
    "text",
    "varchar",
    "ntext",
    "uniqueidentifier",
}

LOB_TYPES = {
    "binary",
    "geography",
    "geometry",
    "hierarchyid",
    "image",
    "sql_variant",
    "varbinary",
    "xml",
}


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class ColumnProfileResult(BaseModel):
    run_id: str
    database_name: str
    schema_name: str
    table_name: str
    column_name: str
    data_type: Optional[str]
    profile_tier: ProfileTier

    total_rows: Optional[int] = None
    non_null_count: Optional[int] = None
    null_rate: Optional[float] = None
    cardinality: Optional[int] = None
    col_min: Optional[str] = None
    col_max: Optional[str] = None

    # PASS 2
    p25: Optional[float] = None
    p75: Optional[float] = None

    # Samples
    top_samples: Optional[List[Dict[str, Any]]] = None

    profiling_status: Literal["SUCCESS", "FAILED"] = "SUCCESS"
    error_message: Optional[str] = None
    profiled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TableProfileResult(BaseModel):
    database_name: str
    schema_name: str
    table_name: str
    columns_profiled: int
    columns_failed: int
    status: Literal["SUCCESS", "PARTIAL", "FAILED", "SKIPPED"]
    duration_seconds: float
    error_message: Optional[str] = None


class ProfilingTable(BaseModel):
    database_name: str
    schema_name: str
    table_name: str
    columns: List[Dict[str, Any]]


# --------------------------------------------------------------------------------------
# Helpers / Config
# --------------------------------------------------------------------------------------

def _copy_state(state: Stage01State) -> Stage01State:
    return state.copy()


def _profiling_max_workers() -> int:
    return max(1, int(os.environ.get("COLUMN_PROFILING_MAX_WORKERS", "4")))


def _profiling_sample_pct() -> float:
    pct = float(os.environ.get("COLUMN_PROFILING_SAMPLE_PCT", "10"))
    return min(max(pct, 0.1), 100.0)


def _high_cardinality_threshold() -> int:
    return max(1, int(os.environ.get("COLUMN_PROFILING_HIGH_CARDINALITY_THRESHOLD", "100")))


def _top_sample_limit() -> int:
    return min(max(1, int(os.environ.get("COLUMN_PROFILING_TOP_SAMPLE_LIMIT", "10"))), 100)


def _quote_identifier(identifier: str) -> str:
    return f"[{identifier.replace(']', ']]')}]"


def _qualified_table(schema: str, table: str) -> str:
    return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"


def _tablesample_clause() -> str:
    return f"TABLESAMPLE({_profiling_sample_pct()} PERCENT)"


def _execute_query(database: str, query: str):
    conn = get_client_connection(database)
    try:
        cur = conn.cursor()
        cur.execute(query)
        return cur.fetchall()
    finally:
        conn.close()


def _row_value(row, name: str):
    if hasattr(row, name):
        return getattr(row, name)
    try:
        return row[name]
    except Exception:
        return None


def _supports_cardinality(data_type: Optional[str]) -> bool:
    return str(data_type or "").lower() not in LOB_TYPES


# --------------------------------------------------------------------------------------
# Tier Classification
# --------------------------------------------------------------------------------------

def classify_profile_tier(column: Dict[str, Any]) -> ProfileTier:
    name = str(column.get("column_name", "")).lower()
    data_type = str(column.get("data_type", "")).lower()
    semantic = str(column.get("semantic_type", "")).upper()

    if semantic in {"ID", "SURROGATE_KEY"} or name == "id" or name.endswith("_id"):
        return "ID"
    if semantic == "AUDIT_TIMESTAMP":
        return "AUDIT"
    if data_type == "bit" or name.startswith(("is_", "has_")):
        return "FLAG"
    if data_type in DATE_TYPES:
        return "DATE"
    if semantic == "MEASURE" or data_type in NUMERIC_TYPES:
        return "MEASURE"
    if semantic == "DIMENSION" or data_type in TEXT_TYPES:
        return "DIMENSION"
    return "DEFAULT"


# --------------------------------------------------------------------------------------
# PASS 1 — Pushdown Profiling
# --------------------------------------------------------------------------------------

def pass1_pushdown_profile(
    database: str,
    schema: str,
    table: str,
    column: str,
    data_type: Optional[str],
    tier: ProfileTier,
) -> Dict[str, Any]:

    col_sql = _quote_identifier(column)
    table_sql = _qualified_table(schema, table)

    exprs = [
        "COUNT_BIG(*) AS total_rows",
        f"COUNT_BIG({col_sql}) AS non_null_count",
    ]

    if tier != "AUDIT" and _supports_cardinality(data_type):
        exprs.append(f"COUNT_BIG(DISTINCT {col_sql}) AS cardinality")

    if tier in {"MEASURE", "DATE"}:
        exprs.append(f"MIN({col_sql}) AS col_min")
        exprs.append(f"MAX({col_sql}) AS col_max")

    query = f"SELECT {', '.join(exprs)} FROM {table_sql}"
    row = _execute_query(database, query)[0]

    total = int(row.total_rows or 0)
    non_null = int(row.non_null_count or 0)
    null_rate = round(1.0 - (non_null / total), 6) if total > 0 else 0.0

    result = {
        "total_rows": total,
        "non_null_count": non_null,
        "null_rate": null_rate,
    }

    if hasattr(row, "cardinality"):
        result["cardinality"] = int(row.cardinality) if row.cardinality is not None else None
    if tier in {"MEASURE", "DATE"}:
        result["col_min"] = str(row.col_min) if row.col_min is not None else None
        result["col_max"] = str(row.col_max) if row.col_max is not None else None

    return result


def pass1_table_pushdown_profile(table_ref: ProfilingTable) -> Dict[str, Dict[str, Any]]:
    """Run pass-1 aggregate metrics for all columns in one table scan."""
    table_sql = _qualified_table(table_ref.schema_name, table_ref.table_name)
    exprs = ["COUNT_BIG(*) AS [total_rows]"]
    column_meta: List[tuple[int, Dict[str, Any], ProfileTier]] = []

    for idx, column in enumerate(table_ref.columns):
        name = str(column["column_name"])
        data_type = column.get("data_type")
        tier = classify_profile_tier(column)
        col_sql = _quote_identifier(name)
        column_meta.append((idx, column, tier))
        exprs.append(f"COUNT_BIG({col_sql}) AS [c{idx}_non_null]")

        if tier != "AUDIT" and _supports_cardinality(data_type):
            exprs.append(f"COUNT_BIG(DISTINCT {col_sql}) AS [c{idx}_cardinality]")

        if tier in {"MEASURE", "DATE"}:
            exprs.append(f"MIN({col_sql}) AS [c{idx}_min]")
            exprs.append(f"MAX({col_sql}) AS [c{idx}_max]")

    query = f"SELECT {', '.join(exprs)} FROM {table_sql}"
    row = _execute_query(table_ref.database_name, query)[0]
    total = int(_row_value(row, "total_rows") or 0)

    results: Dict[str, Dict[str, Any]] = {}
    for idx, column, tier in column_meta:
        name = str(column["column_name"])
        non_null = int(_row_value(row, f"c{idx}_non_null") or 0)
        result: Dict[str, Any] = {
            "total_rows": total,
            "non_null_count": non_null,
            "null_rate": round(1.0 - (non_null / total), 6) if total > 0 else 0.0,
        }

        cardinality = _row_value(row, f"c{idx}_cardinality")
        if cardinality is not None:
            result["cardinality"] = int(cardinality)

        if tier in {"MEASURE", "DATE"}:
            col_min = _row_value(row, f"c{idx}_min")
            col_max = _row_value(row, f"c{idx}_max")
            result["col_min"] = str(col_min) if col_min is not None else None
            result["col_max"] = str(col_max) if col_max is not None else None

        results[name] = result

    return results


# --------------------------------------------------------------------------------------
# PASS 2 — Sampling (MEASURE only)
# --------------------------------------------------------------------------------------

def pass2_measure_sampling(database: str, schema: str, table: str, column: str) -> Dict[str, Optional[float]]:
    col_sql = _quote_identifier(column)
    table_sql = _qualified_table(schema, table)

    query = f"""
        SELECT DISTINCT
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY TRY_CONVERT(float, {col_sql})) OVER () AS p25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY TRY_CONVERT(float, {col_sql})) OVER () AS p75
        FROM {table_sql} {_tablesample_clause()}
        WHERE TRY_CONVERT(float, {col_sql}) IS NOT NULL
    """

    rows = _execute_query(database, query)
    if not rows:
        return {"p25": None, "p75": None}

    r = rows[0]
    return {
        "p25": float(r.p25) if r.p25 is not None else None,
        "p75": float(r.p75) if r.p75 is not None else None,
    }


# --------------------------------------------------------------------------------------
# Column Orchestration (PASS 1 → PASS 2)
# --------------------------------------------------------------------------------------

def profile_column(
    table_ref: ProfilingTable,
    column: Dict[str, Any],
    run_id: str,
    pass1_result: Optional[Dict[str, Any]] = None,
) -> ColumnProfileResult:
    name = str(column["column_name"])
    data_type = column.get("data_type")
    tier = classify_profile_tier(column)

    base = dict(
        run_id=run_id,
        database_name=table_ref.database_name,
        schema_name=table_ref.schema_name,
        table_name=table_ref.table_name,
        column_name=name,
        data_type=data_type,
        profile_tier=tier,
    )

    try:
        if pass1_result is None:
            pass1_result = pass1_pushdown_profile(
                table_ref.database_name,
                table_ref.schema_name,
                table_ref.table_name,
                name,
                data_type,
                tier,
            )
        result = {**base, **pass1_result}

        if tier == "MEASURE":
            result.update(
                pass2_measure_sampling(
                    table_ref.database_name,
                    table_ref.schema_name,
                    table_ref.table_name,
                    name,
                )
            )

        return ColumnProfileResult(**result)

    except Exception as exc:
        return ColumnProfileResult(
            **base,
            profiling_status="FAILED",
            error_message=str(exc),
        )


# --------------------------------------------------------------------------------------
# Table + State Glue
# --------------------------------------------------------------------------------------

def _resolve_tables_for_profiling(state: Stage01State) -> List[ProfilingTable]:
    discovered = state.get("discovered_metadata") or {}
    raw_tables = discovered.get("tables", []) if isinstance(discovered, dict) else []
    resolved: List[ProfilingTable] = []

    for item in raw_tables:
        if not isinstance(item, dict):
            continue
        if item.get("table_status") != "COMPLETED":
            continue

        database = str(item.get("database_name", "")).strip()
        schema = str(item.get("schema_name", "dbo")).strip()
        table = str(item.get("table_name", "")).strip()
        columns = item.get("columns") or []

        if not database or not table or not columns:
            continue

        resolved.append(
            ProfilingTable(
                database_name=database,
                schema_name=schema,
                table_name=table,
                columns=[col for col in columns if isinstance(col, dict)],
            )
        )

    return resolved


def profile_table(table_ref: ProfilingTable, run_id: str) -> tuple[TableProfileResult, List[ColumnProfileResult]]:
    started = datetime.now(timezone.utc)

    if not table_ref.columns:
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        return (
            TableProfileResult(
                database_name=table_ref.database_name,
                schema_name=table_ref.schema_name,
                table_name=table_ref.table_name,
                columns_profiled=0,
                columns_failed=0,
                status="SKIPPED",
                duration_seconds=duration,
                error_message="No columns available for profiling",
            ),
            [],
        )

    try:
        pass1_results = pass1_table_pushdown_profile(table_ref)
    except Exception as exc:
        logger.warning(
            "Table-level profiling failed for %s.%s.%s, falling back to per-column scans: %s",
            table_ref.database_name,
            table_ref.schema_name,
            table_ref.table_name,
            exc,
            extra={"run_id": run_id, "node": "column_profiling"},
        )
        pass1_results = {}

    profiles = [
        profile_column(table_ref, column, run_id, pass1_results.get(str(column["column_name"])))
        for column in table_ref.columns
    ]
    failed = sum(1 for profile in profiles if profile.profiling_status == "FAILED")
    success = len(profiles) - failed

    if success == 0:
        status: Literal["SUCCESS", "PARTIAL", "FAILED", "SKIPPED"] = "FAILED"
    elif failed > 0:
        status = "PARTIAL"
    else:
        status = "SUCCESS"

    duration = (datetime.now(timezone.utc) - started).total_seconds()
    return (
        TableProfileResult(
            database_name=table_ref.database_name,
            schema_name=table_ref.schema_name,
            table_name=table_ref.table_name,
            columns_profiled=success,
            columns_failed=failed,
            status=status,
            duration_seconds=duration,
        ),
        profiles,
    )


def _persist_column_profiles(
    *,
    run_id: str,
    fingerprint: str,
    tables: List[TableProfileResult],
    profiles: List[ColumnProfileResult],
) -> Dict[str, Any]:
    payload = {
        "fingerprint": fingerprint,
        "storage_fingerprint": f"{fingerprint}:COLUMN_PROFILES",
        "run_id": run_id,
        "table_count": len(tables),
        "tables_success": sum(1 for table in tables if table.status == "SUCCESS"),
        "tables_partial": sum(1 for table in tables if table.status == "PARTIAL"),
        "tables_failed": sum(1 for table in tables if table.status == "FAILED"),
        "columns_profiled": sum(table.columns_profiled for table in tables),
        "columns_failed": sum(table.columns_failed for table in tables),
        "profiling_strategy": "sql_pushdown_column_profile_v1",
        "sample_pct": _profiling_sample_pct(),
        "high_cardinality_threshold": _high_cardinality_threshold(),
        "table_results": [table.model_dump(mode="json") for table in tables],
        "column_profiles": [profile.model_dump(mode="json") for profile in profiles],
    }

    ai_store_db_writer(
        run_id=run_id,
        stage="Column Profiling",
        artifact_type="COLUMN_PROFILES",
        payload=payload,
        schema_version="ColumnProfileSummary_v1",
        prompt_version="DETERMINISTIC_SQL_PROFILING_v1",
        faithfulness_status="NOT_APPLICABLE",
        token_count=0,
        input_tokens=0,
        output_tokens=0,
        fingerprint=fingerprint,
    )
    return payload


def column_profiling_node(state: Stage01State) -> Stage01State:
    new_state = _copy_state(state)
    log_context = {"run_id": new_state.get("run_id", "unknown"), "node": "column_profiling"}
    logger.info("START column_profiling_node", extra=log_context)

    if new_state.get("status") == "FAILED":
        logger.warning("Skipping column profiling because pipeline status is FAILED", extra=log_context)
        return new_state

    table_refs = _resolve_tables_for_profiling(new_state)
    if not table_refs:
        logger.info("Skipping column profiling because no discovered metadata is available", extra=log_context)
        new_state.update(
            {
                "column_profiling_status": "SKIPPED",
                "column_profiling_error": "No discovered metadata available for profiling",
            }
        )
        return new_state

    run_id = str(new_state.get("run_id") or "unknown")
    fingerprint = str(new_state.get("fingerprint") or run_id)
    max_workers = min(_profiling_max_workers(), max(len(table_refs), 1))

    table_results: List[TableProfileResult] = []
    column_profiles: List[ColumnProfileResult] = []

    try:
        with timed_stage("column_profiling_tables", **log_context), ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(profile_table, table_ref, run_id): table_ref for table_ref in table_refs}
            for future in as_completed(futures):
                table_result, profiles = future.result()
                table_results.append(table_result)
                column_profiles.extend(profiles)
                partial_state = {
                    **new_state,
                    "column_profiles": {
                        "table_results": [table.model_dump(mode="json") for table in table_results],
                        "column_profiles": [profile.model_dump(mode="json") for profile in column_profiles],
                    },
                    "column_profiling_status": "IN_PROGRESS",
                }
                try:
                    save_checkpoint_state(run_id, partial_state)
                except Exception:
                    logger.warning("Partial profiling checkpoint failed", extra=log_context)

        payload = _persist_column_profiles(
            run_id=run_id,
            fingerprint=fingerprint,
            tables=table_results,
            profiles=column_profiles,
        )
    except Exception as exc:
        logger.error("Column profiling failed: %s", exc, extra=log_context)
        new_state.update(
            {
                "column_profiling_status": "FAILED",
                "column_profiling_error": str(exc),
            }
        )
        return new_state

    failed_tables = sum(1 for table in table_results if table.status == "FAILED")
    partial_tables = sum(1 for table in table_results if table.status == "PARTIAL")
    status = "COMPLETED_WITH_WARNINGS" if (failed_tables or partial_tables) else "COMPLETED"

    new_state.update(
        {
            "column_profiles": payload,
            "column_profiling_status": status,
            "column_profiling_error": None,
        }
    )

    logger.info(
        "END column_profiling_node: tables=%d profiles=%d failed_tables=%d",
        len(table_results),
        len(column_profiles),
        failed_tables,
        extra=log_context,
    )
    return new_state


def build_column_profiling_graph() -> StateGraph:
    graph = StateGraph(Stage01State)
    graph.add_node("column_profiling", column_profiling_node)
    graph.set_entry_point("column_profiling")
    graph.set_finish_point("column_profiling")
    return graph


def compile_column_profiling_graph():
    return build_column_profiling_graph().compile(checkpointer=MemorySaver())
