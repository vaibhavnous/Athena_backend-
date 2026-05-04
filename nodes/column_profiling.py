"""
Deterministic column profiling node for LangGraph.

This node profiles the columns discovered by metadata_discovery using source
database SQL pushdown. It is adapted from the Databricks NB08 notebook into the
repo's pyodbc/Azure SQL runtime style.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Literal, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from state import Stage01State
from utilis.db import ai_store_db_writer, get_client_connection
from utilis.logger import logger


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


class ColumnProfileResult(BaseModel):
    run_id: str
    database_name: str
    schema_name: str
    table_name: str
    column_name: str
    data_type: Optional[str] = None
    profile_tier: ProfileTier
    total_rows: Optional[int] = None
    non_null_count: Optional[int] = None
    null_rate: Optional[float] = None
    cardinality: Optional[int] = None
    col_min: Optional[str] = None
    col_max: Optional[str] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
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
    columns: List[Dict[str, Any]] = Field(default_factory=list)


def _copy_state(state: Stage01State) -> Stage01State:
    return state.copy()


def _profiling_max_workers() -> int:
    raw = os.environ.get("COLUMN_PROFILING_MAX_WORKERS", "4")
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _profiling_sample_pct() -> float:
    raw = os.environ.get("COLUMN_PROFILING_SAMPLE_PCT", "10")
    try:
        pct = float(raw)
    except ValueError:
        return 10.0
    return min(max(pct, 0.1), 100.0)


def _high_cardinality_threshold() -> int:
    raw = os.environ.get("COLUMN_PROFILING_HIGH_CARDINALITY_THRESHOLD", "100")
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


def _top_sample_limit() -> int:
    raw = os.environ.get("COLUMN_PROFILING_TOP_SAMPLE_LIMIT", "10")
    try:
        return min(max(1, int(raw)), 100)
    except ValueError:
        return 10


def _quote_identifier(identifier: str) -> str:
    clean = str(identifier or "").strip()
    if not clean:
        raise ValueError("SQL identifier cannot be empty")
    return f"[{clean.replace(']', ']]')}]"


def _qualified_table(schema_name: str, table_name: str) -> str:
    return f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"


def _tablesample_clause() -> str:
    return f"TABLESAMPLE({_profiling_sample_pct()} PERCENT)"


def _execute_source_query(database_name: str, query: str) -> List[Any]:
    conn = get_client_connection(database_name)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        conn.close()


def _resolve_tables_for_profiling(state: Stage01State) -> List[ProfilingTable]:
    discovered = state.get("discovered_metadata") or {}
    raw_tables = discovered.get("tables", []) if isinstance(discovered, dict) else []
    resolved: List[ProfilingTable] = []

    for item in raw_tables:
        if not isinstance(item, dict):
            continue
        if item.get("table_status") != "COMPLETED":
            continue

        database_name = str(item.get("database_name") or "").strip()
        schema_name = str(item.get("schema_name") or "dbo").strip()
        table_name = str(item.get("table_name") or "").strip()
        columns = item.get("columns") or []

        if not database_name or not table_name or not columns:
            continue

        resolved.append(
            ProfilingTable(
                database_name=database_name,
                schema_name=schema_name,
                table_name=table_name,
                columns=[col for col in columns if isinstance(col, dict)],
            )
        )

    return resolved


def classify_profile_tier(column: Dict[str, Any]) -> ProfileTier:
    semantic_type = str(column.get("semantic_type") or "").upper()
    column_name = str(column.get("column_name") or "").lower()
    data_type = str(column.get("data_type") or "").lower()

    if semantic_type in {"ID", "SURROGATE_KEY"} or column_name == "id" or column_name.endswith("_id"):
        return "ID"
    if semantic_type == "AUDIT_TIMESTAMP" or column_name in {
        "created_at",
        "updated_at",
        "modified_at",
        "created_date",
        "updated_date",
        "modified_date",
    }:
        return "AUDIT"
    if semantic_type == "FLAG" or data_type == "bit" or column_name.startswith(("is_", "has_")):
        return "FLAG"
    if data_type in DATE_TYPES:
        return "DATE"
    if semantic_type == "MEASURE" or data_type in NUMERIC_TYPES:
        return "MEASURE"
    if semantic_type == "DIMENSION" or data_type in TEXT_TYPES:
        return "DIMENSION"
    return "DEFAULT"


def _supports_cardinality(data_type: Optional[str]) -> bool:
    return str(data_type or "").lower() not in LOB_TYPES


def _fetch_pushdown_stats(
    database_name: str,
    schema_name: str,
    table_name: str,
    column_name: str,
    data_type: Optional[str],
    tier: ProfileTier,
) -> Dict[str, Any]:
    column_sql = _quote_identifier(column_name)
    table_sql = _qualified_table(schema_name, table_name)

    expressions = [
        "COUNT_BIG(*) AS total_rows",
        f"COUNT_BIG({column_sql}) AS non_null_count",
    ]
    include_cardinality = tier != "AUDIT" and _supports_cardinality(data_type)
    if include_cardinality:
        expressions.append(f"COUNT_BIG(DISTINCT {column_sql}) AS cardinality")
    if tier in {"MEASURE", "DATE"}:
        expressions.append(f"MIN({column_sql}) AS col_min")
        expressions.append(f"MAX({column_sql}) AS col_max")

    query = f"SELECT {', '.join(expressions)} FROM {table_sql}"
    row = _execute_source_query(database_name, query)[0]

    total_rows = int(row.total_rows or 0)
    non_null_count = int(row.non_null_count or 0)
    null_rate = round(1.0 - (non_null_count / total_rows), 6) if total_rows > 0 else 0.0

    result: Dict[str, Any] = {
        "total_rows": total_rows,
        "non_null_count": non_null_count,
        "null_rate": null_rate,
    }
    if include_cardinality:
        result["cardinality"] = int(row.cardinality) if row.cardinality is not None else None
    if tier in {"MEASURE", "DATE"}:
        result["col_min"] = str(row.col_min) if row.col_min is not None else None
        result["col_max"] = str(row.col_max) if row.col_max is not None else None

    return result


def _fetch_measure_percentiles(
    database_name: str,
    schema_name: str,
    table_name: str,
    column_name: str,
) -> tuple[Optional[float], Optional[float]]:
    column_sql = _quote_identifier(column_name)
    table_sql = _qualified_table(schema_name, table_name)
    query = f"""
        SELECT DISTINCT
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY TRY_CONVERT(float, {column_sql})) OVER () AS p25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY TRY_CONVERT(float, {column_sql})) OVER () AS p75
        FROM {table_sql} {_tablesample_clause()}
        WHERE {column_sql} IS NOT NULL
          AND TRY_CONVERT(float, {column_sql}) IS NOT NULL
    """

    try:
        rows = _execute_source_query(database_name, query)
        if not rows:
            return None, None
        row = rows[0]
        return (
            float(row.p25) if row.p25 is not None else None,
            float(row.p75) if row.p75 is not None else None,
        )
    except Exception as exc:
        logger.warning(
            "Percentile profiling failed for %s.%s.%s: %s",
            schema_name,
            table_name,
            column_name,
            str(exc)[:120],
            extra={"node": "column_profiling"},
        )
        return None, None


def _fetch_top_samples(
    database_name: str,
    schema_name: str,
    table_name: str,
    column_name: str,
) -> Optional[List[Dict[str, Any]]]:
    column_sql = _quote_identifier(column_name)
    table_sql = _qualified_table(schema_name, table_name)
    limit = _top_sample_limit()
    query = f"""
        SELECT TOP ({limit})
            CAST({column_sql} AS nvarchar(4000)) AS sample_value,
            COUNT_BIG(*) AS sample_count
        FROM {table_sql} {_tablesample_clause()}
        WHERE {column_sql} IS NOT NULL
        GROUP BY CAST({column_sql} AS nvarchar(4000))
        ORDER BY sample_count DESC
    """

    try:
        rows = _execute_source_query(database_name, query)
        samples = [
            {
                "value": str(row.sample_value),
                "count": int(row.sample_count or 0),
            }
            for row in rows
            if row.sample_value is not None
        ]
        return samples or None
    except Exception as exc:
        logger.warning(
            "Top-sample profiling failed for %s.%s.%s: %s",
            schema_name,
            table_name,
            column_name,
            str(exc)[:120],
            extra={"node": "column_profiling"},
        )
        return None


def profile_column(table_ref: ProfilingTable, column: Dict[str, Any], run_id: str) -> ColumnProfileResult:
    column_name = str(column.get("column_name") or "")
    data_type = str(column.get("data_type") or "") or None
    tier = classify_profile_tier(column)
    base = {
        "run_id": run_id,
        "database_name": table_ref.database_name,
        "schema_name": table_ref.schema_name,
        "table_name": table_ref.table_name,
        "column_name": column_name,
        "data_type": data_type,
        "profile_tier": tier,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        result = {
            **base,
            **_fetch_pushdown_stats(
                table_ref.database_name,
                table_ref.schema_name,
                table_ref.table_name,
                column_name,
                data_type,
                tier,
            ),
        }

        if tier == "MEASURE":
            p25, p75 = _fetch_measure_percentiles(
                table_ref.database_name,
                table_ref.schema_name,
                table_ref.table_name,
                column_name,
            )
            result["p25"] = p25
            result["p75"] = p75

        if tier in {"FLAG", "DIMENSION", "DEFAULT"}:
            cardinality = result.get("cardinality")
            if cardinality is not None and cardinality <= _high_cardinality_threshold():
                result["top_samples"] = _fetch_top_samples(
                    table_ref.database_name,
                    table_ref.schema_name,
                    table_ref.table_name,
                    column_name,
                )
            elif cardinality is not None and cardinality > _high_cardinality_threshold():
                result["profile_tier"] = "HIGH_CARD_TEXT"

        return ColumnProfileResult(**result)
    except Exception as exc:
        logger.warning(
            "Column profiling failed for %s.%s.%s: %s",
            table_ref.schema_name,
            table_ref.table_name,
            column_name,
            str(exc)[:160],
            extra={"node": "column_profiling"},
        )
        return ColumnProfileResult(
            **base,
            profiling_status="FAILED",
            error_message=str(exc)[:500],
        )


def profile_table(table_ref: ProfilingTable, run_id: str) -> tuple[TableProfileResult, List[ColumnProfileResult]]:
    start = datetime.now(timezone.utc)
    if not table_ref.columns:
        return (
            TableProfileResult(
                database_name=table_ref.database_name,
                schema_name=table_ref.schema_name,
                table_name=table_ref.table_name,
                columns_profiled=0,
                columns_failed=0,
                status="SKIPPED",
                duration_seconds=0.0,
                error_message="No columns available for profiling",
            ),
            [],
        )

    profiles = [profile_column(table_ref, column, run_id) for column in table_ref.columns]
    failed = sum(1 for profile in profiles if profile.profiling_status == "FAILED")
    success = len(profiles) - failed
    duration = (datetime.now(timezone.utc) - start).total_seconds()

    if failed == 0:
        status: Literal["SUCCESS", "PARTIAL", "FAILED", "SKIPPED"] = "SUCCESS"
    elif success > 0:
        status = "PARTIAL"
    else:
        status = "FAILED"

    return (
        TableProfileResult(
            database_name=table_ref.database_name,
            schema_name=table_ref.schema_name,
            table_name=table_ref.table_name,
            columns_profiled=success,
            columns_failed=failed,
            status=status,
            duration_seconds=round(duration, 3),
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
    log_context = {
        "run_id": new_state.get("run_id", "unknown"),
        "node": "column_profiling",
    }

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
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(profile_table, table_ref, run_id): table_ref for table_ref in table_refs}
            for future in as_completed(futures):
                table_ref = futures[future]
                try:
                    table_result, profiles = future.result()
                except Exception as exc:
                    logger.warning(
                        "Table profiling failed for %s.%s.%s: %s",
                        table_ref.database_name,
                        table_ref.schema_name,
                        table_ref.table_name,
                        str(exc)[:160],
                        extra=log_context,
                    )
                    table_result = TableProfileResult(
                        database_name=table_ref.database_name,
                        schema_name=table_ref.schema_name,
                        table_name=table_ref.table_name,
                        columns_profiled=0,
                        columns_failed=len(table_ref.columns),
                        status="FAILED",
                        duration_seconds=0.0,
                        error_message=str(exc)[:500],
                    )
                    profiles = []

                table_results.append(table_result)
                column_profiles.extend(profiles)

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
    status = "COMPLETED" if failed_tables == 0 else "COMPLETED_WITH_WARNINGS"
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
