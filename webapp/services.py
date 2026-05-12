from __future__ import annotations

import json
import os
import uuid
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from utilis.db import config, get_completed_items, get_connection, get_pending_items, timed_stage, update_hitl_items_batch
from utilis.logger import logger


BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("ATHENA_BACKGROUND_WORKERS", "2")))
BACKGROUND_JOBS: Dict[str, Future] = {}
BACKGROUND_JOB_LOCK = threading.Lock()


def _pipeline_schema() -> str:
    return (
        config["azure_sql"].get("pipeline_schema")
        or config["azure_sql"].get("schema_name")
        or "dbo"
    )


def fetch_json_artifact(run_id: str, artifact_type: str) -> Dict[str, Any]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT TOP 1 payload
            FROM [{_pipeline_schema()}].[ai_store]
            WHERE run_id = ? AND artifact_type = ?
            ORDER BY stored_at DESC
            """,
            (run_id, artifact_type),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return {}
        return json.loads(row[0])
    finally:
        conn.close()


def fetch_run_summary(run_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                stage,
                artifact_type,
                faithfulness_status,
                retry_count,
                input_tokens,
                output_tokens,
                token_count,
                cost_usd,
                stored_at
            FROM [{_pipeline_schema()}].[ai_store]
            WHERE run_id = ?
            ORDER BY stored_at
            """,
            (run_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "stage": row[0],
                "artifact_type": row[1],
                "faithfulness_status": row[2],
                "retry_count": row[3],
                "input_tokens": row[4],
                "output_tokens": row[5],
                "token_count": row[6],
                "cost_usd": row[7],
                "stored_at": row[8],
            }
            for row in rows
        ]
    finally:
        conn.close()


def load_checkpoint_state(run_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT TOP 1 full_state_json
            FROM [{_pipeline_schema()}].[kpi_checkpoints]
            WHERE run_id = ?
            ORDER BY checkpoint_at DESC
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    finally:
        conn.close()


def save_checkpoint_state(run_id: str, state: Dict[str, Any]) -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        state_json = json.dumps(state, default=str)
        cursor.execute(
            f"""
            MERGE [{_pipeline_schema()}].[kpi_checkpoints] AS target
            USING (VALUES (?)) AS source (run_id)
            ON target.run_id = source.run_id
            WHEN MATCHED THEN UPDATE SET full_state_json = ?, checkpoint_at = GETUTCDATE()
            WHEN NOT MATCHED THEN INSERT (run_id, full_state_json, checkpoint_at) VALUES (?, ?, GETUTCDATE());
            """,
            (run_id, state_json, run_id, state_json),
        )
        conn.commit()
    finally:
        conn.close()


def mark_run_processing(run_id: str, stage: str) -> None:
    checkpoint = load_checkpoint_state(run_id) or {"run_id": run_id}
    checkpoint.update(
        {
            "run_id": run_id,
            "status": "PROCESSING",
            "background_stage": stage,
        }
    )
    save_checkpoint_state(run_id, checkpoint)


def submit_background(run_id: str, stage: str, fn, *args) -> Future:
    job_key = f"{run_id}:{stage}"
    with BACKGROUND_JOB_LOCK:
        existing = BACKGROUND_JOBS.get(job_key)
        if existing and not existing.done():
            logger.info("Background %s already running for run_id=%s", stage, run_id)
            return existing

        mark_run_processing(run_id, stage)
        future = BACKGROUND_EXECUTOR.submit(fn, *args)
        BACKGROUND_JOBS[job_key] = future

    def _record_background_result(done: Future) -> None:
        try:
            result = done.result()
            checkpoint = load_checkpoint_state(run_id) or {"run_id": run_id}
            if isinstance(result, dict):
                checkpoint.update(result)
            checkpoint.update({"run_id": run_id, "background_stage": None})
            if checkpoint.get("status") == "PROCESSING":
                checkpoint["status"] = checkpoint.get("semantic_enrichment_status") or checkpoint.get("table_nomination_status") or "COMPLETED"
            save_checkpoint_state(run_id, checkpoint)
        except Exception as exc:
            logger.exception("Background %s failed for run_id=%s", stage, run_id)
            checkpoint = load_checkpoint_state(run_id) or {"run_id": run_id}
            checkpoint.update(
                {
                    "run_id": run_id,
                    "status": "FAILED",
                    "background_stage": None,
                    "failed_background_stage": stage,
                    "error": str(exc),
                }
            )
            save_checkpoint_state(run_id, checkpoint)
        finally:
            with BACKGROUND_JOB_LOCK:
                if BACKGROUND_JOBS.get(job_key) is done:
                    BACKGROUND_JOBS.pop(job_key, None)

    future.add_done_callback(_record_background_result)
    return future


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            WITH ai_runs AS (
                SELECT run_id, MAX(stored_at) AS last_activity
                FROM [{_pipeline_schema()}].[ai_store]
                GROUP BY run_id
            ),
            queue_runs AS (
                SELECT run_id, MAX(COALESCE(decided_at, queued_at)) AS last_activity
                FROM [{_pipeline_schema()}].[hitl_review_queue]
                GROUP BY run_id
            ),
            combined AS (
                SELECT run_id, last_activity FROM ai_runs
                UNION
                SELECT run_id, last_activity FROM queue_runs
            )
            SELECT TOP ({limit}) run_id, MAX(last_activity) AS last_activity
            FROM combined
            GROUP BY run_id
            ORDER BY MAX(last_activity) DESC
            """
        )
        rows = cursor.fetchall()
        return [
            {
                "run_id": row[0],
                "last_activity": row[1],
            }
            for row in rows
            if row[0]
        ]
    finally:
        conn.close()


def _table_key(item: Dict[str, Any]) -> str:
    return f"{item.get('database_name', '')}.{item.get('schema_name', '')}.{item.get('table_name', '')}"


def load_bronze_scripts() -> Dict[str, Any]:
    bundle_path = Path(os.getcwd()) / "generated_code" / "bronze" / "bronze_scripts.json"
    if not bundle_path.exists():
        return {"generated_at": None, "scripts": []}

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    scripts: List[Dict[str, Any]] = []
    for item in bundle.get("scripts", []):
        script_path = Path(item.get("script_path", ""))
        script_body = ""
        if script_path.exists():
            script_body = script_path.read_text(encoding="utf-8")
        scripts.append(
            {
                **item,
                "script_body": script_body,
            }
        )

    return {
        "generated_at": bundle.get("generated_at"),
        "scripts": scripts,
    }


def load_silver_scripts() -> Dict[str, Any]:
    bundle_path = Path(os.getcwd()) / "generated_code" / "silver" / "silver_scripts.json"
    if not bundle_path.exists():
        return {"generated_at": None, "scripts": []}

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    scripts: List[Dict[str, Any]] = []
    for item in bundle.get("scripts", []):
        script_path = Path(item.get("script_path", ""))
        script_body = ""
        if script_path.exists():
            script_body = script_path.read_text(encoding="utf-8")
        scripts.append(
            {
                **item,
                "script_body": script_body,
            }
        )

    return {
        "generated_at": bundle.get("generated_at"),
        "scripts": scripts,
    }


def build_pipeline_steps(
    *,
    checkpoint: Dict[str, Any],
    summary: List[Dict[str, Any]],
    pending_gate1: List[Dict[str, Any]],
    completed_gate1: List[Dict[str, Any]],
    nominated_tables: List[Dict[str, Any]],
    certified_tables: List[Dict[str, Any]],
    enriched_payload: Dict[str, Any],
    gate3_payload: Dict[str, Any],
    bronze_generation_completed: bool,
    silver_generation_completed: bool,
) -> List[Dict[str, str]]:
    artifact_types = {str(row.get("artifact_type") or "") for row in summary}
    stages = {str(row.get("stage") or "").lower() for row in summary}

    def has_stage(text: str) -> bool:
        needle = text.lower()
        return any(needle in stage for stage in stages)

    steps = [
        {
            "key": "ingestion",
            "label": "Ingestion",
            "complete": bool(checkpoint.get("fingerprint") or checkpoint.get("brd_text") or summary),
            "detail": "BRD parsed and run created",
        },
        {
            "key": "memory",
            "label": "Memory Lookup",
            "complete": bool(
                checkpoint.get("memory_layer1")
                or checkpoint.get("memory_layer2")
                or has_stage("memory")
                or artifact_types.intersection({"REQUIREMENTS", "REQUIREMENTS_WARN", "KPIS"})
            ),
            "detail": "Exact/semantic memory checked",
        },
        {
            "key": "requirements",
            "label": "Req Extract",
            "complete": bool(artifact_types.intersection({"REQUIREMENTS", "REQUIREMENTS_WARN"})),
            "detail": "Business requirements extracted",
        },
        {
            "key": "kpis",
            "label": "KPI Extract",
            "complete": bool("KPIS" in artifact_types or pending_gate1 or completed_gate1),
            "detail": "KPI candidates generated",
        },
        {
            "key": "gate1",
            "label": "Gate 1",
            "complete": bool("GATE1_CERTIFIED_KPIS" in artifact_types or (completed_gate1 and not pending_gate1)),
            "detail": "Human KPI certification",
        },
        {
            "key": "nomination",
            "label": "Nomination",
            "complete": bool("TABLE_NOMINATIONS" in artifact_types or nominated_tables),
            "detail": "Candidate tables selected",
        },
        {
            "key": "gate2",
            "label": "Gate 2",
            "complete": bool("GATE2_CERTIFIED_TABLES" in artifact_types or certified_tables),
            "detail": "Human table certification",
        },
        {
            "key": "discovery",
            "label": "Metadata Discovery",
            "complete": bool("DISCOVERED_METADATA" in artifact_types or checkpoint.get("metadata_status")),
            "detail": "Table metadata discovered",
        },
        {
            "key": "profiling",
            "label": "Column Profiling",
            "complete": bool("COLUMN_PROFILES" in artifact_types or checkpoint.get("column_profiling_status")),
            "detail": "Column profiles generated",
        },
        {
            "key": "enrichment",
            "label": "Semantic Enrichment",
            "complete": bool("ENRICHED_METADATA" in artifact_types or enriched_payload or checkpoint.get("semantic_enrichment_status")),
            "detail": "Semantic metadata enriched",
        },
        {
            "key": "gate3",
            "label": "Gate 3",
            "complete": bool("GATE3_APPROVED_ENRICHMENT" in artifact_types or gate3_payload),
            "detail": "Human enrichment approval",
        },
        {
            "key": "bronze",
            "label": "Bronze Scripts",
            "complete": bool(bronze_generation_completed),
            "detail": "Bronze scripts generated",
        },
        {
            "key": "silver",
            "label": "Silver Scripts",
            "complete": bool(silver_generation_completed),
            "detail": "Silver transformation scripts generated",
        },
    ]

    last_complete_index = -1
    for index, step in enumerate(steps):
        if step["complete"]:
            last_complete_index = index

    # The pipeline is linear for this UI. If a downstream node completed, every
    # upstream node must have already run even if its individual artifact was not
    # persisted with the exact name this page checks.
    for index, step in enumerate(steps):
        if index <= last_complete_index:
            step["complete"] = True

    first_incomplete_seen = False
    for step in steps:
        if step["complete"]:
            step["state"] = "complete"
        elif not first_incomplete_seen:
            step["state"] = "running"
            first_incomplete_seen = True
        else:
            step["state"] = "pending"

    if checkpoint.get("status") == "FAILED":
        for step in steps:
            if step["state"] == "running":
                step["state"] = "failed"
                break

    return steps


def get_run_context(run_id: str) -> Dict[str, Any]:
    checkpoint = load_checkpoint_state(run_id) or {}
    summary = fetch_run_summary(run_id)
    pending_gate1 = get_pending_items(run_id, 1)
    completed_gate1 = get_completed_items(run_id, 1)
    nominations_payload = fetch_json_artifact(run_id, "TABLE_NOMINATIONS")
    nominated_tables = nominations_payload.get("nominations", []) or []
    gate2_payload = fetch_json_artifact(run_id, "GATE2_CERTIFIED_TABLES")
    certified_tables = gate2_payload.get("certified_tables", []) or []
    enriched_payload = fetch_json_artifact(run_id, "ENRICHED_METADATA")
    gate3_payload = fetch_json_artifact(run_id, "GATE3_APPROVED_ENRICHMENT")
    bronze_generation_completed = any(
        row.get("artifact_type") in {"BRONZE_GENERATION", "BRONZE_SCRIPTS"}
        or str(row.get("stage", "")).lower().startswith("bronze")
        for row in summary
    ) or checkpoint.get("bronze_generation_status") == "COMPLETED"
    silver_generation_completed = any(
        row.get("artifact_type") in {"SILVER_GENERATION", "SILVER_SCRIPTS"}
        or str(row.get("stage", "")).lower().startswith("silver")
        for row in summary
    ) or checkpoint.get("silver_generation_status") == "COMPLETED"
    bronze = load_bronze_scripts() if gate3_payload or bronze_generation_completed else {"generated_at": None, "scripts": []}
    silver = load_silver_scripts() if silver_generation_completed else {"generated_at": None, "scripts": []}

    enriched_columns = enriched_payload.get("columns", []) if isinstance(enriched_payload, dict) else []
    enriched_joins = enriched_payload.get("joins", []) if isinstance(enriched_payload, dict) else []
    semantic_counts: Dict[str, int] = {}
    pii_columns: List[Dict[str, Any]] = []
    join_key_columns: List[Dict[str, Any]] = []
    measure_columns: List[Dict[str, Any]] = []
    for column in enriched_columns:
        semantic_type = str(column.get("semantic_type") or "UNKNOWN")
        semantic_counts[semantic_type] = semantic_counts.get(semantic_type, 0) + 1
        if column.get("is_pii_candidate") or column.get("is_pii"):
            pii_columns.append(column)
        if semantic_type in {"ID", "SURROGATE_KEY"} or column.get("is_join_key"):
            join_key_columns.append(column)
        if column.get("is_measure"):
            measure_columns.append(column)

    next_gate = None
    resume_message = None
    if pending_gate1:
        next_gate = 1
        resume_message = "Gate 1 is pending. Review the KPI items below."
    elif nominated_tables and not certified_tables:
        next_gate = 2
        resume_message = "Gate 2 is pending. Review and certify nominated tables below."
    elif enriched_payload and not gate3_payload:
        next_gate = 3
        resume_message = "Gate 3 is pending. Review enrichment details below."
    elif gate3_payload:
        resume_message = "Gate 3 is complete."
    elif certified_tables and not enriched_payload:
        resume_message = "Gate 2 is certified. Downstream metadata/profiling/enrichment has not completed yet."
    elif completed_gate1 and not nominated_tables:
        resume_message = "Gate 1 is certified. Table nomination has not completed yet."
    elif not summary and not checkpoint:
        resume_message = "No stored state was found for this run ID."

    status = checkpoint.get("status") or checkpoint.get("table_nomination_status") or checkpoint.get("enrichment_review_status") or "UNKNOWN"
    if checkpoint.get("bronze_generation_status") == "COMPLETED":
        status = "PIPELINE_COMPLETED"
    if gate3_payload and bronze_generation_completed:
        status = "PIPELINE_COMPLETED"
    if silver_generation_completed:
        status = "PIPELINE_COMPLETED"
    pipeline_steps = build_pipeline_steps(
        checkpoint=checkpoint,
        summary=summary,
        pending_gate1=pending_gate1,
        completed_gate1=completed_gate1,
        nominated_tables=nominated_tables,
        certified_tables=certified_tables,
        enriched_payload=enriched_payload,
        gate3_payload=gate3_payload,
        bronze_generation_completed=bronze_generation_completed,
        silver_generation_completed=silver_generation_completed,
    )
    current_pipeline_step = next((step for step in pipeline_steps if step["state"] == "running"), None)
    if not current_pipeline_step and status == "PIPELINE_COMPLETED":
        current_pipeline_step = {
            "key": "completed",
            "label": "Pipeline Completed",
            "state": "complete",
            "detail": "All backend stages completed",
        }

    return {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "summary": summary,
        "pending_gate1": pending_gate1,
        "completed_gate1": completed_gate1,
        "nominated_tables": nominated_tables,
        "certified_tables": certified_tables,
        "enriched_metadata": enriched_payload,
        "enriched_columns": enriched_columns,
        "enriched_joins": enriched_joins,
        "semantic_counts": semantic_counts,
        "pii_columns": pii_columns,
        "join_key_columns": join_key_columns,
        "measure_columns": measure_columns,
        "gate3_approved": bool(gate3_payload),
        "bronze_generation_completed": bronze_generation_completed,
        "silver_generation_completed": silver_generation_completed,
        "bronze": bronze,
        "silver": silver,
        "next_gate": next_gate,
        "resume_message": resume_message,
        "status": status,
        "pipeline_steps": pipeline_steps,
        "current_pipeline_step": current_pipeline_step,
    }


def start_pipeline(*, brd_text: Optional[str] = None, input_path: Optional[str] = None, source_databases: Optional[List[str]] = None) -> Dict[str, Any]:
    from graph import app as graph_app

    run_id = str(uuid.uuid4())
    default_source_db = config["azure_sql"].get("source_database") or "insurance"
    initial_state: Dict[str, Any] = {
        "brd_text": brd_text or input_path or "",
        "run_id": run_id,
        "metadata": {},
        "status": "PENDING",
        "source_databases": source_databases or [default_source_db],
    }
    result = graph_app.invoke(initial_state, {"configurable": {"thread_id": run_id}})
    return {
        "run_id": run_id,
        "result": result,
    }


def submit_gate1_review(run_id: str, decisions: List[Dict[str, str]]) -> Dict[str, Any]:
    from nodes.hitl import hitl_review_node
    from nodes.table_nomination import table_nomination_node

    pending = get_pending_items(run_id, 1)
    existing_nomination = fetch_json_artifact(run_id, "TABLE_NOMINATIONS")
    if not pending and existing_nomination.get("nominations"):
        checkpoint = load_checkpoint_state(run_id) or {"run_id": run_id}
        return {
            **checkpoint,
            "run_id": run_id,
            "nominated_tables": existing_nomination.get("nominations", []),
            "table_nomination_status": checkpoint.get("table_nomination_status") or "PENDING",
        }

    decision_map = {item["item_id"]: item for item in decisions}
    batch_updates: List[Dict[str, Optional[str]]] = []

    for item in pending:
        decision = decision_map.get(item["item_id"])
        action = (decision or {}).get("action", "APPROVED").upper()
        kpi = item["kpi"]

        if action == "REJECTED":
            batch_updates.append(
                {
                    "item_id": item["item_id"],
                    "status": "REJECTED",
                    "edited_content": None,
                    "rejection_reason": (decision or {}).get("reason", ""),
                }
            )
            continue

        if action == "EDITED":
            edited = kpi.copy()
            edited["kpi_name"] = (decision or {}).get("name", kpi.get("kpi_name", ""))
            edited["kpi_description"] = (decision or {}).get("description", kpi.get("kpi_description", ""))
            batch_updates.append(
                {
                    "item_id": item["item_id"],
                    "status": "APPROVED",
                    "edited_content": json.dumps(edited),
                    "rejection_reason": None,
                }
            )
            continue

        batch_updates.append(
            {
                "item_id": item["item_id"],
                "status": "APPROVED",
                "edited_content": None,
                "rejection_reason": None,
            }
        )

    update_hitl_items_batch(batch_updates)

    certified = get_completed_items(run_id, 1)
    resumed_input = load_checkpoint_state(run_id) or {"run_id": run_id}
    resumed_input["human_decision"] = "COMPLETED"
    resumed_input["certified_kpis"] = [item["kpi"] for item in certified]
    with timed_stage("gate1_hitl_certification", run_id=run_id, node="webapp"):
        resumed = hitl_review_node(resumed_input)
    if resumed.get("status") == "FAILED":
        raise ValueError(resumed.get("error", "Gate 1 certification failed."))

    save_checkpoint_state(run_id, resumed)

    with timed_stage("table_nomination", run_id=run_id, node="webapp"):
        nominated = table_nomination_node(resumed)
    if nominated.get("status") == "FAILED":
        raise ValueError(
            nominated.get("table_nomination_error", nominated.get("error", "Table nomination failed."))
        )

    save_checkpoint_state(run_id, nominated)
    return nominated


def submit_gate2_review(run_id: str, approved_keys: List[str]) -> Dict[str, Any]:
    from nodes.column_profiling import column_profiling_node
    from nodes.hitl import hitl_table_review_node
    from nodes.metadata_discovery import metadata_discovery_node
    from nodes.semantic_enrichment import semantic_enrichment_node

    tables = fetch_json_artifact(run_id, "TABLE_NOMINATIONS").get("nominations", []) or []
    approved_key_set = set(approved_keys)
    approved = [item for item in tables if _table_key(item) in approved_key_set]

    if not approved:
        raise ValueError("At least one table must be approved for Gate 2.")

    resumed_input = load_checkpoint_state(run_id) or {"run_id": run_id}
    resumed_input["human_table_decision"] = "COMPLETED"
    resumed_input["certified_tables"] = approved
    with timed_stage("gate2_hitl_certification", run_id=run_id, node="webapp"):
        resumed = hitl_table_review_node(resumed_input)
    if resumed.get("status") == "FAILED":
        raise ValueError(resumed.get("error", "Gate 2 certification failed."))
    save_checkpoint_state(run_id, resumed)

    with timed_stage("metadata_discovery", run_id=run_id, node="webapp"):
        discovered = metadata_discovery_node(resumed)
    save_checkpoint_state(run_id, discovered)

    with timed_stage("column_profiling", run_id=run_id, node="webapp"):
        profiled = column_profiling_node(discovered)
    save_checkpoint_state(run_id, profiled)

    with timed_stage("semantic_enrichment", run_id=run_id, node="webapp"):
        enriched = semantic_enrichment_node(profiled)
    save_checkpoint_state(run_id, enriched)
    return enriched


def submit_gate3_review(run_id: str, approve: bool) -> Dict[str, Any]:
    from nodes.bronze_gen import bronze_code_generation_node
    from nodes.hitl import build_hitl_enrichment_review_node
    from nodes.silver_gen import silver_code_generation_node

    metadata = fetch_json_artifact(run_id, "ENRICHED_METADATA")
    if not metadata:
        raise ValueError("No enriched metadata found for this run.")

    enrichment_node = build_hitl_enrichment_review_node()
    state: Dict[str, Any] = {
        "run_id": run_id,
        "enriched_metadata": metadata,
        "semantic_tags_reviewed": approve,
        "pii_classifications_reviewed": approve,
        "join_key_annotations_reviewed": approve,
        "enrichment_review_decision": "APPROVED" if approve else "REJECTED",
    }
    result = enrichment_node(state)
    if result.get("enrichment_review_status") != "COMPLETED":
        return result

    checkpoint_state = load_checkpoint_state(run_id) or {}
    certified_tables = (
        fetch_json_artifact(run_id, "GATE2_CERTIFIED_TABLES").get("certified_tables", [])
        or metadata.get("certified_tables")
        or checkpoint_state.get("certified_tables")
        or []
    )
    if not certified_tables:
        raise ValueError("Bronze generation skipped: no Gate 2 certified tables found.")

    bronze_state: Dict[str, Any] = {
        "run_id": run_id,
        "fingerprint": metadata.get("fingerprint") or checkpoint_state.get("fingerprint") or run_id,
        "certified_tables": certified_tables,
        "discovered_metadata": fetch_json_artifact(run_id, "DISCOVERED_METADATA") or checkpoint_state.get("discovered_metadata") or {},
        "bronze_catalog": os.getenv("BRONZE_CATALOG", "main"),
        "bronze_schema": os.getenv("BRONZE_SCHEMA", "bronze"),
    }
    bronze_result = bronze_code_generation_node(bronze_state)
    silver_state = {
        **checkpoint_state,
        **result,
        **bronze_result,
        "run_id": run_id,
        "enriched_metadata": metadata,
        "silver_catalog": os.getenv("SILVER_CATALOG", os.getenv("BRONZE_CATALOG", "main")),
        "silver_schema": os.getenv("SILVER_SCHEMA", "silver"),
    }
    silver_result = silver_code_generation_node(silver_state)
    final_state = {
        **checkpoint_state,
        **result,
        **bronze_result,
        **silver_result,
        "run_id": run_id,
    }
    if silver_result.get("silver_generation_status") == "COMPLETED":
        final_state["status"] = "PIPELINE_COMPLETED"
    save_checkpoint_state(run_id, final_state)
    return {
        "enrichment_result": result,
        "bronze_result": bronze_result,
        "silver_result": silver_result,
    }


def submit_bronze_generation(run_id: str) -> Dict[str, Any]:
    from nodes.bronze_gen import bronze_code_generation_node

    checkpoint_state = load_checkpoint_state(run_id) or {"run_id": run_id}
    gate2_payload = fetch_json_artifact(run_id, "GATE2_CERTIFIED_TABLES")
    certified_tables = (
        gate2_payload.get("certified_tables", [])
        or checkpoint_state.get("certified_tables")
        or []
    )
    if not certified_tables:
        raise ValueError("Bronze generation failed: no Gate 2 certified tables found.")

    state: Dict[str, Any] = {
        **checkpoint_state,
        "run_id": run_id,
        "certified_tables": certified_tables,
        "discovered_metadata": fetch_json_artifact(run_id, "DISCOVERED_METADATA") or checkpoint_state.get("discovered_metadata") or {},
        "bronze_catalog": os.getenv("BRONZE_CATALOG", "main"),
        "bronze_schema": os.getenv("BRONZE_SCHEMA", "bronze"),
    }
    result = bronze_code_generation_node(state)
    final_state = {**checkpoint_state, **result, "run_id": run_id}
    save_checkpoint_state(run_id, final_state)
    return result


def submit_silver_generation(run_id: str) -> Dict[str, Any]:
    from nodes.silver_gen import silver_code_generation_node

    checkpoint_state = load_checkpoint_state(run_id) or {"run_id": run_id}
    metadata = fetch_json_artifact(run_id, "ENRICHED_METADATA")
    gate3_payload = fetch_json_artifact(run_id, "GATE3_APPROVED_ENRICHMENT")
    if not metadata and isinstance(gate3_payload, dict):
        metadata = gate3_payload.get("enrichment_artifact") or {}

    state: Dict[str, Any] = {
        **checkpoint_state,
        "run_id": run_id,
        "enriched_metadata": metadata,
        "silver_catalog": os.getenv("SILVER_CATALOG", os.getenv("BRONZE_CATALOG", "main")),
        "silver_schema": os.getenv("SILVER_SCHEMA", "silver"),
    }
    result = silver_code_generation_node(state)
    final_state = {**checkpoint_state, **result, "run_id": run_id}
    if result.get("silver_generation_status") == "COMPLETED":
        final_state["status"] = "PIPELINE_COMPLETED"
    save_checkpoint_state(run_id, final_state)
    return result
