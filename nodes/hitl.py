"""
HITL review nodes for LangGraph.
Placed after interrupts. Certify artifacts to `ai_store` when review is completed.
"""

from typing import Callable, Dict, List

from state import Stage01State
from utilis.db import ai_store_db_writer
from utilis.logger import logger


def certify_hitl_enrichment(run_id: str, enrichment_artifact: dict, fingerprint: str | None = None) -> None:
    ai_store_db_writer(
        run_id=run_id,
        stage="HITL Enrichment Certification",
        artifact_type="GATE3_APPROVED_ENRICHMENT",
        payload={
            "fingerprint": fingerprint or run_id,
            "storage_fingerprint": f"{fingerprint or run_id}:GATE3_APPROVED_ENRICHMENT",
            "run_id": run_id,
            "enrichment_artifact": enrichment_artifact,
            "source": "HUMAN_CERTIFIED_ENRICHMENT",
        },
        schema_version="GATE3_v1",
        prompt_version="NB09B_ENRICHMENT_REVIEW_v1",
        faithfulness_status="PASSED",
        token_count=0,
        input_tokens=0,
        output_tokens=0,
        fingerprint=fingerprint or run_id,
    )


def build_hitl_enrichment_review_node() -> Callable[[Stage01State], Stage01State]:
    """
    Gate 3 HITL review for semantic enrichment.
    Auto-approves in dev mode. In prod, waits for reviewed flags/decision in state.
    """
    import os

    def hitl_enrichment_review_node(state: Stage01State) -> Stage01State:
        log_context = {"run_id": state.get("run_id", "unknown"), "node": "enrichment_review"}
        dev_mode = os.getenv("DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on", "dev"}

        if dev_mode:
            logger.info("[DEV] Auto-approving enrichment review", extra=log_context)
            new_state = state.copy()
            new_state["status"] = "GATE3_COMPLETE"
            artifact = state.get("enriched_metadata")
            new_state["enrichment_review_status"] = "COMPLETED"
            new_state["semantic_tags_reviewed"] = True
            new_state["pii_classifications_reviewed"] = True
            new_state["join_key_annotations_reviewed"] = True
            new_state["enrichment_review_decision"] = "APPROVED"
            new_state["enrichment_review_artifact"] = artifact
            certify_hitl_enrichment(state["run_id"], artifact, state.get("fingerprint"))
            return new_state

        if not state.get("semantic_tags_reviewed") or not state.get("pii_classifications_reviewed"):
            logger.info("Enrichment review pending human validation", extra=log_context)
            new_state = state.copy()
            new_state["enrichment_review_status"] = "PENDING"
            new_state["enrichment_review_decision"] = "PENDING"
            return new_state

        if state.get("enrichment_review_decision") == "APPROVED":
            artifact = state.get("enriched_metadata")
            certify_hitl_enrichment(state["run_id"], artifact, state.get("fingerprint"))
            new_state = state.copy()
            new_state["status"] = "GATE3_COMPLETE"
            new_state["enrichment_review_status"] = "COMPLETED"
            new_state["enrichment_review_artifact"] = artifact
            return new_state

        if state.get("enrichment_review_decision") == "REJECTED":
            logger.warning("Enrichment review rejected by human", extra=log_context)
            new_state = state.copy()
            new_state["enrichment_review_status"] = "FAILED"
            new_state["enrichment_review_error"] = "Rejected by reviewer"
            return new_state

        new_state = state.copy()
        new_state["enrichment_review_status"] = "PENDING"
        new_state["enrichment_review_decision"] = "PENDING"
        return new_state

    return hitl_enrichment_review_node


def certify_hitl_result(run_id: str, certified_kpis: List[Dict], fingerprint: str | None = None) -> None:
    ai_store_db_writer(
        run_id=run_id,
        stage="HITL Certification",
        artifact_type="GATE1_CERTIFIED_KPIS",
        payload={
            "fingerprint": fingerprint or run_id,
            "storage_fingerprint": f"{fingerprint or run_id}:GATE1_CERTIFIED_KPIS",
            "run_id": run_id,
            "certified_kpi_count": len(certified_kpis),
            "certified_kpis": certified_kpis,
            "source": "HUMAN_CERTIFIED",
        },
        schema_version="GATE1_v1",
        prompt_version="CLI_REVIEWER_v1",
        faithfulness_status="PASSED",
        token_count=0,
        input_tokens=0,
        output_tokens=0,
        fingerprint=fingerprint or run_id,
    )


def certify_hitl_tables(run_id: str, certified_tables: List[Dict], fingerprint: str | None = None) -> None:
    ai_store_db_writer(
        run_id=run_id,
        stage="HITL Table Certification",
        artifact_type="GATE2_CERTIFIED_TABLES",
        payload={
            "fingerprint": fingerprint or run_id,
            "storage_fingerprint": f"{fingerprint or run_id}:GATE2_CERTIFIED_TABLES",
            "run_id": run_id,
            "certified_table_count": len(certified_tables),
            "certified_tables": certified_tables,
            "source": "HUMAN_CERTIFIED_TABLES",
        },
        schema_version="GATE2_v1",
        prompt_version="CLI_REVIEWER_v1",
        faithfulness_status="PASSED",
        token_count=0,
        input_tokens=0,
        output_tokens=0,
        fingerprint=fingerprint or run_id,
    )


def build_hitl_review_node() -> Callable[[Stage01State], Stage01State]:
    def hitl_review_node(state: Stage01State) -> Stage01State:
        log_context = {"run_id": state.get("run_id", "unknown"), "node": "hitl_review"}
        human_decision = state.get("human_decision")

        if human_decision != "COMPLETED":
            logger.info("HITL review skipped - decision pending", extra=log_context)
            return state

        certified_kpis = state.get("certified_kpis")
        if not certified_kpis:
            logger.warning("No certified KPIs found despite COMPLETED decision", extra=log_context)
            return {**state, "status": "FAILED", "error": "No certified KPIs"}

        run_id = state["run_id"]
        fingerprint = state.get("fingerprint", run_id)
        certify_hitl_result(run_id, certified_kpis, fingerprint)

        logger.info("HITL certified %d KPIs to ai_store", len(certified_kpis), extra=log_context)
        new_state = state.copy()
        new_state["status"] = "GATE1_COMPLETE"
        return new_state

    return hitl_review_node


def build_hitl_table_review_node() -> Callable[[Stage01State], Stage01State]:
    def hitl_table_review_node(state: Stage01State) -> Stage01State:
        log_context = {"run_id": state.get("run_id", "unknown"), "node": "hitl_table_review"}
        human_table_decision = state.get("human_table_decision")

        if human_table_decision != "COMPLETED":
            logger.info("HITL table review skipped - decision pending", extra=log_context)
            return state

        certified_tables = state.get("certified_tables")
        if not certified_tables:
            logger.warning("No certified tables found despite COMPLETED decision", extra=log_context)
            return {**state, "status": "FAILED", "error": "No certified tables after Gate 2"}

        run_id = state["run_id"]
        fingerprint = state.get("fingerprint", run_id)
        certify_hitl_tables(run_id, certified_tables, fingerprint)

        logger.info("HITL certified %d tables to ai_store", len(certified_tables), extra=log_context)
        new_state = state.copy()
        new_state["status"] = "GATE2_COMPLETE"
        return new_state

    return hitl_table_review_node


hitl_review_node = build_hitl_review_node()
hitl_table_review_node = build_hitl_table_review_node()
