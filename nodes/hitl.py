"""
HITL Review Node for LangGraph.
Placed after interrupt. Checks human_decision and certifies to ai_store if COMPLETED.
"""
from typing import Callable, List, Dict
from state import Stage01State
from utilis.db import ai_store_db_writer
from utilis.logger import logger


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
    """
    Gate 2 — Certify table nominations to ai_store after human review.
    """
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
        
        logger.info(f"HITL certified {len(certified_kpis)} KPIs to ai_store", extra=log_context)
        new_state = state.copy()
        new_state["status"] = "GATE1_COMPLETE"
        return new_state
    
    return hitl_review_node


def build_hitl_table_review_node() -> Callable[[Stage01State], Stage01State]:
    """
    Gate 2 HITL Review — certifies table nominations after human review.
    Must be placed after interrupt, following table_nomination_node.
    """
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

        logger.info(f"HITL certified {len(certified_tables)} tables to ai_store", extra=log_context)
        new_state = state.copy()
        new_state["status"] = "GATE2_COMPLETE"
        return new_state

    return hitl_table_review_node


# Singleton instances for direct import
hitl_review_node = build_hitl_review_node()
hitl_table_review_node = build_hitl_table_review_node()
