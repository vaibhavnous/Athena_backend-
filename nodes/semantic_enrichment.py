"""
NB09 — Semantic Enrichment Node

Responsibilities:
- Column semantic classification (rule-first, LLM optional)
- Explicit join metadata capture (declarative, not executed)
- Explicit aggregation policy capture (rules only)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from state import Stage01State
from utilis.logger import logger
from utilis.db import ai_store_db_writer


# ------------------------------------------------------------------------------------
# Semantic Types
# ------------------------------------------------------------------------------------

SemanticType = Literal[
    "MEASURE",
    "DIMENSION",
    "DATE",
    "ID",
    "PII",
    "FLAG",
    "HIGH_CARD_TEXT",
    "AUDIT_TIMESTAMP",
    "SURROGATE_KEY",
    "UNKNOWN",
]

AggType = Literal["SUM", "AVG", "COUNT", "MIN", "MAX", "NONE"]


# ------------------------------------------------------------------------------------
# RULE-BASED SEMANTIC CLASSIFICATION
# ------------------------------------------------------------------------------------

def rule_based_semantic_classification(column: Dict[str, Any]) -> Dict[str, Any]:
    name = str(column.get("column_name", "")).lower()
    data_type = str(column.get("data_type", "")).lower()
    cardinality = column.get("cardinality")

    semantic: SemanticType = "UNKNOWN"
    suggested_agg: AggType = "NONE"

    if name.endswith("_id") or name == "id":
        semantic = "ID"
    elif name.startswith(("is_", "has_")) or data_type == "bit":
        semantic = "FLAG"
        suggested_agg = "COUNT"
    elif name in {"created_at", "updated_at", "modified_at"}:
        semantic = "AUDIT_TIMESTAMP"
    elif data_type in {"date", "datetime", "datetime2"}:
        semantic = "DATE"
    elif data_type in {"int", "bigint", "decimal", "numeric", "float"}:
        semantic = "MEASURE"
        suggested_agg = "SUM"
    elif data_type in {"varchar", "nvarchar", "text"}:
        semantic = "DIMENSION"
        if cardinality and cardinality > 1000:
            semantic = "HIGH_CARD_TEXT"

    is_pii = any(
        k in name for k in ("email", "phone", "mobile", "ssn", "pan", "aadhaar")
    )

    return {
        "semantic_type": semantic,
        "is_measure": semantic == "MEASURE",
        "is_dimension": semantic in {"DIMENSION", "HIGH_CARD_TEXT"},
        "is_pii_candidate": is_pii,
        "suggested_aggregation": suggested_agg,
        "needs_llm": semantic in {"UNKNOWN", "MEASURE", "DIMENSION"},
    }


# ------------------------------------------------------------------------------------
# AGGREGATION POLICY (DECLARATIVE)
# ------------------------------------------------------------------------------------

def build_aggregation_policy(column: Dict[str, Any]) -> Dict[str, Any]:
    semantic = column.get("semantic_type")
    data_type = str(column.get("data_type", "")).lower()
    cardinality = column.get("cardinality")

    policy = {
        "allowed": False,
        "recommended_aggregations": [],
        "forbidden_aggregations": [],
        "requires_deduplication": False,
        "confidence": 0.9,
    }

    if semantic == "MEASURE":
        policy["allowed"] = True
        policy["recommended_aggregations"] = ["SUM"]
        policy["forbidden_aggregations"] = ["COUNT"]
        if cardinality and cardinality > 1_000_000:
            policy["requires_deduplication"] = True

    elif semantic == "FLAG":
        policy["allowed"] = True
        policy["recommended_aggregations"] = ["COUNT"]
        policy["forbidden_aggregations"] = ["SUM", "AVG"]

    elif semantic in {"ID", "SURROGATE_KEY"}:
        policy["allowed"] = False
        policy["confidence"] = 1.0

    elif semantic in {"DATE", "AUDIT_TIMESTAMP"}:
        policy["allowed"] = False

    return policy


# ------------------------------------------------------------------------------------
# OPTIONAL LLM ENRICHMENT (ABSTRACTED)
# ------------------------------------------------------------------------------------

def llm_enrich_column(column: Dict[str, Any], domain_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for LLM enrichment (Azure OpenAI / OpenAI).
    """
    display = column["column_name"].replace("_", " ").title()

    return {
        "business_description": f"{display} used for {domain_context.get('business_objective', 'analytics')}.",
        "suggested_display_name": display,
        "suggested_join_key_for": None,
    }


# ------------------------------------------------------------------------------------
# JOIN DISCOVERY (SAFE, RULE-BASED)
# ------------------------------------------------------------------------------------

def discover_joins(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    joins: List[Dict[str, Any]] = []
    index: Dict[str, List[Dict[str, Any]]] = {}

    for table in tables:
        for col in table["columns"]:
            if col.get("semantic_type") in {"ID", "SURROGATE_KEY"}:
                index.setdefault(col["column_name"], []).append({
                    "table": table["table_name"],
                    "column": col["column_name"],
                    "cardinality": col.get("cardinality"),
                })

    for col_name, refs in index.items():
        if len(refs) < 2:
            continue

        for left in refs:
            for right in refs:
                if left["table"] == right["table"]:
                    continue

                joins.append({
                    "left_table": left["table"],
                    "left_column": left["column"],
                    "right_table": right["table"],
                    "right_column": right["column"],
                    "cardinality": "MANY_TO_ONE",
                    "join_type": "INNER",
                    "confidence": 0.8,
                    "source": "RULES",
                    "certified": False,
                })

    return joins


# ------------------------------------------------------------------------------------
# COLUMN ENRICHMENT ORCHESTRATION
# ------------------------------------------------------------------------------------

def enrich_column(column: Dict[str, Any], domain_context: Dict[str, Any]) -> Dict[str, Any]:
    enriched = {**column}

    rule_result = rule_based_semantic_classification(column)
    enriched.update(rule_result)

    if column.get("embedding_version") != "ENRICHED" and rule_result["needs_llm"]:
        enriched.update(llm_enrich_column(column, domain_context))
        enriched["enrichment_source"] = "LLM"
    else:
        enriched["enrichment_source"] = "RULES"

    enriched["aggregation_policy"] = build_aggregation_policy(enriched)
    return enriched


# ------------------------------------------------------------------------------------
# LANGGRAPH NODE
# ------------------------------------------------------------------------------------

def semantic_enrichment_node(state: Stage01State) -> Stage01State:
    new_state = state.copy()
    discovered = state.get("discovered_metadata", {})
    profiling = state.get("column_profiles", {})

    domain_context = {
        "business_objective": state.get("business_objective"),
        "data_domains": state.get("data_domains"),
    }

    enriched_columns: List[Dict[str, Any]] = []
    enriched_tables: List[Dict[str, Any]] = []

    for table in discovered.get("tables", []):
        cols = []
        for col in table.get("columns", []):
            profile = next(
                (p for p in profiling.get("column_profiles", [])
                 if p["column_name"] == col["column_name"]
                 and p["table_name"] == table["table_name"]),
                {},
            )
            merged = {**col, **profile, "table_name": table["table_name"]}
            enriched = enrich_column(merged, domain_context)
            cols.append(enriched)
            enriched_columns.append(enriched)

        enriched_tables.append({
            "table_name": table["table_name"],
            "columns": cols,
        })

    joins = discover_joins(enriched_tables)

    payload = {
        "run_id": state.get("run_id"),
        "fingerprint": state.get("fingerprint"),
        "certified_tables": state.get("certified_tables", []),
        "enriched_at": datetime.now(timezone.utc).isoformat(),
        "columns": enriched_columns,
        "joins": joins,
    }

    ai_store_db_writer(
        run_id=state.get("run_id"),
        stage="Semantic Enrichment",
        artifact_type="ENRICHED_METADATA",
        payload=payload,
        schema_version="SemanticEnrichment_v1",
        prompt_version="NB09_SEMANTIC_ENRICHMENT_v1",
        faithfulness_status="NOT_APPLICABLE",
        token_count=0,
        input_tokens=0,
        output_tokens=0,
        fingerprint=state.get("fingerprint"),
    )

    new_state["enriched_metadata"] = payload
    new_state["semantic_enrichment_status"] = "COMPLETED"
    return new_state


# ------------------------------------------------------------------------------------
# GRAPH BUILDER
# ------------------------------------------------------------------------------------

def build_semantic_enrichment_graph() -> StateGraph:
    graph = StateGraph(Stage01State)
    graph.add_node("semantic_enrichment", semantic_enrichment_node)
    graph.set_entry_point("semantic_enrichment")
    graph.set_finish_point("semantic_enrichment")
    return graph


def compile_semantic_enrichment_graph():
    return build_semantic_enrichment_graph().compile(checkpointer=MemorySaver())
