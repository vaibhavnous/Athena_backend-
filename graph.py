from langgraph.graph import END, StateGraph
from langgraph.graph.state import StateGraph as _StateGraph  # Avoid name conflict
from langgraph.checkpoint.memory import MemorySaver

from nodes.ingestion import ingestion_node
from nodes.memory_lookup import memory_lookup_node
from nodes.req_extraction import build_req_extraction_node
from nodes.kpi_extraction import kpi_extraction_node
from nodes.hitl import hitl_review_node, hitl_table_review_node
from nodes.metadata_discovery import metadata_discovery_node
from nodes.column_profiling import column_profiling_node
from nodes.table_nomination import table_nomination_node
from nodes.semantic_enrichment import semantic_enrichment_node
from nodes.hitl import build_hitl_enrichment_review_node
from nodes.bronze_gen import bronze_code_generation_node
from nodes.silver_gen import silver_code_generation_node
from state import Stage01State

# Build nodes
req_extraction_node = build_req_extraction_node(llm_provider="azure_openai")

workflow = StateGraph(Stage01State)

workflow.add_node("stage01_ingestion", ingestion_node)
workflow.add_node("memory_lookup", memory_lookup_node)
workflow.add_node("req_extraction", req_extraction_node)
workflow.add_node("kpi_extraction", kpi_extraction_node)
workflow.add_node("hitl_review", hitl_review_node)
workflow.add_node("table_nomination", table_nomination_node)
workflow.add_node("hitl_table_review", hitl_table_review_node)
workflow.add_node("semantic_enrichment", semantic_enrichment_node)
workflow.add_node("enrichment_review", build_hitl_enrichment_review_node())
workflow.add_node("metadata_discovery", metadata_discovery_node)
workflow.add_node("column_profiling", column_profiling_node)
workflow.add_node("bronze_code_generation", bronze_code_generation_node)
workflow.add_node("silver_code_generation", silver_code_generation_node)

workflow.set_entry_point("stage01_ingestion")

# ── Edges ──────────────────────────────────────────────

workflow.add_edge("stage01_ingestion", "memory_lookup")

# Conditional after memory_lookup
def should_skip_extraction(state):
    if state.get("status") == "FAILED":
        return END
    return "req_extraction"

workflow.add_conditional_edges(
    "memory_lookup",
    should_skip_extraction,
)

workflow.add_edge("req_extraction", "kpi_extraction")
workflow.add_edge("kpi_extraction", "hitl_review")

# Conditional after Gate 1 HITL
def route_after_gate1(state):
    if state.get("status") == "FAILED":
        return END
    if state.get("human_decision") != "COMPLETED":
        return END
    return "table_nomination"

workflow.add_conditional_edges(
    "hitl_review",
    route_after_gate1,
)


workflow.add_edge("table_nomination", "hitl_table_review")
workflow.add_edge("hitl_table_review", "metadata_discovery")
workflow.add_edge("metadata_discovery", "column_profiling")
workflow.add_edge("column_profiling", "semantic_enrichment")
workflow.add_edge("semantic_enrichment", "enrichment_review")
workflow.add_edge("enrichment_review", "bronze_code_generation")
workflow.add_edge("bronze_code_generation", "silver_code_generation")
workflow.add_edge("silver_code_generation", END)

# ── Compile with Checkpointer & HITL Interrupts ────────

checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["hitl_review", "hitl_table_review", "enrichment_review"]
)
