from typing import TypedDict, Optional, Dict, Any, List


class Stage01State(TypedDict, total=False):
    """
    Shared pipeline state passed between all LangGraph nodes.
    Fields are progressively populated by each stage.
    """

    # ── Core pipeline metadata ─────────────────────────────
    run_id: Optional[str]
    brd_text: str
    token_estimate: int
    fingerprint: str
    metadata: Dict[str, Any]
    status: str
    error: Optional[str]

    # ── Embedding / Vectorization Flags (NEW) ──────────────
    brd_embedded: Optional[bool]          # BRD stored in ai-store-index
    schema_embedded: Optional[bool]       # schema stored in metadata index
    schema_columns_count: Optional[int]   # total columns embedded

    # ── Stage 02: Requirements Extraction outputs ──────────
    req_business_objective: Optional[str]
    req_data_domains: Optional[List[str]]
    req_reporting_frequency: Optional[str]
    req_target_audience: Optional[str]
    req_constraints: Optional[List[str]]
    req_schema_valid: Optional[bool]
    req_prompt_version: Optional[str]
    req_agent_attempts: Optional[int]
    req_tokens_used: Optional[int]
    req_cost_usd: Optional[float]
    req_faithfulness_status: Optional[str]

    # ── Stage 03: KPI Extraction memory flags ──────────────
    memory_layer1: Optional[bool]
    memory_layer2: Optional[bool]
    prior_kpis: Optional[List[Dict]]
    context_kpis: Optional[List[Dict]]
    rejected_kpis: Optional[List[str]]

    # ── Stage 03: KPI Extraction outputs ───────────────────
    kpis: Optional[List[Dict]]
    kpi_source: Optional[str]  # "MEMORY_LAYER1" | "MEMORY_LAYER2" | "LLM"
    kpi_tokens_used: Optional[int]
    kpi_cost_usd: Optional[float]
    kpi_attempts: Optional[int]

    # ── HITL Fields ────────────────────────────────────────
    extracted_kpis: Optional[List[Dict]]
    human_decision: Optional[str]  # 'PENDING' | 'COMPLETED'
    certified_kpis: Optional[List[Dict]]

    # ── Stage 04: Table Nomination inputs ──────────────────
    source_databases: Optional[List[str]]

    # ── Stage 04: Semantic Search (NEW) ────────────────────
    semantic_matches: Optional[List[Dict]]   # raw vector matches
    semantic_top_k: Optional[int]            # number of matches fetched

    # ── Stage 04: Table Nomination outputs ─────────────────
    nominated_tables: Optional[List[Dict]]
    table_nomination_status: Optional[str]   # 'PENDING' | 'COMPLETE' | 'FAILED'
    table_nomination_error: Optional[str]

    # ── Stage 05: Gate 2 HITL Table Review ─────────────────
    human_table_decision: Optional[str]  # 'PENDING' | 'COMPLETED'
    certified_tables: Optional[List[Dict]]

    discovered_metadata: Optional[Dict[str, Any]]
    metadata_status: Optional[str]  # 'PENDING' | 'COMPLETED' | 'FAILED' | 'SKIPPED'
    metadata_error: Optional[str]

    column_profiles: Optional[Dict[str, Any]]
    column_profiling_status: Optional[str]  # 'COMPLETED' | 'COMPLETED_WITH_WARNINGS' | 'FAILED' | 'SKIPPED'
    column_profiling_error: Optional[str]
