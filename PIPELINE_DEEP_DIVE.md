# Athena Pipeline Deep Dive

This document explains the Athena pipeline in depth so you can walk someone through:

- what the pipeline is trying to do,
- how control flows from node to node,
- what each node writes into shared state,
- what each helper function is responsible for,
- where Human-in-the-Loop (HITL) pauses happen,
- what gets stored in Azure SQL and Pinecone,
- and where the current design has shortcomings.

## 1. Big Picture

The pipeline starts from a BRD-like input and tries to convert it into:

1. structured business requirements,
2. candidate KPIs,
3. human-certified KPIs,
4. candidate source tables,
5. human-certified tables,
6. final discovered metadata for those tables,
7. deterministic column profiles for the discovered columns.

It uses a mix of:

- LangGraph for orchestration,
- Azure SQL for pipeline persistence and HITL queueing,
- Pinecone for vector search,
- sentence-transformer embeddings for semantic retrieval,
- Azure OpenAI for LLM extraction.

At a very high level:

1. ingest the BRD and embed it,
2. check whether we have seen the same BRD before,
3. extract requirements,
4. extract KPIs,
5. pause for human KPI review,
6. nominate tables,
7. pause for human table review,
8. discover detailed metadata,
9. profile the discovered columns.

## 2. Core Orchestration

The graph is defined in `graph.py`.

### Node sequence

The workflow is:

1. `stage01_ingestion`
2. `memory_lookup`
3. `req_extraction`
4. `kpi_extraction`
5. `hitl_review`
6. `table_nomination`
7. `hitl_table_review`
8. `metadata_discovery`
9. `column_profiling`

### Important graph behavior

- The graph entry point is `stage01_ingestion`.
- After `memory_lookup`, the graph only checks whether status is `FAILED`. If not failed, it always routes to `req_extraction`.
- After `kpi_extraction`, the graph goes to `hitl_review`.
- `interrupt_before=["hitl_review", "hitl_table_review"]` means LangGraph pauses before both human-review nodes.
- After human KPI certification, the graph continues into table nomination.
- After human table certification, the graph continues into metadata discovery and then column profiling.

### Key implication

Memory reuse is not implemented as a graph-level skip. Instead, the graph still enters downstream nodes, and those nodes inspect flags like `memory_bypass` and `memory_layer1` to decide whether to reuse old results.

That means orchestration is partly graph-driven and partly embedded inside node logic.

## 3. Shared State Model

The shared contract is `Stage01State` in `state.py`.

This state is the backbone of the whole system. Every node reads from it and writes back into it.

### Core fields

- `run_id`: unique execution identifier
- `brd_text`: parsed BRD content
- `token_estimate`: rough token count
- `fingerprint`: SHA-256 of BRD text
- `metadata`: generic metadata bag
- `status`: pipeline status
- `error`: pipeline error if any

### Memory-related fields

- `memory_layer1`: exact fingerprint hit
- `memory_layer2`: semantic memory hit
- `prior_kpis`: KPIs reused from prior exact match
- `context_kpis`: semantically related KPIs
- `rejected_kpis`: low-confidence historical KPIs used as negative memory

### Requirement output fields

- `req_business_objective`
- `req_data_domains`
- `req_reporting_frequency`
- `req_target_audience`
- `req_constraints`
- `req_schema_valid`
- `req_prompt_version`
- `req_agent_attempts`
- `req_tokens_used`
- `req_cost_usd`
- `req_faithfulness_status`

### KPI output fields

- `kpis`
- `kpi_source`
- `kpi_tokens_used`
- `kpi_cost_usd`
- `kpi_attempts`

### HITL fields

- `extracted_kpis`
- `human_decision`
- `certified_kpis`
- `human_table_decision`
- `certified_tables`

### Table and metadata fields

- `source_databases`
- `semantic_matches`
- `semantic_top_k`
- `nominated_tables`
- `table_nomination_status`
- `table_nomination_error`
- `discovered_metadata`
- `metadata_status`
- `metadata_error`

### State drift worth noticing

`Stage01State` also declares:

- `brd_embedded`
- `schema_embedded`
- `schema_columns_count`

These suggest intended observability, but current code does not populate them.

## 4. CLI and Runtime Entry

The main runtime entry is `cli.py`.

### What the CLI does

- parses arguments,
- builds the initial state,
- imports the compiled LangGraph app,
- invokes the graph with `thread_id=run_id`,
- displays summaries,
- optionally launches Gate 1 or Gate 2 review interactively.

### Initial state creation

`build_initial_state()` constructs:

- `brd_text`
- `run_id`
- `metadata = {}`
- `status = "PENDING"`
- `source_databases`

### Important runtime detail

The graph uses a LangGraph checkpointer, but the human resume path is largely handled manually by the CLI. After a human review, the CLI loads checkpoint state from SQL and then directly calls the HITL node and the next business node, rather than simply resuming the entire graph from LangGraph.

This is important when explaining the system because it is not a pure “single graph execution” design.

## 5. Node 1: Ingestion

File: `nodes/ingestion.py`

Purpose: parse the BRD, validate it, estimate tokens, fingerprint it, store initial run info, embed the BRD, and embed schema metadata.

### Main function

`ingestion_node(state)`

It runs the following helpers in order:

1. `_parse_input`
2. `_acquire_and_validate_brd`
3. `_estimate_and_fingerprint`
4. `_validate_budget`
5. `_chunk_and_embed`
6. `_embed_schema_metadata`

The functions `_validate_pricing_config`, `_validate_schema`, and `_store_and_register` are not called directly here; they are deferred and later triggered from `memory_lookup` via `finalize_ingestion_after_memory`.

### Helper: `_parse_input`

Responsibilities:

- Detect whether `brd_text` is raw text or a file path.
- If the input path exists:
  - parse `.txt`,
  - parse `.docx` paragraphs and tables,
  - otherwise fall back to raw string.
- Write parsed text back into `new_state["brd_text"]`.

Why it matters:

- This is the conversion boundary between external input and normalized BRD text.

### Helper: `_acquire_and_validate_brd`

Responsibilities:

- Ensure a `run_id` exists.
- Check that BRD text is present.
- Check that BRD length is at least 200 characters.
- Mark pipeline as `IN_PROGRESS` if valid.
- Mark pipeline as `FAILED` if empty or too short.

Why it matters:

- This is the earliest business validation checkpoint.

### Helper: `_estimate_and_fingerprint`

Responsibilities:

- Estimate token count using `tiktoken` if available.
- Fall back to a rough `len(text) // 4` estimate if needed.
- Generate SHA-256 fingerprint of BRD text.

Why it matters:

- The fingerprint is a central identity key used throughout memory lookup and persistence.

### Helper: `_validate_budget`

Responsibilities:

- Compare estimated tokens against `TOKEN_BUDGET = 50000`.
- Fail the pipeline if the estimate exceeds budget.

Why it matters:

- This is a cost-control gate.

### Helper: `_validate_pricing_config`

Responsibilities:

- Add estimated pricing metadata to `state["metadata"]`.
- Set `pricing_validated = True`.

Important:

- This does not perform any external pricing lookup.
- It just attaches hardcoded pricing assumptions.

### Helper: `_validate_schema`

Responsibilities:

- Validate BRD content against `BRDSchema` in `schema.py`.
- `BRDSchema` enforces:
  - min length 200,
  - presence of at least one business keyword.

Why it matters:

- This is not SQL schema validation.
- It is validation of whether the input text looks like a business requirements document.

### Helper: `_store_and_register`

Responsibilities:

- Insert a run entry into `brd_run_registry`.
- Insert or update a BRD entry in pipeline `ai_store`.

What it stores:

- `run_id`
- pipeline status
- token count
- raw BRD text
- metadata JSON
- fingerprint

Important nuance:

- This stage writes into pipeline storage after memory logic is done, not inside the first ingestion execution path.

### Helper: `_chunk_and_embed`

Responsibilities:

- Split BRD text into chunks with overlap.
- Generate embeddings with `_embedding_model`.
- Upsert those chunks into Pinecone namespace `global`.

Metadata stored per vector:

- `run_id`
- `fingerprint`
- `source = "BRD"`

Why it matters:

- This is the vector memory base used for semantic recall.

### Helper: `_embed_schema_metadata`

Responsibilities:

- Query source DB `INFORMATION_SCHEMA.COLUMNS`.
- For each column, create a semantic sentence like:
  - `Table claims contains column claim_amount`
- Embed those sentences.
- Upsert into Pinecone index `metadata`, namespace `schema`.

Metadata stored per schema vector:

- `database_name`
- `schema_name`
- `table_name`
- `column_name`
- `type = "schema"`

Why it matters:

- This is the retrieval layer that later grounds KPI extraction and table nomination.

### Ingestion shortcomings

1. Schema embedding only captures columns, not PK/FK or relationship metadata.
2. `_embed_schema_metadata` clears the entire `schema` namespace with `delete_all=True` before upsert, so one run can wipe another run’s schema vectors.
3. The state fields `brd_embedded`, `schema_embedded`, and `schema_columns_count` are never updated.
4. Schema scans can be broader than intended because not all queries are consistently schema-filtered.

## 6. Node 2: Memory Lookup

File: `nodes/memory_lookup.py`

Purpose: detect exact or semantic prior matches, reuse prior artifacts where possible, and complete the delayed ingestion finalization work.

### Main function

`memory_lookup_node(state)`

Flow:

1. If status is `FAILED`, return immediately.
2. Run `_fetch_exact_match`.
3. Apply result using `_apply_match_result`.
4. If not bypassing memory, run `_run_semantic_lookup`.
5. If not bypassing memory, run:
   - `_chunk_and_embed`
   - `finalize_ingestion_after_memory`

### Helper: `_fetch_latest_payload`

Responsibilities:

- Query pipeline `ai_store` for the newest payload by:
  - `fingerprint`
  - `artifact_type`

Used for:

- exact-memory requirement fetch,
- exact-memory KPI fetch,
- semantic-context KPI fetch.

### Helper: `_fetch_exact_match`

Responsibilities:

- Retrieve latest `REQUIREMENTS` or `REQUIREMENTS_WARN`.
- Retrieve latest `KPIS`.
- Return:
  - whether any usable exact match exists,
  - requirements payload,
  - KPI payload.

How it decides a match:

- requirement payload is considered valid if it has `business_objective`
- KPI payload is considered valid if it has `kpis`

### Helper: `_apply_match_result`

Responsibilities:

- If exact match exists:
  - set `memory_layer1 = True`
  - set `memory_bypass = True`
  - hydrate state with prior requirements
  - hydrate state with prior KPIs
  - set `status = "EXACT_MATCH_FOUND"`
- If no exact match:
  - set `memory_layer1 = False`
  - set `memory_bypass = False`
  - set `status = "NO_EXACT_MATCH"`

Why it matters:

- This is where exact-match reuse becomes part of the state contract.

### Helper: `_run_semantic_lookup`

Responsibilities:

- Embed the BRD text with `SentenceTransformer`.
- Query Pinecone namespace `global`.
- If top score is at least `0.75`, mark `memory_layer2 = True`.
- Fetch related KPI context using `_fetch_context_kpis`.
- Fetch historically rejected/low-confidence KPIs via `_fetch_rejected_kpis`.

Why it matters:

- This is “soft memory”: not exact reuse, but prior context to bias extraction.

### Helper: `_fetch_context_kpis`

Responsibilities:

- Query Pinecone for nearby vectors.
- Try to find matched items whose metadata contains `fingerprint`.
- Pull the corresponding KPI payloads from `ai_store`.

### Helper: `_fetch_rejected_kpis`

Responsibilities:

- Read historical KPI artifacts from `ai_store`.
- Gather KPI names with confidence below `0.3`.
- Use them as a negative-memory list to avoid repeating bad KPIs.

### Memory lookup shortcomings

1. The semantic context retrieval likely underperforms because BRD chunk vectors do not appear to store `artifact_type="KPIS"`, yet `_fetch_context_kpis` filters by that field.
2. This node reruns `_chunk_and_embed` even though ingestion already embedded the BRD, creating duplication.
3. Exact memory depends on `ai_store` artifact integrity, but `ai_store` currently overwrites by fingerprint, which weakens multi-artifact retrieval.

## 7. Node 3: Requirement Extraction

File: `nodes/req_extraction.py`

Purpose: convert BRD text into a strict structured requirements object.

### Main function

`req_extraction_node(state)`

Flow:

1. Fail fast if `status == FAILED`.
2. Validate handoff using `handoff_validator`.
3. If `memory_bypass` is true:
   - reuse exact-match requirements,
   - write a fresh `REQUIREMENTS` artifact to DB,
   - return.
4. Otherwise:
   - build LLM prompt,
   - call LLM,
   - parse JSON,
   - validate with `RequirementsSchema`,
   - run faithfulness check,
   - retry if constraints appear ungrounded,
   - write result to DB,
   - update state.

### Helper: `get_llm`

Responsibilities:

- Build the correct LangChain chat model.
- Supports:
  - `azure_openai`
  - `openai`

### Helper: `TokenAccumulator`

Responsibilities:

- Track prompt and completion token usage from LLM callbacks.

Why it matters:

- This drives cost tracking and observability.

### Helper: `compute_cost_usd`

Responsibilities:

- Convert token usage into a rough cost based on hardcoded pricing tables.

### Helper: `handoff_validator`

Responsibilities:

- Ensure required state keys exist before node execution.

Why it matters:

- It protects against silent stage-to-stage state drift.

### Helper: `check_faithfulness`

Responsibilities:

- Check whether each extracted constraint seems to be grounded in the BRD text.
- Uses a simple keyword overlap heuristic.

Important nuance:

- This is a lightweight grounding check, not a robust semantic fact-check.

### Prompt design

The prompt forces a strict JSON object with:

- `business_objective`
- `data_domains`
- `reporting_frequency`
- `target_audience`
- `constraints`

Validation then happens through `RequirementsSchema`.

### Output schema

`RequirementsSchema` in `schema.py` enforces:

- `business_objective`: min length 10
- `reporting_frequency`: one of `daily|weekly|monthly|quarterly|adhoc`
- `target_audience`: min length 5
- default fields:
  - `schema_valid = True`
  - `prompt_version = "PROMPT_REQ_v1"`

### Requirement extraction shortcomings

1. The faithfulness check is heuristic and word-based, so it can miss paraphrases and produce false warnings.
2. `schema_valid` inside requirements is effectively static metadata, not a dynamically computed truth.
3. Memory-bypass mode writes a new `REQUIREMENTS` artifact even though it is just reusing old data, which can blur provenance.

## 8. Node 4: KPI Extraction

File: `nodes/kpi_extraction.py`

Purpose: turn structured requirements into measurable KPIs grounded in available source schema.

### Main function

`kpi_extraction_node(state)`

Flow:

1. Fail fast on `status == FAILED`.
2. Validate required handoff keys.
3. Build requirements object from state.
4. Resolve source DBs.
5. Pull relevant schema context from Pinecone metadata index.
6. If exact-memory KPIs exist, reuse them.
7. Otherwise:
   - prompt the LLM,
   - parse output,
   - validate each KPI,
   - deduplicate,
   - filter rejected names,
   - grounding-check the results,
   - fallback to a weak synthetic KPI if final retry still yields none.
8. Write KPI artifact.
9. Insert KPIs into Gate 1 HITL queue.
10. Save full state checkpoint into SQL.

### Helper: `_build_requirements`

Responsibilities:

- Convert individual `req_*` state fields into a single dict for prompting.

### Helper: `_resolve_source_databases`

Responsibilities:

- Use `state["source_databases"]` if present.
- Otherwise fall back to config defaults.

### Helper: `_fetch_relevant_schema`

Responsibilities:

- Embed BRD text and query Pinecone index `metadata`, namespace `schema`.
- Filter matches down to allowed source DBs.
- Group matches by table and collect top columns.

Why it matters:

- This is the schema grounding layer for KPI extraction.

### Helper: `_format_schema_context`

Responsibilities:

- Render schema retrieval output into prompt-ready text.

Example shape:

- `Available Data Schema:`
- `- claims (claim_id, claim_amount, claim_date, ...)`

### Helper: `_is_measurable_kpi`

Responsibilities:

- Reject KPI text that appears abstract.
- Require measurable vocabulary like:
  - time
  - rate
  - count
  - amount
  - ratio
  - percentage

### Helper: `_grounding_check`

Responsibilities:

- Score whether KPI wording is grounded in requirements or BRD text.
- Keep KPI if:
  - grounding is found, or
  - confidence is at least `0.6`
- Add `grounding = STRONG|WEAK`.

### Helper: `_remove_duplicates_and_rejected`

Responsibilities:

- Deduplicate by `kpi_name`.
- Remove KPIs whose names appear in `rejected_kpis`.

### KPI schema

Each KPI is validated against `KPISchemaItem` in `schema.py`.

It enforces:

- `kpi_name`
- `kpi_description`
- `ai_confidence_score`
- `derivation_type`
- `source_requirement_ref`

And it has an important model rule:

- `explicit` KPIs must have confidence >= `0.7`
- `implicit` KPIs must have confidence >= `0.4`

### HITL integration

This node inserts every extracted KPI into `hitl_review_queue` with `gate_number=1`.

It also sets:

- `extracted_kpis`
- `human_decision = "PENDING"`

That prepares the graph to stop at Gate 1.

### KPI extraction shortcomings

1. The semantic schema grounding uses only table and column embeddings, not relationships or business semantics.
2. Final retry behavior includes `FINAL: Force at least 3 valid KPIs`, which can encourage low-quality salvage generation.
3. If everything fails on the last attempt, the node can fabricate a weak fallback KPI from the business objective.
4. The node logs “Skipping kpi_memory insert; current schema does not support detailed KPI columns”, which hints that memory persistence is incomplete.

## 9. Node 5: HITL Review

File: `nodes/hitl.py`

Purpose: certify human-reviewed KPIs after Gate 1.

### Main function

`hitl_review_node(state)`

Flow:

1. If `human_decision != COMPLETED`, return state untouched.
2. Ensure `certified_kpis` exists.
3. Write `GATE1_CERTIFIED_KPIS` artifact.
4. Set `status = "GATE1_COMPLETE"`.

### Helper: `certify_hitl_result`

Responsibilities:

- Write human-certified KPI payload to `ai_store`.

Why it matters:

- This creates the audited human-approved output for Gate 1.

### Gate 1 lifecycle in practice

Although LangGraph pauses before this node, the actual human review is handled in `cli.py`:

1. Read pending queue items from SQL.
2. Present them to the operator.
3. Approve, reject, or edit them.
4. Save decisions back to SQL.
5. Reload checkpoint state.
6. Manually call `hitl_review_node`.

So the human review flow is implemented mostly outside the graph.

## 10. Node 6: Table Nomination

File: `nodes/table_nomination.py`

Purpose: nominate candidate source tables that can support the certified KPIs.

This is one of the richest nodes in the system.

### Main function

`table_nomination_node(state)`

Flow:

1. Fail if state is already failed.
2. Require `certified_kpis`.
3. Resolve `source_databases`.
4. Extract KPI names.
5. Build domain keywords and synonym-expanded variants.
6. Run lexical search.
7. Run semantic search.
8. Fuse lexical and semantic scores.
9. Expand with FK-linked tables.
10. Expand with lookup-table sweep.
11. Validate nominations with `NominationSchema`.
12. Write `TABLE_NOMINATIONS` artifact.
13. Save state checkpoint.
14. Set:
   - `nominated_tables`
   - `table_nomination_status = "PENDING"`
   - `human_table_decision = "PENDING"`

### Helper: `_extract_kpi_names`

Responsibilities:

- Normalize KPI names whether items are strings or dicts.

### Helper: `_build_keywords`

Responsibilities:

- Tokenize KPI names into lowercase keyword candidates.
- Keep tokens with length >= 3.

### Helper: `_normalize` and `_tokenize_identifier`

Responsibilities:

- Normalize table/column identifiers into underscore-separated tokens.
- Support cross-style matching like camelCase, snake_case, and mixed punctuation.

### Helper: `_expand_keywords`

Responsibilities:

- Add synonym sets for business terms like:
  - claim
  - premium
  - policy
  - customer
  - revenue
  - ratio
  - identifier

Why it matters:

- This is a major reason lexical nomination is stronger than plain keyword search.

### Helper: `_lexical_search`

Responsibilities:

- Query `INFORMATION_SCHEMA.TABLES` joined to `INFORMATION_SCHEMA.COLUMNS`.
- Match keyword variants against table and column names.
- Build per-table lexical scores.
- Track:
  - matched keywords
  - matched columns
  - coverage ratio
  - token-based relevance

### Helper: `_semantic_search`

Responsibilities:

- Query Pinecone `metadata` index using KPI names as semantic queries.
- Aggregate per-table semantic scores from matched column vectors.

### Helper: `_fuse_results`

Responsibilities:

- Combine lexical score, semantic score, coverage ratio, and token frequency.
- Penalize tables without domain overlap.
- Boost tables with both lexical and semantic evidence.
- Normalize scores.
- Assign human-readable nomination reasons.

### Helper: `_fk_resolution`

Responsibilities:

- Query:
  - `INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS`
  - `INFORMATION_SCHEMA.TABLE_CONSTRAINTS`
- Discover tables connected by foreign key relationships.
- Add them as lower-confidence nominations.

This is the one place in the pipeline where FK-based relationship discovery is explicitly used.

### Helper: `_lookup_table_sweep`

Responsibilities:

- Search for likely dimensional/reference tables by prefixes such as:
  - `dim_`
  - `ref_`
  - `lkp_`
  - `lookup_`
  - `code_`
  - `type_`
- Limit to smaller tables based on row count.
- Keep only those with domain-token overlap.

### Table nomination schema

`NominationItem` in `schema.py` enforces:

- `table_name`
- `schema_name`
- `database_name`
- `nomination_reason`
- `confidence_score`
- `matched_keywords`
- `coverage_ratio`

### Table nomination strengths

- Strongest heuristic stage in the pipeline.
- Uses both symbolic and semantic methods.
- Includes relationship expansion.
- Includes lookup dimension sweep.

### Table nomination shortcomings

1. No column-level PK/FK metadata is carried forward into the final nomination payload.
2. Semantic quality depends on the schema vectors, which currently only describe columns as isolated sentences.
3. Confidence scoring is heuristic and not calibrated against observed outcomes.
4. If source DB access is blocked or unavailable, this whole stage weakens sharply.

## 11. Node 7: HITL Table Review

File: `nodes/hitl.py`

Purpose: certify human-reviewed table nominations after Gate 2.

### Main function

`hitl_table_review_node(state)`

Flow:

1. If `human_table_decision != COMPLETED`, return state.
2. Ensure `certified_tables` exists.
3. Write `GATE2_CERTIFIED_TABLES` artifact.
4. Set `status = "GATE2_COMPLETE"`.

### Helper: `certify_hitl_tables`

Responsibilities:

- Store the human-approved tables in `ai_store`.

### Practical flow

As with Gate 1, the actual operator workflow is in `cli.py`, not inside the graph itself.

The CLI:

- fetches nominated tables,
- lets the operator approve/reject them,
- loads checkpoint state,
- sets `human_table_decision = COMPLETED`,
- manually calls `hitl_table_review_node`.

## 12. Node 8: Metadata Discovery

File: `nodes/metadata_discovery.py`

Purpose: fetch deterministic column-level metadata for the final approved tables.

### Main function

`metadata_discovery_node(state)`

Flow:

1. Fail fast if pipeline status is failed.
2. Resolve which tables to inspect:
   - `certified_tables`
   - fallback to `nominated_tables`
3. Open Azure SQL connections grouped by database.
4. For each table:
   - fetch column metadata,
   - append completed/failed table record.
5. Persist final discovered metadata artifact.
6. Write:
   - `discovered_metadata`
   - `metadata_status`
   - `metadata_error`

### Helper: `_resolve_tables_for_discovery`

Responsibilities:

- Normalize tables from state into a consistent structure:
  - `database_name`
  - `schema_name`
  - `table_name`

### Helper: `_build_connection_string`

Responsibilities:

- Construct Azure SQL ODBC string for a specific source DB.

### Helper: `get_azure_sql_connection`

Responsibilities:

- Open the connection for metadata crawling.

### Helper: `_format_data_type`

Responsibilities:

- Convert raw SQL type metadata into a friendly string like:
  - `varchar(100)`
  - `decimal(18,2)`
  - `datetime2(7)`

### Helper: `_fetch_table_columns`

Responsibilities:

- Query `INFORMATION_SCHEMA.COLUMNS`.
- Return per-column details:
  - name
  - type
  - nullable
  - ordinal
  - character length
  - numeric precision
  - scale
  - datetime precision
  - collation
  - default

### Helper: `_close_connections`

Responsibilities:

- Close all opened DB connections safely.

### Helper: `_persist_discovered_metadata`

Responsibilities:

- Store the final metadata artifact in `ai_store`.

### Metadata discovery shortcomings

1. It does not fetch primary keys.
2. It does not fetch foreign keys.
3. It does not fetch indexes, uniqueness, check constraints, or relationship graphs.
4. It is deterministic and reliable for columns, but not rich enough for full SQL generation or join reasoning.

## 13. Node 9: Column Profiling

File: `nodes/column_profiling.py`

Purpose: compute deterministic data-quality and distribution statistics for the columns discovered by metadata discovery.

### Main function

`column_profiling_node(state)`

Flow:

1. Fail fast if pipeline status is failed.
2. Read completed tables from `state["discovered_metadata"]["tables"]`.
3. Profile tables in parallel using source database SQL pushdown.
4. For each column:
   - classify the profiling tier,
   - compute row count, non-null count, null rate,
   - compute cardinality where safe,
   - compute min/max for measures and dates,
   - compute p25/p75 for measures using sampled SQL,
   - compute top samples for low-cardinality dimensions, defaults, and flags.
5. Persist a `COLUMN_PROFILES` artifact to `ai_store`.
6. Write:
   - `column_profiles`
   - `column_profiling_status`
   - `column_profiling_error`

### Helper: `_resolve_tables_for_profiling`

Responsibilities:

- Pull only successfully discovered tables from `discovered_metadata`.
- Normalize them into profiling-ready table references.

### Helper: `classify_profile_tier`

Responsibilities:

- Classify each column as one of:
  - `ID`
  - `AUDIT`
  - `FLAG`
  - `DATE`
  - `MEASURE`
  - `DIMENSION`
  - `DEFAULT`
  - `HIGH_CARD_TEXT`

### Helper: `_fetch_pushdown_stats`

Responsibilities:

- Run source-side aggregate SQL for each column.
- Compute null rate, cardinality, and min/max where applicable.

### Helper: `_fetch_measure_percentiles`

Responsibilities:

- Use source SQL `PERCENTILE_CONT` over a `TABLESAMPLE` to compute p25 and p75 for measure columns.

### Helper: `_fetch_top_samples`

Responsibilities:

- Use source SQL grouping over `TABLESAMPLE` to capture common values for low-cardinality columns.

### Helper: `profile_column`

Responsibilities:

- Orchestrate all profiling work for one column.
- Return a validated `ColumnProfileResult`.

### Helper: `profile_table`

Responsibilities:

- Profile all columns for one table.
- Summarize table-level success, partial, or failed status.

### Helper: `_persist_column_profiles`

Responsibilities:

- Store the profiling summary and per-column profiles as a `COLUMN_PROFILES` artifact.

### Column profiling shortcomings

1. The repo-native implementation does not include the Databricks notebook's Delta cache tables or MLflow logging.
2. Profiling currently targets SQL Server/Azure SQL semantics through `pyodbc`.
3. Per-column profiling is deterministic but can be expensive on very wide or very large tables.
4. Percentiles and top samples are best-effort; failures are captured as warnings or failed column profiles instead of stopping the whole pipeline.

## 14. Persistence Layer

File: `utilis/db.py`

This file is extremely important because it controls how pipeline state and artifacts are persisted.

### Main responsibilities

- Azure SQL config
- connection builders
- source DB query execution
- `ai_store` writer
- HITL queue helpers

### `get_pipeline_connection`

Connects to the pipeline DB used for:

- `ai_store`
- `brd_run_registry`
- `hitl_review_queue`
- `kpi_checkpoints`

### `get_client_connection`

Connects to the source business DB used for:

- schema reads
- table nomination lookup
- metadata discovery

### `execute_source_sql`

Responsibilities:

- Open a source DB connection.
- Execute a read query.
- Return rows or empty list on failure.

Important nuance:

- It only auto-injects source schema filtering for queries that include `INFORMATION_SCHEMA.TABLES`.
- That means some source-schema reads are constrained, but others are not.

### `ai_store_db_writer`

Responsibilities:

- Insert or update artifacts into pipeline `ai_store`.
- Persist:
  - run_id
  - fingerprint
  - stage
  - artifact_type
  - payload
  - schema version
  - prompt version
  - faithfulness status
  - token usage
  - cost
  - timestamp

### Most important storage flaw

This writer checks `record_exists` using only:

- `WHERE fingerprint = ?`

Then updates that row if found.

That means all artifacts for the same fingerprint collide unless the underlying table schema handles it differently outside this code.

Practical consequence:

- `REQUIREMENTS`
- `KPIS`
- `TABLE_NOMINATIONS`
- `GATE1_CERTIFIED_KPIS`
- `GATE2_CERTIFIED_TABLES`
- `DISCOVERED_METADATA`

can overwrite each other for the same BRD fingerprint.

This is the biggest systemic design weakness in the repo.

### HITL helpers

- `insert_hitl_queue_items`
- `get_pending_items`
- `update_hitl_item`
- `get_completed_items`

These functions implement the Gate 1 review queue in SQL.

Table review is not queued the same way; instead the CLI pulls table nominations directly from `ai_store`.

## 15. Schemas and Validation

File: `schema.py`

### `BRDSchema`

Validates:

- minimum length 200
- presence of at least one business-ish keyword

### `RequirementsSchema`

Validates the structured requirement output.

### `KPISchemaItem`

Validates each KPI record and enforces confidence thresholds based on derivation type.

### `NominationItem`

Validates table nomination records and rounds key float fields.

Why this matters:

- A lot of the pipeline reliability is coming from Pydantic validation, not just prompt design.

## 16. HITL Resume Model

The pipeline has two different pause/resume concepts:

1. LangGraph interrupt/checkpointing
2. SQL checkpoint reload plus manual node invocation

### Actual flow

After `kpi_extraction`, the graph pauses before `hitl_review`.

Then the CLI:

1. reads pending queue items,
2. collects human decisions,
3. loads `full_state_json` from `kpi_checkpoints`,
4. sets `human_decision = COMPLETED`,
5. sets `certified_kpis`,
6. manually calls `hitl_review_node`,
7. manually calls `table_nomination_node`.

The same pattern is repeated for Gate 2 and metadata discovery.

### Why this matters

If you explain this as “LangGraph handles everything automatically after pause,” that would be inaccurate. The graph is important, but the resume behavior is partially implemented in the CLI.

## 17. Main Shortcomings Summary

These are the issues most worth calling out to the team.

### 1. Artifact overwrite risk in `ai_store`

The persistence model updates by fingerprint alone, which can destroy stage separation.

### 2. Global schema-vector deletion

Every schema embedding run can wipe previously embedded schema vectors for all runs.

### 3. Mixed orchestration model

The pipeline is half LangGraph and half manual CLI-driven continuation.

### 4. Weak semantic memory reuse

Layer 2 retrieval logic depends on vector metadata that may not match what is actually stored.

### 5. Incomplete metadata richness

Final metadata discovery only captures columns and types, not PK/FK/index structures.

### 6. Duplicated embedding work

BRD embedding happens both in ingestion and again in memory lookup on the non-bypass path.

### 7. Configuration and security concerns

DB credentials are present as defaults in source code, which is risky.

### 8. State/model drift

Some state fields exist but are not populated, which suggests unfinished observability.

### 9. CLI maintenance debt

The CLI contains duplicate functions and old/dead logic, which makes reasoning harder.

## 18. Best Way To Explain This Pipeline Verbally

If you need a concise but accurate way to describe it to the team, use something like this:

“First we normalize and validate the BRD, fingerprint it, and build both BRD and schema embeddings. Then we check whether we already processed the same or a similar BRD. If not, we extract structured requirements, then grounded KPIs, and pause for human review. After KPI approval, we nominate candidate source tables using lexical search, semantic search, foreign-key expansion, and lookup-table discovery. Then we pause for table review. After tables are approved, we do deterministic metadata discovery and column profiling for the final selected tables.”

Then add the honest qualifier:

“Conceptually the pipeline is strong, but the biggest technical risks today are artifact persistence collisions, globally destructive schema indexing, and a split between graph orchestration and manual resume logic.”

## 19. Suggested Next Improvements

If the team asks what should be fixed first, this is the order I would recommend:

1. Fix `ai_store` keying so artifacts are uniquely stored by at least `(fingerprint, artifact_type)` or a dedicated storage fingerprint.
2. Remove `delete_all=True` from schema embedding and make schema vectors incremental or namespaced safely.
3. Decide whether HITL resume should be graph-native or CLI-native, then simplify around one model.
4. Make semantic memory metadata consistent so Layer 2 retrieval can actually work as intended.
5. Extend metadata discovery to include PK/FK and relationship-level metadata.
6. Populate currently unused state observability fields.
7. Clean the CLI to remove duplicated/dead code and make operator flow easier to trust.

## 20. Files To Study First

If you want to understand the whole system quickly, read in this order:

1. `graph.py`
2. `state.py`
3. `nodes/ingestion.py`
4. `nodes/memory_lookup.py`
5. `nodes/req_extraction.py`
6. `nodes/kpi_extraction.py`
7. `nodes/hitl.py`
8. `nodes/table_nomination.py`
9. `nodes/metadata_discovery.py`
10. `nodes/column_profiling.py`
11. `utilis/db.py`
12. `schema.py`
13. `cli.py`

## 21. Final Take

This repo already has the skeleton of a serious multi-stage analytics pipeline:

- it has deterministic stages,
- typed shared state,
- LLM guardrails,
- semantic retrieval,
- two HITL gates,
- table discovery heuristics,
- and deterministic metadata extraction.
- and deterministic column profiling.

Its biggest challenge is not lack of ideas. It is consistency:

- consistency of persistence,
- consistency of orchestration,
- consistency of vector metadata,
- and consistency of final metadata richness.

Once those are tightened, the pipeline becomes much easier to trust, explain, and extend.
