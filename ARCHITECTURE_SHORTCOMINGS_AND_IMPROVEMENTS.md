# Architecture Shortcomings and Further Improvements

This document focuses only on the architecture of the Athena pipeline. It is meant to help explain where the current design is strong, where it is fragile, and what should be improved next from a system-design point of view.

The companion file `PIPELINE_DEEP_DIVE.md` explains how the pipeline works node by node. This file looks at the architecture across nodes.

## 1. Current Architecture Summary

Athena is a multi-stage BRD-to-metadata pipeline. It uses:

- LangGraph for orchestration.
- Azure SQL for persistent artifacts, run registry, HITL queues, and checkpoints.
- Pinecone for BRD and schema embeddings.
- Azure OpenAI for requirements and KPI extraction.
- Human-in-the-loop gates for KPI and table certification.
- Source Azure SQL databases for schema and metadata discovery.

The current flow is:

```text
BRD Input
  -> Ingestion
  -> Memory Lookup
  -> Requirement Extraction
  -> KPI Extraction
  -> Gate 1 HITL Review
  -> Table Nomination
  -> Gate 2 HITL Review
  -> Metadata Discovery
  -> Column Profiling
  -> Final Metadata Artifact
```

At a conceptual level, this architecture is good: it separates document understanding, memory reuse, KPI generation, human validation, table discovery, and deterministic metadata extraction.

The main architectural issue is that the boundaries between orchestration, persistence, memory, and HITL resume are not clean enough yet.

## 2. High-Level Strengths

Before discussing shortcomings, these are the architectural strengths worth preserving.

### 2.1 Stage-based design

The pipeline is split into clear nodes:

- ingestion,
- memory lookup,
- requirements extraction,
- KPI extraction,
- HITL certification,
- table nomination,
- metadata discovery.

This makes the system easier to reason about than a single large agent.

### 2.2 Shared state contract

`Stage01State` gives the pipeline a central handoff model. Every node reads and writes to the same state object, which helps keep the flow explicit.

### 2.3 Human validation gates

The two HITL gates are architecturally valuable:

- Gate 1 protects KPI quality.
- Gate 2 protects source table selection.

This is important because the pipeline uses LLMs and heuristic matching, both of which need human review before downstream automation.

### 2.4 Hybrid table nomination

Table nomination combines:

- lexical matching,
- semantic matching,
- FK relationship expansion,
- lookup-table sweep.

This is a stronger design than relying only on LLM guesses or only on vector search.

### 2.5 Deterministic metadata and profiling stages

The final metadata and profiling stages do not ask the LLM to invent schema or statistics. They read source database metadata and source-side aggregate statistics directly. That is the right architectural direction.

## 3. Core Architectural Shortcomings

## 3.1 Artifact storage is keyed too broadly

### Current design

The `ai_store` writer checks whether a row exists using only:

```sql
WHERE fingerprint = ?
```

If a record exists for the same fingerprint, it updates that row.

### Why this is a problem

A single BRD fingerprint can produce many separate artifacts:

- `REQUIREMENTS`
- `KPIS`
- `GATE1_CERTIFIED_KPIS`
- `TABLE_NOMINATIONS`
- `GATE2_CERTIFIED_TABLES`
- `DISCOVERED_METADATA`

These are different logical artifacts, but the writer treats the fingerprint as if it identifies one artifact.

That creates a serious risk that later stages overwrite earlier stages.

### Architectural impact

- Memory lookup may not find the expected `REQUIREMENTS` and `KPIS` artifacts.
- Run summary may show incomplete or misleading history.
- Auditability is weakened because the pipeline cannot reliably preserve all stage outputs.
- Re-running the same BRD can mutate prior artifacts instead of creating a clean lineage.

### Recommended improvement

Use an artifact identity model instead of fingerprint-only identity.

Recommended key:

```text
storage_fingerprint = fingerprint + ":" + artifact_type
```

or a proper relational unique key:

```text
(fingerprint, artifact_type, schema_version)
```

For full lineage, prefer:

```text
(run_id, fingerprint, artifact_type, artifact_version)
```

### Target design

Each stage writes its own immutable or versioned artifact.

Example:

```text
862bb...:REQUIREMENTS
862bb...:KPIS
862bb...:TABLE_NOMINATIONS
862bb...:DISCOVERED_METADATA
```

This makes memory lookup, audit, and debugging much more reliable.

## 3.2 Orchestration is split between LangGraph and CLI

### Current design

LangGraph defines the main pipeline and interrupts before HITL nodes. But after review, the CLI manually:

- loads checkpoint state from SQL,
- sets human decision fields,
- directly calls HITL node functions,
- directly calls downstream nodes.

### Why this is a problem

There are two sources of truth for pipeline execution:

- the graph,
- the CLI resume logic.

This makes it harder to reason about what the pipeline is doing after a human gate.

### Architectural impact

- Resume behavior is harder to test.
- Future non-CLI interfaces will need to duplicate orchestration logic.
- LangGraph checkpoints are not being used as cleanly as intended.
- Node execution order can diverge between normal graph runs and manual review flows.

### Recommended improvement

Choose one resume model and design around it.

Preferred architecture:

```text
Graph owns orchestration.
CLI only updates human decisions and resumes the graph.
```

The CLI should not directly call business nodes like `table_nomination_node`.

### Target design

1. Graph pauses before HITL.
2. CLI writes human decisions into persistent state or command payload.
3. Graph resumes from checkpoint.
4. Graph executes `hitl_review`.
5. Graph routes to the next node.

This keeps orchestration centralized.

## 3.3 Memory architecture is not clearly separated

### Current design

Memory lookup does several jobs:

- exact artifact lookup,
- semantic BRD lookup,
- rejected KPI lookup,
- BRD embedding,
- delayed ingestion finalization.

### Why this is a problem

The node mixes memory retrieval, vector indexing, and ingestion finalization.

### Architectural impact

- Harder to test memory independently.
- Duplicate embedding behavior appears in both ingestion and memory lookup.
- Memory behavior becomes difficult to explain.
- Future memory strategies will be harder to add cleanly.

### Recommended improvement

Split memory into clearer components:

```text
Ingestion Indexer
  -> handles BRD/schema embedding

Exact Memory Resolver
  -> resolves exact fingerprint artifacts

Semantic Memory Retriever
  -> retrieves similar historical context

Negative Memory Retriever
  -> retrieves rejected or low-quality historical outputs
```

These can still be implemented as one node initially, but the internal architecture should separate responsibilities.

## 3.4 Schema vector indexing is globally destructive

### Current design

Schema embedding clears the entire Pinecone `schema` namespace before upserting new vectors.

### Why this is a problem

If multiple runs or multiple source databases depend on the same metadata index, one run can erase schema vectors used by another run.

### Architectural impact

- Concurrent runs can interfere with each other.
- Multi-client or multi-database operation becomes unsafe.
- KPI extraction and table nomination can suddenly lose schema grounding.
- Reproducibility becomes weak because vector state changes globally.

### Recommended improvement

Use scoped vector namespaces or stable vector IDs.

Possible namespace strategy:

```text
schema:{source_database}
schema:{client_id}:{source_database}
schema:{environment}:{source_database}
```

Possible ID strategy:

```text
{database}.{schema}.{table}.{column}
```

Then upsert incrementally instead of deleting all.

### Target design

Schema embeddings should be treated as a durable catalog index, not a per-run scratchpad.

## 3.5 Metadata model is too shallow for downstream SQL intelligence

### Current design

Final metadata discovery fetches column-level metadata from `INFORMATION_SCHEMA.COLUMNS`.

It captures:

- column name,
- data type,
- nullability,
- ordinal position,
- precision,
- scale,
- collation,
- default.

### What is missing

It does not fetch:

- primary keys,
- foreign keys,
- referenced tables,
- referenced columns,
- unique constraints,
- indexes,
- check constraints,
- row counts,
- table descriptions,
- column descriptions,
- sample values,
- cardinality,
- relationship graph.

### Architectural impact

The metadata artifact is useful for column awareness, but not enough for reliable SQL generation or join planning.

Without PK/FK and relationship metadata:

- join paths are uncertain,
- fact/dimension modeling is weaker,
- KPI computability checks are incomplete,
- downstream SQL generation may hallucinate joins.

### Recommended improvement

Create a richer metadata contract:

```text
TableMetadata
  -> columns
  -> primary_keys
  -> foreign_keys
  -> unique_constraints
  -> indexes
  -> row_count
  -> relationships
```

Foreign key example:

```json
{
  "constraint_name": "FK_claim_policy",
  "source_schema": "dbo",
  "source_table": "claim",
  "source_column": "policy_id",
  "referenced_schema": "dbo",
  "referenced_table": "policy",
  "referenced_column": "policy_id"
}
```

## 3.6 Table nomination produces tables but not join-ready context

### Current design

Table nomination returns candidate tables with:

- table name,
- schema,
- database,
- confidence score,
- nomination reason,
- matched keywords,
- coverage ratio.

### Why this is incomplete

The nomination tells us which tables are likely useful, but not how they connect.

### Architectural impact

Later stages cannot easily answer:

- Which table is the fact table?
- Which tables are dimensions?
- Which joins are valid?
- Which relationships were used to include a table?
- Which columns are important for the KPI?

### Recommended improvement

Extend nomination output to include:

- matched columns,
- relationship evidence,
- candidate join paths,
- table role classification,
- KPI-to-table mapping.

Example target shape:

```json
{
  "table_name": "claims",
  "role": "fact",
  "matched_kpis": ["Claim Settlement Rate"],
  "matched_columns": ["claim_id", "settlement_date", "claim_status"],
  "relationship_evidence": [
    {
      "type": "foreign_key",
      "from": "claims.policy_id",
      "to": "policies.policy_id"
    }
  ]
}
```

## 3.7 State contract has drifted from implementation

### Current design

`Stage01State` includes fields such as:

- `brd_embedded`,
- `schema_embedded`,
- `schema_columns_count`.

But these fields are not populated by the active nodes.

### Why this is a problem

The state model advertises observability that does not exist.

### Architectural impact

- Operators may assume a stage completed when there is no explicit state signal.
- Future nodes may rely on fields that are never set.
- Debugging becomes harder because the state contract is not fully truthful.

### Recommended improvement

Either:

- populate these fields consistently, or
- remove them until they are implemented.

Recommended additions:

```text
brd_embedded: true|false
schema_embedded: true|false
schema_columns_count: int
schema_embedding_namespace: string
schema_embedding_version: string
```

## 3.8 Persistence and checkpointing are mixed

### Current design

The pipeline stores:

- artifacts in `ai_store`,
- full state snapshots in `kpi_checkpoints`,
- HITL queue items in `hitl_review_queue`,
- run registry entries in `brd_run_registry`.

### Why this is not enough

These stores exist, but their responsibilities are not sharply defined.

### Architectural impact

- It is unclear whether `ai_store` is an artifact log, memory store, or latest-state table.
- `kpi_checkpoints` only appears after KPI and table nomination stages.
- There is no clear event log for all stage transitions.
- Reconstructing a run timeline is difficult.

### Recommended improvement

Define three separate storage concepts:

```text
Artifact Store
  -> immutable/versioned stage outputs

State Checkpoint Store
  -> resumable pipeline state

Event Log
  -> stage start/end/error events
```

This separation makes operations and debugging much cleaner.

## 3.9 Error handling is inconsistent across boundaries

### Current design

Some nodes fail the state with `status = FAILED`.
Some helpers catch exceptions and return empty lists.
Some persistence failures are considered critical.
Some checkpoint failures are considered non-critical.

### Why this is a problem

The pipeline does not have a consistent error policy.

### Architectural impact

- A source DB failure may silently produce empty nominations.
- A vector lookup failure may silently weaken extraction quality.
- A DB write failure may terminate the run.
- Operators may not know whether a result is complete or degraded.

### Recommended improvement

Use explicit degradation states.

Example:

```text
status = COMPLETED_WITH_WARNINGS
warnings = [
  {
    "stage": "schema_embedding",
    "severity": "warning",
    "message": "Source DB query failed"
  }
]
```

Recommended severity model:

- `FAILED`: cannot continue safely.
- `DEGRADED`: can continue, but output quality is reduced.
- `WARNING`: non-critical issue.
- `PASSED`: no issue.

## 3.10 Configuration and secret management are not production-grade

### Current design

Database configuration includes environment variable fallbacks, but also hardcoded default credentials.

### Why this is a problem

Secrets in code are a security risk.

### Architectural impact

- Credential leakage risk.
- Harder to move safely between local, staging, and production.
- Risk of accidentally connecting to the wrong environment.

### Recommended improvement

Move to environment-only configuration or a secret manager.

Recommended production model:

- Azure Key Vault for secrets.
- Environment variables for non-secret config.
- No credential defaults in code.
- Separate config profiles for local, dev, staging, production.

## 3.11 Observability is too log-heavy and not metric-driven

### Current design

The pipeline logs many events and writes token counts/costs into `ai_store`.

### What is missing

There is no clear metrics model for:

- stage duration,
- vector query latency,
- LLM latency,
- source DB query latency,
- HITL wait time,
- retry count by stage,
- degradation count,
- nomination precision after review,
- KPI approval rate.

### Architectural impact

It is hard to answer:

- Which stage is slow?
- Which stage fails most often?
- Are table nominations improving?
- Is memory reuse actually reducing cost?
- How much human review effort is needed?

### Recommended improvement

Add a `pipeline_events` or `pipeline_metrics` table.

Example metrics:

```text
run_id
stage
event_type
started_at
ended_at
duration_ms
status
warning_count
input_count
output_count
token_count
cost_usd
```

## 4. Proposed Target Architecture

A cleaner target architecture would look like this:

```text
Input Adapter
  -> BRD Parser
  -> Validation Layer
  -> Fingerprint Service
  -> Memory Resolver
       -> Exact Artifact Store
       -> Semantic Memory Index
       -> Negative Memory Store
  -> Requirement Extractor
  -> KPI Extractor
       -> Schema Retrieval Service
       -> KPI Validator
  -> HITL Gateway 1
  -> Table Discovery Service
       -> Lexical Search
       -> Semantic Search
       -> Relationship Resolver
       -> Lookup Table Resolver
  -> HITL Gateway 2
  -> Metadata Crawler
       -> Columns
       -> PK/FK
       -> Indexes
       -> Relationships
  -> Column Profiler
       -> Null rates
       -> Cardinality
       -> Min/max
       -> Percentiles
       -> Top samples
  -> Artifact Store
  -> Event Log / Metrics
```

## 5. Recommended Improvements by Priority

## Priority 1: Fix artifact identity

### Problem

Artifacts can overwrite each other because persistence is keyed by fingerprint.

### Improvement

Use a unique artifact identity:

```text
artifact_id = fingerprint + ":" + artifact_type
```

or:

```text
UNIQUE(run_id, artifact_type)
UNIQUE(fingerprint, artifact_type, schema_version)
```

### Expected benefit

- Reliable memory lookup.
- Reliable audit history.
- Cleaner run summaries.
- Safer reruns.

## Priority 2: Make schema indexing safe

### Problem

Schema embedding deletes the whole Pinecone namespace.

### Improvement

Use scoped namespaces:

```text
schema:{database}
schema:{client}:{database}
```

or stable vector IDs with incremental upsert.

### Expected benefit

- Safe multi-run behavior.
- Safe multi-database behavior.
- Stable schema grounding.

## Priority 3: Make HITL resume graph-native

### Problem

CLI manually calls downstream nodes after review.

### Improvement

Let LangGraph own post-HITL continuation.

### Expected benefit

- Cleaner orchestration.
- Easier testing.
- Future UI/API support becomes simpler.

## Priority 4: Add relationship-rich metadata

### Problem

Metadata discovery is column-only.

### Improvement

Fetch:

- primary keys,
- foreign keys,
- referenced columns,
- indexes,
- unique constraints,
- row counts.

### Expected benefit

- Better SQL generation.
- Better join planning.
- Better KPI computability analysis.

## Priority 5: Cleanly separate memory responsibilities

### Problem

Memory lookup mixes too many responsibilities.

### Improvement

Separate:

- exact memory,
- semantic memory,
- rejected KPI memory,
- indexing.

### Expected benefit

- Easier testing.
- Easier debugging.
- Better future extensibility.

## Priority 6: Add architecture-level observability

### Problem

Logs exist, but structured metrics are limited.

### Improvement

Add event and metrics tables.

### Expected benefit

- Faster debugging.
- Better cost tracking.
- Better quality measurement.

## Priority 7: Harden configuration and secrets

### Problem

Credentials are present as source-code defaults.

### Improvement

Use environment-only secrets or Azure Key Vault.

### Expected benefit

- Safer deployments.
- Cleaner environment separation.

## 6. Architecture Risk Register

| Risk | Severity | Why It Matters | Recommended Fix |
|---|---:|---|---|
| Artifact overwrite by fingerprint | Critical | Can lose requirements, KPIs, nominations, metadata | Key artifacts by fingerprint plus artifact type |
| Global schema vector deletion | Critical | One run can erase schema context for others | Scoped namespaces or incremental upsert |
| CLI/manual graph continuation | High | Orchestration is split and harder to test | Resume through LangGraph |
| Missing PK/FK in final metadata | High | Weak join reasoning and SQL generation | Add relationship metadata crawler |
| Weak semantic memory metadata | High | Layer 2 memory may not retrieve useful context | Align vector metadata and query filters |
| Hardcoded credential defaults | High | Security and environment risk | Use secret manager/env-only config |
| Inconsistent error policy | Medium | Failures can be silent or unclear | Add warnings/degraded states |
| State fields not populated | Medium | State contract is misleading | Populate or remove fields |
| Duplicate embedding work | Medium | Extra cost and inconsistent vector state | Make indexing single-responsibility |
| CLI duplicate/dead code | Low/Medium | Maintenance burden | Refactor CLI review flow |

## 7. Suggested Future Architecture Milestones

### Milestone 1: Storage correctness

Goal:

- Make every artifact durable, unique, and retrievable.

Deliverables:

- `artifact_id` or `storage_fingerprint`.
- DB migration for artifact uniqueness.
- Updated `ai_store_db_writer`.
- Memory lookup updated to use artifact identity.

### Milestone 2: Safe schema catalog

Goal:

- Make schema embeddings durable and non-destructive.

Deliverables:

- Per-source namespace strategy.
- Stable vector IDs.
- Schema index versioning.
- No global delete during normal runs.

### Milestone 3: Graph-native HITL

Goal:

- Make LangGraph the only owner of stage ordering.

Deliverables:

- CLI writes decisions.
- Graph resumes from checkpoint.
- No direct CLI calls to business nodes.

### Milestone 4: Relationship-aware metadata

Goal:

- Make metadata rich enough for SQL planning.

Deliverables:

- PK extraction.
- FK extraction.
- Index extraction.
- Relationship graph payload.
- Table role hints.

### Milestone 5: Observability and quality metrics

Goal:

- Make pipeline quality measurable.

Deliverables:

- Stage event table.
- Duration metrics.
- Approval/rejection rates.
- Memory hit rates.
- Cost per run.
- Nomination precision after Gate 2.

## 8. Better End-State Data Model

Recommended durable entities:

```text
runs
  run_id
  fingerprint
  status
  created_at
  completed_at

artifacts
  artifact_id
  run_id
  fingerprint
  artifact_type
  artifact_version
  payload
  created_at

checkpoints
  run_id
  checkpoint_id
  graph_node
  full_state_json
  created_at

hitl_items
  item_id
  run_id
  gate_number
  artifact_id
  status
  original_content
  edited_content
  reviewer
  decided_at

pipeline_events
  event_id
  run_id
  stage
  event_type
  severity
  message
  started_at
  ended_at
  duration_ms

schema_catalog
  database_name
  schema_name
  table_name
  column_name
  data_type
  is_primary_key
  is_foreign_key
  referenced_table
  referenced_column
```

This model separates run identity, artifacts, checkpoints, human work, events, and schema metadata.

## 9. How To Explain The Architecture Shortcomings

A clear way to explain it:

“The pipeline has a good staged architecture, but some infrastructure boundaries need to be tightened. The biggest issue is that artifacts are not stored as separate durable outputs, so stages can overwrite each other by fingerprint. The second issue is that schema embeddings are treated like temporary run data even though they are really shared catalog data. The third issue is that HITL resume is split between LangGraph and CLI code. Once we fix artifact identity, schema indexing, and graph-native resume, the rest of the pipeline becomes much easier to trust and extend.”

## 10. Final Recommendation

Do not start by adding more LLM logic. Start by hardening the architecture underneath the existing flow.

Recommended order:

1. Fix artifact storage identity.
2. Make schema indexing non-destructive.
3. Make HITL resume graph-native.
4. Add PK/FK and relationship metadata.
5. Add structured events and metrics.
6. Clean up configuration and secrets.

Once these are done, future improvements like SQL generation, KPI computability scoring, lineage views, and automated dashboard generation will have a much stronger foundation.
