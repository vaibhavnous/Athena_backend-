"""
Table nomination node.

Uses certified KPIs to nominate source tables from business databases such as
`insurance`, while keeping ai_store and HITL metadata in the pipeline DB.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Set

from langgraph.graph import StateGraph
from pinecone import Pinecone

from nodes.ingestion import _embedding_model
from schema import NominationItem, NominationSchema
from state import Stage01State
from utilis.db import (
    ai_store_db_writer,
    config as db_config,
    execute_source_sql,
    get_client_connection,
    get_pipeline_connection,
)
from utilis.logger import logger

PLATFORM_TABLES: Set[str] = {"ai_store", "brd_run_registry", "hitl_review_queue"}

SCORE_DUAL_MATCH = 1.0
SCORE_LEXICAL_ONLY = 0.9
SCORE_SEMANTIC_ONLY = 0.8
SCORE_FK_RESOLVED = 0.75
SCORE_LOOKUP_SWEEP = 0.70

REASON_DUAL_MATCH = "Dual Match (Keyword + Semantic)"
REASON_LEXICAL_ONLY = "Exact Schema Keyword Match"
REASON_SEMANTIC_ONLY = "Semantic Vector Match"
REASON_FK_RESOLVED = "FK Resolution (related to nominated table)"
REASON_LOOKUP_SWEEP = "Lookup Table Sweep (dim/ref/lkp)"

LOOKUP_PREFIXES = ("dim_", "ref_", "lkp_", "lookup_", "code_", "type_")
LOOKUP_MAX_ROWS = 10_000
NOISE_PREFIXES = ("tmp_", "log_", "audit_")

SYNONYMS: Dict[str, List[str]] = {
    "claim": ["claim", "claims", "settlement", "loss"],
    "premium": ["premium", "payment", "amount"],
    "policy": ["policy", "contract", "coverage"],
    "customer": ["customer", "member", "client", "insured"],
    "revenue": ["revenue", "sales", "income"],
    "ratio": ["ratio", "rate", "percentage"],
    "identifier": ["identifier", "id", "key", "reference"],
}


def _extract_kpi_names(certified_kpis: List[Any]) -> List[str]:
    names: List[str] = []
    for item in certified_kpis:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("kpi_name") or item.get("name")
            if name:
                names.append(str(name))
    return names


def _build_keywords(kpi_names: List[str]) -> List[str]:
    keywords: Set[str] = set()
    for name in kpi_names:
        for token in re.split(r"[^a-zA-Z0-9_]", name):
            token = token.strip().lower()
            if len(token) >= 3:
                keywords.add(token)
    return sorted(keywords)


def _normalize(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return re.sub(r"_+", "_", text.strip("_")).lower()


def _tokenize_identifier(text: str) -> List[str]:
    normalized = _normalize(text)
    return [token for token in normalized.split("_") if token]


def _expand_keywords(keywords: List[str]) -> Dict[str, Set[str]]:
    expanded: Dict[str, Set[str]] = {}
    for keyword in keywords:
        variants = set(SYNONYMS.get(keyword, [keyword]))
        variants.add(keyword)
        expanded[keyword] = {_normalize(variant) for variant in variants if variant}
    return expanded


def _domain_keywords(kpi_names: List[str]) -> Set[str]:
    return {_normalize(token) for token in _build_keywords(kpi_names)}


def _build_domain_tokens(lexical_results: List[Dict[str, Any]]) -> Set[str]:
    token_freq: Dict[str, int] = {}
    for row in lexical_results:
        table_tokens = _tokenize_identifier(row["table_name"])
        for token in table_tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

    return {token for token, count in token_freq.items() if count >= 2}


def _build_table_token_frequency_scores(lexical_results: List[Dict[str, Any]]) -> Dict[str, float]:
    token_freq: Dict[str, int] = {}
    for row in lexical_results:
        for token in set(_tokenize_identifier(row["table_name"])):
            token_freq[token] = token_freq.get(token, 0) + 1

    max_freq = max(token_freq.values(), default=0)
    table_scores: Dict[str, float] = {}
    if max_freq == 0:
        return table_scores

    for row in lexical_results:
        key = f"{row['database_name'].lower()}.{row['schema_name']}.{row['table_name']}"
        table_tokens = set(_tokenize_identifier(row["table_name"]))
        if not table_tokens:
            table_scores[key] = 0.0
            continue
        avg_freq = sum(token_freq[token] for token in table_tokens) / len(table_tokens)
        table_scores[key] = round(avg_freq / max_freq, 4)

    return table_scores


def _has_domain_overlap(table_tokens: Set[str], column_tokens: Set[str], domain_tokens: Set[str]) -> bool:
    combined = table_tokens | column_tokens
    return bool(combined & domain_tokens)


def _best_match_weight(variants: Set[str], table_tokens: Set[str], column_tokens: Set[str]) -> tuple[float, bool]:
    best_weight = 0.0
    matched_in_column = False

    for variant in variants:
        for token in column_tokens:
            if token == variant:
                return 0.2, True
            if token.startswith(variant) or variant.startswith(token):
                best_weight = max(best_weight, 0.12)
                matched_in_column = True
            elif variant in token or token in variant:
                best_weight = max(best_weight, 0.05)
                matched_in_column = True

        for token in table_tokens:
            if token == variant:
                best_weight = max(best_weight, 0.1)
            elif token.startswith(variant) or variant.startswith(token):
                best_weight = max(best_weight, 0.06)
            elif variant in token or token in variant:
                best_weight = max(best_weight, 0.02)

    return best_weight, matched_in_column


def _lexical_search(kpi_keywords: List[str], source_databases: List[str]) -> List[Dict[str, Any]]:
    if not kpi_keywords or not source_databases:
        return []

    keyword_set = {kw.lower() for kw in kpi_keywords}
    expanded_keywords = _expand_keywords(sorted(keyword_set))
    domain_tokens: Set[str] = set()
    for variants in expanded_keywords.values():
        domain_tokens.update(variants)
    variant_set = sorted({variant for variants in expanded_keywords.values() for variant in variants})
    like_clauses = " OR ".join(["(c.COLUMN_NAME LIKE ? OR t.TABLE_NAME LIKE ?)" for _ in variant_set])

    params: List[str] = []
    for variant in variant_set:
        params.extend([f"%{variant}%", f"%{variant}%"])

    query = f"""
        SELECT
            t.TABLE_CATALOG AS database_name,
            t.TABLE_SCHEMA AS schema_name,
            t.TABLE_NAME AS table_name,
            c.COLUMN_NAME AS column_name
        FROM INFORMATION_SCHEMA.TABLES t
        JOIN INFORMATION_SCHEMA.COLUMNS c
            ON t.TABLE_CATALOG = c.TABLE_CATALOG
            AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
            AND t.TABLE_NAME = c.TABLE_NAME
        WHERE t.TABLE_TYPE = 'BASE TABLE'
          AND ({like_clauses})
    """

    all_results: List[Dict[str, Any]] = []
    for source_db in source_databases:
        rows = execute_source_sql(source_db, query, tuple(params))
        table_map: Dict[str, Dict[str, Any]] = {}

        for row in rows:
            db = (row.database_name or source_db).lower()
            schema = (row.schema_name or "dbo").lower()
            table = row.table_name.lower()
            column = str(row.column_name)

            if table in PLATFORM_TABLES or table.startswith(NOISE_PREFIXES):
                continue

            key = f"{db}.{schema}.{table}"
            entry = table_map.setdefault(
                key,
                {
                    "database_name": db,
                    "schema_name": schema,
                    "table_name": table,
                    "lexical_score": 0.0,
                    "matched_keywords": set(),
                    "matched_columns": set(),
                    "_all_columns_seen": set(),
                    "_table_tokens": set(_tokenize_identifier(table)),
                    "_column_tokens": set(),
                    "_column_match_hits": 0,
                },
            )

            column_tokens = set(_tokenize_identifier(column))
            entry["_all_columns_seen"].add(column)
            entry["_column_tokens"].update(column_tokens)

            overlap = _has_domain_overlap(entry["_table_tokens"], column_tokens, domain_tokens)
            domain_multiplier = 1.0 if overlap else 0.2

            for keyword, variants in expanded_keywords.items():
                weight, matched_in_column = _best_match_weight(
                    variants,
                    entry["_table_tokens"],
                    column_tokens,
                )
                if weight <= 0:
                    continue

                # Boost strong column matches (ground truth signal)
                if matched_in_column:
                    weight *= 1.5

                entry["lexical_score"] += weight * domain_multiplier
                entry["matched_keywords"].add(keyword)
                if matched_in_column:
                    entry["matched_columns"].add(column)
                    entry["_column_match_hits"] += 1

        for entry in table_map.values():
            total_keywords = len(keyword_set) or 1
            keyword_coverage = len(entry["matched_keywords"]) / total_keywords
            column_coverage = len(entry["matched_columns"]) / max(1, len(entry["_all_columns_seen"]))
            coverage = 0.5 * keyword_coverage + 0.5 * column_coverage
            entry["coverage_ratio"] = round(coverage, 4)
            entry["lexical_score"] += coverage * 0.35
            if entry["_column_match_hits"] > 0:
                entry["lexical_score"] += min(0.1, entry["_column_match_hits"] * 0.02)
            if not entry["matched_keywords"]:
                entry["lexical_score"] = 0.0

            # Slight boost for strong grounding tables
            if entry["coverage_ratio"] > 0.5:
                entry["lexical_score"] += 0.05

            entry["lexical_score"] = round(min(entry["lexical_score"], 1.0), 4)
            entry["matched_keywords"] = sorted(entry["matched_keywords"])
            entry["matched_columns"] = sorted(entry["matched_columns"])
            entry.pop("_all_columns_seen", None)
            entry.pop("_table_tokens", None)
            entry.pop("_column_tokens", None)
            entry.pop("_column_match_hits", None)

        all_results.extend([entry for entry in table_map.values() if entry["lexical_score"] > 0])

    return all_results


def _semantic_search(combined_kpi_string: str, source_databases: List[str]) -> List[Dict[str, Any]]:
    kpi_queries = [part.strip() for part in combined_kpi_string.split(";") if part.strip()]
    if not kpi_queries:
        return []

    if _embedding_model is None:
        logger.warning(
            "Semantic search skipped: embedding model not initialized",
            extra={"node": "table_nomination", "pass": "semantic"},
        )
        return []

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("metadata")
    except Exception as e:
        logger.error(f"Pinecone init failed: {e}", extra={"node": "table_nomination", "pass": "semantic"})
        return []

    table_map: Dict[str, Dict[str, Any]] = {}
    source_set = {db.lower() for db in source_databases}

    for kpi_query in kpi_queries:
        try:
            query_vector = _embedding_model.embed_query(kpi_query)
            results = index.query(
                vector=query_vector,
                top_k=30,
                include_metadata=True,
                namespace="schema",
            )
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}", extra={"node": "table_nomination", "pass": "semantic"})
            continue

        matches = getattr(results, "matches", None)
        if matches is None and isinstance(results, dict):
            matches = results.get("matches", [])
        matches = matches or []

        for match in matches:
            raw_score = float(getattr(match, "score", 0.0))
            score = max(0.0, min(raw_score, 1.0))
            meta = getattr(match, "metadata", {}) or {}

            db = str(meta.get("database_name", "")).lower()
            if db not in source_set:
                continue

            table = str(meta.get("table_name", "")).lower()
            schema = str(meta.get("schema_name", "dbo")).lower()
            key = f"{db}.{schema}.{table}"

            if key not in table_map:
                table_map[key] = {
                    "database_name": db,
                    "schema_name": schema,
                    "table_name": table,
                    "semantic_score": score,
                    "matched_columns": set(),
                }

            table_map[key]["semantic_score"] = max(table_map[key]["semantic_score"], score)

            if meta.get("column_name"):
                table_map[key]["matched_columns"].add(str(meta["column_name"]))

    return [{**value, "matched_columns": sorted(value["matched_columns"])} for value in table_map.values()]


def _fuse_results(
    lexical_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    source_databases: List[str],
) -> Dict[str, Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    source_set = {db.lower() for db in source_databases}
    domain_tokens = _build_domain_tokens(lexical_results)
    token_frequency_scores = _build_table_token_frequency_scores(lexical_results)

    for row in lexical_results:
        db = row["database_name"].lower()
        if db not in source_set:
            continue

        key = f"{db}.{row['schema_name']}.{row['table_name']}"
        fused[key] = {
            "database_name": db,
            "schema_name": row["schema_name"],
            "table_name": row["table_name"],
            "lexical_score": row.get("lexical_score", 0.0),
            "semantic_score": 0.0,
            "matched_keywords": list(set(row.get("matched_keywords", []))),
            "matched_columns": list(set(row.get("matched_columns", []))),
            "coverage_ratio": row.get("coverage_ratio", 0.0),
            "token_frequency_score": token_frequency_scores.get(key, 0.0),
        }

    for row in semantic_results:
        db = row["database_name"].lower()
        if db not in source_set:
            continue

        key = f"{db}.{row['schema_name']}.{row['table_name']}"

        if key not in fused:
            fused[key] = {
                "database_name": db,
                "schema_name": row["schema_name"],
                "table_name": row["table_name"],
                "lexical_score": 0.0,
                "semantic_score": row["semantic_score"],
                "matched_keywords": [],
                "matched_columns": [],
                "coverage_ratio": 0.0,
                "token_frequency_score": token_frequency_scores.get(key, 0.0),
            }
        else:
            fused[key]["semantic_score"] = max(
                fused[key]["semantic_score"],
                row["semantic_score"],
            )

        for column_name in row.get("matched_columns", []):
            fused[key]["matched_columns"].append(column_name)

    for row in fused.values():
        row["matched_keywords"] = list(set(row.get("matched_keywords", [])))
        row["matched_columns"] = list(set(row.get("matched_columns", [])))

    for row in fused.values():
        lex = row["lexical_score"]
        sem = row["semantic_score"]
        coverage = row.get("coverage_ratio", 0.0)
        token_frequency = row.get("token_frequency_score", 0.0)
        table_tokens = set(_tokenize_identifier(row["table_name"]))
        domain_overlap = bool(table_tokens & domain_tokens)

        final_score = (
            0.4 * lex
            + 0.3 * sem
            + 0.2 * coverage
            + 0.1 * token_frequency
        )

        if not domain_overlap:
            final_score *= 0.6

        if lex > 0 and sem > 0:
            final_score += 0.3

        if row.get("matched_columns"):
            final_score += 0.1

        if coverage < 0.3 and sem < 0.4:
            final_score *= 0.6

        row["confidence_score"] = round(final_score, 4)

        if sem > 0 and lex > 0:
            row["nomination_reason"] = REASON_DUAL_MATCH
        elif sem > 0:
            row["nomination_reason"] = REASON_SEMANTIC_ONLY
        else:
            row["nomination_reason"] = REASON_LEXICAL_ONLY

        row["matched_keywords"] = sorted(row["matched_keywords"])
        row["coverage_ratio"] = round(row.get("coverage_ratio", 0.0), 4)

    max_score = max((row["confidence_score"] for row in fused.values()), default=0.0)
    if max_score > 0:
        for row in fused.values():
            row["confidence_score"] = round(row["confidence_score"] / max_score, 4)

    logger.info(
        "Fusion complete: total=%d",
        len(fused),
        extra={"node": "table_nomination", "pass": "fusion"},
    )

    return fused


def _fk_resolution(nominated_tables: List[str], source_databases: List[str]) -> List[Dict[str, Any]]:
    if not nominated_tables or not source_databases:
        return []

    resolved: List[Dict[str, Any]] = []
    nominated_set = {table.lower() for table in nominated_tables}
    seen: Set[str] = set(nominated_set)

    placeholders = ",".join("?" for _ in nominated_set)
    params = tuple(nominated_set) + tuple(nominated_set)

    query = f"""
    SELECT
        FK.TABLE_NAME AS source_table,
        PK.TABLE_NAME AS referenced_table,
        FK.TABLE_SCHEMA AS source_schema,
        PK.TABLE_SCHEMA AS referenced_schema
    FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS RC
    JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS FK
        ON RC.CONSTRAINT_NAME = FK.CONSTRAINT_NAME
    JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS PK
        ON RC.UNIQUE_CONSTRAINT_NAME = PK.CONSTRAINT_NAME
    WHERE FK.TABLE_NAME IN ({placeholders})
       OR PK.TABLE_NAME IN ({placeholders})
    """

    for db in source_databases:
        conn = None
        try:
            conn = get_client_connection(database_name=db)
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"FK query failed for {db}: {e}", extra={"node": "table_nomination", "pass": "fk_resolution"})
            continue

        for row in rows:
            src_table = row.source_table.lower()
            ref_table = row.referenced_table.lower()
            src_schema = row.source_schema.lower()
            ref_schema = row.referenced_schema.lower()

            if src_table in nominated_set and ref_table not in PLATFORM_TABLES and ref_table not in seen:
                seen.add(ref_table)
                resolved.append(
                    {
                        "database_name": db.lower(),
                        "schema_name": ref_schema,
                        "table_name": ref_table,
                        "confidence_score": SCORE_FK_RESOLVED,
                        "nomination_reason": REASON_FK_RESOLVED,
                        "matched_keywords": [],
                        "coverage_ratio": 0.0,
                    }
                )

            if ref_table in nominated_set and src_table not in PLATFORM_TABLES and src_table not in seen:
                seen.add(src_table)
                resolved.append(
                    {
                        "database_name": db.lower(),
                        "schema_name": src_schema,
                        "table_name": src_table,
                        "confidence_score": SCORE_FK_RESOLVED,
                        "nomination_reason": REASON_FK_RESOLVED,
                        "matched_keywords": [],
                        "coverage_ratio": 0.0,
                    }
                )
        conn.close()

    logger.info(
        "FK resolution: %d additional tables found from %d nominated tables",
        len(resolved),
        len(nominated_tables),
        extra={"node": "table_nomination", "pass": "fk_resolution"},
    )

    return resolved


def _lookup_table_sweep(
    source_databases: List[str],
    already_nominated: Set[str],
    domain_tokens: Set[str],
    max_rows: int = LOOKUP_MAX_ROWS,
) -> List[Dict[str, Any]]:
    prefix_clauses = " OR ".join(["LOWER(t.name) LIKE ?" for _ in LOOKUP_PREFIXES])
    params = tuple(f"{prefix}%" for prefix in LOOKUP_PREFIXES) + (max_rows,)

    query = f"""
        SELECT
            DB_NAME() AS database_name,
            s.name AS schema_name,
            t.name AS table_name
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        JOIN sys.partitions p ON t.object_id = p.object_id
        WHERE ({prefix_clauses})
          AND p.index_id IN (0, 1)
          AND p.rows <= ?
    """

    swept: List[Dict[str, Any]] = []
    for source_db in source_databases:
        rows = execute_source_sql(source_db, query, params)
        for row in rows:
            table = row.table_name.lower()
            if table in PLATFORM_TABLES or table in already_nominated:
                continue
            table_tokens = set(_tokenize_identifier(table))
            if domain_tokens and not (table_tokens & domain_tokens):
                continue
            swept.append(
                {
                    "database_name": (row.database_name or source_db).lower(),
                    "schema_name": (row.schema_name or "dbo").lower(),
                    "table_name": table,
                    "confidence_score": SCORE_LOOKUP_SWEEP,
                    "nomination_reason": REASON_LOOKUP_SWEEP,
                    "matched_keywords": [],
                    "coverage_ratio": 0.0,
                }
            )
    return swept


def build_table_nomination_node() -> Callable[[Stage01State], Stage01State]:
    def table_nomination_node(state: Stage01State) -> Stage01State:
        log_context = {"run_id": state.get("run_id", "unknown"), "node": "table_nomination"}

        if state.get("status") == "FAILED":
            return state

        certified_kpis = state.get("certified_kpis")
        if not certified_kpis:
            return {
                **state,
                "status": "FAILED",
                "table_nomination_status": "FAILED",
                "table_nomination_error": "Missing certified_kpis",
            }

        source_databases = state.get("source_databases")
        if not source_databases:
            default_db = (
                db_config.get("azure_sql", {}).get("source_database")
                or db_config.get("azure_sql", {}).get("target_catalog")
                or db_config.get("azure_sql", {}).get("database_name")
            )
            source_databases = [default_db] if default_db else []

        if not source_databases:
            return {
                **state,
                "status": "FAILED",
                "table_nomination_status": "FAILED",
                "table_nomination_error": "Missing source_databases",
            }

        run_id = state["run_id"]
        fingerprint = state.get("fingerprint", run_id)
        kpi_names = _extract_kpi_names(certified_kpis)
        keywords = _build_keywords(kpi_names)
        domain_tokens = {_normalize(token) for token in keywords}
        for variants in _expand_keywords(keywords).values():
            domain_tokens.update(variants)

        lexical_results = _lexical_search(keywords, source_databases)
        semantic_results = _semantic_search("; ".join(kpi_names), source_databases)
        fused = _fuse_results(lexical_results, semantic_results, source_databases)

        if not fused:
            return {
                **state,
                "status": "FAILED",
                "table_nomination_status": "FAILED",
                "table_nomination_error": "No tables found.",
            }

        nominated_table_names = [row["table_name"] for row in fused.values()]
        for row in _fk_resolution(nominated_table_names, source_databases):
            fused[f"{row['database_name']}.{row['schema_name']}.{row['table_name']}"] = row

        for row in _lookup_table_sweep(source_databases, {item["table_name"] for item in fused.values()}, domain_tokens):
            fused[f"{row['database_name']}.{row['schema_name']}.{row['table_name']}"] = row

        all_nominations = sorted(fused.values(), key=lambda item: item["confidence_score"], reverse=True)
        validated = NominationSchema(nominations=[NominationItem(**nom) for nom in all_nominations])

        payload = {
            "fingerprint": fingerprint,
            "storage_fingerprint": f"{fingerprint}:TABLE_NOMINATIONS",
            "run_id": run_id,
            "nomination_count": len(validated.nominations),
            "nominations": [n.model_dump(mode="json") for n in validated.nominations],
            "source_databases": source_databases,
            "kpi_names": kpi_names,
        }
        ai_store_db_writer(
            run_id=run_id,
            stage="Table Nomination",
            artifact_type="TABLE_NOMINATIONS",
            payload=payload,
            schema_version="NominationSchema_v2",
            prompt_version="FIVE_PASS_HYBRID_v1",
            faithfulness_status="PASSED",
            token_count=0,
            input_tokens=0,
            output_tokens=0,
            fingerprint=fingerprint,
        )

        new_state = state.copy()
        new_state.update(
            {
                "nominated_tables": [n.model_dump(mode="json") for n in validated.nominations],
                "table_nomination_status": "PENDING",
                "table_nomination_error": None,
                "human_table_decision": "PENDING",
                "source_databases": source_databases,
                "semantic_matches": semantic_results,
                "semantic_top_k": len(semantic_results),
            }
        )

        try:
            conn = get_pipeline_connection()
            try:
                db_schema = (
                    db_config.get("azure_sql", {}).get("pipeline_schema")
                    or db_config.get("azure_sql", {}).get("schema_name")
                    or "dbo"
                )
                cursor = conn.cursor()
                state_json = json.dumps(new_state, default=str)
                cursor.execute(
                    f"""
                    MERGE [{db_schema}].[kpi_checkpoints] AS target
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
        except Exception as exc:
            logger.warning("Table nomination checkpoint save failed: %s", exc, extra=log_context)

        logger.info(
            "END table_nomination_node: nominated=%d tables from %s",
            len(validated.nominations),
            source_databases,
            extra=log_context,
        )
        return new_state

    return table_nomination_node


table_nomination_node = build_table_nomination_node()


def build_nomination_graph() -> StateGraph:
    graph = StateGraph(Stage01State)
    graph.add_node("table_nomination", table_nomination_node)
    graph.set_entry_point("table_nomination")
    graph.set_finish_point("table_nomination")
    return graph
