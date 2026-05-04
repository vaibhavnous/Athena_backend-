import json
import io
import os
from typing import Any, Dict, List, Optional, Tuple
from contextlib import redirect_stderr, redirect_stdout

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from nodes.ingestion import _chunk_and_embed, finalize_ingestion_after_memory
from state import Stage01State
from utilis.db import config, get_pipeline_connection
from utilis.logger import logger


load_dotenv()
DEV_MODE = os.getenv("DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on", "dev"}

db_conf = config.get("azure_sql", {})
db_schema = db_conf.get("schema_name", "dbo")
pinecone_conf = config.get("pinecone", {})

if DEV_MODE:
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
else:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")


def _copy_state(state: Stage01State) -> Stage01State:
    return state.copy()


def _log_context(state: Stage01State) -> dict:
    return {
        "run_id": state.get("run_id", "unknown"),
        "node": "memory_lookup",
        "fingerprint": state.get("fingerprint"),
    }


def _fetch_latest_payload(fingerprint: str, artifact_types: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    conn = get_pipeline_connection()
    try:
        cursor = conn.cursor()
        placeholders = ", ".join("?" for _ in artifact_types)
        cursor.execute(
            f"""
            SELECT TOP 1 payload
            FROM [{db_schema}].[ai_store]
            WHERE fingerprint = ? AND artifact_type IN ({placeholders})
            ORDER BY stored_at DESC
            """,
            (fingerprint, *artifact_types),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
    finally:
        conn.close()
    return None


def _fetch_exact_match(fingerprint: str, state: Stage01State) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    if state.get("skip_db", False):
        logger.info("DB skipped (test mode)")
        return False, {}, {}

    requirements_payload = _fetch_latest_payload(
        fingerprint,
        ("REQUIREMENTS", "REQUIREMENTS_WARN"),
    ) or {}
    kpis_payload = _fetch_latest_payload(
        fingerprint,
        ("KPIS",),
    ) or {}

    has_requirements = bool(requirements_payload.get("business_objective"))
    has_kpis = bool(kpis_payload.get("kpis"))
    return has_requirements or has_kpis, requirements_payload, kpis_payload


def _fetch_context_kpis(embeddings: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    kpis = []
    try:
        pc = Pinecone(api_key=pinecone_conf.get("api_key") or os.getenv("PINECONE_API_KEY"))
        index = pc.Index(pinecone_conf.get("index_name", "ai-store-index"))
        res = index.query(
            vector=embeddings,
            top_k=top_k,
            include_metadata=True,
            filter={"artifact_type": "KPIS"},
            namespace="global",
        )
        for match in res.matches:
            fp = match.metadata.get("fingerprint")
            if fp:
                payload = _fetch_latest_payload(fp, ("KPIS",)) or {}
                kpis.extend(payload.get("kpis", [])[:3])
    except Exception as e:
        logger.warning("Context KPIs fetch failed: %s", str(e))
    return kpis


def _fetch_rejected_kpis(fingerprint: str, limit: int = 10) -> List[str]:
    rejected = []
    conn = get_pipeline_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT TOP (?) payload
            FROM [{db_schema}].[ai_store]
            WHERE fingerprint != ? AND artifact_type = 'KPIS' AND (faithfulness_status = 'FAILED' OR cost_usd = 0)
            ORDER BY stored_at DESC
            """,
            (limit, fingerprint),
        )
        for row in cursor.fetchall():
            payload = json.loads(row[0])
            kpis = payload.get("kpis", [])
            rejected.extend([kpi["kpi_name"] for kpi in kpis if kpi.get("ai_confidence_score", 1.0) < 0.3])
    finally:
        conn.close()
    return rejected[:20]


def _apply_match_result(
    state: Stage01State,
    is_match: bool,
    requirements_payload: Dict[str, Any],
    kpis_payload: Dict[str, Any],
    log_context: dict,
) -> Stage01State:
    new_state = state.copy()
    if is_match:
        stored_kpis = kpis_payload.get("kpis", [])
        logger.info(
            "Layer 1 hit: exact match reused (has_requirements=%s has_kpis=%s)",
            bool(requirements_payload),
            bool(stored_kpis),
            extra=log_context,
        )
        new_state.update({
            "memory_layer1": True,
            "memory_bypass": True,
            "prior_kpis": stored_kpis,
            "kpis": stored_kpis,
            "kpi_source": "MEMORY_LAYER1" if stored_kpis else new_state.get("kpi_source"),
            "req_business_objective": requirements_payload.get("business_objective", new_state.get("req_business_objective")),
            "req_data_domains": requirements_payload.get("data_domains", new_state.get("req_data_domains", [])),
            "req_reporting_frequency": requirements_payload.get("reporting_frequency", new_state.get("req_reporting_frequency")),
            "req_target_audience": requirements_payload.get("target_audience", new_state.get("req_target_audience")),
            "req_constraints": requirements_payload.get("constraints", new_state.get("req_constraints", [])),
            "req_schema_valid": requirements_payload.get("schema_valid", new_state.get("req_schema_valid")),
            "req_prompt_version": requirements_payload.get("prompt_version", new_state.get("req_prompt_version")),
            "status": "EXACT_MATCH_FOUND",
        })
    else:
        logger.info("Layer 1 miss", extra=log_context)
        new_state.update({
            "memory_layer1": False,
            "memory_bypass": False,
            "status": "NO_EXACT_MATCH",
        })
    return new_state


def _run_semantic_lookup(state: Stage01State, log_context: dict) -> Stage01State:
    new_state = _copy_state(state)
    logger.info("START: semantic lookup", extra=log_context)

    try:
        pinecone_api_key = pinecone_conf.get("api_key") or os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            logger.warning("No Pinecone API key", extra=log_context)
            new_state["memory_layer2"] = False
            return new_state

        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(pinecone_conf.get("index_name", "ai-store-index"))

        text = new_state.get("brd_text", "").strip()
        if not text:
            logger.warning("No text for embedding", extra=log_context)
            new_state["memory_layer2"] = False
            return new_state

        emb = emb_model.encode(text).tolist()
        namespace = "global"

        res = pinecone_index.query(
            vector=emb,
            top_k=1,
            include_metadata=True,
            namespace=namespace,
        )

        log_context["query_namespace"] = namespace

        if res.matches:
            score = float(res.matches[0].score)
            logger.info("Semantic score %.3f", score, extra=log_context)

            if score >= 0.75:
                new_state["memory_layer2"] = True
                new_state["context_kpis"] = _fetch_context_kpis(emb)
                logger.info("Layer2 KPI context: %d KPIs", len(new_state["context_kpis"]), extra=log_context)
            else:
                new_state["memory_layer2"] = False
        else:
            new_state["memory_layer2"] = False

        new_state["rejected_kpis"] = _fetch_rejected_kpis(new_state["fingerprint"])

    except Exception as e:
        logger.error("Semantic lookup error: %s", str(e), extra=log_context)
        new_state["memory_layer2"] = False

    return new_state


def memory_lookup_node(state: Stage01State) -> Stage01State:
    new_state = _copy_state(state)
    log_context = _log_context(new_state)

    logger.info("START memory_lookup + KPI memory", extra=log_context)

    if new_state.get("status") == "FAILED":
        return new_state

    is_match, requirements_payload, kpis_payload = _fetch_exact_match(new_state.get("fingerprint"), new_state)
    new_state = _apply_match_result(new_state, is_match, requirements_payload, kpis_payload, log_context)

    new_state["memory_layer2"] = False
    if not new_state.get("memory_bypass", False):
        new_state = _run_semantic_lookup(new_state, log_context)

    if not new_state.get("memory_bypass", False):
        logger.info("RUNNING EMBEDDING", extra=log_context)
        new_state = _chunk_and_embed(new_state)
        new_state = finalize_ingestion_after_memory(new_state)

    logger.info(
        "END memory_lookup: layer1=%s layer2=%s prior_n=%d context_n=%d rejected_n=%d",
        new_state.get("memory_layer1"),
        new_state.get("memory_layer2"),
        len(new_state.get("prior_kpis", [])),
        len(new_state.get("context_kpis", [])),
        len(new_state.get("rejected_kpis", [])),
        extra=log_context,
    )
    return new_state
