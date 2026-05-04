import json
from typing import Any, Dict

from utilis.db import get_pipeline_connection, config
from utilis.logger import logger


def ai_store_db_writer(
    run_id: str,
    stage: str,
    artifact_type: str,
    payload: Dict[str, Any],
    schema_version: str,
    prompt_version: str,
    faithfulness_status: str,
    faithfulness_warn_count: int = 0,
    retry_count: int = 0,
    token_count: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """
    Write AI artifact to existing ai_store table (no schema changes).
    """
    db_conf = config["azure_sql"]
    db_schema = db_conf["schema_name"]
    
    log_context = {
        "run_id": run_id,
        "node": "ai_store_writer",
        "stage": stage,
        "artifact_type": artifact_type,
    }
    
    conn = get_pipeline_connection()
    try:
        cursor = conn.cursor()
        
        # UPSERT using MERGE (exact match on fingerprint ONLY)
        cursor.execute(
            f"""
            INSERT INTO [{db_schema}].[ai_store] (
                run_id,
                fingerprint,
                stored_at,
                payload,
                stage,
                artifact_type,
                schema_version,
                prompt_version,
                faithfulness_status,
                faithfulness_warn_count,
                retry_count,
                token_count,
                input_tokens,
                output_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            WHEN NOT MATCHED THEN
                INSERT (fingerprint, stored_at, payload, stage, artifact_type, 
                        schema_version, prompt_version, faithfulness_status, 
                        faithfulness_warn_count, retry_count, token_count, 
                        input_tokens, output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            # USING param (Only the unique ID needed here)
            payload["fingerprint"] if "fingerprint" in payload else run_id,
            
            # WHEN MATCHED params
            payload.get("stored_at", "2024-01-01"), # Update the timestamp
            json.dumps(payload),
            stage,
            artifact_type,
            schema_version,
            prompt_version,
            faithfulness_status,
            faithfulness_warn_count,
            retry_count,
            token_count,
            input_tokens,
            output_tokens,
            
            # WHEN NOT MATCHED params (repeat all)
            payload["fingerprint"] if "fingerprint" in payload else run_id,
            payload.get("stored_at", "2024-01-01"),
            json.dumps(payload),
            stage,
            artifact_type,
            schema_version,
            prompt_version,
            faithfulness_status,
            faithfulness_warn_count,
            retry_count,
            token_count,
            input_tokens,
            output_tokens,
        )
        
        conn.commit()
        logger.info("✅ ai_store written (stage=%s)", stage, extra=log_context)
        
    except Exception as e:
        logger.error(f"ai_store write failed: {e}", extra=log_context)
        raise
    finally:
        conn.close()
