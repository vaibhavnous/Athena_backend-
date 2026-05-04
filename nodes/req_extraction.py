from __future__ import annotations

import os
from dotenv import load_dotenv

# MUST be called before any LangChain or OpenAI imports
load_dotenv()

import json
from typing import Any, Callable, Dict, List, Optional

import pydantic
from pydantic import BaseModel, Field
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from state import Stage01State
from schema import RequirementsSchema
from utilis.logger import logger
from utilis.db import ai_store_db_writer


class TokenAccumulator(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.total_input: int = 0
        self.total_output: int = 0

    @property
    def total(self) -> int:
        return self.total_input + self.total_output

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage = response.llm_output.get("token_usage", {})
            self.total_input += usage.get("prompt_tokens", 0)
            self.total_output += usage.get("completion_tokens", 0)
        except AttributeError:
            pass

    def reset(self) -> None:
        self.total_input = 0
        self.total_output = 0


_LLM_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-35-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "_default": {"input": 0.001, "output": 0.002},
}

_ACTIVE_MODEL: str = "gpt-4o-mini"


def set_active_model(model_name: str) -> None:
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model_name


def compute_cost_usd(input_tokens: int, output_tokens: int) -> float:
    pricing = _LLM_PRICING.get(_ACTIVE_MODEL, _LLM_PRICING["_default"])
    cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
    return round(cost, 6)


def get_llm(provider: str = "azure_openai", model: str | None = None, temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
    _model = model or _ACTIVE_MODEL

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=_model, 
            temperature=temperature, 
            api_version="2024-02-15-preview",
            **kwargs
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=_model, temperature=temperature, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def handoff_validator(stage_label: str, state: Dict[str, Any], required_keys: List[str]) -> None:
    missing = [k for k in required_keys if state.get(k) is None]
    if missing:
        raise ValueError(f"[{stage_label}] Handoff validation failed. Missing required keys: {missing}")
    logger.debug("[%s] Handoff validation passed (%d keys)", stage_label, len(required_keys), extra={"node": "req_extraction"})


def check_faithfulness(constraints: List[str], brd_text: str) -> List[str]:
    brd_lower = brd_text.lower()
    ungrounded: List[str] = []
    for constraint in constraints:
        key_words = [w for w in constraint.lower().split() if len(w) > 4]
        if key_words and not any(w in brd_lower for w in key_words):
            logger.warning("Faithfulness WARN — constraint not in BRD: %r", constraint, extra={"node": "req_extraction"})
            ungrounded.append(constraint)
    return ungrounded


PROMPT_REQ_V1 = """\
You are a business requirements analyst. Read the BRD below and extract structured requirements.

BRD TEXT:
{brd_text}

Return a JSON object with EXACTLY these fields — no markdown fences, no explanation:
{{
  "business_objective": "<one sentence: the primary business goal>",
  "data_domains": ["<domain1>", "<domain2>"],
  "reporting_frequency": "<daily|weekly|monthly|quarterly|adhoc>",
  "target_audience": "<who will consume these KPIs>",
  "constraints": ["<constraint1>", "<constraint2>"]
}}

Rules:
- reporting_frequency MUST be exactly one of: daily, weekly, monthly, quarterly, adhoc
- constraints MUST be verifiable in the BRD text — do not invent constraints
- data_domains MUST be actual business domains mentioned in the BRD
- Return ONLY the JSON object"""

SYSTEM_MSG = "You are a precise business analyst. Return only valid JSON."


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rsplit("```", 1)[0].strip()
    return raw


def build_req_extraction_node(
    llm: Optional[BaseChatModel] = None,
    llm_provider: str = "azure_openai",
    max_retries: int = 2,
) -> Callable[[Stage01State], Stage01State]:
    
    _llm = llm or get_llm(provider=llm_provider)

    def req_extraction_node(state: Stage01State) -> Stage01State:
        log_context = {"run_id": state.get("run_id", "unknown"), "node": "req_extraction"}
        
        # Early exit guards
        if state.get("status") == "FAILED":
            logger.warning("Skipping req_extraction: status=FAILED", extra=log_context)
            return state
        handoff_validator("Requirement Extraction", state, ["run_id", "brd_text", "fingerprint"])

        brd_text = state["brd_text"]
        run_id = state["run_id"]
        fingerprint = state["fingerprint"]

        if state.get("memory_bypass"):
            logger.info("Requirement Extraction - reusing exact-match requirements", extra=log_context)
            payload = {
                "business_objective": state.get("req_business_objective", ""),
                "data_domains": state.get("req_data_domains", []),
                "reporting_frequency": state.get("req_reporting_frequency", ""),
                "target_audience": state.get("req_target_audience", ""),
                "constraints": state.get("req_constraints", []),
                "schema_valid": state.get("req_schema_valid", True),
                "prompt_version": state.get("req_prompt_version", "PROMPT_REQ_v1"),
                "fingerprint": fingerprint,
                "run_id": run_id,
                "cost_usd": 0.0,
                "source": "MEMORY_LAYER1",
            }
            ai_store_db_writer(
                run_id=run_id,
                stage="Requirement Extraction",
                artifact_type="REQUIREMENTS",
                payload=payload,
                schema_version="RequirementsSchema_v1",
                prompt_version="PROMPT_REQ_v1",
                faithfulness_status=state.get("req_faithfulness_status", "PASSED") or "PASSED",
                retry_count=0,
                token_count=0,
                input_tokens=0,
                output_tokens=0,
                fingerprint=fingerprint,
            )
            new_state = state.copy()
            new_state.update({
                "req_agent_attempts": 0,
                "req_tokens_used": 0,
                "req_cost_usd": 0.0,
                "req_faithfulness_status": state.get("req_faithfulness_status", "PASSED") or "PASSED",
            })
            return new_state

        logger.info("Requirement Extraction — starting extraction (run_id=%s)", run_id, extra=log_context)

        token_acc = TokenAccumulator()
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            prompt_text = PROMPT_REQ_V1.format(brd_text=brd_text[:12_000])
            if attempt > 0 and last_error:
                prompt_text += f"\n\n--- PREVIOUS ATTEMPT FAILED ---\nError: {last_error}\nFix this specific issue and return valid JSON."

            logger.info("Requirement Extraction — attempt %d / %d", attempt + 1, max_retries + 1, extra=log_context)

            try:
                response = _llm.invoke(
                    [SystemMessage(content=SYSTEM_MSG), HumanMessage(content=prompt_text)],
                    config={"callbacks": [token_acc]},
                )

                raw = _strip_fences(response.content)
                parsed = json.loads(raw)
                result = RequirementsSchema(**parsed)

                ungrounded = check_faithfulness(result.constraints, brd_text)

                if ungrounded and attempt < max_retries:
                    last_error = f"These constraints are NOT in the BRD: {ungrounded}. Only include constraints explicitly stated in the BRD."
                    logger.warning("Retrying due to ungrounded constraints: %s", ungrounded, extra=log_context)
                    continue

                for c in ungrounded:
                    result.constraints.remove(c)

                faithfulness_status = "WARN" if ungrounded else "PASSED"
                
                # Calculate cost before payload construction so we can log it inside the DB json
                cost = compute_cost_usd(token_acc.total_input, token_acc.total_output)

                # Store to DB
                ai_store_db_writer(
                    run_id=run_id,
                    stage="Requirement Extraction",
                    artifact_type="REQUIREMENTS" if not ungrounded else "REQUIREMENTS_WARN",
                    payload={
                        **result.model_dump(),
                        "fingerprint": fingerprint,
                        "run_id": run_id,
                        "cost_usd": cost,
                    },
                    schema_version="RequirementsSchema_v1",
                    prompt_version="PROMPT_REQ_v1",
                    faithfulness_status=faithfulness_status,
                    faithfulness_warn_count=len(ungrounded),
                    retry_count=attempt,
                    token_count=token_acc.total,
                    input_tokens=token_acc.total_input,
                    output_tokens=token_acc.total_output,
                    fingerprint=fingerprint,
                )

                logger.info("Requirement Extraction — success (attempt=%d tokens=%d)", attempt + 1, token_acc.total, extra=log_context)

                new_state = state.copy()
                new_state.update({
                    "req_business_objective": result.business_objective,
                    "req_data_domains": result.data_domains,
                    "req_reporting_frequency": result.reporting_frequency,
                    "req_target_audience": result.target_audience,
                    "req_constraints": result.constraints,
                    "req_schema_valid": result.schema_valid,
                    "req_prompt_version": result.prompt_version,
                    "req_agent_attempts": attempt + 1,
                    "req_tokens_used": token_acc.total,
                    "req_cost_usd": cost,
                    "req_faithfulness_status": faithfulness_status,
                })

                return new_state

            except (json.JSONDecodeError, pydantic.ValidationError) as exc:
                last_error = str(exc)
                logger.warning("Attempt %d failed (parse/validate): %s", attempt + 1, last_error[:200], extra=log_context)

            except Exception as exc:
                last_error = str(exc)[:200]
                logger.error("Attempt %d failed (unexpected): %s", attempt + 1, last_error, extra=log_context)

        # If we exit the loop, it means max retries were exhausted
        ai_store_db_writer(
            run_id=run_id,
            stage="Requirement Extraction",
            artifact_type="REQUIREMENTS_FAILED",
            payload={
                "error": last_error,
                "attempts": max_retries + 1,
                "fingerprint": fingerprint,
            },
            schema_version="RequirementsSchema_v1",
            prompt_version="PROMPT_REQ_v1",
            faithfulness_status="FAILED",
            retry_count=max_retries,
            token_count=token_acc.total,
            input_tokens=token_acc.total_input,
            output_tokens=token_acc.total_output,
            fingerprint=fingerprint,
        )
        logger.error("Requirement Extraction failed after %d attempts", max_retries + 1, extra=log_context)
        return {**state, "status": "FAILED", "error": f"Req extraction failed: {last_error}"}

    return req_extraction_node


req_extraction_node = build_req_extraction_node()
