from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List
from enum import Enum
import re

class BRDSchema(BaseModel):
    """Pydantic model to enforce business requirement standards."""
    content: str = Field(..., min_length=200)
    
    @field_validator("content")
    @classmethod
    def check_business_keywords(cls, v: str) -> str:
        keywords = ["requirement", "business", "user", "system", "feature", "process", "workflow"]
        if not any(word in v.lower() for word in keywords):
            raise ValueError("Text does not contain standard business requirements keywords.")
        return v

class RequirementsSchema(BaseModel):
    business_objective: str = Field(..., min_length=10)
    data_domains: List[str] = Field(default_factory=list)
    reporting_frequency: str = Field(..., pattern=r"^(daily|weekly|monthly|quarterly|adhoc)$")
    target_audience: str = Field(..., min_length=5)
    constraints: List[str] = Field(default_factory=list)
    schema_valid: bool = True
    prompt_version: str = "PROMPT_REQ_v1"

class DerivationType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"

class KPISchemaItem(BaseModel):
    """Strict KPI schema per requirements."""
    kpi_name: str = Field(..., min_length=1, max_length=200)
    kpi_description: str = Field(..., min_length=10, max_length=1000)
    ai_confidence_score: float = Field(ge=0.0, le=1.0)
    derivation_type: DerivationType
    source_requirement_ref: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def validate_kpi(self):
        if self.derivation_type == DerivationType.EXPLICIT and self.ai_confidence_score < 0.7:
            raise ValueError("Explicit KPIs must have confidence >= 0.7")
        if self.derivation_type == DerivationType.IMPLICIT and self.ai_confidence_score < 0.4:
            raise ValueError("Implicit KPIs must have confidence >= 0.4")
        return self

class KPISchema(BaseModel):
    """List of KPIs with max 25."""
    kpis: List[KPISchemaItem] = Field(..., max_items=25)


# ── Table Nomination Schema ────────────────────────────

class NominationItem(BaseModel):
    """A single table nomination result from hybrid search."""
    table_name: str = Field(..., min_length=1, description="Name of the nominated table")
    schema_name: str = Field(..., min_length=1, description="SQL schema (e.g., dbo, metadata)")
    database_name: str = Field(..., min_length=1, description="Source database name")
    nomination_reason: str = Field(..., min_length=1, description="Human-readable match reason")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Fusion confidence score")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords matched in table or columns")
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of keywords covered by this table")

    @field_validator("confidence_score")
    @classmethod
    def check_confidence_precision(cls, v: float) -> float:
        return round(v, 4)

    @field_validator("coverage_ratio")
    @classmethod
    def check_coverage_precision(cls, v: float) -> float:
        return round(v, 4)


class NominationSchema(BaseModel):
    """Container for all table nominations produced by the hybrid search."""
    nominations: List[NominationItem] = Field(default_factory=list, max_items=100)
