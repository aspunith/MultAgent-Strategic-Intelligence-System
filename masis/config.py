"""
MASIS Configuration — Central configuration management with environment variable loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", str(DATA_DIR / "documents")))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))


# ---------------------------------------------------------------------------
# Pydantic Settings
# ---------------------------------------------------------------------------
class ModelConfig(BaseModel):
    """LLM model routing configuration."""
    primary_model: str = Field(default_factory=lambda: os.getenv("PRIMARY_MODEL", "gpt-4o"))
    secondary_model: str = Field(default_factory=lambda: os.getenv("SECONDARY_MODEL", "gpt-4o-mini"))
    temperature_reasoning: float = 0.1
    temperature_creative: float = 0.4


class RAGConfig(BaseModel):
    """Retrieval-Augmented Generation settings."""
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    top_k_semantic: int = 8
    top_k_keyword: int = 5
    top_k_final: int = 6          # After re-ranking / fusion


class AgentConfig(BaseModel):
    """Agent behaviour guardrails."""
    max_research_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCH_ITERATIONS", "5"))
    )
    max_skeptic_challenges: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SKEPTIC_CHALLENGES", "3"))
    )
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    )
    rate_limit_rpm: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_RPM", "60"))
    )


class HITLConfig(BaseModel):
    """Human-in-the-Loop settings."""
    enabled: bool = Field(
        default_factory=lambda: os.getenv("HITL_ENABLED", "true").lower() == "true"
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("HITL_TIMEOUT_SECONDS", "300"))
    )


class MASISConfig(BaseModel):
    """Top-level configuration aggregating all sub-configs."""
    models: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)


@lru_cache(maxsize=1)
def get_config() -> MASISConfig:
    """Singleton accessor for the global config."""
    return MASISConfig()
