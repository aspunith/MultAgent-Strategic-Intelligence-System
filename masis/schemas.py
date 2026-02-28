"""
MASIS Schemas — Typed Pydantic models for structured agent I/O, state, and citations.

Every data object flowing between agents is defined here so that:
  1. Agents produce predictable, parseable outputs.
  2. The citation engine can trace every claim back to evidence.
  3. The evaluation layer can score faithfulness automatically.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    SKEPTIC = "skeptic"
    SYNTHESIZER = "synthesizer"
    HUMAN = "human"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_HUMAN_INPUT = "needs_human_input"


class ConfidenceLevel(str, Enum):
    HIGH = "high"          # >=0.8
    MEDIUM = "medium"      # 0.5–0.8
    LOW = "low"            # <0.5


# ──────────────────────────────────────────────────────────────
# Evidence & Citation
# ──────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single chunk retrieved from the vector store."""
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_document: str          # Filename or URL
    page_or_section: str = ""     # Page number, heading, etc.
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0  # From retrieval


class Citation(BaseModel):
    """Links a specific claim to its supporting evidence."""
    citation_id: str = Field(default_factory=lambda: f"cite-{uuid.uuid4().hex[:8]}")
    claim: str                    # The statement being supported
    evidence: list[DocumentChunk] # One or more chunks backing the claim
    confidence: float = 0.0       # 0-1 score


# ──────────────────────────────────────────────────────────────
# Sub-task & Plan
# ──────────────────────────────────────────────────────────────

class SubTask(BaseModel):
    """A discrete unit of work created by the Supervisor."""
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    description: str
    assigned_to: AgentRole
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)   # task_ids
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0


class TaskPlan(BaseModel):
    """The DAG of sub-tasks produced by the Supervisor."""
    plan_id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    original_query: str
    sub_tasks: list[SubTask] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ──────────────────────────────────────────────────────────────
# Agent Messages (shared whiteboard)
# ──────────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    """A single message on the shared whiteboard between agents."""
    sender: AgentRole
    content: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ──────────────────────────────────────────────────────────────
# Skeptic Critique
# ──────────────────────────────────────────────────────────────

class CritiqueIssue(BaseModel):
    """A single issue found by the Skeptic."""
    issue_type: str          # "hallucination", "logical_gap", "weak_evidence", "contradiction"
    description: str
    affected_claim: str = ""
    severity: str = "medium"  # "low", "medium", "high"
    suggested_action: str = ""


class CritiqueResult(BaseModel):
    """Full output of the Skeptic agent."""
    issues: list[CritiqueIssue] = Field(default_factory=list)
    overall_assessment: str = ""
    passes_review: bool = False
    confidence_score: float = 0.0


# ──────────────────────────────────────────────────────────────
# Final Output
# ──────────────────────────────────────────────────────────────

class FinalReport(BaseModel):
    """The structured output returned to the user."""
    query: str
    executive_summary: str
    detailed_analysis: str
    recommendations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    audit_trail: list[AgentMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# HITL
# ──────────────────────────────────────────────────────────────

class HITLRequest(BaseModel):
    """Request for human clarification."""
    reason: str
    question_to_user: str
    context_summary: str = ""
    options: list[str] = Field(default_factory=list)
