"""
MASIS State — The shared whiteboard (TypedDict) for LangGraph.

This is the single source of truth flowing through the DAG.
All agents read from and write to this state.
Context growth is controlled via message truncation and summarization.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from operator import add

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from masis.schemas import (
    AgentMessage,
    AgentRole,
    Citation,
    CritiqueResult,
    FinalReport,
    HITLRequest,
    SubTask,
    TaskPlan,
)


def _merge_messages(
    existing: list[AgentMessage], new: list[AgentMessage]
) -> list[AgentMessage]:
    """Append new messages while capping total length to prevent unbounded growth."""
    MAX_MESSAGES = 50
    merged = existing + new
    if len(merged) > MAX_MESSAGES:
        # Keep first 5 (original context) + last MAX_MESSAGES-5
        merged = merged[:5] + merged[-(MAX_MESSAGES - 5):]
    return merged


class MASISState(BaseModel):
    """
    The full runtime state of a MASIS execution.

    LangGraph nodes receive this state, may mutate their assigned fields,
    and return a partial dict with updates.
    """

    # ── User query ──────────────────────────────────────────
    original_query: str = ""
    clarified_query: str = ""           # After HITL or query rewriting

    # ── Task planning (Supervisor) ──────────────────────────
    task_plan: Optional[TaskPlan] = None
    current_task_id: Optional[str] = None

    # ── Shared whiteboard (message passing) ─────────────────
    messages: list[AgentMessage] = Field(default_factory=list)

    # ── Retrieved evidence (Researcher) ─────────────────────
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    research_iterations: int = 0

    # ── Critique (Skeptic) ──────────────────────────────────
    critique: Optional[CritiqueResult] = None
    skeptic_rounds: int = 0

    # ── Citations (Citation Engine) ─────────────────────────
    citations: list[Citation] = Field(default_factory=list)

    # ── Draft & final output (Synthesizer) ──────────────────
    draft_response: str = ""
    final_report: Optional[FinalReport] = None

    # ── HITL ────────────────────────────────────────────────
    hitl_request: Optional[HITLRequest] = None
    hitl_response: Optional[str] = None
    awaiting_human: bool = False

    # ── Control flow ────────────────────────────────────────
    next_agent: Optional[AgentRole] = None
    should_end: bool = False
    error_log: list[str] = Field(default_factory=list)
    iteration_count: int = 0           # Global safety counter
    max_iterations: int = 15           # Hard ceiling

    class Config:
        arbitrary_types_allowed = True
