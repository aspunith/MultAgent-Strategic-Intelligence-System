"""
Unit tests for MASIS schemas, state, config, and citation engine.
These tests do NOT require an API key — they validate structure and logic only.
"""

import pytest
from masis.schemas import (
    AgentMessage,
    AgentRole,
    Citation,
    ConfidenceLevel,
    CritiqueIssue,
    CritiqueResult,
    DocumentChunk,
    FinalReport,
    HITLRequest,
    SubTask,
    TaskPlan,
    TaskStatus,
)
from masis.state import MASISState
from masis.config import MASISConfig, get_config


# ──────────────────────────────────────────────────────────────
# Schema Tests
# ──────────────────────────────────────────────────────────────

class TestSchemas:
    def test_document_chunk_creation(self):
        chunk = DocumentChunk(
            source_document="report.pdf",
            content="Revenue grew 12%",
            relevance_score=0.95,
        )
        assert chunk.source_document == "report.pdf"
        assert chunk.chunk_id  # Auto-generated
        assert chunk.relevance_score == 0.95

    def test_citation_links_claim_to_evidence(self):
        chunk = DocumentChunk(source_document="data.md", content="Sales: $2.3B")
        citation = Citation(
            claim="Revenue was $2.3 billion",
            evidence=[chunk],
            confidence=0.9,
        )
        assert len(citation.evidence) == 1
        assert citation.confidence == 0.9
        assert citation.citation_id.startswith("cite-")

    def test_subtask_creation(self):
        task = SubTask(
            description="Retrieve financial data",
            assigned_to=AgentRole.RESEARCHER,
        )
        assert task.status == TaskStatus.PENDING
        assert task.retries == 0
        assert task.task_id.startswith("task-")

    def test_task_plan_with_dependencies(self):
        t1 = SubTask(description="Research", assigned_to=AgentRole.RESEARCHER)
        t2 = SubTask(
            description="Critique",
            assigned_to=AgentRole.SKEPTIC,
            depends_on=[t1.task_id],
        )
        plan = TaskPlan(original_query="Test", sub_tasks=[t1, t2])
        assert len(plan.sub_tasks) == 2
        assert t1.task_id in t2.depends_on

    def test_critique_result(self):
        issue = CritiqueIssue(
            issue_type="hallucination",
            description="Fabricated revenue figure",
            severity="high",
        )
        critique = CritiqueResult(
            issues=[issue],
            overall_assessment="Failed review",
            passes_review=False,
            confidence_score=0.3,
        )
        assert not critique.passes_review
        assert len(critique.issues) == 1

    def test_final_report_structure(self):
        report = FinalReport(
            query="What is Acme's revenue?",
            executive_summary="$2.3B in Q3 2025",
            detailed_analysis="Detailed breakdown...",
            recommendations=["Invest in cloud", "Reduce hardware"],
            confidence=ConfidenceLevel.HIGH,
        )
        assert report.confidence == ConfidenceLevel.HIGH
        assert len(report.recommendations) == 2

    def test_hitl_request(self):
        req = HITLRequest(
            reason="Ambiguous query",
            question_to_user="Which division?",
            options=["Cloud", "Hardware", "Both"],
        )
        assert len(req.options) == 3


# ──────────────────────────────────────────────────────────────
# State Tests
# ──────────────────────────────────────────────────────────────

class TestState:
    def test_initial_state(self):
        state = MASISState(original_query="Test query")
        assert state.original_query == "Test query"
        assert state.iteration_count == 0
        assert not state.should_end
        assert state.messages == []

    def test_state_serialization(self):
        state = MASISState(
            original_query="Test",
            messages=[
                AgentMessage(sender=AgentRole.SUPERVISOR, content="Planning...")
            ],
        )
        d = state.model_dump()
        assert isinstance(d, dict)
        assert d["original_query"] == "Test"
        assert len(d["messages"]) == 1

    def test_max_iterations_guard(self):
        state = MASISState(
            original_query="Test",
            iteration_count=15,
            max_iterations=15,
        )
        assert state.iteration_count >= state.max_iterations


# ──────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        cfg = MASISConfig()
        assert cfg.models.primary_model  # Not empty
        assert cfg.rag.chunk_size > 0
        assert cfg.agents.max_research_iterations > 0
        assert isinstance(cfg.hitl.enabled, bool)

    def test_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2


# ──────────────────────────────────────────────────────────────
# Citation Engine Tests
# ──────────────────────────────────────────────────────────────

class TestCitationEngine:
    def test_validate_report_with_valid_citations(self):
        from masis.citation_engine import CitationEngine

        chunks = [
            {"chunk_id": "c1", "source": "report.md", "content": "Revenue: $2.3B"},
            {"chunk_id": "c2", "source": "market.md", "content": "Market share: 4.2%"},
        ]
        engine = CitationEngine(chunks)

        report = FinalReport(
            query="Test",
            executive_summary="Summary",
            detailed_analysis="Revenue is $2.3B [Source 1]",
            citations=[
                Citation(
                    claim="Revenue is $2.3B",
                    evidence=[DocumentChunk(chunk_id="c1", source_document="report.md", content="Revenue: $2.3B")],
                    confidence=0.9,
                )
            ],
        )

        audit = engine.validate_report(report)
        assert audit.valid_citations == 1
        assert audit.coverage_score == 1.0

    def test_detect_orphaned_citations(self):
        from masis.citation_engine import CitationEngine

        engine = CitationEngine([])
        report = FinalReport(
            query="Test",
            executive_summary="Summary",
            detailed_analysis="No sources here",
            citations=[
                Citation(claim="Unsupported claim", evidence=[], confidence=0.0)
            ],
        )

        audit = engine.validate_report(report)
        assert len(audit.orphaned_claims) == 1
        assert audit.coverage_score == 0.0
