"""
MASIS Evaluation Framework — LLM-as-a-Judge for measuring system quality.

Core Metrics:
  • Faithfulness  — No hallucinations; claims match evidence
  • Relevance     — Answer matches user intent
  • Completeness  — All aspects of the question addressed

Additional Metrics:
  • Citation Quality — Citations are valid and specific
  • Consistency     — Repeated runs produce stable conclusions
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from masis.llm_utils import get_primary_llm, invoke_llm_structured
from masis.schemas import FinalReport


# ──────────────────────────────────────────────────────────────
# Evaluation Schemas
# ──────────────────────────────────────────────────────────────

class MetricScore(BaseModel):
    """A single evaluation metric result."""
    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    issues: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Complete evaluation of a MASIS run."""
    query: str
    faithfulness: MetricScore
    relevance: MetricScore
    completeness: MetricScore
    citation_quality: MetricScore
    overall_score: float = 0.0
    grade: str = ""  # A/B/C/D/F
    summary: str = ""


# ──────────────────────────────────────────────────────────────
# Judge Prompts
# ──────────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are an evaluation judge assessing FAITHFULNESS.

Faithfulness measures whether the response contains ONLY information supported by the provided evidence.
Any claim not backed by the evidence is a hallucination.

Rate on a scale of 0.0 to 1.0:
  1.0 = Every claim is directly supported by evidence
  0.5 = Some claims are supported, some are not
  0.0 = Most claims are fabricated

QUERY: {query}

RESPONSE:
{response}

EVIDENCE:
{evidence}

Evaluate faithfulness and return a MetricScore with:
- metric_name: "faithfulness"
- score: 0.0-1.0
- reasoning: Explain your assessment
- issues: List any hallucinated claims"""

RELEVANCE_PROMPT = """You are an evaluation judge assessing RELEVANCE.

Relevance measures whether the response actually answers what the user asked.
Off-topic information or tangential answers reduce relevance.

Rate on a scale of 0.0 to 1.0:
  1.0 = Directly and fully addresses the user's question
  0.5 = Partially relevant, some off-topic content
  0.0 = Does not address the user's question at all

QUERY: {query}

RESPONSE:
{response}

Evaluate relevance and return a MetricScore with:
- metric_name: "relevance"
- score: 0.0-1.0
- reasoning: Explain your assessment
- issues: List any irrelevant sections"""

COMPLETENESS_PROMPT = """You are an evaluation judge assessing COMPLETENESS.

Completeness measures whether ALL aspects of the user's question are addressed.
Break the query into sub-questions and check each is answered.

Rate on a scale of 0.0 to 1.0:
  1.0 = Every aspect of the question is thoroughly addressed
  0.5 = Some aspects addressed, others missing
  0.0 = The response barely covers the question

QUERY: {query}

RESPONSE:
{response}

Evaluate completeness and return a MetricScore with:
- metric_name: "completeness"
- score: 0.0-1.0
- reasoning: Explain your assessment
- issues: List any missing aspects"""

CITATION_QUALITY_PROMPT = """You are an evaluation judge assessing CITATION QUALITY.

Citation quality measures whether:
1. Claims are properly attributed to specific sources
2. Citations are relevant to the claims they support
3. There are no orphaned claims (important claims without citations)

Rate on a scale of 0.0 to 1.0:
  1.0 = Every important claim has a relevant, specific citation
  0.5 = Some citations present but incomplete coverage
  0.0 = No citations or all citations are irrelevant

QUERY: {query}

RESPONSE:
{response}

NUMBER OF CITATIONS: {num_citations}

Evaluate citation quality and return a MetricScore with:
- metric_name: "citation_quality"
- score: 0.0-1.0
- reasoning: Explain your assessment
- issues: List any citation problems"""


# ──────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────

class MASISEvaluator:
    """LLM-as-a-Judge evaluator for MASIS outputs."""

    def __init__(self):
        self._llm = get_primary_llm(temperature=0.0)

    def evaluate(
        self,
        report: FinalReport,
        evidence_text: str = "",
    ) -> EvaluationResult:
        """Run all evaluation metrics against a FinalReport."""
        full_response = (
            f"{report.executive_summary}\n\n"
            f"{report.detailed_analysis}\n\n"
            f"Recommendations:\n" + "\n".join(f"• {r}" for r in report.recommendations)
        )

        # Run each metric
        faithfulness = self._evaluate_metric(
            FAITHFULNESS_PROMPT.format(
                query=report.query,
                response=full_response[:4000],
                evidence=evidence_text[:4000],
            )
        )

        relevance = self._evaluate_metric(
            RELEVANCE_PROMPT.format(
                query=report.query,
                response=full_response[:4000],
            )
        )

        completeness = self._evaluate_metric(
            COMPLETENESS_PROMPT.format(
                query=report.query,
                response=full_response[:4000],
            )
        )

        citation_quality = self._evaluate_metric(
            CITATION_QUALITY_PROMPT.format(
                query=report.query,
                response=full_response[:4000],
                num_citations=len(report.citations),
            )
        )

        # Compute overall score (weighted average)
        overall = (
            faithfulness.score * 0.35
            + relevance.score * 0.25
            + completeness.score * 0.25
            + citation_quality.score * 0.15
        )

        grade = self._score_to_grade(overall)

        return EvaluationResult(
            query=report.query,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            citation_quality=citation_quality,
            overall_score=round(overall, 3),
            grade=grade,
            summary=(
                f"Overall: {grade} ({overall:.1%}). "
                f"Faithfulness: {faithfulness.score:.1%}, "
                f"Relevance: {relevance.score:.1%}, "
                f"Completeness: {completeness.score:.1%}, "
                f"Citations: {citation_quality.score:.1%}."
            ),
        )

    def _evaluate_metric(self, prompt: str) -> MetricScore:
        """Run a single metric evaluation."""
        try:
            return invoke_llm_structured(
                self._llm,
                "You are a strict but fair evaluation judge. Return structured scores.",
                prompt,
                MetricScore,
            )
        except Exception as exc:
            return MetricScore(
                metric_name="error",
                score=0.0,
                reasoning=f"Evaluation failed: {exc}",
                issues=[str(exc)],
            )

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        return "F"
