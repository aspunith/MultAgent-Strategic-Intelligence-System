"""
Skeptic Agent — "The Auditor"

Responsibilities:
  • Actively search for hallucinations and logical gaps
  • Challenge assumptions and weak evidence
  • Detect contradictions between different sources
  • Verify that claims are actually supported by retrieved evidence
  • Assess overall confidence and decide if findings pass review

Uses: GPT-4 class model (primary) for rigorous logical analysis.
"""

from __future__ import annotations

from masis.config import get_config
from masis.llm_utils import get_primary_llm, invoke_llm_structured
from masis.schemas import (
    AgentMessage,
    AgentRole,
    CritiqueIssue,
    CritiqueResult,
    TaskStatus,
)
from masis.state import MASISState


# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

SKEPTIC_SYSTEM = """You are the Skeptic agent ("The Auditor") of a Multi-Agent Strategic Intelligence System.

Your job is to critically evaluate research findings for quality, accuracy, and logical soundness.

You will receive:
  1. The ORIGINAL USER QUERY
  2. RESEARCH FINDINGS from the Researcher agent
  3. The RAW RETRIEVED CHUNKS (evidence) that the findings are based on

Perform these checks:

1. HALLUCINATION DETECTION
   - Is every claim in the findings actually supported by the retrieved chunks?
   - Are there any statements that go beyond what the evidence says?
   - Are any statistics, names, or facts fabricated?

2. LOGICAL VALIDATION
   - Do the conclusions logically follow from the evidence?
   - Are there logical fallacies or unsupported leaps?

3. CONTRADICTION CHECK
   - Do any retrieved chunks contradict each other?
   - If so, are both viewpoints acknowledged?
   - Is there a reasonable resolution proposed?

4. EVIDENCE QUALITY
   - Is the evidence sufficient to answer the query?
   - Are critical aspects of the question left unaddressed?
   - Are the sources reliable and relevant?

5. COMPLETENESS
   - Are all parts of the user's question addressed?
   - Are there obvious follow-up questions that should be anticipated?

For each issue found, classify it as:
  - "hallucination": Claim not supported by evidence
  - "logical_gap": Missing reasoning step or logical fallacy
  - "weak_evidence": Claim supported by insufficient or questionable evidence
  - "contradiction": Conflicting information between sources
  - "incomplete": Missing coverage of part of the query

Rate severity as "low", "medium", or "high".

Set passes_review to true ONLY if there are no high-severity issues and
the overall quality is acceptable.

Set confidence_score between 0 and 1 based on overall quality."""


# ──────────────────────────────────────────────────────────────
# Few-shot examples for structured output guidance
# ──────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLE = """
EXAMPLE INPUT:
Query: "What is the company's revenue growth?"
Research: "Revenue grew 25% year-over-year reaching $5B."
Evidence: [Source 1: "Revenue increased 15% to $4.5B"]

EXAMPLE OUTPUT:
{
  "issues": [
    {
      "issue_type": "hallucination",
      "description": "Research claims 25% growth to $5B, but source says 15% to $4.5B",
      "affected_claim": "Revenue grew 25% year-over-year reaching $5B",
      "severity": "high",
      "suggested_action": "Correct to match source: 15% growth to $4.5B"
    }
  ],
  "overall_assessment": "Critical hallucination found — revenue figures fabricated.",
  "passes_review": false,
  "confidence_score": 0.2
}
"""


# ──────────────────────────────────────────────────────────────
# Node Function
# ──────────────────────────────────────────────────────────────

def skeptic_node(state: MASISState) -> dict:
    """
    Validate and critique research findings.
    Returns structured CritiqueResult with issues and confidence score.
    """
    cfg = get_config()

    # Collect all researcher messages as findings
    research_messages = [
        m for m in state.messages if m.sender == AgentRole.RESEARCHER
    ]
    if not research_messages:
        return {
            "critique": CritiqueResult(
                issues=[],
                overall_assessment="No research findings to evaluate.",
                passes_review=False,
                confidence_score=0.0,
            ),
            "skeptic_rounds": state.skeptic_rounds + 1,
            "messages": [
                AgentMessage(
                    sender=AgentRole.SKEPTIC,
                    content="No research findings available to critique.",
                )
            ],
        }

    findings = "\n\n".join(m.content for m in research_messages)

    # Format retrieved chunks as evidence
    evidence_parts = []
    for i, chunk in enumerate(state.retrieved_chunks[:15], 1):  # Cap to prevent context overflow
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")[:500]  # Truncate each chunk
        evidence_parts.append(f"[Evidence {i} from {source}]: {content}")
    evidence_str = "\n\n".join(evidence_parts) if evidence_parts else "(No raw evidence available)"

    user_prompt = f"""ORIGINAL QUERY: {state.clarified_query or state.original_query}

RESEARCH FINDINGS:
{findings[:4000]}

RAW RETRIEVED EVIDENCE:
{evidence_str}

{FEW_SHOT_EXAMPLE}

Now perform your critique:"""

    llm = get_primary_llm()
    critique = invoke_llm_structured(llm, SKEPTIC_SYSTEM, user_prompt, CritiqueResult)

    # Mark current task
    if state.task_plan and state.current_task_id:
        for task in state.task_plan.sub_tasks:
            if task.task_id == state.current_task_id:
                task.result = critique.overall_assessment
                task.status = TaskStatus.COMPLETED

    summary = (
        f"Review {'PASSED' if critique.passes_review else 'FAILED'} "
        f"(confidence: {critique.confidence_score:.2f}). "
        f"Issues found: {len(critique.issues)}."
    )

    return {
        "critique": critique,
        "skeptic_rounds": state.skeptic_rounds + 1,
        "task_plan": state.task_plan,
        "messages": [
            AgentMessage(
                sender=AgentRole.SKEPTIC,
                content=summary,
                metadata={
                    "passes_review": critique.passes_review,
                    "confidence": critique.confidence_score,
                    "issue_count": len(critique.issues),
                    "issues": [i.model_dump() for i in critique.issues[:5]],
                },
            )
        ],
    }
