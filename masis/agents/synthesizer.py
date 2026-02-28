"""
Synthesizer Agent — "The Writer"

Responsibilities:
  • Produce the final response with executive summary, analysis, and recommendations
  • Include inline citations for every substantive claim
  • Integrate skeptic feedback to address identified issues
  • Structure output as a FinalReport with full audit trail

Uses: GPT-4o-mini (secondary) model for drafting; Citation Engine for tracing.
"""

from __future__ import annotations

from masis.config import get_config
from masis.llm_utils import get_secondary_llm, get_primary_llm, invoke_llm
from masis.schemas import (
    AgentMessage,
    AgentRole,
    Citation,
    ConfidenceLevel,
    DocumentChunk,
    FinalReport,
    TaskStatus,
)
from masis.state import MASISState


# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are the Synthesizer agent ("The Writer") of a Multi-Agent Strategic Intelligence System.

Your job is to produce the FINAL response to the user's query based on:
  1. Research findings from the Researcher agent
  2. Critique and validation from the Skeptic agent
  3. All retrieved evidence chunks

CRITICAL RULES:
- Every substantive claim MUST include a citation reference like [Source N] 
  that traces back to a specific retrieved chunk.
- If the Skeptic identified issues (hallucinations, gaps), you MUST address them:
  • Correct hallucinated claims
  • Acknowledge evidence gaps honestly
  • Present contradictory evidence from both sides
- Do NOT add information beyond what the evidence supports.
- Structure your response as follows:

EXECUTIVE SUMMARY:
(2-3 sentence overview answering the core question)

DETAILED ANALYSIS:
(Thorough analysis with [Source N] citations throughout)

RECOMMENDATIONS:
(Actionable recommendations, each on its own line, prefixed with "• ")

CONFIDENCE ASSESSMENT:
(State your confidence in the answer: HIGH / MEDIUM / LOW and explain why)

EVIDENCE GAPS:
(What the available evidence does NOT cover, if anything)"""

CITATION_EXTRACTION = """Extract all citation references from the following text.
For each [Source N] reference, identify:
1. The claim it supports
2. The source number N

Return a list in this format:
CLAIM: <the claim>
SOURCE: <N>

If there are no citations, return "NO_CITATIONS"."""


# ──────────────────────────────────────────────────────────────
# Node Function
# ──────────────────────────────────────────────────────────────

def synthesizer_node(state: MASISState) -> dict:
    """
    Produce the final report by synthesizing research, skeptic feedback, and evidence.
    """
    # Gather all research findings
    research_messages = [m for m in state.messages if m.sender == AgentRole.RESEARCHER]
    findings = "\n\n".join(m.content for m in research_messages) if research_messages else "(No research available)"

    # Gather skeptic feedback
    skeptic_messages = [m for m in state.messages if m.sender == AgentRole.SKEPTIC]
    critique_text = "\n\n".join(m.content for m in skeptic_messages) if skeptic_messages else "(No critique available)"

    # Format evidence
    evidence_parts = []
    for i, chunk in enumerate(state.retrieved_chunks[:12], 1):
        source = chunk.get("source", "unknown")
        cid = chunk.get("chunk_id", "n/a")
        content = chunk.get("content", "")[:600]
        evidence_parts.append(f"[Source {i}: {source} | ID: {cid}]\n{content}")
    evidence_str = "\n\n---\n\n".join(evidence_parts) if evidence_parts else "(No evidence)"

    user_prompt = f"""ORIGINAL QUERY: {state.clarified_query or state.original_query}

RESEARCH FINDINGS:
{findings[:5000]}

SKEPTIC CRITIQUE:
{critique_text[:2000]}

AVAILABLE EVIDENCE:
{evidence_str}

Now produce the final report:"""

    # Use primary model for final synthesis to ensure quality
    llm = get_primary_llm(temperature=0.3)
    response = invoke_llm(llm, SYNTHESIS_SYSTEM, user_prompt)

    # Parse response into structured sections
    exec_summary, detailed, recs, confidence = _parse_synthesis(response)

    # Build citations from retrieved chunks
    citations = _build_citations(response, state.retrieved_chunks)

    # ── Explicit confidence aggregation ──────────────────────
    # Formula: final_confidence = 0.6 * skeptic_confidence + 0.25 * citation_ratio + 0.15 * evidence_coverage
    #   - skeptic_confidence: Skeptic agent's validation score (0-1)
    #   - citation_ratio:     fraction of claims that have [Source N] citations
    #   - evidence_coverage:  fraction of retrieved chunks actually cited
    skeptic_confidence = state.critique.confidence_score if state.critique else 0.0

    # Citation ratio: how many citations vs expected (at least 1 per evidence chunk used)
    expected_citations = max(len(state.retrieved_chunks), 1)
    citation_ratio = min(len(citations) / expected_citations, 1.0)

    # Evidence coverage: how many unique chunks are cited
    cited_sources = {c.source_chunk_id for c in citations if c.source_chunk_id}
    total_chunks = max(len(state.retrieved_chunks), 1)
    evidence_coverage = min(len(cited_sources) / total_chunks, 1.0)

    final_confidence = (
        0.60 * skeptic_confidence
        + 0.25 * citation_ratio
        + 0.15 * evidence_coverage
    )

    # Map aggregated score to confidence level
    if final_confidence >= 0.8:
        conf_level = ConfidenceLevel.HIGH
    elif final_confidence >= 0.5:
        conf_level = ConfidenceLevel.MEDIUM
    else:
        conf_level = ConfidenceLevel.LOW

    final_report = FinalReport(
        query=state.clarified_query or state.original_query,
        executive_summary=exec_summary,
        detailed_analysis=detailed,
        recommendations=recs,
        citations=citations,
        confidence=conf_level,
        audit_trail=state.messages,
        metadata={
            "research_iterations": state.research_iterations,
            "skeptic_rounds": state.skeptic_rounds,
            "total_chunks_retrieved": len(state.retrieved_chunks),
            "confidence_breakdown": {
                "skeptic_confidence": round(skeptic_confidence, 3),
                "citation_ratio": round(citation_ratio, 3),
                "evidence_coverage": round(evidence_coverage, 3),
                "final_score": round(final_confidence, 3),
            },
        },
    )

    # Mark task complete
    if state.task_plan and state.current_task_id:
        for task in state.task_plan.sub_tasks:
            if task.task_id == state.current_task_id:
                task.result = "Final report generated"
                task.status = TaskStatus.COMPLETED

    return {
        "draft_response": response,
        "final_report": final_report,
        "citations": citations,
        "should_end": True,
        "task_plan": state.task_plan,
        "messages": [
            AgentMessage(
                sender=AgentRole.SYNTHESIZER,
                content=f"Final report generated. Confidence: {conf_level.value}. "
                        f"Citations: {len(citations)}.",
                citations=citations,
            )
        ],
    }


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _parse_synthesis(text: str) -> tuple[str, str, list[str], str]:
    """Parse the LLM response into structured sections."""
    sections = {
        "EXECUTIVE SUMMARY:": "",
        "DETAILED ANALYSIS:": "",
        "RECOMMENDATIONS:": "",
        "CONFIDENCE ASSESSMENT:": "",
    }

    current_section = None
    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        matched = False
        for header in sections:
            if stripped.upper().startswith(header.rstrip(":")):
                current_section = header
                # Check if content is on the same line
                rest = stripped[len(header):].strip() if stripped.upper().startswith(header) else ""
                if rest:
                    sections[header] = rest
                matched = True
                break
        if not matched and current_section:
            sections[current_section] += line + "\n"

    exec_summary = sections["EXECUTIVE SUMMARY:"].strip() or text[:500]
    detailed = sections["DETAILED ANALYSIS:"].strip() or text
    rec_text = sections["RECOMMENDATIONS:"].strip()
    recs = [
        r.strip().lstrip("•-* ").strip()
        for r in rec_text.split("\n")
        if r.strip() and r.strip() not in ("", "-")
    ]
    confidence = sections["CONFIDENCE ASSESSMENT:"].strip()

    return exec_summary, detailed, recs, confidence


def _build_citations(response_text: str, chunks: list[dict]) -> list[Citation]:
    """Build citation objects linking claims to evidence chunks."""
    citations: list[Citation] = []
    import re

    # Find all [Source N] references in the response
    refs = re.findall(r"\[Source\s*(\d+)[^\]]*\]", response_text)
    seen_sources: set[int] = set()

    for ref_str in refs:
        try:
            idx = int(ref_str) - 1  # Convert to 0-based
        except ValueError:
            continue

        if idx in seen_sources or idx < 0 or idx >= len(chunks):
            continue
        seen_sources.add(idx)

        chunk = chunks[idx]

        # Find the sentence containing this reference
        pattern = rf"[^.]*\[Source\s*{ref_str}[^\]]*\][^.]*\.?"
        claim_match = re.search(pattern, response_text)
        claim_text = claim_match.group(0).strip() if claim_match else f"Reference to Source {ref_str}"

        doc_chunk = DocumentChunk(
            chunk_id=chunk.get("chunk_id", f"chunk-{idx}"),
            source_document=chunk.get("source", "unknown"),
            content=chunk.get("content", "")[:500],
            metadata=chunk.get("metadata", {}),
            relevance_score=chunk.get("rrf_score", 0.0) if isinstance(chunk.get("rrf_score"), (int, float)) else 0.0,
        )

        citations.append(Citation(
            claim=claim_text,
            evidence=[doc_chunk],
            confidence=chunk.get("semantic_score", 0.5) if isinstance(chunk.get("semantic_score"), (int, float)) else 0.5,
        ))

    return citations
