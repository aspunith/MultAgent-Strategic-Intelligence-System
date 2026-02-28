"""
MASIS Citation Engine — Enforces traceability from claims to evidence.

Every recommendation in the final report must be traceable to:
  - A specific document chunk, OR
  - An external data point

The Citation Engine:
  1. Validates that all citations reference real retrieved chunks
  2. Scores citation quality (relevance, specificity)
  3. Flags unsupported claims
  4. Generates a citation report for audit purposes
"""

from __future__ import annotations

import re
from typing import Any

from masis.schemas import Citation, DocumentChunk, FinalReport


class CitationEngine:
    """Enforces and validates citation traceability."""

    def __init__(self, retrieved_chunks: list[dict[str, Any]]):
        self._chunks = {
            chunk.get("chunk_id", f"idx-{i}"): chunk
            for i, chunk in enumerate(retrieved_chunks)
        }
        self._chunk_list = retrieved_chunks

    def validate_report(self, report: FinalReport) -> CitationAudit:
        """
        Validate all citations in a FinalReport.
        Returns a CitationAudit with issues and statistics.
        """
        issues: list[str] = []
        valid_citations = 0
        orphaned_claims: list[str] = []

        # Check each citation
        for cit in report.citations:
            if not cit.evidence:
                issues.append(f"Citation '{cit.citation_id}' has no evidence chunks")
                orphaned_claims.append(cit.claim)
                continue

            for ev in cit.evidence:
                if ev.chunk_id not in self._chunks and not any(
                    c.get("chunk_id") == ev.chunk_id for c in self._chunk_list
                ):
                    issues.append(
                        f"Citation '{cit.citation_id}' references unknown chunk '{ev.chunk_id}'"
                    )
                else:
                    valid_citations += 1

        # Check for claims in the text that lack citations
        uncited = self._find_uncited_claims(report.detailed_analysis)

        return CitationAudit(
            total_citations=len(report.citations),
            valid_citations=valid_citations,
            issues=issues,
            orphaned_claims=orphaned_claims,
            uncited_statements=uncited,
            coverage_score=valid_citations / max(len(report.citations), 1),
        )

    def _find_uncited_claims(self, text: str) -> list[str]:
        """Find sentences that make factual claims but lack [Source N] references."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        uncited: list[str] = []

        # Heuristics for factual claims (contains numbers, percentages, names, etc.)
        claim_patterns = [
            r'\d+%',           # Percentages
            r'\$[\d,]+',       # Dollar amounts
            r'\d{4}',          # Years
            r'according to',
            r'reported that',
            r'found that',
            r'shows that',
            r'indicates',
        ]

        for sentence in sentences:
            is_claim = any(re.search(p, sentence, re.IGNORECASE) for p in claim_patterns)
            has_citation = bool(re.search(r'\[Source\s*\d+', sentence))

            if is_claim and not has_citation:
                uncited.append(sentence.strip())

        return uncited

    def enrich_citations(self, citations: list[Citation]) -> list[Citation]:
        """Enrich citations with full chunk content from the store."""
        enriched = []
        for cit in citations:
            new_evidence = []
            for ev in cit.evidence:
                if ev.chunk_id in self._chunks:
                    chunk_data = self._chunks[ev.chunk_id]
                    ev.content = chunk_data.get("content", ev.content)
                    ev.source_document = chunk_data.get("source", ev.source_document)
                new_evidence.append(ev)
            cit.evidence = new_evidence
            enriched.append(cit)
        return enriched


class CitationAudit:
    """Result of citation validation."""

    def __init__(
        self,
        total_citations: int,
        valid_citations: int,
        issues: list[str],
        orphaned_claims: list[str],
        uncited_statements: list[str],
        coverage_score: float,
    ):
        self.total_citations = total_citations
        self.valid_citations = valid_citations
        self.issues = issues
        self.orphaned_claims = orphaned_claims
        self.uncited_statements = uncited_statements
        self.coverage_score = coverage_score

    def to_dict(self) -> dict:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "issues": self.issues,
            "orphaned_claims": self.orphaned_claims,
            "uncited_statements_count": len(self.uncited_statements),
            "coverage_score": self.coverage_score,
            "passes_audit": len(self.issues) == 0 and self.coverage_score >= 0.7,
        }

    def __repr__(self) -> str:
        return (
            f"CitationAudit(valid={self.valid_citations}/{self.total_citations}, "
            f"coverage={self.coverage_score:.2f}, issues={len(self.issues)})"
        )
