"""
Researcher Agent — "The Librarian"

Responsibilities:
  • Own the RAG pipeline
  • Execute hybrid retrieval (semantic + keyword)
  • Summarize and organize evidence relevant to the current sub-task
  • Detect insufficient evidence and signal for additional search
  • Prevent infinite search loops via iteration cap

Uses: GPT-4o-mini (secondary) for summarization; hybrid search for retrieval.
"""

from __future__ import annotations

from masis.config import get_config
from masis.llm_utils import get_secondary_llm, invoke_llm
from masis.rag import hybrid_search, format_context
from masis.schemas import AgentMessage, AgentRole, TaskStatus
from masis.state import MASISState


# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

RESEARCH_SYSTEM = """You are the Researcher agent ("The Librarian") of a Multi-Agent Strategic Intelligence System.

Your job is to gather, organize, and summarize evidence retrieved from a document corpus.

You have been given:
  1. The ORIGINAL USER QUERY
  2. Your SPECIFIC SUB-TASK description
  3. RETRIEVED CONTEXT from hybrid search (semantic + keyword)

Rules:
- Focus ONLY on evidence relevant to your sub-task.
- For each key finding, note the [Source N] reference so the Citation Engine can trace it.
- If the retrieved context is insufficient, clearly state what additional information is needed.
- Do NOT fabricate information. If the documents don't contain the answer, say so explicitly.
- Organize your findings in a structured format:
  1. Key Findings (with source references)
  2. Gaps / Insufficient Evidence
  3. Contradictions between sources (if any)
- If two documents contradict each other, present BOTH viewpoints with their sources.
- Keep your output concise but complete.

IMPORTANT: You are NOT the final answer. You are gathering evidence for other agents to review."""

SUFFICIENCY_CHECK = """Given the following research findings, assess if there is sufficient evidence 
to answer the user's question. Respond with exactly "SUFFICIENT" or "INSUFFICIENT: <what's missing>"."""


# ──────────────────────────────────────────────────────────────
# Node Function
# ──────────────────────────────────────────────────────────────

def researcher_node(state: MASISState) -> dict:
    """
    Execute research for the current sub-task.
    Performs hybrid retrieval, summarizes findings, and checks sufficiency.
    """
    cfg = get_config()

    # Safety: prevent infinite research loops
    if state.research_iterations >= cfg.agents.max_research_iterations:
        return {
            "research_iterations": state.research_iterations,
            "messages": [
                AgentMessage(
                    sender=AgentRole.RESEARCHER,
                    content="Max research iterations reached. Proceeding with available evidence.",
                    metadata={"capped": True},
                )
            ],
        }

    # Determine the search query — use sub-task description + original query
    task_desc = ""
    if state.task_plan and state.current_task_id:
        for task in state.task_plan.sub_tasks:
            if task.task_id == state.current_task_id:
                task_desc = task.description
                break

    search_query = f"{state.clarified_query} {task_desc}".strip()
    if not search_query:
        search_query = state.original_query

    # Step 1: Hybrid retrieval
    chunks = hybrid_search(search_query)

    # Step 2: Summarize and organize with LLM
    llm = get_secondary_llm()
    context_str = format_context(chunks) if chunks else "(No documents found in corpus)"

    user_prompt = f"""ORIGINAL QUERY: {state.clarified_query or state.original_query}

SUB-TASK: {task_desc or 'General research'}

RETRIEVED CONTEXT:
{context_str}"""

    research_output = invoke_llm(llm, RESEARCH_SYSTEM, user_prompt)

    # Step 3: Sufficiency check
    sufficiency = invoke_llm(
        llm,
        SUFFICIENCY_CHECK,
        f"Query: {state.clarified_query}\n\nFindings:\n{research_output[:2000]}",
    )

    is_sufficient = sufficiency.strip().upper().startswith("SUFFICIENT")

    # Mark current task result
    if state.task_plan and state.current_task_id:
        for task in state.task_plan.sub_tasks:
            if task.task_id == state.current_task_id:
                task.result = research_output
                task.status = TaskStatus.COMPLETED

    return {
        "retrieved_chunks": state.retrieved_chunks + chunks,
        "research_iterations": state.research_iterations + 1,
        "task_plan": state.task_plan,
        "messages": [
            AgentMessage(
                sender=AgentRole.RESEARCHER,
                content=research_output,
                metadata={
                    "chunks_retrieved": len(chunks),
                    "sufficient": is_sufficient,
                    "search_query": search_query,
                },
            )
        ],
    }
