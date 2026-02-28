"""
Supervisor Agent — "The Brain"

Responsibilities:
  • Decompose the user query into a DAG of sub-tasks
  • Route tasks to appropriate specialist agents
  • Monitor execution state and decide retry/escalate/stop
  • Detect when Human-in-the-Loop is needed (ambiguity, low confidence)
  • Prevent agentic drift by anchoring sub-tasks to the original intent

Uses: GPT-4 class model (primary) for complex reasoning.
"""

from __future__ import annotations

from masis.config import get_config
from masis.llm_utils import get_primary_llm, invoke_llm, invoke_llm_structured
from masis.schemas import (
    AgentMessage,
    AgentRole,
    HITLRequest,
    SubTask,
    TaskPlan,
    TaskStatus,
)
from masis.state import MASISState


# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

DECOMPOSITION_SYSTEM = """You are the Supervisor agent ("The Brain") of a Multi-Agent Strategic Intelligence System.

Your job is to decompose a user query into a directed acyclic graph (DAG) of sub-tasks.
Each sub-task must be assigned to one of these specialist agents:
  - researcher: Evidence gathering via hybrid search (semantic + keyword)
  - skeptic: Hallucination detection, logical validation, contradiction checking
  - synthesizer: Final answer construction with citations

Rules:
1. Always start with at least one 'researcher' task to gather evidence.
2. After research, always include a 'skeptic' task to validate findings.
3. End with a 'synthesizer' task to produce the final response.
4. If the query is ambiguous or has multiple interpretations, output a task
   marked as 'needs_human_input' with a clear question for the user.
5. Keep sub-tasks focused — each should have a single objective.
6. Use depends_on to encode ordering constraints (task IDs).
7. Do NOT attempt to answer the query yourself — only plan.

Respond with a JSON object matching the TaskPlan schema."""

ROUTING_SYSTEM = """You are the Supervisor agent monitoring task execution.

Given the current state of the task plan and agent outputs so far,
decide what happens next. Your options:
- Route to the next pending task
- Retry a failed task (max {max_retries} retries)
- Request human clarification if confidence is below {threshold}
- Mark the plan as complete if all tasks are done or should_end is warranted

Respond with one of: "route:<task_id>", "retry:<task_id>", "hitl:<reason>", "complete"

Also detect AGENTIC DRIFT: if any agent's output has diverged from the original query intent,
flag it and re-focus by describing the drift."""

QUERY_REWRITE_SYSTEM = """You are a query understanding specialist.
Rewrite the user's query to be clear, unambiguous, and self-contained.
Keep the original intent intact. If the query is already clear, return it unchanged.
Only return the rewritten query, nothing else."""


# ──────────────────────────────────────────────────────────────
# Node Functions (called by LangGraph)
# ──────────────────────────────────────────────────────────────

def supervisor_plan(state: MASISState) -> dict:
    """
    Entry node: Decompose user query into a task DAG.
    Also rewrites the query for clarity.
    """
    cfg = get_config()
    llm = get_primary_llm()

    # Step 1: Rewrite query for clarity
    clarified = invoke_llm(llm, QUERY_REWRITE_SYSTEM, state.original_query)
    if not clarified.strip():
        clarified = state.original_query

    # Step 2: Decompose into task plan
    task_plan = invoke_llm_structured(
        llm,
        DECOMPOSITION_SYSTEM,
        f"User query: {clarified}",
        TaskPlan,
    )
    task_plan.original_query = clarified

    # Check if any sub-task needs human input
    hitl_tasks = [t for t in task_plan.sub_tasks if t.status == TaskStatus.NEEDS_HUMAN_INPUT]
    if hitl_tasks and cfg.hitl.enabled:
        hitl_req = HITLRequest(
            reason="Query is ambiguous — supervisor needs clarification",
            question_to_user=hitl_tasks[0].description,
            context_summary=f"Original query: {state.original_query}",
        )
        return {
            "clarified_query": clarified,
            "task_plan": task_plan,
            "hitl_request": hitl_req,
            "awaiting_human": True,
            "messages": [
                AgentMessage(
                    sender=AgentRole.SUPERVISOR,
                    content=f"Created task plan with {len(task_plan.sub_tasks)} sub-tasks. "
                            f"HITL required: {hitl_req.question_to_user}",
                )
            ],
        }

    # Find the first task to route
    first_task = _find_next_task(task_plan)

    return {
        "clarified_query": clarified,
        "task_plan": task_plan,
        "current_task_id": first_task.task_id if first_task else None,
        "next_agent": first_task.assigned_to if first_task else None,
        "messages": [
            AgentMessage(
                sender=AgentRole.SUPERVISOR,
                content=f"Decomposed query into {len(task_plan.sub_tasks)} sub-tasks. "
                        f"Routing to {first_task.assigned_to.value if first_task else 'none'}.",
            )
        ],
    }


def supervisor_route(state: MASISState) -> dict:
    """
    Routing node: After an agent completes, decide what to do next.
    Handles retries, drift detection, and completion.
    """
    cfg = get_config()
    plan = state.task_plan

    if plan is None:
        return {"should_end": True, "error_log": ["No task plan exists"]}

    # Safety valve: prevent infinite loops
    if state.iteration_count >= state.max_iterations:
        return {
            "should_end": True,
            "messages": [
                AgentMessage(
                    sender=AgentRole.SUPERVISOR,
                    content=f"Max iterations ({state.max_iterations}) reached. Forcing completion.",
                )
            ],
        }

    # Mark current task complete if it was in progress
    if state.current_task_id:
        for task in plan.sub_tasks:
            if task.task_id == state.current_task_id and task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.COMPLETED

    # Check for failed tasks that need retry
    for task in plan.sub_tasks:
        if task.status == TaskStatus.FAILED and task.retries < 2:
            task.retries += 1
            task.status = TaskStatus.PENDING
            return {
                "task_plan": plan,
                "current_task_id": task.task_id,
                "next_agent": task.assigned_to,
                "iteration_count": state.iteration_count + 1,
                "messages": [
                    AgentMessage(
                        sender=AgentRole.SUPERVISOR,
                        content=f"Retrying task {task.task_id} (attempt {task.retries}): {task.description}",
                    )
                ],
            }

    # If skeptic review failed, send back to researcher for more evidence
    if (
        state.critique is not None
        and not state.critique.passes_review
        and state.skeptic_rounds < cfg.agents.max_skeptic_challenges
    ):
        # Re-queue a researcher task for the gaps identified
        gaps = "; ".join(i.description for i in state.critique.issues[:3])
        new_task = SubTask(
            description=f"Address skeptic concerns: {gaps}",
            assigned_to=AgentRole.RESEARCHER,
        )
        plan.sub_tasks.append(new_task)
        # Also re-queue skeptic after
        skeptic_task = SubTask(
            description="Re-validate after additional research",
            assigned_to=AgentRole.SKEPTIC,
            depends_on=[new_task.task_id],
        )
        plan.sub_tasks.append(skeptic_task)

        return {
            "task_plan": plan,
            "current_task_id": new_task.task_id,
            "next_agent": AgentRole.RESEARCHER,
            "iteration_count": state.iteration_count + 1,
            "critique": None,  # Reset for next round
            "messages": [
                AgentMessage(
                    sender=AgentRole.SUPERVISOR,
                    content=f"Skeptic found issues. Sending back to Researcher: {gaps}",
                )
            ],
        }

    # Find next pending task whose dependencies are met
    next_task = _find_next_task(plan)

    if next_task is None:
        # All tasks done
        return {
            "task_plan": plan,
            "should_end": True,
            "iteration_count": state.iteration_count + 1,
            "messages": [
                AgentMessage(
                    sender=AgentRole.SUPERVISOR,
                    content="All sub-tasks completed. Ending pipeline.",
                )
            ],
        }

    return {
        "task_plan": plan,
        "current_task_id": next_task.task_id,
        "next_agent": next_task.assigned_to,
        "iteration_count": state.iteration_count + 1,
        "messages": [
            AgentMessage(
                sender=AgentRole.SUPERVISOR,
                content=f"Routing to {next_task.assigned_to.value}: {next_task.description}",
            )
        ],
    }


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _find_next_task(plan: TaskPlan) -> SubTask | None:
    """Find the next pending task whose dependencies are all completed."""
    completed_ids = {
        t.task_id for t in plan.sub_tasks if t.status == TaskStatus.COMPLETED
    }
    for task in plan.sub_tasks:
        if task.status == TaskStatus.PENDING:
            deps_met = all(d in completed_ids for d in task.depends_on)
            if deps_met:
                task.status = TaskStatus.IN_PROGRESS
                return task
    return None
