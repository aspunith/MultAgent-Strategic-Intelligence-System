"""
MASIS Orchestration Graph — LangGraph DAG wiring all agents together.

Architecture:
  ┌──────────┐
  │  START    │
  └────┬─────┘
       │
  ┌────▼──────────┐
  │  Supervisor    │◄──────────────────────────┐
  │  (Plan/Route)  │                           │
  └────┬──────────┘                            │
       │ (routes based on next_agent)          │
       ├───────────────┬───────────────┐       │
  ┌────▼─────┐  ┌──────▼──────┐  ┌────▼──────┐│
  │Researcher│  │   Skeptic   │  │Synthesizer││
  │  (RAG)   │  │  (Critique) │  │ (Output)  ││
  └────┬─────┘  └──────┬──────┘  └────┬──────┘│
       │               │              │        │
       └───────────────┴──────────────┘        │
                       │                       │
                  ┌────▼─────┐                 │
                  │Supervisor │─────────────────┘
                  │  (Route)  │
                  └────┬─────┘
                       │
                 ┌─────▼──────┐
                 │    END      │
                 └─────────────┘
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph

from masis.schemas import AgentRole
from masis.state import MASISState
from masis.agents.supervisor import supervisor_plan, supervisor_route
from masis.agents.researcher import researcher_node
from masis.agents.skeptic import skeptic_node
from masis.agents.synthesizer import synthesizer_node


# ──────────────────────────────────────────────────────────────
# State Adapter — Convert MASISState (Pydantic) to/from dict
# ──────────────────────────────────────────────────────────────

def _wrap_node(func):
    """Wrap an agent node to handle Pydantic state conversion."""
    def wrapper(state: dict) -> dict:
        masis_state = MASISState(**state)
        result = func(masis_state)
        # Merge result into existing state
        new_state = masis_state.model_dump()
        if isinstance(result, dict):
            # Handle messages append
            if "messages" in result and "messages" in new_state:
                existing = new_state["messages"]
                new_msgs = result["messages"]
                if isinstance(new_msgs, list) and len(new_msgs) > 0:
                    if hasattr(new_msgs[0], "model_dump"):
                        new_msgs = [m.model_dump() for m in new_msgs]
                result["messages"] = [
                    m.model_dump() if hasattr(m, "model_dump") else m
                    for m in existing
                ] + new_msgs
            # Convert Pydantic models in result to dicts
            for key, val in result.items():
                if hasattr(val, "model_dump"):
                    result[key] = val.model_dump()
                elif isinstance(val, list):
                    result[key] = [
                        item.model_dump() if hasattr(item, "model_dump") else item
                        for item in val
                    ]
            new_state.update(result)
        return new_state
    return wrapper


# ──────────────────────────────────────────────────────────────
# Routing Logic
# ──────────────────────────────────────────────────────────────

def _should_continue_after_plan(state: dict) -> str:
    """After supervisor plans, decide next step."""
    if state.get("awaiting_human"):
        return "hitl_pause"
    next_agent = state.get("next_agent")
    if next_agent:
        if isinstance(next_agent, str):
            return next_agent
        return next_agent.value if hasattr(next_agent, "value") else str(next_agent)
    return "end"


def _should_continue_after_route(state: dict) -> str:
    """After supervisor routes, decide next step."""
    if state.get("should_end"):
        return "end"
    if state.get("awaiting_human"):
        return "hitl_pause"
    next_agent = state.get("next_agent")
    if next_agent:
        if isinstance(next_agent, str):
            return next_agent
        return next_agent.value if hasattr(next_agent, "value") else str(next_agent)
    return "end"


def _after_agent(state: dict) -> str:
    """After any specialist agent, route back to supervisor for decision."""
    if state.get("should_end"):
        return "end"
    return "supervisor_route"


# ──────────────────────────────────────────────────────────────
# HITL Pause Node
# ──────────────────────────────────────────────────────────────

def hitl_pause_node(state: dict) -> dict:
    """
    Pause point for human-in-the-loop.
    In production, this would be an interrupt.
    For now, it prompts on stdin.
    """
    from masis.hitl import handle_hitl_request
    masis_state = MASISState(**state)
    updates = handle_hitl_request(masis_state)
    state.update(updates)
    return state


# ──────────────────────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────────────────────

def build_graph() -> CompiledStateGraph:
    """Build and compile the MASIS orchestration graph."""
    graph = StateGraph(dict)

    # --- Add nodes ---
    graph.add_node("supervisor_plan", _wrap_node(supervisor_plan))
    graph.add_node("supervisor_route", _wrap_node(supervisor_route))
    graph.add_node("researcher", _wrap_node(researcher_node))
    graph.add_node("skeptic", _wrap_node(skeptic_node))
    graph.add_node("synthesizer", _wrap_node(synthesizer_node))
    graph.add_node("hitl_pause", hitl_pause_node)

    # --- Entry point ---
    graph.add_edge(START, "supervisor_plan")

    # --- After planning: route to first agent or HITL ---
    graph.add_conditional_edges(
        "supervisor_plan",
        _should_continue_after_plan,
        {
            "researcher": "researcher",
            "skeptic": "skeptic",
            "synthesizer": "synthesizer",
            "hitl_pause": "hitl_pause",
            "end": END,
        },
    )

    # --- After each agent: back to supervisor for routing ---
    for agent_name in ["researcher", "skeptic", "synthesizer"]:
        graph.add_conditional_edges(
            agent_name,
            _after_agent,
            {
                "supervisor_route": "supervisor_route",
                "end": END,
            },
        )

    # --- After supervisor routes: to next agent, HITL, or end ---
    graph.add_conditional_edges(
        "supervisor_route",
        _should_continue_after_route,
        {
            "researcher": "researcher",
            "skeptic": "skeptic",
            "synthesizer": "synthesizer",
            "hitl_pause": "hitl_pause",
            "end": END,
        },
    )

    # --- After HITL: back to supervisor to re-plan ---
    graph.add_edge("hitl_pause", "supervisor_plan")

    return graph.compile()


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

def run_masis(query: str) -> dict:
    """
    Execute the full MASIS pipeline for a given query.
    Returns the final state dict containing the FinalReport.
    """
    graph = build_graph()
    initial_state = MASISState(original_query=query).model_dump()
    final_state = graph.invoke(initial_state)
    return final_state
