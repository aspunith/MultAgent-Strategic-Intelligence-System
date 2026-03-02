"""
MASIS Human-in-the-Loop (HITL) — Handles pause/resume for human clarification.

Triggers when:
  • Supervisor detects ambiguous query
  • Confidence score falls below threshold
  • Contradictions require human judgment
  • Skeptic flags critical unresolvable issues
"""

from __future__ import annotations

from masis.config import get_config
from masis.schemas import AgentMessage, AgentRole, HITLRequest
from masis.state import MASISState

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def handle_hitl_request(state: MASISState) -> dict:
    """
    Present HITL request to the user and collect their response.
    Returns state updates to resume the pipeline.
    """
    cfg = get_config()

    if not cfg.hitl.enabled or state.hitl_request is None:
        return {
            "awaiting_human": False,
            "hitl_request": None,
        }

    req = state.hitl_request

    # Pause the CLI spinner so the prompt is visible
    from masis.cli import _active_progress
    if _active_progress is not None:
        _active_progress.stop()

    # Display the request to the user
    console.print()
    console.print(Panel(
        f"[bold yellow]The system needs your input to proceed.[/bold yellow]\n\n"
        f"[bold]Reason:[/bold] {req.reason}\n\n"
        f"[bold]Context:[/bold] {req.context_summary}\n\n"
        f"[bold cyan]Question:[/bold cyan] {req.question_to_user}",
        title="🤚 Human Input Required",
        border_style="yellow",
    ))

    if req.options:
        console.print("\n[bold]Options:[/bold]")
        for i, opt in enumerate(req.options, 1):
            console.print(f"  {i}. {opt}")
        console.print()

    # Collect response
    response = Prompt.ask("[bold green]Your response[/bold green]")

    # Show confirmation and resume spinner
    console.print(f"\n[dim]Your input received. Resuming pipeline...[/dim]\n")
    if _active_progress is not None:
        _active_progress.start()

    return {
        "hitl_response": response,
        "awaiting_human": False,
        "hitl_request": None,
        "clarified_query": f"{state.clarified_query or state.original_query} [User clarification: {response}]",
        "messages": [
            AgentMessage(
                sender=AgentRole.HUMAN,
                content=response,
                metadata={"hitl_reason": req.reason},
            ).model_dump()
        ],
    }


def should_trigger_hitl(state: MASISState) -> bool:
    """Check if HITL should be triggered based on current state."""
    cfg = get_config()

    if not cfg.hitl.enabled:
        return False

    if not state.critique:
        return False

    critical_issues = [
        i for i in state.critique.issues
        if i.severity == "high" and i.issue_type in ("contradiction", "hallucination")
    ]

    # Low confidence AND at least one contradiction/hallucination → needs human judgment
    if (
        state.critique.confidence_score < cfg.agents.confidence_threshold
        and len(critical_issues) >= 1
    ):
        return True

    # Multiple critical issues regardless of confidence
    if len(critical_issues) >= 2:
        return True

    return False
