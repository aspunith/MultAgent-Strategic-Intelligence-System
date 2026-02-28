"""
MASIS CLI — Command-line interface for the Multi-Agent Strategic Intelligence System.

Commands:
  masis query "your question here"      Run the full MASIS pipeline
  masis ingest [--dir PATH]             Ingest documents into the vector store
  masis evaluate                        Evaluate the last run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from masis.config import get_config, DOCUMENT_DIR

console = Console()


def _print_report(final_state: dict) -> None:
    """Pretty-print the final report."""
    report_data = final_state.get("final_report")
    if not report_data:
        console.print("[red]No final report generated.[/red]")
        if final_state.get("error_log"):
            for err in final_state["error_log"]:
                console.print(f"  [red]Error:[/red] {err}")
        return

    # Handle both dict and FinalReport
    if hasattr(report_data, "model_dump"):
        report = report_data.model_dump()
    elif isinstance(report_data, dict):
        report = report_data
    else:
        console.print(str(report_data))
        return

    # Executive Summary
    console.print()
    console.print(Panel(
        report.get("executive_summary", "N/A"),
        title="[bold blue]Executive Summary[/bold blue]",
        border_style="blue",
    ))

    # Detailed Analysis
    console.print()
    console.print(Panel(
        Markdown(report.get("detailed_analysis", "N/A")),
        title="[bold green]Detailed Analysis[/bold green]",
        border_style="green",
    ))

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        console.print()
        console.print("[bold magenta]Recommendations:[/bold magenta]")
        for i, rec in enumerate(recs, 1):
            console.print(f"  {i}. {rec}")

    # Citations
    citations = report.get("citations", [])
    if citations:
        console.print()
        table = Table(title="Citation Trail", show_lines=True)
        table.add_column("ID", style="dim", width=12)
        table.add_column("Claim", max_width=50)
        table.add_column("Source", style="cyan", max_width=30)
        table.add_column("Confidence", justify="right")

        for cit in citations:
            cit_id = cit.get("citation_id", "?")
            claim = cit.get("claim", "?")[:80]
            evidence = cit.get("evidence", [])
            source = evidence[0].get("source_document", "?") if evidence else "?"
            conf = f"{cit.get('confidence', 0):.0%}"
            table.add_row(cit_id, claim, source, conf)

        console.print(table)

    # Confidence & Metadata
    console.print()
    conf = report.get("confidence", "unknown")
    meta = report.get("metadata", {})
    console.print(f"[bold]Confidence:[/bold] {conf}")
    console.print(f"[bold]Research iterations:[/bold] {meta.get('research_iterations', '?')}")
    console.print(f"[bold]Skeptic rounds:[/bold] {meta.get('skeptic_rounds', '?')}")
    console.print(f"[bold]Chunks retrieved:[/bold] {meta.get('total_chunks_retrieved', '?')}")

    # Audit Trail Summary
    audit = report.get("audit_trail", [])
    if audit:
        console.print()
        console.print(f"[bold]Audit trail:[/bold] {len(audit)} agent messages recorded.")


def _run_query(query: str, evaluate: bool = False) -> None:
    """Execute the MASIS pipeline."""
    from masis.graph import run_masis

    console.print(Panel(
        f"[bold]{query}[/bold]",
        title="[bold cyan]MASIS Query[/bold cyan]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running MASIS pipeline...", total=None)
        final_state = run_masis(query)
        progress.update(task, description="Complete!")

    _print_report(final_state)

    # Optional evaluation
    if evaluate:
        _run_evaluation(final_state)


def _run_evaluation(final_state: dict) -> None:
    """Run LLM-as-Judge evaluation on the output."""
    from masis.evaluation import MASISEvaluator
    from masis.schemas import FinalReport
    from masis.rag import format_context

    report_data = final_state.get("final_report")
    if not report_data:
        console.print("[red]No report to evaluate.[/red]")
        return

    if isinstance(report_data, dict):
        report = FinalReport(**report_data)
    else:
        report = report_data

    chunks = final_state.get("retrieved_chunks", [])
    evidence_text = format_context(chunks) if chunks else ""

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating with LLM-as-Judge...", total=None)
        evaluator = MASISEvaluator()
        result = evaluator.evaluate(report, evidence_text)
        progress.update(task, description="Evaluation complete!")

    console.print()
    console.print(Panel(
        f"[bold]{result.summary}[/bold]",
        title=f"[bold yellow]Evaluation: Grade {result.grade}[/bold yellow]",
        border_style="yellow",
    ))

    # Detailed metric table
    table = Table(title="Metric Breakdown", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Reasoning", max_width=60)

    for metric in [result.faithfulness, result.relevance, result.completeness, result.citation_quality]:
        score_color = "green" if metric.score >= 0.8 else "yellow" if metric.score >= 0.6 else "red"
        table.add_row(
            metric.metric_name,
            f"[{score_color}]{metric.score:.1%}[/{score_color}]",
            metric.reasoning[:100],
        )

    console.print(table)


def _ingest_docs(doc_dir: str | None = None) -> None:
    """Ingest documents into the vector store."""
    from masis.rag import ingest_documents

    path = Path(doc_dir) if doc_dir else DOCUMENT_DIR
    path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Ingesting documents from:[/bold] {path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=None)
        count = ingest_documents(path)
        progress.update(task, description="Done!")

    console.print(f"[green]Ingested {count} chunks into the vector store.[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="masis",
        description="Multi-Agent Strategic Intelligence System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # query command
    query_parser = subparsers.add_parser("query", help="Run the MASIS pipeline")
    query_parser.add_argument("question", type=str, help="The question to analyze")
    query_parser.add_argument("--evaluate", "-e", action="store_true", help="Run LLM-as-Judge evaluation")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--dir", type=str, default=None, help="Document directory path")

    args = parser.parse_args()

    if args.command == "query":
        _run_query(args.question, evaluate=args.evaluate)
    elif args.command == "ingest":
        _ingest_docs(args.dir)


if __name__ == "__main__":
    main()
