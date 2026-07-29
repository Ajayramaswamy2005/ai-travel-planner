"""CLI entrypoint using typer + rich."""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .agents.supervisor import parse_request
from .config import settings
from .graph.workflow import plan_from_text, run_plan
from .models import TravelRequest

console = Console()
app = typer.Typer(help="AI Multi-Agent Travel Planner CLI", add_completion=False)

logging.basicConfig(level=settings.log_level)


@app.command()
def plan(
    destination: str = typer.Option(..., "--destination", "-d", help="Trip destination."),
    days: int = typer.Option(3, "--days", help="Number of days."),
    budget: float | None = typer.Option(None, "--budget", help="Total budget."),
    currency: str = typer.Option("USD", "--currency", help="Currency code."),
    travelers: int = typer.Option(1, "--travelers", help="Number of travelers."),
    preferences: list[str] = typer.Option([], "--pref", help="Preferences (repeatable)."),
) -> None:
    """Generate a full travel plan from structured options."""
    request = TravelRequest(
        destination=destination,
        days=days,
        budget=budget,
        currency=currency,
        travelers=travelers,
        preferences=preferences,
        raw=f"{days}-day trip to {destination} for {travelers} traveler(s)"
        + (f" budget {budget} {currency}" if budget else ""),
    )
    state = asyncio.run(run_plan(request))

    console.print(Panel.fit(f"[bold]Travel Plan: {destination}[/bold]"))
    for name, out in state.agent_outputs.items():
        console.print(f"[cyan]{name}[/cyan] tools: {', '.join(out.tools_used)}")
    console.print()
    console.print(Panel(state.final_plan, title="Final Plan", border_style="green"))
    if state.errors:
        console.print(f"[red]Errors:[/red] {state.errors}")


@app.command()
def chat() -> None:
    """Interactive chat that runs the planner on free text."""
    console.print("[bold green]Travel Planner Chat[/bold green] (type 'exit' to quit)")
    session = None
    while True:
        message = Prompt.ask("[bold]You[/bold]")
        if message.strip().lower() in {"exit", "quit"}:
            break
        state = asyncio.run(plan_from_text(message, session_id=session or "cli"))
        session = "cli"
        console.print(Panel(state.final_plan, title="Planner", border_style="cyan"))
        console.print(f"[dim]destination={state.request.destination} "
                      f"days={state.request.days} "
                      f"budget={state.request.budget} {state.request.currency}[/dim]\n")


@app.command()
def tools() -> None:
    """List the MCP tools available to the agents."""
    from .mcp_server.registry import TOOL_REGISTRY

    console.print("[bold]MCP Tools[/bold]")
    for name in TOOL_REGISTRY:
        console.print(f"  - {name}")


if __name__ == "__main__":
    app()