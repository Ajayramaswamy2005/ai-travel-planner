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
logging.getLogger("httpx").setLevel(logging.WARNING)


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
def tools(
    mcp: bool = typer.Option(False, "--mcp", help="Ask the MCP server instead of the local registry."),
) -> None:
    """List the MCP tools available to the agents."""
    if mcp:
        from .mcp_server.client import list_tools_mcp

        names = asyncio.run(list_tools_mcp())
    else:
        from .mcp_server.registry import TOOL_REGISTRY

        names = list(TOOL_REGISTRY)

    console.print("[bold]MCP Tools[/bold]")
    for name in names:
        console.print(f"  - {name}")


@app.command()
def bench(
    destination: str = typer.Option("Paris, France", "--destination", "-d"),
    days: int = typer.Option(4, "--days"),
    runs: int = typer.Option(5, "--runs", help="Measured runs per mode."),
    tool_latency_ms: int | None = typer.Option(None, "--tool-latency-ms", help="Mock tool latency."),
    llm_latency_ms: int | None = typer.Option(None, "--llm-latency-ms", help="Mock LLM latency."),
) -> None:
    """Compare end-to-end latency of parallel vs sequential execution."""
    from .benchmark import benchmark

    if tool_latency_ms is not None:
        settings.mock_tool_latency_ms = tool_latency_ms
    if llm_latency_ms is not None:
        settings.mock_llm_latency_ms = llm_latency_ms

    request = TravelRequest(destination=destination, days=days, budget=1500)
    result = asyncio.run(benchmark(request, runs=runs))

    mode = "mock" if settings.use_mock_fallback else "live APIs"
    console.print(f"[bold]Benchmark[/bold] ({mode}, transport={settings.tool_transport}, runs={runs})")
    console.print(f"  sequential: {result.sequential_mean:.2f}s mean")
    console.print(f"  parallel:   {result.parallel_mean:.2f}s mean")
    console.print(f"  [green]latency reduction: {result.reduction_pct:.1f}%[/green]")


if __name__ == "__main__":
    app()