"""Latency benchmark: parallel vs sequential tool/agent execution.

Runs the same request through the workflow in both modes and reports the
end-to-end latency reduction. With API keys configured the numbers reflect
real tool/LLM latency; in mock mode, latency comes from MOCK_TOOL_LATENCY_MS and
MOCK_LLM_LATENCY_MS, so the result only shows the effect of the orchestration.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from .graph.workflow import run_plan
from .models import TravelRequest


@dataclass
class BenchResult:
    sequential_s: list[float]
    parallel_s: list[float]

    @property
    def sequential_mean(self) -> float:
        return statistics.mean(self.sequential_s)

    @property
    def parallel_mean(self) -> float:
        return statistics.mean(self.parallel_s)

    @property
    def reduction_pct(self) -> float:
        return (1 - self.parallel_mean / self.sequential_mean) * 100


async def _timed(request: TravelRequest, *, parallel: bool) -> float:
    start = time.perf_counter()
    await run_plan(request, parallel=parallel)
    return time.perf_counter() - start


async def benchmark(request: TravelRequest, *, runs: int = 5, warmup: int = 1) -> BenchResult:
    """Alternate sequential/parallel runs so drift (rate limits, caches) hits both equally."""
    for _ in range(warmup):
        await _timed(request, parallel=True)
    seq: list[float] = []
    par: list[float] = []
    for _ in range(runs):
        seq.append(await _timed(request, parallel=False))
        par.append(await _timed(request, parallel=True))
    return BenchResult(sequential_s=seq, parallel_s=par)
