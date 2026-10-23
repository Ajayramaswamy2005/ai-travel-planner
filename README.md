# AI Multi-Agent Travel Planner

A multi-agent travel planning system built with **LangGraph**, **FastAPI**, a custom **MCP server**, and **Groq**.
Specialized agents (itinerary, hotel, weather, budget) collaborate under a supervisor to produce complete travel plans.

## Stack
- **Orchestration:** LangGraph supervisor that fans out to worker agents in parallel
- **LLM:** Groq via `langchain-groq` (mock fallback when no API key)
- **Tools:** Reusable MCP server exposing Google Maps, Weather, SerpAPI, and Currency APIs
  as standardized tools; agents call them in-process or over MCP (stdio / streamable HTTP)
- **API:** FastAPI
- **UI:** Gradio chat interface
- **Memory:** Shared agent memory in the LangGraph state, persisted per session by a checkpointer

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt

cp .env.example .env       # fill in keys (optional; mocks used when missing)
```

### Run the MCP server (standalone)
```bash
python -m app.mcp_server.server                                          # stdio
python -m app.mcp_server.server --transport streamable-http --port 8765  # shared, http://127.0.0.1:8765/mcp
```

Any MCP-capable agent can connect to it. To make the planner's own agents use it
instead of in-process calls, set `TOOL_TRANSPORT=mcp` (and `MCP_SERVER_URL` to
reuse a running HTTP server; without it, a server is spawned over stdio):
```bash
TOOL_TRANSPORT=mcp MCP_SERVER_URL=http://127.0.0.1:8765/mcp python -m app.cli plan -d Rome
python -m app.cli tools --mcp     # list the tools the MCP server advertises
```

## How it works
- **Parallel execution.** The supervisor node returns every selected worker, so
  LangGraph runs the itinerary, hotel, weather and budget agents in the same
  superstep; inside each agent, its tool calls run concurrently with
  `asyncio.gather`. Worker results are merged into the shared state by reducers.
- **Shared agent memory.** All agents read and write one LangGraph state
  (request, agent outputs, messages, errors). An `InMemorySaver` checkpointer
  keeps that state per `session_id` (`thread_id`), so later turns see earlier
  turns' conversation. Swap in a SQLite/Postgres checkpointer to persist it.

### Benchmark: parallel vs sequential
```bash
python -m app.cli bench --runs 5                          # live APIs if keys are set
python -m app.cli bench --tool-latency-ms 300 --llm-latency-ms 800   # mock mode, simulated latency
```
Sequential mode runs the same graph with one worker and one tool call at a time.
In mock mode the reduction depends entirely on the simulated latencies, so quote
numbers measured against the live APIs.

### Tests
```bash
pytest
```

### Run the FastAPI backend
```bash
python -m app.api.main
# or: uvicorn app.api.main:app --reload
```

### Run the Gradio UI
```bash
python -m app.ui
```

### Run the CLI
```bash
python -m app.cli plan --destination "Paris, France" --days 4 --budget 1500
python -m app.cli chat
```

## Project structure
```
app/
  config.py              # settings (env)
  llm.py                 # Groq LLM wrapper + mock fallback
  memory.py             # shared conversational memory
  models.py             # shared state schemas
  mcp_server/
    server.py            # MCP server entrypoint (stdio / streamable HTTP)
    client.py            # MCP client used when TOOL_TRANSPORT=mcp
    registry.py          # tool registry + parallel/sequential dispatch
    tools/               # maps, weather, serp, currency tool implementations
  agents/
    supervisor.py        # routes requests to worker agents
    itinerary.py
    hotel.py
    weather_agent.py
    budget.py
  graph/
    workflow.py          # LangGraph fan-out workflow + checkpointer memory
  benchmark.py           # parallel vs sequential latency benchmark
  api/main.py            # FastAPI app
  ui.py                  # Gradio chat UI
  cli.py                 # CLI
```

## Mock mode
When `USE_MOCK_FALLBACK=true` or the relevant API key is missing, every tool
and the LLM return deterministic mock data, so the system runs end-to-end
without any external credentials.