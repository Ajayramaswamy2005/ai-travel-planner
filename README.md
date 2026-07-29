# AI Multi-Agent Travel Planner

A multi-agent travel planning system built with **LangGraph**, **FastAPI**, a custom **MCP server**, and **Groq**.
Specialized agents (itinerary, hotel, weather, budget) collaborate under a supervisor to produce complete travel plans.

## Stack
- **Orchestration:** LangGraph (supervisor + worker agents)
- **LLM:** Groq via `langchain-groq` (mock fallback when no API key)
- **Tools:** Custom MCP server exposing Google Maps, Weather, SerpAPI, and Currency APIs
- **API:** FastAPI
- **UI:** Gradio chat interface
- **Memory:** Shared conversational memory across agents

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt

cp .env.example .env       # fill in keys (optional; mocks used when missing)
```

### Run the MCP server (standalone)
```bash
python -m app.mcp_server.server
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
    server.py            # FastMCP server entrypoint
    registry.py          # tool registry used by server + in-process agents
    tools/               # maps, weather, serp, currency tool implementations
  agents/
    supervisor.py        # routes requests to worker agents
    itinerary.py
    hotel.py
    weather_agent.py
    budget.py
  graph/
    workflow.py          # LangGraph supervisor workflow
  api/main.py            # FastAPI app
  ui.py                  # Gradio chat UI
  cli.py                 # CLI
```

## Mock mode
When `USE_MOCK_FALLBACK=true` or the relevant API key is missing, every tool
and the LLM return deterministic mock data, so the system runs end-to-end
without any external credentials.