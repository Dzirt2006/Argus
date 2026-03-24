# CLAUDE.md

Local home agent. vLLM + Qwen3.5-4B + LangChain/LangGraph + MCP + Wyoming voice.

## Stack

- Python 3.12+, async-first, type hints, Pydantic v2
- LangChain + LangGraph for agent, `ChatOpenAI` as LLM client
- vLLM in Docker (`vllm/vllm-openai`) → OpenAI-compat API at `http://vllm:8000/v1`
- `langchain-mcp-adapters` for MCP tool discovery
- FastMCP for MCP servers, streamable-http transport
- Custom guardrails layer (tool allowlist, path validation, confirmation gate)
- Docker Compose orchestration, `uv` for deps, `structlog` for logging

## Structure

```
agent/           → LangGraph agent, guardrails, CLI, API
mcp_servers/     → one dir per server (filesystem/, system/, search/, etc.)
voice/           → wake → STT → agent → TTS pipeline
data/            → gitignored persistent data
```

## Conventions

- Config via `pydantic-settings` + `.env`, never hardcoded
- MCP tools return error strings, never raise to the LLM
- Tests with `pytest` + `pytest-asyncio`
- Docker images based on `python:3.12-slim`

## Agent pattern

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

llm = ChatOpenAI(model="qwen3.5-4b", base_url="http://vllm:8000/v1", api_key="not-needed")
model = llm.bind_tools(tools)

graph = StateGraph(MessagesState)
graph.add_node("agent", lambda s: {"messages": [model.invoke(s["messages"])]})
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
```

## MCP server pattern

```python
from fastmcp import FastMCP
mcp = FastMCP("server-name")

@mcp.tool()
async def tool_name(param: str) -> str:
    """Description the LLM reads to decide when to use this tool."""
    try:
        return "Success: ..."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
```

## vLLM flags

```
--model Qwen/Qwen3.5-4B --enable-auto-tool-choice --tool-call-parser qwen3_coder
--reasoning-parser qwen3 --language-model-only --enable-prefix-caching
--gpu-memory-utilization 0.70 --max-model-len 16384
```