"""Search MCP server — web search via SearXNG.

Self-hosted metasearch (Google, Bing, DuckDuckGo, etc.). No API key needed.

Tools:
  - web_search: general web search
  - web_search_news: recent news search
"""

import os

from fastmcp import FastMCP
import httpx

mcp = FastMCP("search")

_SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")


def _search(query: str, categories: str = "general", max_results: int = 5) -> list[dict]:
    """Query SearXNG and return results."""
    resp = httpx.get(
        f"{_SEARXNG_URL}/search",
        params={
            "q": query,
            "format": "json",
            "categories": categories,
            "language": "en",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])[:max_results]


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Use this for factual questions, current events, looking up people/places/things,
    or anything the user asks that requires up-to-date information.

    Args:
        query: The search query — be specific for better results.
               Example: "current weather Madison WI" or "Python 3.13 release date"
        max_results: Number of results to return (1-10, default 5).
    """
    results = _search(query, categories="general", max_results=max_results)

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"**{r.get('title', '')}**")
        lines.append(r.get("url", ""))
        lines.append(r.get("content", ""))
        lines.append("")

    return "\n".join(lines).strip()


@mcp.tool()
def web_search_news(query: str, max_results: int = 5) -> str:
    """Search for recent news articles.

    Use this when the user asks about current events, breaking news,
    or recent developments on any topic.

    Args:
        query: The news search query.
               Example: "AI regulation 2026" or "NBA playoffs"
        max_results: Number of results to return (1-10, default 5).
    """
    results = _search(query, categories="news", max_results=max_results)

    if not results:
        return "No news found."

    lines = []
    for r in results:
        lines.append(f"**{r.get('title', '')}**")
        lines.append(f"Source: {r.get('engine', 'unknown')} | {r.get('publishedDate', '')}")
        lines.append(r.get("content", ""))
        lines.append(r.get("url", ""))
        lines.append("")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)
