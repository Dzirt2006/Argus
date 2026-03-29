"""Search MCP server — web search via DuckDuckGo.

No API key needed. Tools:
  - web_search: general web search
  - web_search_news: recent news search
"""

from fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("search", port=8003)


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
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"**{r['title']}**")
        lines.append(r.get("href", ""))
        lines.append(r.get("body", ""))
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
    with DDGS() as ddgs:
        results = list(ddgs.news(query, max_results=max_results))

    if not results:
        return "No news found."

    lines = []
    for r in results:
        lines.append(f"**{r['title']}**")
        lines.append(f"Source: {r.get('source', 'unknown')} | {r.get('date', '')}")
        lines.append(r.get("body", ""))
        lines.append(r.get("url", ""))
        lines.append("")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
