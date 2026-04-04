# Search MCP Server

Web search via self-hosted SearXNG metasearch engine. Aggregates results from Google, Bing, DuckDuckGo, and others.

## Tools

- **web_search** — general web search
- **web_search_news** — recent news articles

## Setup

### 1. Start SearXNG

```bash
docker compose up searxng -d
```

SearXNG runs on port 8080. Verify it's working: `http://localhost:8080`

### 2. Configuration

SearXNG settings live in `searxng/settings.yml` at the project root. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `search.formats` | `[html, json]` | Must include `json` for the MCP server to work |
| `search.default_lang` | `en` | Language for search results |
| `engines` | Google, Bing, DDG, Wikipedia, Google News | Which search engines to aggregate |

To disable/enable an engine, edit `searxng/settings.yml`:
```yaml
engines:
  - name: google
    engine: google
    disabled: false   # set to true to disable
```

Rate limiting is disabled in `searxng/limiter.toml` for local use.

### 3. Run the search server

**Docker (via compose):**
```bash
docker compose up search -d
```

**Local dev:**
```bash
SEARXNG_URL=http://localhost:8080 python mcp_servers/search/server.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARXNG_URL` | `http://localhost:8080` | SearXNG instance URL |

## Troubleshooting

- **No results** — Check SearXNG is running: `curl http://localhost:8080/search?q=test&format=json`
- **Google blocking** — Google may rate-limit SearXNG with heavy use. Bing and DuckDuckGo will still work as fallbacks. You can also add a delay in `searxng/settings.yml` under the google engine config.
- **Timeout errors** — The search server uses a 10s timeout for SearXNG requests. If SearXNG is slow on first start (downloading engine configs), retry after a few seconds.
