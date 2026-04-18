from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.config import settings


def create_mcp_client() -> MultiServerMCPClient:
    servers = {}
    for name, url in settings.mcp_servers.items():
        servers[name] = {
            "url": f"{url}/mcp",
            "transport": "streamable_http",
        }
    if settings.ha_url and settings.ha_token:
        servers["homeassistant"] = {
            "url": f"{settings.ha_url.rstrip('/')}/mcp_server/sse",
            "transport": "sse",
            "headers": {"Authorization": f"Bearer {settings.ha_token}"},
        }
    return MultiServerMCPClient(servers)
