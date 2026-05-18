from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.config import settings


def create_mcp_client() -> MultiServerMCPClient:
    servers = {}
    for name, url in settings.mcp_servers.items():
        servers[name] = {
            "url": f"{url}/mcp",
            "transport": "streamable_http",
        }
    return MultiServerMCPClient(servers)
