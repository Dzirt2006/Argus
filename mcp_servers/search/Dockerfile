FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system fastmcp httpx

COPY mcp_servers/search/server.py ./server.py

CMD ["python", "server.py"]
