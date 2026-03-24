FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system fastmcp

COPY mcp_servers/filesystem/server.py ./server.py

CMD ["python", "server.py"]
