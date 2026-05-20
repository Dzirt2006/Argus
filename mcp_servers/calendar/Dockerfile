FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system fastmcp google-api-python-client google-auth-oauthlib google-auth-httplib2

COPY mcp_servers/calendar/server.py ./server.py

CMD ["python", "server.py"]
