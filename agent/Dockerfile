FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

COPY agent/ ./agent/
COPY docs/ ./docs/

CMD ["python", "-m", "agent"]
