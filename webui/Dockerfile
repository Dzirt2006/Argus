FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY webui/requirements.txt ./requirements.txt
RUN uv pip install --system -r requirements.txt

COPY webui/ ./webui/

EXPOSE 8009

CMD ["uvicorn", "webui.app:app", "--host", "0.0.0.0", "--port", "8009"]
