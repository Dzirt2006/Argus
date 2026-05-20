FROM python:3.12-slim

WORKDIR /app

# mpv handles playback; ffmpeg is its decoder backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    mpv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system fastmcp ytmusicapi yt-dlp

COPY mcp_servers/media/server.py ./server.py

CMD ["python", "server.py"]
