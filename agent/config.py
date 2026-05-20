import time
from pathlib import Path

import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings

log = structlog.get_logger("agent.config")

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}

    vllm_url: str = "http://vllm:8000/v1"
    model_name: str = ""
    system_prompt_path: str = "docs/system_prompt.md"
    max_steps: int = 10
    vllm_timeout: int = 300  # seconds to wait for vLLM to come up
    ha_url: str = ""
    ha_token: str = ""
    mcp_servers: dict[str, str] = Field(default_factory=lambda: {
        "memory": "http://localhost:8007",
        "switches": "http://localhost:8008",
        "filesystem": "http://localhost:8001",
        "system": "http://localhost:8002",
        "search": "http://localhost:8003",
        "weather": "http://localhost:8004",
        "calendar": "http://localhost:8005",
        # "media": "http://localhost:8006",     # not yet in compose
    })

    memory_enabled: bool = True
    memory_sqlite_path: str = "/data/memory.db"
    memory_summary_min_turns: int = 2          # skip summarizing very short sessions
    memory_summaries_inject_n: int = 20        # how many recent summaries to inject per turn


settings = Settings()


def wait_for_vllm() -> str:
    """Poll vLLM /v1/models until a model is available. Returns the model ID."""
    url = f"{settings.vllm_url}/models"
    deadline = time.monotonic() + settings.vllm_timeout
    delay = 1.0

    while True:
        try:
            resp = httpx.get(url, timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                model_id = models[0]["id"]
                log.info("vllm_ready", model=model_id)
                return model_id
        except (httpx.HTTPError, httpx.ConnectError, KeyError):
            pass

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"vLLM not ready after {settings.vllm_timeout}s at {url}"
            )

        log.info("vllm_waiting", retry_in=delay, url=url)
        time.sleep(delay)
        delay = min(delay * 2, 15.0)
