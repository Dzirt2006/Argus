from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    vllm_url: str = "http://vllm:8000/v1"
    model_name: str = "qwen3.5-4b"
    system_prompt_path: str = "docs/system_prompt.md"
    max_steps: int = 10
    mcp_servers: dict[str, str] = Field(default_factory=lambda: {
        "filesystem": "http://filesystem:8001",
        "system": "http://system:8002",
        "search": "http://search:8003",
        "weather": "http://weather:8004",
        "calendar": "http://calendar:8005",
        "media": "http://media:8006",
    })


settings = Settings()
