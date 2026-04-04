from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    vllm_url: str = "http://vllm:8000/v1"
    model_name: str = "qwen3.5-4b"
    system_prompt_path: str = "docs/system_prompt.md"
    max_steps: int = 10
    mcp_servers: dict[str, str] = Field(default_factory=lambda: {
        # "filesystem": "http://localhost:8001",
        # "system": "http://localhost:8002",
        "search": "http://localhost:8003",
        "weather": "http://localhost:8004",
        # "calendar": "http://localhost:8005",
        # "media": "http://localhost:8006",
    })


settings = Settings()
