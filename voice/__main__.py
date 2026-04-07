"""Entry point: python -m voice"""

import asyncio
import uuid
from pathlib import Path

from langchain_core.messages import SystemMessage

from agent.agent import create_agent
from agent.config import settings, wait_for_vllm
from agent.mcp import create_mcp_client
from voice.pipeline import VoicePipeline


async def _async_setup():
    """Async init: connect to MCP servers, load tools."""
    client = create_mcp_client()
    return await client.get_tools()


def main():
    print("[1/5] Waiting for vLLM...")
    model_name = wait_for_vllm()
    print(f"[2/5] vLLM ready — model: {model_name}")

    prompt_path = Path(settings.system_prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8")
    print(f"[3/5] System prompt loaded ({len(system_prompt)} chars)")

    tools = asyncio.run(_async_setup())
    print(f"[4/5] MCP tools loaded: {[t.name for t in tools]}")

    agent = create_agent(tools, model_name=model_name)
    messages = [SystemMessage(content=system_prompt)]
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("[5/5] Starting voice pipeline — say 'Hey Jarvis' to begin")
    pipeline = VoicePipeline(agent, config, messages)
    pipeline.run()


if __name__ == "__main__":
    main()
