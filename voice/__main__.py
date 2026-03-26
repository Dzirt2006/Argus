"""Entry point: python -m voice"""

import asyncio
import uuid
from pathlib import Path

from agent.agent import create_agent
from agent.config import settings
from agent.mcp import create_mcp_client
from voice.pipeline import VoicePipeline


async def main():
    prompt_path = Path(settings.system_prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    from langchain_core.messages import SystemMessage

    async with create_mcp_client() as client:
        tools = client.get_tools()
        agent = create_agent(tools)
        messages = [SystemMessage(content=system_prompt)]
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        pipeline = VoicePipeline(agent, config, messages)
        pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
