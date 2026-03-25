import asyncio
import time
import uuid
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from agent.agent import create_agent
from agent.config import settings
from agent.mcp import create_mcp_client
from agent.tracing import setup_logging, new_request_id

setup_logging()
log = structlog.get_logger("agent.cli")


def _handle_interrupts(interrupts: list) -> list[Command]:
    """Prompt the user for each pending confirmation and return Commands."""
    commands = []
    for irq in interrupts:
        payload = irq.value
        desc = payload.get("description", "unknown action")
        print(f"\n⚠  Confirmation required:\n   {desc}")
        try:
            answer = input("   Allow? [y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        commands.append(Command(resume=answer))
    return commands


async def main():
    prompt_path = Path(settings.system_prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    async with create_mcp_client() as client:
        tools = client.get_tools()
        log.info("tools_loaded", count=len(tools), names=[t.name for t in tools])

        agent = create_agent(tools)
        messages = [SystemMessage(content=system_prompt)]

        # Each conversation turn gets a unique thread_id so the checkpointer
        # can track state for interrupt/resume.
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("Argus ready. Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.strip().lower() in ("quit", "exit"):
                break
            if not user_input.strip():
                continue

            request_id = new_request_id()
            structlog.contextvars.bind_contextvars(request_id=request_id)
            t0 = time.monotonic()

            messages.append(HumanMessage(content=user_input))
            result = await agent.ainvoke({"messages": messages}, config)

            # Handle confirmation interrupts — the graph may pause multiple
            # times if the LLM called several destructive tools.
            while result.get("__interrupt__"):
                commands = _handle_interrupts(result["__interrupt__"])
                for cmd in commands:
                    result = await agent.ainvoke(cmd, config)

            messages = result["messages"]
            duration = time.monotonic() - t0
            log.info("turn_done", request_id=request_id, duration=round(duration, 3))
            structlog.contextvars.unbind_contextvars("request_id")

            print(f"\n{messages[-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())
