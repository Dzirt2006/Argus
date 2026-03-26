"""Voice pipeline orchestrator.

wake word → record → STT → agent → TTS → speaker

Run directly:  python -m voice.pipeline
"""

from __future__ import annotations

import asyncio
import time

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from agent.tracing import setup_logging, new_request_id
from voice.audio import play_audio, record_until_silence
from voice.config import voice_settings
from voice.stt import transcribe
from voice.tts import synthesize
from voice.wakeword import listen_for_wakeword

setup_logging()
log = structlog.get_logger("voice.pipeline")


class VoicePipeline:
    """Wires wake word detection to the LangGraph agent with voice I/O."""

    def __init__(self, agent, config: dict, messages: list):
        self.agent = agent
        self.config = config
        self.messages = messages

    def run(self) -> None:
        """Start the wake word listener. Blocks forever."""
        log.info("pipeline_start", wakeword=voice_settings.wakeword_model)
        print("Listening for wake word...\n")
        listen_for_wakeword(on_wake=self._on_wake)

    def _on_wake(self) -> None:
        """Called when the wake word is detected."""
        asyncio.run(self._handle_turn())

    async def _handle_turn(self) -> None:
        """Record → transcribe → agent → speak."""
        request_id = new_request_id()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        t0 = time.monotonic()

        # --- STT ---
        print("Listening...")
        audio = record_until_silence()
        if len(audio) < voice_settings.sample_rate * 0.3:
            log.info("recording_too_short", request_id=request_id)
            structlog.contextvars.unbind_contextvars("request_id")
            return

        text = transcribe(audio)
        if not text:
            log.info("stt_empty", request_id=request_id)
            structlog.contextvars.unbind_contextvars("request_id")
            return

        log.info("user_said", request_id=request_id, text=text)
        print(f"You: {text}")

        # --- Agent ---
        self.messages.append(HumanMessage(content=text))
        result = await self.agent.ainvoke({"messages": self.messages}, self.config)

        # Handle confirmation interrupts via voice
        while result.get("__interrupt__"):
            result = await self._handle_voice_confirm(result)

        self.messages = result["messages"]
        response_text = self.messages[-1].content
        log.info("agent_response", request_id=request_id, text_length=len(response_text))
        print(f"Argus: {response_text}")

        # --- TTS ---
        response_audio, sr = synthesize(response_text)
        play_audio(response_audio, sample_rate=sr)

        duration = time.monotonic() - t0
        log.info("turn_done", request_id=request_id, duration=round(duration, 3))
        structlog.contextvars.unbind_contextvars("request_id")
        print("Listening for wake word...\n")

    async def _handle_voice_confirm(self, result: dict) -> dict:
        """Speak the confirmation question, listen for yes/no."""
        for irq in result.get("__interrupt__", []):
            desc = irq.value.get("description", "an action")
            question = f"Should I proceed with {desc}?"
            log.info("confirm_asking", description=desc)

            # Speak the question
            q_audio, sr = synthesize(question)
            play_audio(q_audio, sample_rate=sr)

            # Listen for answer
            print("Waiting for confirmation...")
            answer_audio = record_until_silence()
            answer_text = transcribe(answer_audio).lower().strip()
            log.info("confirm_answer", text=answer_text)

            approved = answer_text in ("yes", "yeah", "yep", "sure", "go ahead", "do it", "y")
            resume_value = "y" if approved else "n"
            result = await self.agent.ainvoke(Command(resume=resume_value), self.config)

        return result
