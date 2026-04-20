"""Voice pipeline orchestrator.

wake word → record → STT → agent → TTS → speaker

Run directly:  python -m voice.pipeline
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from agent.tracing import setup_logging, new_request_id
from voice.audio import play_audio, record_until_silence, warm_up as warm_up_vad
from voice.config import voice_settings
from voice.stt import transcribe, warm_up as warm_up_stt
from voice.tts import synthesize
from voice.wakeword import listen_for_wakeword

setup_logging()
log = structlog.get_logger("voice.pipeline")


def _make_beep() -> tuple[np.ndarray, int]:
    """Generate a short beep tone to signal 'start speaking'."""
    sr = 16000
    duration = 0.15
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 880 * t)
    return tone, sr


class VoicePipeline:
    """Wires wake word detection to the LangGraph agent with voice I/O."""

    def __init__(self, agent, config: dict, messages: list):
        self.agent = agent
        self.config = config
        self.messages = messages
        self._beep, self._beep_sr = _make_beep()

    def run(self) -> None:
        """Start the wake word listener. Blocks forever."""
        print("Loading Whisper model (one-time)...")
        warm_up_stt()
        print("Whisper ready.")
        warm_up_vad()
        print("VAD ready.\n")

        log.info("pipeline_start", wakeword=voice_settings.wakeword_model)
        print(f"👂 Listening for wake word (\"{voice_settings.wakeword_model}\")...\n")
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
        print("🎤 Recording... (speak now)")
        play_audio(self._beep, sample_rate=self._beep_sr)
        audio = record_until_silence()
        duration_sec = round(len(audio) / voice_settings.sample_rate, 1)
        print(f"🎤 Recorded {duration_sec}s of audio")

        if len(audio) < voice_settings.sample_rate * 0.3:
            print("⚠  Too short, ignoring.")
            log.info("recording_too_short", request_id=request_id)
            structlog.contextvars.unbind_contextvars("request_id")
            return

        print("📝 Transcribing...")
        text = transcribe(audio)
        if not text:
            print("⚠  Could not transcribe, ignoring.")
            log.info("stt_empty", request_id=request_id)
            structlog.contextvars.unbind_contextvars("request_id")
            return

        log.info("user_said", request_id=request_id, text=text)
        print(f"👤 You said: \"{text}\"")

        # --- Agent ---
        print("🤖 Thinking...")
        self.messages.append(HumanMessage(content=text))
        result = await self.agent.ainvoke({"messages": self.messages}, self.config)

        # Handle confirmation interrupts via voice
        while result.get("__interrupt__"):
            result = await self._handle_voice_confirm(result)

        self.messages = result["messages"]
        response_text = self.messages[-1].content
        log.info("agent_response", request_id=request_id, text_length=len(response_text))
        print(f"🤖 Argus: {response_text}")

        # --- TTS ---
        print("🔊 Speaking...")
        response_audio, sr = synthesize(response_text)
        play_audio(response_audio, sample_rate=sr)

        duration = time.monotonic() - t0
        log.info("turn_done", request_id=request_id, duration=round(duration, 3))
        structlog.contextvars.unbind_contextvars("request_id")
        print(f"✅ Turn done ({round(duration, 1)}s)\n")
        print("👂 Listening for wake word...\n")

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
