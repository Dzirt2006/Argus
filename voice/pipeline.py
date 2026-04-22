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

from agent.summarizer import summarize_and_store
from agent.tracing import setup_logging, new_request_id
from voice.audio import play_audio, record_until_silence, warm_up as warm_up_vad
from voice.config import voice_settings
from voice.stt import transcribe, warm_up as warm_up_stt
from voice.tts import synthesize
from voice.wakeword import listen_for_wakeword

setup_logging()
log = structlog.get_logger("voice.pipeline")


def _is_question(text: str) -> bool:
    """Heuristic: treat a trailing '?' as 'agent wants a reply'."""
    return text.rstrip().endswith("?") if text else False


_TONE_SR = 16000


def _make_tone(freq: float, duration: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(_TONE_SR * duration), dtype=np.float32)
    return amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_error_tone() -> np.ndarray:
    """Descending two-tone buzz to signal failure."""
    return np.concatenate([_make_tone(440, 0.12), _make_tone(220, 0.18)])


class VoicePipeline:
    """Wires wake word detection to the LangGraph agent with voice I/O."""

    def __init__(self, agent, config: dict, messages: list):
        self.agent = agent
        self.config = config
        self.messages = messages
        self._start_tone = _make_tone(880, 0.15)
        self._end_tone = _make_tone(660, 0.12)
        self._error_tone = _make_error_tone()

    def _play_tone(self, tone: np.ndarray) -> None:
        play_audio(tone, sample_rate=_TONE_SR)

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
        """Record → transcribe → agent → speak, looping while agent asks follow-ups."""
        request_id = new_request_id()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        t0 = time.monotonic()

        session_start_idx = len(self.messages)
        followup = False
        for turn_idx in range(voice_settings.max_followup_turns + 1):
            # --- STT ---
            if followup:
                print("🎤 Recording follow-up... (speak now)")
            else:
                print("🎤 Recording... (speak now)")
            self._play_tone(self._start_tone)
            start_timeout = voice_settings.followup_start_timeout if followup else None
            audio = record_until_silence(start_timeout=start_timeout)

            if len(audio) == 0 and followup:
                print("⚠  No follow-up heard, back to wake word.")
                log.info("followup_timeout", request_id=request_id)
                break

            duration_sec = round(len(audio) / voice_settings.sample_rate, 1)
            print(f"🎤 Recorded {duration_sec}s of audio")

            if len(audio) < voice_settings.sample_rate * 0.3:
                print("⚠  Too short, ignoring.")
                log.info("recording_too_short", request_id=request_id)
                self._play_tone(self._error_tone)
                break

            self._play_tone(self._end_tone)
            print("📝 Transcribing...")
            text = transcribe(audio)
            if not text:
                print("⚠  Could not transcribe, ignoring.")
                log.info("stt_empty", request_id=request_id)
                self._play_tone(self._error_tone)
                break

            log.info("user_said", request_id=request_id, text=text, followup=followup)
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

            if not _is_question(response_text):
                break

            if turn_idx >= voice_settings.max_followup_turns:
                log.info("followup_max_reached", request_id=request_id)
                break

            followup = True
            log.info("followup_continue", request_id=request_id)

        duration = time.monotonic() - t0
        log.info("turn_done", request_id=request_id, duration=round(duration, 3))

        session_messages = self.messages[session_start_idx:]
        if session_messages:
            try:
                summarize_and_store(session_messages, session_id=request_id)
            except Exception as e:
                log.warning("summary_failed", error=str(e))

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
