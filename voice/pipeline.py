"""Voice pipeline orchestrator.

wake word → record → STT → agent (streaming) → sentence TTS → speaker

Run directly:  python -m voice.pipeline
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import structlog
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command

from agent.summarizer import extract_and_store_facts, summarize_and_store
from agent.tracing import setup_logging, new_request_id
from voice.audio import play_audio, record_until_silence, warm_up as warm_up_vad
from voice.config import voice_settings
from voice.sentence_buffer import SentenceBuffer
from voice.stt import transcribe, warm_up as warm_up_stt
from voice.tts import synthesize
from voice.wakeword import listen_for_wakeword

setup_logging()
log = structlog.get_logger("voice.pipeline")


# Bounded so the producer doesn't run wildly ahead of the speaker.
_TTS_QUEUE_MAX = 8


def _is_question(text: str) -> bool:
    """Heuristic: treat a trailing '?' as 'agent wants a reply'."""
    return text.rstrip().endswith("?") if text else False


def _chunk_text(chunk: AIMessageChunk) -> str:
    """Extract the visible content from an AIMessageChunk.

    Qwen3's reasoning-parser ships chain-of-thought in ``additional_kwargs``
    (as ``reasoning_content``); only ``chunk.content`` is meant to be spoken.
    Content can be a string or a list of content blocks (multimodal).
    """
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


_TONE_SR = 16000


def _make_tone(freq: float, duration: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(_TONE_SR * duration), dtype=np.float32)
    return amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_error_tone() -> np.ndarray:
    """Descending two-tone buzz to signal failure."""
    return np.concatenate([_make_tone(440, 0.12), _make_tone(220, 0.18)])


async def _tts_consumer(queue: "asyncio.Queue[str | None]", request_id: str) -> None:
    """Pull sentences off ``queue`` and play each in order.

    Terminates when it receives the ``None`` sentinel. Audio plays in the
    order sentences were enqueued.
    """
    loop = asyncio.get_running_loop()
    while True:
        sentence = await queue.get()
        if sentence is None:
            queue.task_done()
            return
        try:
            audio, sr = await loop.run_in_executor(None, synthesize, sentence)
            await loop.run_in_executor(None, play_audio, audio, sr)
        except Exception as e:
            log.warning("tts_sentence_failed", request_id=request_id, error=str(e))
        finally:
            queue.task_done()


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

            # --- Agent (streaming) ---
            print("🤖 Thinking...")
            self.messages.append(HumanMessage(content=text))
            response_text = await self._stream_and_speak(
                {"messages": self.messages}, request_id
            )

            # Pull the final checkpointed state for next turn's input.
            state = await self.agent.aget_state(self.config)
            self.messages = state.values["messages"]
            log.info(
                "agent_response", request_id=request_id, text_length=len(response_text)
            )
            print(f"🤖 Argus: {response_text}")

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
            try:
                extract_and_store_facts(session_messages, session_id=request_id)
            except Exception as e:
                log.warning("facts_failed", error=str(e))

        structlog.contextvars.unbind_contextvars("request_id")
        print(f"✅ Turn done ({round(duration, 1)}s)\n")
        print("👂 Listening for wake word...\n")

    async def _stream_and_speak(self, stream_input, request_id: str) -> str:
        """Stream the agent run, speak sentences as they form, handle interrupts.

        Loops over the graph stream — every time the graph pauses on an
        interrupt() we speak the confirmation question, capture a yes/no via
        STT, resume with Command(resume=...) and continue streaming. Returns
        the visible response text from the final AI message.
        """
        current_input = stream_input
        final_text = ""

        while True:
            final_text, pending_interrupts = await self._stream_one_phase(
                current_input, request_id
            )
            if not pending_interrupts:
                break

            # Resume one interrupt at a time. After each resume, the graph
            # continues and may pause again on another interrupt; the outer
            # while-loop catches that on the next pass.
            irq = pending_interrupts[0]
            resume_value = await self._voice_confirm_one(irq)
            current_input = Command(resume=resume_value)

        return final_text

    async def _stream_one_phase(
        self, stream_input, request_id: str
    ) -> tuple[str, list]:
        """Run one segment of the graph stream until END or interrupt.

        Returns the visible text spoken in this phase and any pending
        interrupts found in the final state (empty list if the graph ran
        through to END).
        """
        buffer = SentenceBuffer()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_TTS_QUEUE_MAX)
        consumer = asyncio.create_task(_tts_consumer(queue, request_id))

        spoken_chars = 0
        sentences_emitted = 0
        agent_node_seen = False
        stream_failed = False

        try:
            async for mode, payload in self.agent.astream(
                stream_input,
                self.config,
                stream_mode=["messages", "values"],
            ):
                if mode != "messages":
                    continue
                chunk, meta = payload
                # Only stream tokens from the 'agent' LLM node — skip tool
                # outputs, summarizer side-channels, etc.
                if meta.get("langgraph_node") != "agent":
                    continue
                if not isinstance(chunk, AIMessageChunk):
                    continue
                agent_node_seen = True

                text = _chunk_text(chunk)
                if not text:
                    continue
                spoken_chars += len(text)

                for sentence in buffer.feed(text):
                    sentences_emitted += 1
                    await queue.put(sentence)

            tail = buffer.flush()
            if tail:
                sentences_emitted += 1
                await queue.put(tail)
        except BaseException:
            stream_failed = True
            raise
        finally:
            # On the happy path block on the sentinel so we play out everything
            # already queued. On the failure path the consumer may be stuck in
            # a blocking executor call (synthesize/play_audio) with the queue
            # full — `put(None)` would deadlock — so drop the sentinel
            # non-blockingly and cancel the consumer instead.
            if stream_failed:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                consumer.cancel()
                try:
                    await consumer
                except (asyncio.CancelledError, Exception):
                    pass
            else:
                await queue.put(None)
                await consumer

        # After the stream finishes, query the checkpointed state to see if the
        # graph paused on an interrupt (e.g. destructive tool confirmation).
        state = await self.agent.aget_state(self.config)
        pending_interrupts = []
        for task in state.tasks:
            for irq in getattr(task, "interrupts", []) or []:
                pending_interrupts.append(irq)

        log.info(
            "stream_phase_done",
            request_id=request_id,
            chars=spoken_chars,
            sentences=sentences_emitted,
            agent_seen=agent_node_seen,
            pending_interrupts=len(pending_interrupts),
        )

        # We concatenate raw chunk text (not buffered sentences) so the caller
        # gets the exact response — important for the trailing-? heuristic.
        return self._reconstruct_text_from_state(state), pending_interrupts

    def _reconstruct_text_from_state(self, state) -> str:
        """Pull the last AI message text from the graph state.

        Used to derive the response text for follow-up detection. Falls back
        to '' if the last message isn't an AI message (e.g. interrupt mid-tool).
        """
        messages = state.values.get("messages", []) if state.values else []
        if not messages:
            return ""
        last = messages[-1]
        content = getattr(last, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts)
        return ""

    async def _voice_confirm_one(self, irq) -> str:
        """Speak the interrupt's question and capture a yes/no answer.

        Returns "y" or "n" — the value passed back via Command(resume=...).
        """
        payload = irq.value if hasattr(irq, "value") else {}
        desc = payload.get("description", "an action") if isinstance(payload, dict) else "an action"
        question = f"Should I proceed with {desc}?"
        log.info("confirm_asking", description=desc)

        loop = asyncio.get_running_loop()
        q_audio, sr = await loop.run_in_executor(None, synthesize, question)
        await loop.run_in_executor(None, play_audio, q_audio, sr)

        print("Waiting for confirmation...")
        answer_audio = await loop.run_in_executor(None, record_until_silence, None)
        answer_text = (await loop.run_in_executor(None, transcribe, answer_audio)).lower().strip()
        log.info("confirm_answer", text=answer_text)

        approved = answer_text in ("yes", "yeah", "yep", "sure", "go ahead", "do it", "y")
        return "y" if approved else "n"
