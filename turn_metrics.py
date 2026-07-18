#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Per bot-message latency metrics → CSV (one row per assistant reply)."""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

from session_turn import set_last_user_message

DEFAULT_CSV_PATH = Path(__file__).parent / "metrics" / "bot_messages.csv"
BOT_METRICS_CSV = Path(os.environ.get("BOT_METRICS_CSV", DEFAULT_CSV_PATH))

_LOG_LOCK = Lock()

CSV_HEADER = [
    "session_id",
    "pipeline_mode",
    "input_type",
    "user_question",
    "user_message_date",
    "user_message_time",
    "llm_first_token_date",
    "llm_first_token_time",
    "first_audio_heard_date",
    "first_audio_heard_time",
    "stt_latency_ms",
    "llm_ttft_ms",
    "full_text_response_ms",
    "tts_first_audio_ms",
    "first_audio_heard_ms",
    "audio_complete_ms",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _split_datetime(dt: datetime | None) -> tuple[str, str]:
    if dt is None:
        return "", ""
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%d"), f"{dt.strftime('%H:%M:%S')}.{ms:03d}"


def _delta_ms(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return ""
    return f"{(end - start) * 1000:.1f}"


def _is_citation_text(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.startswith("Source:") or stripped.startswith("Sources:")


def _extract_user_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
    return " ".join(parts).strip()


@dataclass
class _ActiveTurn:
    session_id: str
    pipeline_mode: str
    input_type: str = ""
    user_message: str = ""
    bot_text: str = ""
    skip_tts: bool = False

    t0_mono: float | None = None
    t1_mono: float | None = None
    t2_mono: float | None = None
    t3_mono: float | None = None
    t4_mono: float | None = None
    t5_mono: float | None = None
    t6_mono: float | None = None

    t0_wall: datetime | None = None
    t2_wall: datetime | None = None
    t3_wall: datetime | None = None
    t4_wall: datetime | None = None
    t5_wall: datetime | None = None
    t6_wall: datetime | None = None

    llm_segment_had_text: bool = False
    awaiting_user_text: bool = False


class TurnMetricsObserver(BaseObserver):
    """Observe pipeline frames and append one CSV row per bot reply."""

    def __init__(
        self,
        *,
        session_id: str = "",
        pipeline_mode: str = "",
        max_frames: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._pipeline_mode = pipeline_mode
        self._turn: _ActiveTurn | None = None
        self._skip_tts = False
        self._processed_frames: set[int] = set()
        self._frame_history: deque[int] = deque(maxlen=max_frames)

    def _seen(self, frame: Frame) -> bool:
        if frame.id in self._processed_frames:
            return True
        self._processed_frames.add(frame.id)
        self._frame_history.append(frame.id)
        if len(self._processed_frames) > len(self._frame_history):
            self._processed_frames = set(self._frame_history)
        return False

    def _start_turn(
        self,
        *,
        input_type: str,
        user_message: str = "",
        t0_wall: datetime | None = None,
    ) -> None:
        if self._turn is not None and self._turn.t3_mono is not None:
            self._write_row(self._turn)
        self._turn = _ActiveTurn(
            session_id=self._session_id,
            pipeline_mode=self._pipeline_mode,
            input_type=input_type,
            user_message=user_message[:500],
            skip_tts=self._skip_tts,
            t0_mono=time.perf_counter(),
            t0_wall=t0_wall or _utc_now(),
            awaiting_user_text=not bool(user_message.strip()),
        )

    def _ensure_turn(self, input_type: str) -> _ActiveTurn:
        if self._turn is None:
            self._start_turn(input_type=input_type)
        return self._turn

    def _mark_stt(self) -> None:
        turn = self._turn
        if turn is None or turn.t1_mono is not None:
            return
        turn.t1_mono = time.perf_counter()

    def _mark_llm_first_token(self) -> None:
        turn = self._turn
        if turn is None or turn.t2_mono is not None:
            return
        turn.t2_mono = time.perf_counter()
        turn.t2_wall = _utc_now()

    def _mark_full_text(self) -> None:
        turn = self._turn
        if turn is None or turn.t3_mono is not None:
            return
        turn.t3_mono = time.perf_counter()
        turn.t3_wall = _utc_now()

    def _mark_tts_first_audio(self) -> None:
        turn = self._turn
        if turn is None or turn.t4_mono is not None:
            return
        turn.t4_mono = time.perf_counter()
        turn.t4_wall = _utc_now()

    def _mark_first_audio_heard(self) -> None:
        turn = self._turn
        if turn is None or turn.t5_mono is not None:
            return
        turn.t5_mono = time.perf_counter()
        turn.t5_wall = _utc_now()

    def _mark_audio_complete(self) -> None:
        turn = self._turn
        if turn is None or turn.t6_mono is not None:
            return
        turn.t6_mono = time.perf_counter()
        turn.t6_wall = _utc_now()

    def _commit_if_ready(self, *, text_only: bool = False) -> None:
        turn = self._turn
        if turn is None:
            return
        if turn.t2_mono is None and turn.t3_mono is None:
            return
        if text_only or turn.skip_tts:
            if turn.t3_mono is None:
                return
            self._write_row(turn)
            self._turn = None
            return
        if turn.t6_mono is not None:
            self._write_row(turn)
            self._turn = None

    def _write_row(self, turn: _ActiveTurn) -> None:
        t1_base = turn.t1_mono if turn.t1_mono is not None else turn.t0_mono
        row = {
            "session_id": turn.session_id,
            "pipeline_mode": turn.pipeline_mode,
            "input_type": turn.input_type,
            "user_question": turn.user_message,
            "user_message_date": _split_datetime(turn.t0_wall)[0],
            "user_message_time": _split_datetime(turn.t0_wall)[1],
            "llm_first_token_date": _split_datetime(turn.t2_wall)[0],
            "llm_first_token_time": _split_datetime(turn.t2_wall)[1],
            "first_audio_heard_date": _split_datetime(turn.t5_wall)[0],
            "first_audio_heard_time": _split_datetime(turn.t5_wall)[1],
            "stt_latency_ms": _delta_ms(turn.t0_mono, turn.t1_mono),
            "llm_ttft_ms": _delta_ms(t1_base, turn.t2_mono),
            "full_text_response_ms": _delta_ms(t1_base, turn.t3_mono),
            "tts_first_audio_ms": _delta_ms(turn.t3_mono, turn.t4_mono),
            "first_audio_heard_ms": _delta_ms(turn.t0_mono, turn.t5_mono),
            "audio_complete_ms": _delta_ms(turn.t0_mono, turn.t6_mono),
        }

        BOT_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            needs_header = (
                not BOT_METRICS_CSV.exists() or BOT_METRICS_CSV.stat().st_size == 0
            )
            if not needs_header:
                first_line = BOT_METRICS_CSV.read_text(encoding="utf-8").splitlines()[0]
                if first_line.strip() != ",".join(CSV_HEADER):
                    body = BOT_METRICS_CSV.read_text(encoding="utf-8").splitlines(True)
                    BOT_METRICS_CSV.write_text(
                        ",".join(CSV_HEADER) + "\n" + "".join(body[1:]),
                        encoding="utf-8",
                    )
            with BOT_METRICS_CSV.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)

        # Per-call JSON conversation log (text + voice timings)
        try:
            from conversation_log import _utc_iso, record_server_turn

            def _iso(dt: datetime | None) -> str | None:
                return _utc_iso(dt) if dt else None

            has_voice = turn.t5_wall is not None or turn.t6_wall is not None
            record_server_turn(
                turn.session_id or "unknown",
                pipeline_mode=turn.pipeline_mode,
                input_type=turn.input_type or "text",
                user_text=turn.user_message,
                bot_text=turn.bot_text,
                user_sent_at=_iso(turn.t0_wall),
                bot_text_first_token_at=_iso(turn.t2_wall),
                bot_text_complete_at=_iso(turn.t3_wall),
                bot_voice_first_heard_at=_iso(turn.t5_wall) if has_voice else None,
                bot_voice_complete_at=_iso(turn.t6_wall) if has_voice else None,
                server_extra={
                    "stt_latency_ms": row["stt_latency_ms"],
                    "llm_ttft_ms": row["llm_ttft_ms"],
                    "full_text_response_ms": row["full_text_response_ms"],
                    "tts_first_audio_ms": row["tts_first_audio_ms"],
                    "first_audio_heard_ms": row["first_audio_heard_ms"],
                    "audio_complete_ms": row["audio_complete_ms"],
                    "tts_started_at": _iso(turn.t4_wall),
                },
            )
        except Exception as e:
            logger.warning(f"conversation_log write failed: {e}")

        logger.info(
            "bot_message_metrics "
            f"stt={row['stt_latency_ms']}ms "
            f"ttft={row['llm_ttft_ms']}ms "
            f"text={row['full_text_response_ms']}ms "
            f"voice={row['first_audio_heard_ms']}ms"
        )

    def _update_user_text(self, text: str, *, final: bool) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        set_last_user_message(cleaned)
        turn = self._turn
        if turn is None:
            return
        turn.user_message = cleaned[:500]
        if final:
            turn.awaiting_user_text = False

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if self._seen(frame):
            return

        if isinstance(frame, InterimTranscriptionFrame):
            self._update_user_text(frame.text, final=False)
        elif isinstance(frame, TranscriptionFrame):
            self._update_user_text(frame.text, final=True)
            self._mark_stt()

        if data.direction != FrameDirection.DOWNSTREAM:
            return

        if isinstance(frame, LLMConfigureOutputFrame):
            self._skip_tts = bool(frame.skip_tts)
            if self._turn is not None:
                self._turn.skip_tts = self._skip_tts
            return

        if isinstance(frame, LLMMessagesAppendFrame):
            text = _extract_user_text(frame.messages)
            if not text or text == "Please introduce yourself briefly.":
                return
            self._start_turn(input_type="text", user_message=text)
            set_last_user_message(text)
            # Typed input has no STT; treat input time as STT complete.
            self._mark_stt()
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._start_turn(input_type="voice", user_message="", t0_wall=_utc_now())
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            if self._turn is not None:
                self._turn.llm_segment_had_text = False
            return

        if isinstance(frame, LLMTextFrame):
            text = (frame.text or "").strip()
            if not text or _is_citation_text(text):
                return
            turn = self._turn
            if turn is None:
                return
            turn.llm_segment_had_text = True
            turn.bot_text = (turn.bot_text + text).strip()
            self._mark_llm_first_token()
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            turn = self._turn
            if turn is None:
                return
            if turn.llm_segment_had_text:
                self._mark_full_text()
                if turn.skip_tts:
                    self._commit_if_ready(text_only=True)
            turn.llm_segment_had_text = False
            return

        if isinstance(frame, TTSStartedFrame):
            self._mark_tts_first_audio()
            return

        if isinstance(frame, TTSAudioRawFrame):
            self._mark_tts_first_audio()
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            self._mark_first_audio_heard()
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._mark_audio_complete()
            self._commit_if_ready(text_only=False)
            return
