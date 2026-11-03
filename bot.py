"""ReceptionAi — voice upgrade of the existing LangGraph agent.

Wraps the same ``Main.py`` / ``Tools.py`` appointment flow:

- **cascade** (default): Deepgram STT → Main.py LangGraph (DeepSeek) → Deepgram TTS
- **realtime**: OpenAI Realtime speech-to-speech + the same Tools.py function calls

UI mode dropdown on ``/app/`` or ``/app-lite/`` selects the combo above.

Run with conda env ``hosmanag``::

    conda activate hosmanag
    python bot.py
    BOT_MODE=realtime python bot.py
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import (
    OpenAIRealtimeLLMService,
    OpenAIRealtimeLLMSettings,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

from interrupt_trace import InterruptTraceProcessor
from speculation import UserTurnText, make_transcript_tap
from turn_detection import (
    build_flux_stt,
    build_turn_setup,
    clamp_stop_secs,
    silence_turn_strategies,
    vad_params,
)
from turn_metrics import TurnMetricsObserver
from user_mute_safe import SafeBotSpeakingMuteStrategy
from voice_bridge import (
    realtime_system_prompt,
    get_shared_reception_graph,
    reception_function_schemas,
    make_langgraph_processor,
    register_reception_tools_on_llm,
)

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env", override=True)

# Loguru → app.log (stdlib logging from graph/tools already writes here).
logger.add(
    _ROOT / "app.log",
    level="INFO",
    format="{time:HH:mm:ss} - {level} - {message}",
    enqueue=True,
    rotation="20 MB",
    retention=5,
)

try:
    from service_settings import apply_settings_to_env

    apply_settings_to_env()
except Exception:
    pass


# Kept under their old names: tests and callers import them from bot.
_clamp_stop_secs = clamp_stop_secs
_vad_params = vad_params


def _user_turn_strategies(
    stop_secs: float | None = None,
    *,
    shared_state: dict | None = None,
) -> UserTurnStrategies:
    """Silence method's strategies — see ``turn_detection.silence_turn_strategies``."""
    return silence_turn_strategies(stop_secs, shared_state=shared_state)


def build_cascade_pipeline(
    transport: BaseTransport,
    context: LLMContext,
    thread_id: str,
    *,
    call_id: str = "",
    stop_secs: float | None = None,
    shared_state: dict | None = None,
    turn_detector: str = "silence",
    max_silence_secs: float | None = None,
    speculative: bool = False,
) -> Pipeline:
    """Deepgram STT → existing Main.py LangGraph (GLM) → Deepgram TTS.

    ``turn_detector`` picks the end-of-turn method (``turn_detection.py``);
    ``speculative`` starts the agent at the early end-of-turn signal.
    """
    # Boost month / phone / booking phrases so DOB-style dates and numbers transcribe cleaner.
    stt_keyterms = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "phone", "appointment", "doctor", "prescription", "medicine",
        "tomorrow", "today", "morning", "afternoon", "evening",
    ]
    call_key = call_id or thread_id
    state = shared_state if shared_state is not None else {}
    setup = build_turn_setup(
        turn_detector,
        stop_secs=stop_secs,
        max_silence_secs=max_silence_secs,
        speculative=speculative,
        shared_state=state,
    )
    stt = build_flux_stt(
        keyterms=stt_keyterms,
        speculative=speculative,
        max_silence_secs=setup.max_silence_secs,
    ) if setup.uses_flux else DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_STT_LANGUAGE", "en"),
            interim_results=True,
            smart_format=True,
            punctuate=True,
            keyterm=stt_keyterms,
        ),
    )
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice=os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en"),
    )

    graph_app = get_shared_reception_graph(llm_provider=os.getenv("LLM_PROVIDER", "deepseek"))
    turn_text = UserTurnText()
    agent = make_langgraph_processor(
        thread_id,
        graph_app,
        call_id=call_key,
        shared_state=state,
        speculation=setup.speculation,
        turn_text=turn_text,
    )
    # Mirrors transcripts for VAD-pause speculation (the aggregator swallows them).
    transcript_tap = make_transcript_tap(
        turn_text,
        on_update=agent.on_transcript_update if setup.speculation == "vad" else None,
    )
    if setup.uses_flux:

        @stt.event_handler("on_eager_end_of_turn")
        async def _on_eager_end_of_turn(_service, transcript):
            await agent.on_early_end_of_turn(transcript)

        @stt.event_handler("on_turn_resumed")
        async def _on_turn_resumed(_service):
            await agent.on_turn_resumed()

    interrupt_trace = InterruptTraceProcessor(call_key, state)

    logger.info(
        f"Cascade turn detector={setup.detector} speculative={setup.speculation or 'off'} "
        f"VAD params={setup.vad} end_of_speech_wait={_clamp_stop_secs(stop_secs)} "
        f"max_silence={setup.max_silence_secs}"
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=setup.vad),
            # While TTS is playing, ignore mic activity so barge-in from echo
            # cannot cut the second half of a multi-sentence reply. Also unmute
            # on InterruptionFrame so a cut reply cannot leave the mic stuck off.
            user_mute_strategies=[SafeBotSpeakingMuteStrategy()],
            # Chosen end-of-turn method (turn_detection.build_turn_setup). The
            # silence method keeps the explicit SpeechTimeout stop — Pipecat's
            # default LocalSmartTurnAnalyzerV3 would ignore stop_secs.
            user_turn_strategies=setup.strategies,
        ),
    )

    @user_aggregator.event_handler("on_user_turn_started")
    async def _on_user_turn_started(aggregator, strategy):
        try:
            from conversation_log import record_timeline_event

            record_timeline_event(
                call_key,
                event_type="user_turn_started",
                user_text=str(state.get("last_heard_transcript") or state.get("last_user_text") or ""),
                bot_text=str(state.get("last_bot_text") or ""),
                strategy=type(strategy).__name__ if strategy is not None else "",
                phase=(
                    "agent_busy"
                    if state.get("busy")
                    else "bot_speaking"
                    if state.get("bot_speaking")
                    else "idle"
                ),
                source="server",
            )
        except Exception:
            pass

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def _on_user_turn_stopped(aggregator, strategy, message):
        try:
            from conversation_log import record_timeline_event

            content = getattr(message, "content", None) or getattr(message, "text", None) or ""
            record_timeline_event(
                call_key,
                event_type="user_turn_stopped",
                user_text=str(content or state.get("last_heard_transcript") or "")[:500],
                bot_text=str(state.get("last_bot_text") or ""),
                strategy=type(strategy).__name__ if strategy is not None else "",
                phase=(
                    "agent_busy"
                    if state.get("busy")
                    else "bot_speaking"
                    if state.get("bot_speaking")
                    else "idle"
                ),
                source="server",
            )
            logger.info(
                f"User turn stopped ({call_key}): strategy={type(strategy).__name__} "
                f"text={str(content)[:120]!r}"
            )
        except Exception:
            pass

    return Pipeline(
        [
            transport.input(),
            stt,
            transcript_tap,
            user_aggregator,
            interrupt_trace,
            agent,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )


def build_realtime_pipeline(transport: BaseTransport, context: LLMContext) -> Pipeline:
    """OpenAI Realtime + ReceptionAi Tools.py handlers only."""
    tool_schemas = reception_function_schemas()

    # Resolved per session so an industry change in Admin takes effect on the next call.
    realtime_system = realtime_system_prompt()

    llm = OpenAIRealtimeLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAIRealtimeLLMSettings(
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1-mini"),
            system_instruction=realtime_system,
            session_properties=SessionProperties(
                instructions=realtime_system,
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(),
                        turn_detection=SemanticTurnDetection(),
                    ),
                    output=AudioOutput(
                        voice=os.getenv("OPENAI_REALTIME_VOICE", "marin"),
                    ),
                ),
                tools=tool_schemas,
            ),
        ),
    )

    context.set_tools(register_reception_tools_on_llm(llm))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    return Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    body = runner_args.body or {}
    if not isinstance(body, dict):
        body = {}
    mode = str(body.get("mode") or os.getenv("BOT_MODE", "cascade")).lower()
    if mode == "realtime":
        try:
            from service_settings import assert_realtime_allowed, load_settings

            assert_realtime_allowed(load_settings())
        except RuntimeError:
            logger.warning("Realtime requested but disabled in admin settings; using cascade")
            mode = "cascade"
    try:
        stop_secs = float(os.getenv("VAD_STOP_SECS", "0.2"))
    except (TypeError, ValueError):
        stop_secs = 0.2
    turn_detector = os.getenv("VOICE_TURN_DETECTOR", "silence")
    max_silence_secs: float | None = None
    speculative = os.getenv("VOICE_SPECULATIVE", "0").strip().lower() in ("1", "true", "yes", "on")
    try:
        from service_settings import clamp_vad_stop_secs, load_settings

        voice_settings = load_settings()
        stop_secs = clamp_vad_stop_secs(voice_settings.get("vad_stop_secs", stop_secs))
        turn_detector = str(voice_settings.get("turn_detector") or turn_detector)
        max_silence_secs = voice_settings.get("max_silence_secs")
        speculative = bool(voice_settings.get("voice_speculative", speculative))
    except Exception:
        stop_secs = _clamp_stop_secs(stop_secs)
    stop_secs = _clamp_stop_secs(stop_secs)
    webrtc_session_id = str(getattr(runner_args, "session_id", None) or "")
    # Prefer client call_id so browser logs and server logs share one JSON file.
    call_id = str(
        body.get("call_id")
        or body.get("session_id")
        or webrtc_session_id
        or f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    # Stable conversation memory across calls in the same browser window.
    # call_id is per-call for logs; thread_id is the LangGraph checkpoint key.
    thread_id = str(
        body.get("thread_id")
        or body.get("conversation_id")
        or body.get("user_id")
        or call_id
    )

    logger.info(
        f"Starting ReceptionAi voice (mode={mode}, stop_secs={stop_secs}, "
        f"turn_detector={turn_detector}, speculative={speculative}, "
        f"call_id={call_id}, thread_id={thread_id}, webrtc_session={webrtc_session_id or 'n/a'})"
    )

    try:
        from conversation_log import start_call

        start_call(
            call_id,
            pipeline_mode=mode,
            session_id=call_id,
            channel="web_app",
            user_id=str(body.get("user_id") or "usr_anonymous"),
            audio_codec="opus",
            extra={"webrtc_session_id": webrtc_session_id} if webrtc_session_id else None,
        )
    except Exception as e:
        logger.warning(f"conversation_log start failed: {e}")

    context = LLMContext()

    if mode == "realtime":
        pipeline = build_realtime_pipeline(transport, context)
    elif mode == "cascade":
        shared_state: dict = {
            "last_user_text": "",
            "last_bot_text": "",
            "last_heard_transcript": "",
            "busy": False,
            "bot_speaking": False,
        }
        pipeline = build_cascade_pipeline(
            transport,
            context,
            thread_id,
            call_id=call_id,
            stop_secs=stop_secs,
            shared_state=shared_state,
            turn_detector=turn_detector,
            max_silence_secs=max_silence_secs,
            speculative=speculative,
        )
        try:
            from llm_message_dump import write_session_runtime_meta

            vad = _vad_params(stop_secs)
            write_session_runtime_meta(
                call_id,
                start_secs=vad.start_secs,
                stop_secs=vad.stop_secs,
                extra={
                    "confidence": vad.confidence,
                    "min_volume": vad.min_volume,
                    "pipeline_mode": mode,
                    "thread_id": thread_id,
                    "turn_detector": turn_detector,
                    "max_silence_secs": max_silence_secs,
                    "voice_speculative": speculative,
                },
            )
        except Exception as e:
            logger.debug(f"session runtime meta skipped: {e}")
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected 'cascade' or 'realtime'")

    metrics_observer = TurnMetricsObserver(session_id=call_id, pipeline_mode=mode)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[metrics_observer],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        try:
            from conversation_log import end_call

            end_call(call_id)
        except Exception:
            pass
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def _mount_custom_client() -> None:
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pipecat.runner.run import app

    from admin_routes import router as admin_router
    from log_routes import router as log_router
    from tool_routes import router as tool_router

    app.mount("/app", StaticFiles(directory=_ROOT / "client", html=True), name="custom-client")
    app.mount(
        "/app-lite",
        StaticFiles(directory=_ROOT / "client-lite", html=True),
        name="custom-client-lite",
    )
    app.mount(
        "/shared",
        StaticFiles(directory=_ROOT / "shared"),
        name="shared-assets",
    )
    app.mount(
        "/admin/static",
        StaticFiles(directory=_ROOT / "admin"),
        name="admin-static",
    )
    app.include_router(admin_router)
    app.include_router(log_router)
    app.include_router(tool_router)

    # Landing page: static Next.js export from web/ (`cd web && npm run build`).
    # Its "Request a call" button opens /app/?call=1. Falls back to the older
    # Python-rendered landing.py page when web/out has not been built.
    web_out = _ROOT / "web" / "out"
    if (web_out / "_next").is_dir():
        app.mount(
            "/_next",
            StaticFiles(directory=web_out / "_next"),
            name="landing-assets",
        )

    @app.get("/", include_in_schema=False)
    async def root_chooser():
        index = web_out / "index.html"
        if index.is_file():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        # Rendered per request so an industry change in Admin shows up immediately.
        from landing import render

        return HTMLResponse(render())

    @app.get("/api/industry", include_in_schema=False)
    async def industry_info():
        from industry_profile import all_profiles, describe

        return {"ok": True, "active": describe(), "available": all_profiles()}


if __name__ == "__main__":
    from pipecat.runner.run import main

    _mount_custom_client()
    main()
