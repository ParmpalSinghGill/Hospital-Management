"""Hospital AI — voice upgrade of the existing LangGraph agent.

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
from pipecat.audio.vad.vad_analyzer import VADParams
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
from pipecat.turns.user_start import VADUserTurnStartStrategy
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
from turn_metrics import TurnMetricsObserver
from user_mute_safe import SafeBotSpeakingMuteStrategy
from voice_bridge import (
    REALTIME_SYSTEM,
    get_shared_hospital_graph,
    hospital_function_schemas,
    make_langgraph_processor,
    register_hospital_tools_on_llm,
)

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env", override=True)

try:
    from service_settings import apply_settings_to_env

    apply_settings_to_env()
except Exception:
    pass


def _vad_params_for_mode(vad_mode: str) -> VADParams:
    """Return VAD params for client-selected sensitivity.

    sensitive — Pipecat defaults (faster barge-in, more echo false-triggers)
    stable    — harder to trigger (better against speaker echo cutting TTS)
    """
    mode = (vad_mode or "sensitive").strip().lower()
    if mode in ("stable", "less_sensitive", "low"):
        # Keep stop_secs near Pipecat's 0.2 default. Values >= STT p99 (~0.35s)
        # collapse the STT wait to 0s and break later turn detection / voice.
        return VADParams(confidence=0.85, start_secs=0.35, stop_secs=0.2, min_volume=0.65)
    # Default: stock Silero sensitivity
    return VADParams()


def build_cascade_pipeline(
    transport: BaseTransport,
    context: LLMContext,
    thread_id: str,
    *,
    call_id: str = "",
    vad_mode: str = "sensitive",
    shared_state: dict | None = None,
) -> Pipeline:
    """Deepgram STT → existing Main.py LangGraph (GLM) → Deepgram TTS."""
    # Boost month / phone / booking phrases so DOB-style dates and numbers transcribe cleaner.
    stt_keyterms = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "phone", "appointment", "doctor", "prescription", "medicine",
        "tomorrow", "today", "morning", "afternoon", "evening",
    ]
    stt = DeepgramSTTService(
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

    call_key = call_id or thread_id
    state = shared_state if shared_state is not None else {}
    graph_app = get_shared_hospital_graph(llm_provider=os.getenv("LLM_PROVIDER", "deepseek"))
    agent = make_langgraph_processor(
        thread_id, graph_app, call_id=call_key, shared_state=state
    )
    interrupt_trace = InterruptTraceProcessor(call_key, state)

    vad_params = _vad_params_for_mode(vad_mode)
    logger.info(f"Cascade VAD mode={vad_mode!r} params={vad_params}")

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
            # While TTS is playing, ignore mic activity so barge-in from echo
            # cannot cut the second half of a multi-sentence reply. Also unmute
            # on InterruptionFrame so a cut reply cannot leave the mic stuck off.
            user_mute_strategies=[SafeBotSpeakingMuteStrategy()],
            # Drop TranscriptionUserTurnStartStrategy: delayed STT of earlier
            # speech was starting a new turn mid-TTS and cutting audio + chat text.
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
            ),
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

    return Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            interrupt_trace,
            agent,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )


def build_realtime_pipeline(transport: BaseTransport, context: LLMContext) -> Pipeline:
    """OpenAI Realtime + hospital Tools.py handlers only."""
    tool_schemas = hospital_function_schemas()

    llm = OpenAIRealtimeLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAIRealtimeLLMSettings(
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
            system_instruction=REALTIME_SYSTEM,
            session_properties=SessionProperties(
                instructions=REALTIME_SYSTEM,
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

    context.set_tools(register_hospital_tools_on_llm(llm))
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
    vad_mode = str(body.get("vad_mode") or os.getenv("VAD_MODE", "sensitive")).lower()
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
        f"Starting Hospital voice (mode={mode}, vad_mode={vad_mode}, call_id={call_id}, "
        f"thread_id={thread_id}, webrtc_session={webrtc_session_id or 'n/a'})"
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
            vad_mode=vad_mode,
            shared_state=shared_state,
        )
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


_CHOOSER_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DBC Care — Appointments</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,650&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
<style>
  :root {
    --teal: #1a6b63; --teal-deep: #0f4a45; --ink: #14302c; --muted: #6a857f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; color: var(--ink);
    font-family: "Plus Jakarta Sans", system-ui, sans-serif;
    background:
      radial-gradient(1000px 520px at 8% -8%, #c8e8e1 0%, transparent 55%),
      radial-gradient(800px 420px at 100% 0%, #d5e8f2 0%, transparent 50%),
      linear-gradient(180deg, #eef6f3, #f7faf8 50%, #eef2f0);
  }
  main {
    min-height: 100vh; display: flex; flex-direction: column;
    justify-content: center; padding: 2.5rem 1.5rem; max-width: 920px; margin: 0 auto;
  }
  .mark {
    width: 3rem; height: 3rem; border-radius: 14px;
    background: linear-gradient(145deg, var(--teal), var(--teal-deep));
    position: relative; margin-bottom: 1.1rem;
  }
  .mark::after {
    content: ""; position: absolute; inset: 0; margin: auto; width: 1.1rem; height: 1.1rem;
    background:
      linear-gradient(#e8f6f3,#e8f6f3) center/0.25rem 100% no-repeat,
      linear-gradient(#e8f6f3,#e8f6f3) center/100% 0.25rem no-repeat;
  }
  .brand {
    font-family: "Fraunces", Georgia, serif; font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 650; letter-spacing: -0.03em; color: var(--teal-deep); margin: 0 0 0.4rem;
  }
  .lead { margin: 0 0 2rem; max-width: 28rem; color: var(--muted); font-size: 1.05rem; line-height: 1.5; }
  .ways { display: grid; gap: 0.9rem; }
  @media (min-width: 720px) { .ways { grid-template-columns: 1fr 1fr; } }
  @media (min-width: 980px) { .ways { grid-template-columns: 1fr 1fr 1fr 1fr; } }
  a.way {
    display: block; text-decoration: none; color: inherit;
    padding: 1.35rem 1.2rem; border-radius: 18px;
    background: rgba(255,255,255,0.85); border: 1px solid rgba(20,48,44,0.1);
    box-shadow: 0 18px 40px rgba(15,74,69,0.07);
    transition: transform 0.18s ease, border-color 0.18s ease;
  }
  a.way:hover { transform: translateY(-3px); border-color: rgba(26,107,99,0.35); }
  a.way .name { font-family: "Fraunces", Georgia, serif; font-size: 1.25rem; font-weight: 650; color: var(--teal-deep); margin-bottom: 0.3rem; }
  a.way .desc { font-size: 0.9rem; color: var(--muted); line-height: 1.4; }
</style>
</head>
<body>
  <main>
    <div class="mark" aria-hidden="true"></div>
    <h1 class="brand">DBC Care</h1>
    <p class="lead">One chat window for appointments — type or call. Pick providers in Admin.</p>
    <div class="ways">
      <a class="way" href="/app/">
        <div class="name">Open chat</div>
        <div class="desc">Single thread for messages and live calls.</div>
      </a>
      <a class="way" href="/admin/">
        <div class="name">Admin</div>
        <div class="desc">Choose Cascade / Realtime, LLMs, and debug timings.</div>
      </a>
      <a class="way" href="/app-lite/">
        <div class="name">Voice only</div>
        <div class="desc">Optional lean audio desk.</div>
      </a>
      <a class="way" href="/client/">
        <div class="name">Tech Console</div>
        <div class="desc">Pipecat prebuilt WebRTC debug client.</div>
      </a>
    </div>
  </main>
</body>
</html>
"""


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
        "/admin/static",
        StaticFiles(directory=_ROOT / "admin"),
        name="admin-static",
    )
    app.include_router(admin_router)
    app.include_router(log_router)
    app.include_router(tool_router)

    @app.get("/", include_in_schema=False)
    async def root_chooser():
        return HTMLResponse(_CHOOSER_HTML)


if __name__ == "__main__":
    from pipecat.runner.run import main

    _mount_custom_client()
    main()
