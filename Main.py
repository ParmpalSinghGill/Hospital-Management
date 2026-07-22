"""Hospital AI assistant entrypoint (CLI) and public graph exports."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

from agent_turn import run_turn
from hospital_agents import (  # noqa: F401
    HospitalName,
    build_booking_agent,
    build_cancellation_agent,
    build_general_agent,
    build_prescription_agent,
    build_reschedule_agent,
    build_router_agent,
)
from hospital_graph import GraphState, build_graph  # noqa: F401
from hospital_routing import (  # noqa: F401
    _is_greeting_only,
    _is_short_affirmation,
    _sticky_route_from_history,
    _wants_prescriptions,
)

logfile = "app.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    # Truncate only when running the CLI — importing Main from bot.py must not
    # wipe voice logs mid-session.
    open(logfile, "w").close()
    try:
        from service_settings import apply_settings_to_env

        apply_settings_to_env()
    except Exception:
        pass

    from conversation_log import _utc_iso, append_or_update_turn, end_call, start_call

    # Prepare per-session chat log file (legacy jsonl + structured call JSON)
    base_dir = os.path.dirname(__file__)
    chats_dir = os.path.join(base_dir, "chats")
    os.makedirs(chats_dir, exist_ok=True)
    session_name = datetime.now().strftime("chat-%Y%m%d-%H%M%S.jsonl")
    import hospital_graph

    hospital_graph.CHAT_LOG_PATH = os.path.join(chats_dir, session_name)
    try:
        with open(hospital_graph.CHAT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"session_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    except Exception as e:
        logging.error(f"Failed to initialize chat log file: {e}")

    app = build_graph()

    # Create a unique thread ID for this session
    thread_id = f"hospital_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    call_id = thread_id
    start_call(
        call_id,
        pipeline_mode="cli",
        session_id=thread_id,
        channel="cli",
        user_id="usr_cli",
        audio_codec="none",
    )

    def _logged_turn(user_input: str) -> str:
        user_sent = _utc_iso()
        append_or_update_turn(
            call_id,
            {
                "mode": "text",
                "input_type": "text",
                "user_text": user_input,
                "user_sent_at": user_sent,
                "bot_received_at": user_sent,
            },
            new_turn=True,
        )
        text_start = _utc_iso()
        turn = run_turn(app, user_input, thread_id, call_id=call_id)
        reply = turn.text if hasattr(turn, "text") else str(turn)
        agent_name = getattr(turn, "agent", "") or ""
        text_end = _utc_iso()
        append_or_update_turn(
            call_id,
            {
                "bot_text": reply,
                "agent_name": agent_name,
                "bot_text_first_token_at": text_start,
                "bot_text_first_shown_at": text_start,
                "bot_text_complete_at": text_end,
            },
            new_turn=False,
        )
        return reply

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        print(_logged_turn(user_input))
        end_call(call_id)
        return

    print("Hospital Agent ready. Type your request (Ctrl+C to exit).\n")
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue
            print(_logged_turn(user_input))

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        end_call(call_id)



if __name__ == "__main__":
    main()
