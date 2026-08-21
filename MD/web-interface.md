# Web interface (chat + voice)

Browser UI for text chat and voice calls. One process serves the patient app, Admin, and tool-discovery HTTP routes.

## What to run

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent
python bot.py
```

Optional Realtime-only shortcut (same UI):

```bash
python bot_realtime.py
# or
BOT_MODE=realtime python bot.py
```

Default URL: **http://localhost:7860**

## Pages

| Path | Purpose |
|------|---------|
| http://localhost:7860/app/ | Main UI — type or call in one thread |
| http://localhost:7860/admin/ | Providers, voice pipeline, debug |
| http://localhost:7860/ | Home links |
| http://localhost:7860/app-lite/ | Lean voice-only desk |
| http://localhost:7860/toollist/ | Human-readable tool catalog |
| http://localhost:7860/api/tools | JSON tool metadata (discovery only) |

## Keys needed

| Pipeline | Required in `.env` |
|----------|-------------------|
| Cascade (default) | `DEEPGRAM_API_KEY`, `DEEPSEEK_API_KEY` (or other Cascade LLM) |
| Realtime | `OPENAI_API_KEY` |

Pick Cascade vs Realtime in **Admin** after the server starts (see [Admin](admin.md)).

## How to interact

1. Start `python bot.py`
2. Open **Admin** → sign in (`Admin` / `12345` by default) → set **Default voice pipeline** and Save
3. Open **/app/**
4. **Text:** type in the chat box (e.g. “Book me with Cardiology tomorrow at 10am”)
5. **Voice:** connect / call from the same window; speak naturally

The header **Backend** chip shows which pipeline Admin selected.

## Example prompts

- “Book me with Cardiology tomorrow at 10am. My name is Priya.”
- “Which doctors are in Orthopedics?”
- “Cancel appointment APT-0012”
- “Reschedule APT-0012 to Friday at 3pm”

## Notes

- Chats are logged under `chats/sessions/` with channel `web_app` (voice may use `voice` / desk channels).
- Tool audits append to `chats/tool_call.json`.
- This process does **not** start Telegram, Discord, or MCP — run those separately if needed.
