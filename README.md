# DBC Care — Hospital AI Assistant

Natural-language hospital appointment assistant for **text chat** and **voice calls**.
Book, cancel, and reschedule visits using a LangGraph multi-agent workflow, with Cascade
or OpenAI Realtime voice pipelines and a password-protected Admin console.

---

## How to run each module

Full step-by-step guides (what to run + how to interact) live in **[`MD/`](MD/README.md)**:

| I want… | Guide | Run |
|---------|-------|-----|
| Browser chat + voice | [Web interface](MD/web-interface.md) | `python bot.py` → http://localhost:7860/app/ |
| Terminal booking only | [CLI](MD/cli.md) | `python Main.py` |
| Cursor / Claude Desktop tools | [MCP local](MD/mcp-local.md) | `python mcp_server.py` |
| ChatGPT over the internet | [MCP + Cloudflare](MD/mcp-cloudflare.md) | `mcp_server.py --transport streamable-http` + `cloudflared` |
| Telegram patients | [Telegram](MD/telegram.md) | `python telegram_bot.py` |
| Discord patients | [Discord](MD/discord.md) | `python discord_bot.py` |
| Change Cascade / Realtime / LLM | [Admin](MD/admin.md) | open `/admin/` after `bot.py` |
| Seed / clear demo data | [Database](MD/database.md) | `python MakeDataBase.py …` |

You can run several channels at once (separate terminals). They all share `Tools.py` and `dataset/hospital.db`.

---

## Features

- **Text CLI** — interactive booking desk (`Main.py`)
- **Telegram bot** — book / cancel / reschedule over Telegram (`telegram_bot.py`)
- **Discord bot** — book / cancel / reschedule over Discord DMs or mentions (`discord_bot.py`)
- **MCP server** — same tools for ChatGPT / Cursor (`mcp_server.py`)
- **Voice Cascade** — Deepgram STT → LangGraph (DeepSeek by default) → Deepgram TTS
- **Voice Realtime** — OpenAI Realtime speech-to-speech + appointment tools
- **Web UIs** — Chat & Call (`/app/`), Voice Desk (`/app-lite/`), home chooser (`/`)
- **Admin** — API credit status + choose which provider powers each service
- **SQLite data store** — doctors, patients, appointments, prescriptions in `dataset/hospital.db`
- **Seed utility** — generate sample bookings (`MakeDataBase.py`)

---

## Architecture

```
User (text / voice / Telegram / Discord / MCP)
        │
        ├─ CLI ──────────────► Main.py LangGraph (router → booking/cancel/reschedule)
        │                              │
        │                              ▼
        │                         Tools.py ──► dataset/hospital.db (SQLite)
        │
        ├─ Telegram (telegram_bot.py) ─► same LangGraph + Tools.py
        ├─ Discord (discord_bot.py) ───► same LangGraph + Tools.py
        ├─ MCP (mcp_server.py) ────────► Tools.py (stdio or HTTP)
        │
        └─ Voice / Web (bot.py)
               ├─ Cascade: Deepgram → same LangGraph → Deepgram
               └─ Realtime: OpenAI Realtime + Tools.py function calls
```

### Appointment tools

| Tool | Purpose |
|------|---------|
| `lookup_patient` | Find patient by phone/name; returns past doctors for returning patients |
| `list_doctors` | Filter doctors by department or name |
| `book_appointment` | Book after validating doctor + free time (creates/finds patient by phone) |
| `cancel_appointment` | Cancel by ID (e.g. `APT-0001`) |
| `reschedule_appointment` | Move to a new time if free |
| `get_prescriptions` | Look up medicines (id, or name+phone together) |

After `python bot.py`, tool **metadata** (not execution) is at http://localhost:7860/api/tools and `/toollist/`.  
To **call** tools from ChatGPT/Cursor, use the [MCP guides](MD/README.md).

### Data

| Store | Contents |
|-------|----------|
| `dataset/hospital.db` | SQLite: `doctors`, `patients`, `appointments`, `prescriptions` |

---

## Requirements

- Python **3.11+** (tested on 3.12)
- Conda env **`hosmanag`** (recommended)
- API keys (see below)

---

## Setup

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

Seed doctors / sample bookings — see [Database](MD/database.md):

```bash
python MakeDataBase.py --clear-all
python MakeDataBase.py --days 3 --per-doctor 4
```

---

## Environment variables

Copy `.env.example` → `.env`. Never commit `.env`.

| Variable | Used by | Notes |
|----------|---------|--------|
| `GROQ_API_KEY` | CLI LLM (default) | Text agent |
| `DEEPGRAM_API_KEY` | Cascade STT + TTS | Required for Cascade voice |
| `DEEPGRAM_MANAGE_API_KEY` | Admin balance | Optional Owner/Admin key with `billing:read` |
| `DEEPGRAM_VOICE` | Cascade TTS | Default `aura-2-thalia-en` |
| `GLM_API_KEY` | Optional Cascade LLM | Zhipu / BigModel |
| `GLM_MODEL` | GLM chat model | Default `glm-4-flash` |
| `GLM_BASE_URL` | GLM API | Default China endpoint; use Z.ai URL if needed |
| `DEEPSEEK_API_KEY` | Cascade LLM (default) | Required for default Cascade |
| `DEEPSEEK_MODEL` | DeepSeek chat model | Default `deepseek-chat` |
| `DEEPSEEK_BASE_URL` | DeepSeek API | Default `https://api.deepseek.com` |
| `OPENAI_API_KEY` | Realtime (+ optional chat LLM) | Required for Realtime |
| `OPENAI_REALTIME_MODEL` | Realtime | Default `gpt-realtime` |
| `OPENAI_REALTIME_VOICE` | Realtime | Default `marin` |
| `OPENAI_CHAT_MODEL` | Optional OpenAI chat LLM | Default `gpt-4o-mini` |
| `LLM_PROVIDER` | LangGraph agents | `glm` / `groq` / `openai` / `deepseek` (overridden by Admin) |
| `BOT_MODE` | Voice default | `cascade` or `realtime` |
| `ADMIN_USER` / `ADMIN_PASS` | Admin login | Defaults `Admin` / `12345` |
| `DAILY_API_KEY` | Optional | Daily transport for production |
| `TELEGRAM_BOT_TOKEN` | [Telegram](MD/telegram.md) | From [@BotFather](https://t.me/BotFather) |
| `DISCORD_BOT_TOKEN` | [Discord](MD/discord.md) | From [Developer Portal](https://discord.com/developers/applications) |
| `MCP_TRANSPORT` / `MCP_HOST` / `MCP_PORT` | [MCP](MD/mcp-local.md) | Defaults for remote MCP (`streamable-http`, `127.0.0.1`, `8000`) |

Admin settings also persist to `admin_settings.json` (gitignored).

---

## Voice pipelines

| Mode | STT | LLM | TTS | Flow |
|------|-----|-----|-----|------|
| **Cascade** (default) | Deepgram | DeepSeek `deepseek-chat`* | Deepgram Aura | speech → text → agent → speech |
| **Realtime** | OpenAI Realtime | `gpt-realtime`* | OpenAI voice | speech ↔ speech |

\*Configurable in [Admin](MD/admin.md). Cascade runs the same LangGraph graph as CLI; Realtime registers the same `Tools.py` functions on the OpenAI session.

---

## Project layout

```
Hospital_Ai_Assistent/
├── Main.py                 # LangGraph hospital agents (CLI)
├── Tools.py                # Appointment tools
├── database.py             # SQLite data layer (dataset/hospital.db)
├── Model.py                # Groq / GLM / OpenAI LLM factory
├── agent_turn.py           # Shared invoke helper
├── MakeDataBase.py         # Seed / clear SQLite dataset
│
├── bot.py                  # Voice entry (Cascade + Realtime)
├── bot_realtime.py         # Realtime shortcut
├── telegram_bot.py         # Telegram text booking bot
├── discord_bot.py          # Discord text booking bot
├── mcp_server.py           # MCP server (ChatGPT/Cursor) — wraps Tools.py
├── voice.py                # Alias entry
├── voice_bridge.py         # Bridge Main/Tools ↔ Pipecat
│
├── service_settings.py     # Admin provider settings
├── admin_credits.py        # API credit / status probes
├── admin_routes.py         # Admin API + auth
├── admin/                  # Admin UI
├── client/                 # Chat & Call UI (/app/)
├── client-lite/            # Voice Desk UI (/app-lite/)
│
├── MD/                     # Per-module run & interact guides
│   └── README.md
├── dataset/                # hospital.db (SQLite)
├── chats/
│   ├── sessions/           # one sess-*.json per call (gitignored)
│   └── tool_call.json      # all tool-call audits
├── requirements.txt
├── .env.example
└── README.md
```

---

## Call / chat logs

Each web or CLI session writes JSON under `chats/sessions/`.  
All tool invocations (any channel) go to `chats/tool_call.json`.  
List recent sessions: `GET /api/call-logs/recent` (when `bot.py` is running).

---

## Example prompts

- “Book me with Cardiology tomorrow at 10am. My name is Priya.”
- “Which doctors are in Orthopedics?”
- “Cancel appointment APT-0012”
- “Reschedule APT-0012 to Friday at 3pm”

---

## Tech stack

| Layer | Stack |
|-------|--------|
| Agent | LangGraph + LangChain |
| Voice | Pipecat (WebRTC) |
| Cascade | Deepgram + DeepSeek |
| Realtime | OpenAI Realtime |
| CLI LLM | Groq (default) |
| Web | Static HTML/CSS/JS + FastAPI routes (Pipecat runner) |
| Storage | SQLite (`dataset/hospital.db`) |

---

## Security notes

- Keep `.env` and `admin_settings.json` out of git.
- Default Admin password (`12345`) is for **local demos only** — change it before any shared deployment.
- Appointment data is local SQLite for demos — not a hardened production DB.
- Public MCP tunnels can write to your DB — protect before production.

---

## License / credit

Voice UI patterns adapted from Pipecat-style WebRTC clients.  
Hospital agent logic and appointment tools are project-specific.
