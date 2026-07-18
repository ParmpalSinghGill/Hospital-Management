# DBC Care — Hospital AI Assistant

Natural-language hospital appointment assistant for **text chat** and **voice calls**.
Book, cancel, and reschedule visits using a LangGraph multi-agent workflow, with Cascade
or OpenAI Realtime voice pipelines and a password-protected Admin console.

---

## Features

- **Text CLI** — interactive booking desk (`Main.py`)
- **Voice Cascade** — Deepgram STT → LangGraph (DeepSeek by default) → Deepgram TTS
- **Voice Realtime** — OpenAI Realtime speech-to-speech + appointment tools
- **Web UIs** — Chat & Call (`/app/`), Voice Desk (`/app-lite/`), home chooser (`/`)
- **Admin** — API credit status + choose which provider powers each service
- **SQLite data store** — doctors, patients, appointments, prescriptions in `dataset/hospital.db`
- **Seed utility** — generate sample bookings (`MakeDataBase.py`)

---

## Architecture

```
User (text / voice)
        │
        ├─ CLI ──────────────► Main.py LangGraph (router → booking/cancel/reschedule)
        │                              │
        │                              ▼
        │                         Tools.py ──► dataset/hospital.db (SQLite)
        │
        └─ Voice (bot.py)
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

### Data

| Store | Contents |
|-------|----------|
| `dataset/hospital.db` | SQLite DB: `doctors`, `patients`, `appointments`, `prescriptions` |
| `dataset/*.json` (optional) | Legacy files; imported once if the DB is empty |

Tables mirror the old JSON shape (`doctor_id`, `patient_id`, `APT-XXXX`, `RX-XXXX`, status, time as `YYYY-MM-DD HH:MM`).

---

## Requirements

- Python **3.11+** (project tested on 3.12)
- Conda env **`hosmanag`** (recommended)
- API keys (see [Environment](#environment-variables))

---

## Setup

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

Seed doctors (if missing) and optional sample bookings:

```bash
# Ensure SQLite exists / reseed doctors + clear tables
python MakeDataBase.py --clear-all

# Generate random bookings for the next 3 days
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

Admin settings also persist to `admin_settings.json` (gitignored).

---

## Run

One server, one chat window. Providers are chosen in **Admin**, not in the chat UI.

```bash
conda activate hosmanag
python bot.py
```

Open **http://localhost:7860/app/**

| Path | Purpose |
|------|---------|
| `/app/` | **Main UI** — chat + call in one thread |
| `/admin/` | Select Cascade vs Realtime, LLM/STT/TTS, debug mode |
| `/` | Home links |
| `/app-lite/` | Optional lean voice-only page |

Flow:

1. Start `python bot.py`
2. In `/admin/` (Admin / 12345) set **Default voice pipeline** and LLM/STT/TTS → Save
3. Open `/app/` — chat or call; the header **Backend** chip shows the Admin selection

Optional developer CLI (same tools, no web UI):

```bash
python Main.py
```

### Admin

1. Go to http://localhost:7860/admin/
2. Sign in: **Admin** / **12345** (change via `ADMIN_USER` / `ADMIN_PASS`)
3. Set **Default voice pipeline** (Cascade or Realtime) — this is the chat window backend
4. Set STT / TTS / Cascade LLM (and CLI LLM if you use `Main.py`)
5. Toggle **Debugging mode** for send → first text / first audio timings
6. Save, then reload `/app/`

Optional Realtime-only process shortcut: `python bot_realtime.py` (still use `/app/` UI).

---

## Voice pipelines

| Mode | STT | LLM | TTS | Flow |
|------|-----|-----|-----|------|
| **Cascade** (default) | Deepgram | DeepSeek `deepseek-chat`* | Deepgram Aura | speech → text → agent → speech |
| **Realtime** | OpenAI Realtime | `gpt-realtime`* | OpenAI voice | speech ↔ speech |

\*Configurable in Admin. Admin’s **Chat window backend** selects the **whole** combo — do not mix Cascade STT with Realtime LLM.

Cascade runs the **same** `Main.py` LangGraph graph (router + specialists).  
Realtime registers the same `Tools.py` functions on the OpenAI session.

---

## MakeDataBase.py

Utility to seed / clear the SQLite database (`dataset/hospital.db`).

```bash
# Clear appointments only
python MakeDataBase.py --clear-appointments

# Clear all tables, then re-seed doctors
python MakeDataBase.py --clear-all

# Seed next N days (includes today; skips past slots today)
python MakeDataBase.py --days 5 --per-doctor 6

# Explicit dates
python MakeDataBase.py 2026-07-16 2026-07-17 --per-doctor 4

# Inclusive range
python MakeDataBase.py 2026-07-16 2026-07-20 --range --per-doctor 3

# Options
#   --start 09:00 --end 17:00 --slot-minutes 10 --seed 42
```

Slots are **10 minutes** by default (09:00–17:00). Patients and prescriptions live in SQLite and are referenced by `patient_id`.  
Running the seeder does **not** happen automatically when the bot starts. On first run with an empty DB, existing `dataset/*.json` files are migrated automatically.

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
├── dataset/                # hospital.db (+ optional legacy *.json for migration)
├── turn_metrics.py         # Latency metrics (CSV) + feeds call JSON
├── conversation_log.py     # Per-call JSON conversation + timings
├── log_routes.py           # Client timing ingest API
├── session_turn.py         # Turn helpers for metrics
├── chats/                  # call-*.json logs (gitignored)
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Call / chat logs

Each web or CLI session writes a JSON file under ``chats/``::

    chats/sess-YYYYMMDD-HHMMSS_<session_id>.json

Schema (one file per session)::

    {
      "session_id": "sess_...",
      "user_id": "usr_...",
      "session_start_time": "...Z",
      "session_end_time": "...Z",
      "device_info": { "channel": "web_app", "audio_codec": "opus" },
      "interactions": [
        {
          "turn_number": 1,
          "mode": "text",                 // or "voice_and_text"
          "payload": {
            "user_input_text": "...",
            "bot_response_text": "..."
          },
          "complexity_metrics": {
            "input_tokens": 8,
            "output_tokens": 14
          },
          "timestamps": {
            "user_sent_time": "...Z",
            "bot_received_time": "...Z",
            "first_token_visible_to_user": "...Z",
            "full_response_visible_to_user": "...Z",
            "first_audio_heard_by_user": "...Z",   // voice only
            "audio_playback_ended": "...Z"         // voice only
          }
        }
      ]
    }

List recent sessions: ``GET /api/call-logs/recent``

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

---

## License / credit

Voice UI patterns adapted from Pipecat-style WebRTC clients.  
Hospital agent logic and appointment tools are project-specific.
