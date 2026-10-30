# Telegram bot

Book, cancel, and reschedule appointments over Telegram text chat. Uses the same LangGraph + `Tools.py` flow as the web / CLI.

## What to run

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent
python telegram_bot.py
```

Keep this process running while you use Telegram.

## Setup

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token
2. Add to `.env`:

   ```bash
   TELEGRAM_BOT_TOKEN=123456:ABC...
   ```

3. Run `python telegram_bot.py`
4. You should see: `Telegram bot ready: @YourBotName`

## Keys needed

| Variable | Required |
|----------|----------|
| `TELEGRAM_BOT_TOKEN` | Yes |
| LLM keys (`DEEPSEEK_API_KEY` etc.) | Yes — same agent LLM as other text channels |

## How to interact

1. Open Telegram → search your bot → **Start**
2. Commands:

   | Command | Effect |
   |---------|--------|
   | `/start` | Welcome |
   | `/help` | How it works |
   | `/reset` | Fresh conversation |

3. Chat in plain language, e.g.:

   - “Book a dental appointment tomorrow at 11 AM”
   - “Cancel APT-0012”
   - “Reschedule APT-0012 to Friday at 3pm”

The bot will ask for phone / name when it needs to look up or create a patient record.

## Notes

- Chats log under `chats/` with channel `telegram`
- Does not start the web UI — run `python bot.py` separately if you want Admin / browser
- Prefer private chats for patient details
