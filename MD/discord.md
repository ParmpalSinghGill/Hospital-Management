# Discord bot

Book, cancel, and reschedule over Discord. Same LangGraph + `Tools.py` as Telegram and the web app.

## What to run

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent
pip install -r requirements.txt   # includes discord.py
python discord_bot.py
```

Keep this process running. You should see a log like: `Discord bot online as …`

## Setup

1. Create an application + bot at the [Discord Developer Portal](https://discord.com/developers/applications)
2. Copy the bot token into `.env`:

   ```bash
   DISCORD_BOT_TOKEN=...
   ```

3. Invite the bot (OAuth2 → URL Generator):
   - Scope: `bot`
   - Permissions: **Send Messages**, **Read Message History**, **View Channels**
4. Run `python discord_bot.py`

### Message Content Intent (optional)

DMs and `@mentions` work **without** the privileged Message Content Intent.

Only enable Message Content Intent in the portal (and set `DISCORD_MESSAGE_CONTENT_INTENT=1` in `.env`) if you need the bot to read every guild message without a mention.

## Keys needed

| Variable | Required |
|----------|----------|
| `DISCORD_BOT_TOKEN` | Yes |
| LLM keys | Yes — same as other text channels |

## How to interact

1. **Preferred:** open a **DM** with the bot
2. Or in a server channel: **@mention** the bot, then your request
3. Commands (slash or bang):

   | Command | Effect |
   |---------|--------|
   | `/start` or `!start` | Welcome |
   | `/help` or `!help` | How it works |
   | `/reset` or `!reset` | Fresh conversation |

4. Example message:

   ```text
   book a dental appointment tomorrow at 11 AM
   ```

In servers, messages without an @mention are ignored (privacy / noise control).

## Notes

- Chats log under `chats/` with channel `discord`
- Prefer DMs for phone numbers and health details
- Independent of `bot.py` / Telegram / MCP — run only what you need
