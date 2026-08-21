# Module guides

How to run and use each part of DBC Care. Start from the [main README](../README.md) for setup, then open the guide for the channel you want.

| Guide | What it covers | Command |
|-------|----------------|---------|
| [Web interface](web-interface.md) | Chat + voice in the browser, Admin | `python bot.py` |
| [CLI text desk](cli.md) | Terminal booking (no browser) | `python Main.py` |
| [MCP local](mcp-local.md) | MCP for Cursor / Claude Desktop | `python mcp_server.py` |
| [MCP + Cloudflare](mcp-cloudflare.md) | Remote MCP for ChatGPT | `mcp_server.py` + `cloudflared` |
| [Telegram bot](telegram.md) | Book via Telegram chat | `python telegram_bot.py` |
| [Discord bot](discord.md) | Book via Discord DM / @mention | `python discord_bot.py` |
| [Admin & providers](admin.md) | Choose Cascade / Realtime / LLM keys | via `/admin/` |
| [Database seeding](database.md) | Seed / clear `hospital.db` | `python MakeDataBase.py` |

## Quick chooser

```
Want a browser chat/call?     →  Web interface
Want Cursor/Claude tools?     →  MCP local (stdio)
Want ChatGPT to book?         →  MCP + Cloudflare
Want Telegram patients?       →  Telegram bot
Want Discord patients?        →  Discord bot
Want a quick terminal test?   →  CLI
```

All channels share the same appointment tools and SQLite DB (`dataset/hospital.db`). You can run several at once (e.g. web on 7860 + MCP on 8000 + Discord) as separate processes.
