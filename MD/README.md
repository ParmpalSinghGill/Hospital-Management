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

## HIPAA track

Compliance work is documented step by step under [hipaa/](hipaa/):

| Step | Doc | Status |
|------|-----|--------|
| 1 | [PHI data-flow map](hipaa/01-phi-data-flow-map.md) | done — 2026-08-21 |
| 2 | [Security Risk Assessment](hipaa/02-security-risk-assessment.md) — risk register, safeguard gap analysis, vendor BAA table, remediation plan | done — 2026-08-21 |
| 3 | [Technical remediation report](hipaa/03-technical-remediation-report.md) — what was changed in code, verification, what remains | done — 2026-08-21 |
| 5 | [Security Risk Assessment R2](hipaa/05-security-risk-assessment-r2.md) — re-score of the hardened code + Langfuse tracing | done — 2026-08-24 |
| 4 | Vendor BAAs, hosting, policies & training (administrative track) | next |
