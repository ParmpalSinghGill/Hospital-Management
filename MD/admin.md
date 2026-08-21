# Admin & providers

Password-protected console to choose which voice pipeline and LLM/STT/TTS providers power the system.

## What to run

Admin is served by the web process:

```bash
conda activate hosmanag
python bot.py
```

Then open **http://localhost:7860/admin/**

Default login: **Admin** / **12345** (override with `ADMIN_USER` / `ADMIN_PASS` in `.env`).

## How to interact

1. Sign in
2. Set **Default voice pipeline**: Cascade or Realtime  
   This controls the **/app/** chat window backend (whole combo — do not mix Cascade STT with Realtime LLM)
3. Set STT / TTS / Cascade LLM (and CLI LLM if you use `Main.py`)
4. Optional: toggle **Debugging mode** for latency timings
5. **Save**, then reload `/app/`

Settings persist in `admin_settings.json` (gitignored).

## Credit / status

The Admin home can show API credit / connectivity probes for configured providers (needs the relevant keys; Deepgram balance may need `DEEPGRAM_MANAGE_API_KEY` with `billing:read`).

## Related pages (same server)

| Path | Purpose |
|------|---------|
| `/admin/appointments` | Browse appointments |
| `/admin/doctors` | Doctors |
| `/admin/patients` | Patients |
| `/admin/messages` | Message / session views |

## Notes

- Change the default password before any shared deployment
- Telegram / Discord / MCP do not require Admin to start, but they share the same DB and (for LangGraph text) LLM provider settings where applicable
