# CLI text desk

Terminal booking desk using the same LangGraph agents and `Tools.py` as the web / bots. No browser, no voice.

## What to run

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent
python Main.py
```

## Keys needed

Whatever `LLM_PROVIDER` / Admin CLI LLM uses — typically `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, or `OPENAI_API_KEY`.

## How to interact

1. Run `python Main.py`
2. Type requests at the prompt and press Enter
3. Follow the agent’s questions (phone, name, doctor, time)
4. Exit with Ctrl+C when done

Example:

```text
You: Book a dental appointment tomorrow at 11 AM
Bot: …asks for phone / name / confirms slot…
```

## Notes

- Same SQLite DB as web and bots.
- Useful for quick agent debugging without starting `bot.py`.
- Admin’s CLI LLM setting applies when Admin settings are present (`admin_settings.json`).
