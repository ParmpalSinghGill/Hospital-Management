# Database seeding

SQLite store at `dataset/hospital.db` (doctors, patients, appointments, prescriptions). Seeding is **manual** — it does not run when bots start.

## What to run

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent

# Clear all tables, then re-seed doctors
python MakeDataBase.py --clear-all

# Random bookings for the next 3 days
python MakeDataBase.py --days 3 --per-doctor 4
```

## More options

```bash
# Clear appointments only
python MakeDataBase.py --clear-appointments

# Next N days (includes today; skips past slots today)
python MakeDataBase.py --days 5 --per-doctor 6

# Explicit dates
python MakeDataBase.py 2026-07-16 2026-07-17 --per-doctor 4

# Inclusive date range
python MakeDataBase.py 2026-07-16 2026-07-20 --range --per-doctor 3
```

Optional flags: `--start 09:00 --end 17:00 --slot-minutes 10 --seed 42`

Default slots are **10 minutes**, clinic hours **09:00–17:00**.

## How this relates to other modules

Every channel (web, CLI, Telegram, Discord, MCP) reads/writes this same DB. Seed once, then exercise any interface.

## Notes

- Demo data only — not a hardened production database
- Keep backups if you care about real bookings before `--clear-all`
