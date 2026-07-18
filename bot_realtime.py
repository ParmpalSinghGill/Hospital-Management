#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-quickstart - OpenAI Realtime (speech-to-speech) entry point.

Convenience wrapper around bot.py that forces ``BOT_MODE=realtime``. The actual
pipeline lives in bot.py (see ``build_realtime_pipeline``); this just lets you run
the Realtime bot without setting the env var yourself.

Required env vars:
- OPENAI_API_KEY

Optional env vars:
- OPENAI_REALTIME_MODEL  (default: gpt-realtime)
- OPENAI_REALTIME_VOICE  (default: marin)

Run the bot using::

    conda activate hosmanag
    python bot_realtime.py
"""

import os

os.environ["BOT_MODE"] = "realtime"

# Re-export bot() so the runner's _get_bot_module() finds the entry point here.
from bot import _mount_custom_client, bot  # noqa: E402

__all__ = ["bot"]


if __name__ == "__main__":
    from pipecat.runner.run import main

    _mount_custom_client()
    main()
