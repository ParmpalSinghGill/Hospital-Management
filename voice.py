#!/usr/bin/env python3
"""Alias for ``python bot.py`` (conda env: hosmanag)."""
from bot import _mount_custom_client

if __name__ == "__main__":
    from pipecat.runner.run import main

    _mount_custom_client()
    main()
