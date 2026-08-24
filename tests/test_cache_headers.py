"""UI files must always revalidate; APIs must never be cached (no stale Admin choices)."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/admin/api/ui-config", "no-store"),
        ("/admin/api/settings", "no-store"),
        ("/api/tools/x", "no-store"),
        ("/", "no-cache"),
        ("/app/", "no-cache"),
        ("/app/app.js", "no-cache"),
        ("/app-lite/index.html", "no-cache"),
        ("/admin/", "no-cache"),
        ("/shared/x.js", "no-cache"),
        ("/start", None),
        ("/ws", None),
    ],
)
def test_cache_control_policy(path, expected):
    bot = pytest.importorskip("bot")
    assert bot._cache_control_for(path) == expected
