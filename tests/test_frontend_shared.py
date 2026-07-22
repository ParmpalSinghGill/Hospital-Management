"""Frontend shared-token / format-helper sanity checks."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_tokens_exist_and_are_imported():
    tokens = (ROOT / "shared" / "tokens.css").read_text()
    assert "--teal:" in tokens
    assert "--font-display:" in tokens

    for rel in ("client/style.css", "admin/style.css", "client-lite/style.css"):
        css = (ROOT / rel).read_text()
        assert '@import url("/shared/tokens.css")' in css
        # Local :root fallbacks keep UI visible if /shared is not mounted.
        assert "--teal:" in css


def test_client_format_module_exports():
    text = (ROOT / "client" / "lib" / "format.js").read_text()
    for name in (
        "formatLatency",
        "formatExactTime",
        "averageLatency",
        "normalizeBotBubbleText",
        "isDuplicateBotText",
    ):
        assert f"export function {name}" in text

    app = (ROOT / "client" / "app.js").read_text()
    assert 'from "./lib/format.js"' in app


def test_bot_mounts_shared_static():
    bot = (ROOT / "bot.py").read_text()
    assert '"/shared"' in bot
    assert "shared-assets" in bot
