"""Hospital appointment MCP server — reuses Tools.py (no duplicate tools).

Exposes every LangChain tool from ``tool_catalog.ALL_TOOLS`` over MCP so
ChatGPT / Cursor / other MCP clients can discover and call them.

Transports:
  stdio            — local MCP clients (Cursor, Claude Desktop)
  streamable-http  — remote HTTPS (Cloudflare Tunnel → ChatGPT Developer mode)
  sse              — legacy remote transport (some ChatGPT setups)

Run::

    conda activate hosmanag
    python mcp_server.py                          # stdio
    python mcp_server.py --transport streamable-http --port 8000
    python mcp_server.py --transport sse --port 8000

Remote access: HTTP/SSE transports **require** ``MCP_API_KEY`` (clients send
``Authorization: Bearer <key>``). Publish only through a host/tunnel that is
covered by a BAA (see MD/mcp-cloudflare.md); never a quick trycloudflare URL
with real patient data.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env", override=True)

import security_config as sc  # noqa: E402
from audit_log import audit  # noqa: E402
from tool_catalog import ALL_TOOLS, build_tool_catalog  # noqa: E402


INSTRUCTIONS = (
    "You are connected to the DBC Care hospital appointment tools. "
    "Use these tools to look up patients, list doctors, book / cancel / "
    "reschedule appointments, and fetch prescriptions. "
    "Always verify identity with phone before booking. "
    "Never invent appointment IDs, patient IDs, or doctor names — "
    "only use values returned by tools."
)


def _wrap_langchain_tool(lc_tool: Any) -> Callable[..., str]:
    """Bind MCP handler to the existing LangChain tool (same implementation)."""
    import time

    def _handler(**kwargs: Any) -> str:
        # Drop unset optionals so defaults in Tools.py still apply.
        args = {k: v for k, v in kwargs.items() if v is not None}
        t0 = time.perf_counter()
        error = ""
        result: Any = None
        try:
            result = lc_tool.invoke(args)
        except Exception as exc:  # noqa: BLE001 — surface to MCP client
            error = str(exc)
            result = {"ok": False, "message": error}
        duration_ms = (time.perf_counter() - t0) * 1000.0
        try:
            from conversation_log import record_tool_call

            record_tool_call(
                tool=str(lc_tool.name),
                arguments=args,
                result=result if not error else None,
                source="mcp",
                call_id="mcp",
                duration_ms=duration_ms,
                ok=False if error else None,
                error=error,
            )
        except Exception:  # noqa: BLE001 — never break MCP on logging
            pass
        if error:
            return json.dumps({"ok": False, "message": error})
        if isinstance(result, str):
            return result
        return json.dumps(result)

    _handler.__name__ = str(lc_tool.name)
    _handler.__doc__ = (lc_tool.description or "").strip() or None
    # Preserve arg schema from the original @tool function for MCP discovery.
    original = getattr(lc_tool, "func", None) or lc_tool
    _handler.__signature__ = inspect.signature(original)
    annotations = dict(getattr(original, "__annotations__", {}) or {})
    annotations["return"] = str
    _handler.__annotations__ = annotations
    return _handler


def _transport_security(*, allow_tunnel: bool) -> TransportSecuritySettings:
    """Allow Cloudflare / public Host headers when tunneling.

    FastMCP rejects unknown Host headers by default (DNS-rebinding guard),
    which breaks trycloudflare.com until hosts are allow-listed.
    """
    if allow_tunnel or os.getenv("MCP_ALLOW_TUNNEL", "").lower() in ("1", "true", "yes"):
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    extra = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", *extra],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "https://chatgpt.com", *extra],
    )


def _browser_home_html(*, public_url: str | None = None, oauth: bool = False) -> str:
    rows = []
    for tool in ALL_TOOLS:
        name = tool.name
        desc = (tool.description or "").strip().split("\n", 1)[0]
        rows.append(f"<tr><td><code>{name}</code></td><td>{desc}</td></tr>")
    base = (public_url or "").rstrip("/") or "https://YOUR-HOST.example.com"
    auth_note = (
        "<p><strong>Demo OAuth is ON (dev only)</strong>.</p>"
        if oauth
        else "<p><strong>Bearer auth</strong> — clients must send "
        "<code>Authorization: Bearer &lt;MCP_API_KEY&gt;</code>. "
        "Publish only through a BAA-covered host; never expose real patient data via a "
        "quick tunnel.</p>"
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Hospital MCP</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.45;max-width:52rem}}
 code{{background:#f3f3f3;padding:.1rem .35rem;border-radius:4px}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}}
 th,td{{border:1px solid #ccc;padding:.5rem .75rem;vertical-align:top;text-align:left}}
 th{{background:#f4f4f4}}
 .box{{background:#f8f8f8;padding:1rem;border-radius:8px;margin:1rem 0}}
</style></head><body>
<h1>Hospital AI — MCP server</h1>
<p>This page is for humans. ChatGPT / MCP clients use the protocol endpoint below.</p>
{auth_note}
<div class="box">
  <strong>ChatGPT Developer mode URL</strong><br/>
  Paste: <code>{base}/mcp</code><br/>
  JSON tool list: <a href="/toollist"><code>/toollist</code></a>
</div>
<table><thead><tr><th>Tool</th><th>Description</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def build_mcp(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    stateless_http: bool = True,
    allow_tunnel: bool = True,
    public_url: str | None = None,
    enable_oauth: bool = False,
) -> FastMCP:
    """Create FastMCP app with all hospital tools registered once."""
    auth_kwargs: dict[str, Any] = {}
    if enable_oauth:
        if sc.hipaa_mode():
            raise SystemExit(
                "The demo OAuth provider auto-approves every client and is refused while "
                "HIPAA_MODE is on. Use MCP_API_KEY bearer auth (default) or integrate a real IdP."
            )
        from mcp_demo_oauth import DemoOAuthProvider  # local/dev only

        base = (public_url or "").rstrip("/")
        if not base.startswith("https://"):
            raise SystemExit(
                "OAuth mode needs a public HTTPS base URL, e.g.\n"
                "  python mcp_server.py --transport streamable-http --oauth "
                "--public-url https://YOUR_TUNNEL.trycloudflare.com"
            )
        # resource= must match the connector origin ChatGPT uses.
        auth_kwargs = {
            "auth_server_provider": DemoOAuthProvider(),
            "auth": AuthSettings(
                issuer_url=AnyHttpUrl(base),
                resource_server_url=AnyHttpUrl(base),
                client_registration_options=ClientRegistrationOptions(
                    enabled=True,
                    valid_scopes=["mcp"],
                    default_scopes=["mcp"],
                ),
                required_scopes=["mcp"],
            ),
        }

    mcp = FastMCP(
        name="hospital-ai-assistant",
        instructions=INSTRUCTIONS,
        host=host,
        port=port,
        streamable_http_path="/mcp",
        sse_path="/sse",
        message_path="/messages/",
        # Stateless is easier behind Cloudflare Tunnel / load balancers.
        stateless_http=stateless_http,
        transport_security=_transport_security(allow_tunnel=allow_tunnel),
        **auth_kwargs,
    )

    for lc_tool in ALL_TOOLS:
        mcp.add_tool(
            _wrap_langchain_tool(lc_tool),
            name=lc_tool.name,
            description=(lc_tool.description or "").strip(),
        )

    @mcp.custom_route("/", methods=["GET"])
    async def home(_request: Request) -> Response:
        return HTMLResponse(_browser_home_html(public_url=public_url, oauth=enable_oauth))

    @mcp.custom_route("/toollist", methods=["GET"])
    async def toollist(_request: Request) -> Response:
        return JSONResponse(build_tool_catalog())

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> Response:
        return JSONResponse(
            {
                "ok": True,
                "tools": len(ALL_TOOLS),
                "oauth": enable_oauth,
                "public_url": public_url,
            }
        )

    return mcp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hospital AI Assistant MCP server (wraps Tools.py).",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="MCP transport (default: stdio, or MCP_TRANSPORT env).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="Bind host for HTTP/SSE (default 127.0.0.1; use 0.0.0.0 for LAN).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "8000")),
        help="Bind port for HTTP/SSE (default 8000).",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print registered tool names and exit (no server).",
    )
    parser.add_argument(
        "--allow-tunnel",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MCP_ALLOW_TUNNEL", "").lower() in ("1", "true", "yes"),
        help=(
            "Disable the DNS-rebinding Host guard for public tunnels (default: off). "
            "Prefer MCP_ALLOWED_HOSTS=<your-host> instead."
        ),
    )
    parser.add_argument(
        "--public-url",
        default=os.getenv("MCP_PUBLIC_URL", ""),
        help="Public HTTPS base (Cloudflare URL). Required with --oauth.",
    )
    parser.add_argument(
        "--oauth",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MCP_OAUTH", "").lower() in ("1", "true", "yes"),
        help="Enable the DEMO OAuth provider (dev only; refused in HIPAA_MODE).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    public_url = (args.public_url or "").strip() or None
    mcp = build_mcp(
        host=args.host,
        port=args.port,
        allow_tunnel=bool(args.allow_tunnel),
        public_url=public_url,
        enable_oauth=bool(args.oauth),
    )

    if args.list_tools:
        for tool in ALL_TOOLS:
            desc = (tool.description or "").strip().split("\n", 1)[0]
            print(f"- {tool.name}: {desc}")
        print(f"\n{len(ALL_TOOLS)} tools (from Tools.py via tool_catalog.ALL_TOOLS)")
        return

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    api_key = sc.mcp_api_key()
    if not api_key and not args.oauth:
        raise SystemExit(
            "HTTP/SSE MCP transports require authentication. Set MCP_API_KEY=<long random "
            "secret> (clients send 'Authorization: Bearer <key>'), or run --transport stdio."
        )

    endpoint = (
        f"http://{args.host}:{args.port}/mcp"
        if args.transport == "streamable-http"
        else f"http://{args.host}:{args.port}/sse"
    )
    pub = public_url or f"http://{args.host}:{args.port}"
    print(
        f"Hospital MCP server ({args.transport}) on {endpoint}\n"
        f"Browser page: {pub}/\n"
        f"ChatGPT URL:  {pub}/mcp\n"
        f"Auth: {'demo OAuth (dev only)' if args.oauth else 'Bearer MCP_API_KEY required'}\n"
        f"Tools: {', '.join(t.name for t in ALL_TOOLS)}",
        file=sys.stderr,
    )
    _serve_http(mcp, args, api_key=api_key)


class _BearerAuthMiddleware:
    """Require ``Authorization: Bearer <MCP_API_KEY>`` on every request except /health."""

    def __init__(self, app: Any, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        stype = scope.get("type")
        # "lifespan" = uvicorn's internal startup/shutdown events, not a network request.
        if stype == "lifespan" or not self.api_key:
            await self.app(scope, receive, send)
            return
        if stype == "websocket":
            # No websocket routes exist today; refuse rather than trust that stays true.
            audit("mcp.auth", actor="anonymous", outcome="deny", detail={"path": "ws"})
            await send({"type": "websocket.close", "code": 1008})
            return
        path = scope.get("path") or ""
        if path in ("/health",):
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        auth = headers.get("authorization", "")
        ok = auth.startswith("Bearer ") and _ct_eq(auth[7:].strip(), self.api_key)
        if not ok:
            audit("mcp.auth", actor="anonymous", outcome="deny", detail={"path": path[:80]})
            resp = JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
            await resp(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _ct_eq(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())


def _serve_http(mcp: FastMCP, args: argparse.Namespace, *, api_key: str) -> None:
    import uvicorn

    app = mcp.streamable_http_app() if args.transport == "streamable-http" else mcp.sse_app()
    wrapped: Any = _BearerAuthMiddleware(app, api_key) if api_key else app
    uvicorn.run(wrapped, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
