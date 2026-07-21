"""HTTP tool discovery for external systems (metadata only, no invocation)."""

from __future__ import annotations

import html
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from tool_catalog import build_tool_catalog, get_tool_by_name

router = APIRouter(tags=["tools"])


def _toollist_html(catalog: dict[str, Any]) -> str:
    rows = []
    for tool in catalog.get("tools") or []:
        name = html.escape(str(tool.get("name") or ""))
        desc = html.escape(str(tool.get("description") or ""))
        agents = ", ".join(html.escape(a) for a in (tool.get("agents") or []))
        params = tool.get("parameters") or {}
        required = ", ".join(html.escape(r) for r in (params.get("required") or []))
        rows.append(
            f"<tr><td><code>{name}</code></td>"
            f"<td>{desc}</td>"
            f"<td>{agents or '—'}</td>"
            f"<td><code>{required or '—'}</code></td></tr>"
        )

    pretty = html.escape(json.dumps(catalog, indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hospital tools</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; vertical-align: top; }}
    th {{ background: #f4f4f4; text-align: left; }}
    code {{ font-size: 0.92em; }}
    pre {{ background: #f8f8f8; padding: 1rem; overflow: auto; border-radius: 6px; }}
    a {{ color: #0b57d0; }}
  </style>
</head>
<body>
  <h1>Hospital appointment tools</h1>
  <p>
    Machine-readable JSON:
    <a href="/api/tools">/api/tools</a>
    · per tool:
    <a href="/api/tools/book_appointment">/api/tools/&lt;name&gt;</a>
  </p>
  <table>
    <thead>
      <tr><th>Name</th><th>Description</th><th>Agents</th><th>Required params</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <h2>Full JSON</h2>
  <pre>{pretty}</pre>
</body>
</html>"""


@router.get("/api/tools")
async def list_tools() -> dict[str, Any]:
    """Return all tool names, descriptions, and JSON Schema parameters."""
    return build_tool_catalog()


@router.get("/api/tools/{name}")
async def get_tool(name: str) -> dict[str, Any]:
    """Return metadata for one tool."""
    entry = get_tool_by_name(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")
    return {"ok": True, "tool": entry}


@router.get("/toollist/", response_class=HTMLResponse)
async def toollist_page() -> HTMLResponse:
    """Human-readable tool catalog (same data as /api/tools)."""
    catalog = build_tool_catalog()
    return HTMLResponse(_toollist_html(catalog))


@router.get("/toollist", include_in_schema=False)
async def toollist_alias() -> HTMLResponse:
    return await toollist_page()
