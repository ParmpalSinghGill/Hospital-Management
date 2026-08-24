# MCP over HTTP (remote clients, e.g. ChatGPT)

Run the MCP server over HTTP so a remote MCP client can discover and call the booking tools.

> **HIPAA note.** The MCP tools read and write patient data. A remote MCP endpoint is
> therefore a PHI system: it must be **authenticated**, reached only over **HTTPS**, and
> published only through infrastructure covered by a **BAA** (e.g. Cloudflare *Enterprise*
> with Access/Tunnel under BAA, or your own TLS reverse proxy). Quick `trycloudflare.com`
> tunnels and "No Authentication" connectors are **not** acceptable with real patient data.
> The former `/tool-calls` endpoint (raw tool arguments/results) has been removed.

## 1. Start the server (bearer-token auth is mandatory)

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent

export MCP_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
export MCP_ALLOWED_HOSTS=mcp.example-clinic.org      # the public hostname clients will use
python mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8000
```

* Without `MCP_API_KEY` the server refuses to start on HTTP/SSE transports.
* Every request except `GET /health` must carry `Authorization: Bearer <MCP_API_KEY>`.
* Unauthenticated attempts are written to the audit log (`mcp.auth` / `deny`).
* The demo OAuth provider (`--oauth`) auto-approves every client; it is refused while
  `HIPAA_MODE=1` and exists only for local experiments with synthetic data.

## 2. Publish it

Pick one:

| Option | When | Notes |
|---|---|---|
| Reverse proxy (nginx/Caddy) with TLS on your own host | You control the host | Terminate HTTPS, forward to `127.0.0.1:8000`, keep `MCP_ALLOWED_HOSTS` set |
| Cloudflare Tunnel + Access (**Enterprise plan, BAA executed**) | You want Cloudflare in front | Named tunnel, Access policy in front, still keep the bearer key |
| stdio only (`python mcp_server.py`) | Cursor / Claude Desktop on the same machine | No network exposure at all — preferred for local use |

Only `MCP_ALLOWED_HOSTS` should be used to admit your public hostname. `--allow-tunnel`
(disables the DNS-rebinding guard) is off by default and should stay off.

## 3. Connect a client

Configure the connector with:

* URL: `https://<your-host>/mcp` (Streamable HTTP) — or `/sse` for legacy clients
* Auth: **API key / bearer token** = your `MCP_API_KEY`

Ask the client to list doctors or book / cancel / reschedule; confirm write actions when
prompted. Remember the tools enforce identity (phone **and** full name) — the client cannot
read a record from a phone number alone.

| Purpose | URL |
|---------|-----|
| Browser check | `https://<host>/` |
| JSON tool list | `https://<host>/toollist` |
| Health (no auth) | `https://<host>/health` |
| **MCP** | `https://<host>/mcp` |
| SSE (legacy) | `https://<host>/sse` |

## Notes

- Restart MCP after code updates.
- Same tools and DB as [local MCP](mcp-local.md) and the web UI; every tool call is audited.
- See [HIPAA docs](hipaa/) for the full security posture.
