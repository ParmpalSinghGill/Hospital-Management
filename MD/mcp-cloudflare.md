# MCP + Cloudflare (ChatGPT)

Run the MCP server over HTTP and expose it with a Cloudflare Tunnel so **ChatGPT** (Developer mode) can discover and call booking tools.

## What to run

You need **two terminals**.

### Terminal 1 — MCP HTTP server

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent

python mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8000
```

Legacy SSE (if needed):

```bash
python mcp_server.py --transport sse --host 127.0.0.1 --port 8000
```

### Terminal 2 — Cloudflare Tunnel

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Copy the HTTPS URL Cloudflare prints (e.g. `https://xxxx.trycloudflare.com`).

## Keys / install

- Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)
- No Discord/Telegram tokens needed
- Protect the public URL before production (booking tools write to your DB)

## How to interact (ChatGPT)

1. Open the tunnel root in a browser once: `https://<your-tunnel>/` (human check)
2. In **ChatGPT → Settings → Developer mode** (plan-dependent), create an app/connector
3. Prefer authentication: **No Authentication**
4. Paste the MCP URL:

   ```text
   https://<your-tunnel>/mcp
   ```

5. Ask ChatGPT to list doctors or book/cancel/reschedule; confirm write actions when prompted

| Purpose | URL |
|---------|-----|
| Browser check | `https://<tunnel>/` |
| JSON tool list | `https://<tunnel>/toollist` |
| Health | `https://<tunnel>/health` |
| Recent tool calls | `https://<tunnel>/tool-calls?limit=50` |
| **ChatGPT MCP** | `https://<tunnel>/mcp` |
| SSE (legacy) | `https://<tunnel>/sse` |

## OAuth error (“Couldn’t register… sign-in service”)

ChatGPT tried OAuth, but you ran without a sign-in service.

**Option A:** In the connector, choose **No Authentication** and paste `https://<tunnel>/mcp`.

**Option B:** Demo OAuth (local/dev only):

```bash
python mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8000 \
  --allow-tunnel --oauth \
  --public-url https://YOUR_TUNNEL.trycloudflare.com
```

Use the **same** hostname in `--public-url` and in ChatGPT, then reconnect.

## Notes

- Restart MCP after code updates (`--allow-tunnel` is on by default).
- Local stdio MCP cannot be reached from chatgpt.com — always use HTTP + tunnel for ChatGPT.
- Same tools and DB as [local MCP](mcp-local.md) and the web UI.
