# MCP local (Cursor / Claude Desktop)

Expose the same `Tools.py` appointment tools over **MCP stdio** so a local MCP client (Cursor, Claude Desktop) can list and call them.

## What to run

```bash
conda activate hosmanag
cd /path/to/Hospital_Ai_Assistent
pip install -r requirements.txt   # includes mcp[cli]

# See tools without starting a server
python mcp_server.py --list-tools

# Stdio MCP (default) — for Cursor / Claude Desktop
python mcp_server.py
```

Do **not** use this mode for ChatGPT in the browser — chatgpt.com cannot reach local stdio. For ChatGPT use [MCP + Cloudflare](mcp-cloudflare.md).

## Keys needed

No extra MCP-specific key. Tool calls that book/cancel need a working DB (and the LLM is on the **client** side — Cursor/Claude — not this process).

## How to interact (Cursor)

1. Add an MCP server entry that runs:

   ```bash
   python /path/to/Hospital_Ai_Assistent/mcp_server.py
   ```

   (Use your conda/`hosmanag` Python if needed.)

2. Restart Cursor / reload MCP
3. In chat, ask Cursor to use hospital tools, e.g. “List cardiology doctors” or “Book an appointment…”
4. Confirm write actions when the client prompts

## How to interact (Claude Desktop)

Add under MCP servers in Claude Desktop config something like:

```json
{
  "mcpServers": {
    "hospital": {
      "command": "python",
      "args": ["/path/to/Hospital_Ai_Assistent/mcp_server.py"]
    }
  }
}
```

Then ask Claude to call the hospital tools.

## Available tools (same as web)

| Tool | Purpose |
|------|---------|
| `lookup_patient` | Find patient by phone/name |
| `list_doctors` | Filter doctors |
| `book_appointment` | Book a visit |
| `cancel_appointment` | Cancel by ID |
| `reschedule_appointment` | Move a visit |
| `get_prescriptions` | Look up medicines |

## Notes

- `bot.py` (7860) and `mcp_server.py` are separate; both share `Tools.py` + SQLite.
- Every tool call is audited in `chats/tool_call.json` with source `mcp`.
- For a public HTTPS endpoint (ChatGPT), see [MCP + Cloudflare](mcp-cloudflare.md).
