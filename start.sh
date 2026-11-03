#!/usr/bin/env bash
# Start ReceptionAi: landing page (/), voice + chat client (/app/), admin (/admin/).
#
# Usage:
#   ./start.sh            # start the server on http://localhost:7860
#   ./start.sh --rebuild  # rebuild the landing page (web/) first, then start
#
# Override the interpreter with PYTHON=/path/to/python ./start.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Pick a Python that has the project's dependencies -----------------------
pick_python() {
  local candidates=(
    "${PYTHON:-}"
    "$ROOT/.venv/bin/python"
    "$HOME/anaconda3/envs/hosmanag/bin/python"
    "$(command -v python3 || true)"
    "$(command -v python || true)"
  )
  for py in "${candidates[@]}"; do
    [[ -n "$py" && -x "$py" ]] || continue
    if "$py" -c "import pipecat" >/dev/null 2>&1; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "No Python with pipecat installed was found." >&2
  echo "Install deps (pip install -r requirements.txt) or set PYTHON=/path/to/python." >&2
  exit 1
fi
echo "Python: $PY"

# --- Landing page (static Next.js export in web/out) --------------------------
build_web() {
  if ! command -v node >/dev/null 2>&1; then
    echo "Node.js not found; skipping landing page build (/ falls back to landing.py)." >&2
    return 0
  fi
  echo "Building landing page (web/)..."
  (
    cd "$ROOT/web"
    # Repo sits on NTFS, which has no symlinks, hence --no-bin-links.
    [[ -d node_modules/next ]] || npm install --no-bin-links --no-audit --no-fund
    node node_modules/next/dist/bin/next build
  )
}

if [[ "${1:-}" == "--rebuild" || ! -f "$ROOT/web/out/index.html" ]]; then
  build_web
fi

# --- Run ----------------------------------------------------------------------
[[ -f "$ROOT/.env" ]] || echo "Warning: .env not found (copy .env.example and add API keys)." >&2

echo "Starting ReceptionAi on http://localhost:7860"
echo "  Landing  http://localhost:7860/"
echo "  Call     http://localhost:7860/app/"
echo "  Admin    http://localhost:7860/admin/"
exec "$PY" bot.py
