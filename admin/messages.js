highlightNav("messages");

const sessionsList = document.getElementById("sessions-list");
const sessionsMeta = document.getElementById("sessions-meta");
const sessionsPanel = document.getElementById("sessions-panel");
const viewerPanel = document.getElementById("viewer-panel");
const viewerTitle = document.getElementById("viewer-title");
const viewerMeta = document.getElementById("viewer-meta");
const viewerRuntime = document.getElementById("viewer-runtime");
const passedBody = document.getElementById("passed-body");
const responseBody = document.getElementById("response-body");
const prevBtn = document.getElementById("prev-pair");
const nextBtn = document.getElementById("next-pair");
const backBtn = document.getElementById("back-sessions");
const deleteSessionBtn = document.getElementById("delete-session");
const deleteAllBtn = document.getElementById("delete-all-sessions");
const sortSessionsBtn = document.getElementById("sort-sessions");

let currentSession = "";
let currentN = null;
let nextN = null;
let prevN = null;
/** @type {"newest" | "oldest"} */
let sessionSort = "newest";
/** @type {Array<Record<string, any>>} */
let cachedSessions = [];

function sessionSortLabel() {
  return sessionSort === "newest" ? "Newest first" : "Oldest first";
}

function updateSortButton() {
  if (!sortSessionsBtn) return;
  sortSessionsBtn.textContent = sessionSortLabel();
  sortSessionsBtn.setAttribute("aria-pressed", sessionSort === "newest" ? "true" : "false");
  sortSessionsBtn.title =
    sessionSort === "newest"
      ? "Showing latest sessions first — click for oldest first"
      : "Showing oldest sessions first — click for newest first";
}

function sortedSessions(sessions) {
  const rows = [...(sessions || [])];
  rows.sort((a, b) => {
    const am = Number(a?.mtime) || 0;
    const bm = Number(b?.mtime) || 0;
    return sessionSort === "newest" ? bm - am : am - bm;
  });
  return rows;
}

function renderSessions(sessions) {
  cachedSessions = sessions || [];
  const rows = sortedSessions(cachedSessions);
  sessionsMeta.textContent = rows.length
    ? `${rows.length} session${rows.length === 1 ? "" : "s"} (${sessionSortLabel().toLowerCase()})`
    : "No saved LLM message sessions yet.";
  updateSortButton();
  if (!rows.length) {
    sessionsList.innerHTML = `<p class="hint">Enable “Save LLM messages” in Settings, restart the bot, then run a chat.</p>`;
    return;
  }
  sessionsList.innerHTML = "";
  for (const s of rows) {
    const row = document.createElement("div");
    row.className = "llm-session-row-wrap";

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "llm-session-row";
    const timing =
      s.start_secs != null || s.stop_secs != null
        ? `start=${formatSecs(s.start_secs)} · stop=${formatSecs(s.stop_secs)}`
        : "";
    openBtn.innerHTML = `
      <strong>${escapeHtml(s.session_id)}</strong>
      <span>${escapeHtml(s.pair_count)} LLM call${s.pair_count === 1 ? "" : "s"}</span>
      <span class="muted">${escapeHtml(s.first_at || "")} → ${escapeHtml(s.last_at || "")}</span>
      ${timing ? `<span class="muted">${escapeHtml(timing)}</span>` : ""}
    `;
    openBtn.addEventListener("click", () => openSession(s.session_id));

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "llm-session-delete";
    delBtn.title = "Delete this session’s LLM dumps";
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!confirm(`Delete all LLM dumps for\n${s.session_id}?`)) return;
      delBtn.disabled = true;
      try {
        await deleteSessionById(s.session_id);
        if (currentSession === s.session_id) showSessions();
        await loadSessions();
      } catch (err) {
        delBtn.disabled = false;
        alert(err.message || "Could not delete session.");
      }
    });

    row.appendChild(openBtn);
    row.appendChild(delBtn);
    sessionsList.appendChild(row);
  }
}

function showSessions() {
  sessionsPanel.hidden = false;
  viewerPanel.hidden = true;
  currentSession = "";
  currentN = null;
}

function showViewer() {
  sessionsPanel.hidden = true;
  viewerPanel.hidden = false;
}

function formatSecs(value) {
  if (value == null || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return `${n.toFixed(2)}s`;
}

function renderRuntimeMeta(meta = {}) {
  if (!viewerRuntime) return;
  const client = meta.client_timings && typeof meta.client_timings === "object" ? meta.client_timings : {};
  const rows = [
    ["start_secs", formatSecs(meta.start_secs)],
    ["stop_secs", formatSecs(meta.stop_secs)],
  ];
  if (meta.confidence != null) rows.push(["confidence", String(meta.confidence)]);
  if (meta.min_volume != null) rows.push(["min_volume", String(meta.min_volume)]);
  if (meta.pipeline_mode) rows.push(["pipeline", String(meta.pipeline_mode)]);
  const firstTextMs = client.first_text_latency_ms ?? meta.first_text_latency_ms;
  const firstAudioMs = client.first_audio_latency_ms ?? meta.first_audio_latency_ms;
  const firstText = client.first_text ?? meta.first_text;
  const firstSpeech = client.first_speech ?? meta.first_speech;
  if (firstTextMs != null) rows.push(["first_text_ms", `${Number(firstTextMs).toFixed(0)}ms`]);
  if (firstAudioMs != null) rows.push(["first_audio_ms", `${Number(firstAudioMs).toFixed(0)}ms`]);
  if (firstText) rows.push(["first_text", String(firstText)]);
  if (firstSpeech) rows.push(["first_speech", String(firstSpeech)]);
  viewerRuntime.innerHTML = rows
    .map(
      ([k, v]) =>
        `<div class="llm-runtime-row"><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`
    )
    .join("");
}

async function deleteSessionById(sessionId) {
  const id = String(sessionId || "").trim();
  if (!id) throw new Error("Missing session id");
  const path = `/admin/api/llm-messages/sessions/${encodeURIComponent(id)}`;
  try {
    return await api(`${path}/delete`, { method: "POST" });
  } catch (err) {
    // Older bot process may only have DELETE until restart.
    return api(path, { method: "DELETE" });
  }
}

async function deleteAllSessionsRequest() {
  try {
    return await api("/admin/api/llm-messages/delete-all", { method: "POST" });
  } catch (err) {
    return api("/admin/api/llm-messages/sessions", { method: "DELETE" });
  }
}

async function loadSessions() {
  sessionsList.innerHTML = `<p class="hint">Loading…</p>`;
  try {
    const data = await api("/admin/api/llm-messages/sessions");
    renderSessions(data.sessions || []);
  } catch (err) {
    sessionsList.innerHTML = `<p class="error">${escapeHtml(err.message || err)}</p>`;
  }
}

async function openSession(sessionId) {
  currentSession = sessionId;
  try {
    const data = await api(`/admin/api/llm-messages/sessions/${encodeURIComponent(sessionId)}`);
    const pairs = data.pairs || [];
    if (!pairs.length) {
      alert("No pairs in this session.");
      return;
    }
    await openPair(Number(pairs[0].n));
  } catch (err) {
    alert(err.message || String(err));
  }
}

async function openPair(n) {
  if (!currentSession || n == null) return;
  try {
    const data = await api(
      `/admin/api/llm-messages/sessions/${encodeURIComponent(currentSession)}/pairs/${n}`
    );
    const pair = data.pair || {};
    currentN = pair.n;
    prevN = data.prev_n;
    nextN = data.next_n;
    showViewer();
    viewerTitle.textContent = `Message ${pair.n}`;
    viewerMeta.textContent = [
      currentSession,
      pair.node ? `node=${pair.node}` : "",
      pair.at || "",
      `call ${data.index + 1} of ${data.total}`,
    ]
      .filter(Boolean)
      .join(" · ");
    renderRuntimeMeta({
      ...(pair.session_meta || {}),
      ...(pair.meta || {}),
      start_secs: pair.start_secs ?? pair.meta?.start_secs,
      stop_secs: pair.stop_secs ?? pair.meta?.stop_secs,
      client_timings: pair.meta?.client_timings || pair.client_timings,
    });
    passedBody.textContent = pair.passed || "(empty)";
    responseBody.textContent = pair.response || "(empty)";
    prevBtn.disabled = prevN == null;
    nextBtn.disabled = nextN == null;
  } catch (err) {
    alert(err.message || String(err));
  }
}

async function deleteCurrentSession() {
  if (!currentSession) return;
  if (!confirm(`Delete all LLM dumps for\n${currentSession}?`)) return;
  if (deleteSessionBtn) deleteSessionBtn.disabled = true;
  try {
    await deleteSessionById(currentSession);
    showSessions();
    await loadSessions();
  } catch (err) {
    alert(err.message || "Could not delete session.");
  } finally {
    if (deleteSessionBtn) deleteSessionBtn.disabled = false;
  }
}

async function deleteAllSessions() {
  if (!confirm("Delete ALL saved LLM message sessions? This cannot be undone.")) return;
  if (deleteAllBtn) deleteAllBtn.disabled = true;
  try {
    const data = await deleteAllSessionsRequest();
    showSessions();
    await loadSessions();
    if (data?.deleted != null) {
      sessionsMeta.textContent = `Deleted ${data.deleted} session folder${data.deleted === 1 ? "" : "s"}.`;
    }
  } catch (err) {
    alert(err.message || "Could not delete all sessions.");
  } finally {
    if (deleteAllBtn) deleteAllBtn.disabled = false;
  }
}

document.getElementById("refresh-sessions")?.addEventListener("click", () => {
  loadSessions().catch((err) => alert(err.message || String(err)));
});
sortSessionsBtn?.addEventListener("click", () => {
  sessionSort = sessionSort === "newest" ? "oldest" : "newest";
  renderSessions(cachedSessions);
});
updateSortButton();
if (deleteAllBtn) {
  deleteAllBtn.addEventListener("click", (e) => {
    e.preventDefault();
    deleteAllSessions();
  });
} else {
  console.warn("Delete all button missing from DOM — hard-refresh Admin → Messages");
}
if (deleteSessionBtn) {
  deleteSessionBtn.addEventListener("click", (e) => {
    e.preventDefault();
    deleteCurrentSession();
  });
}
backBtn?.addEventListener("click", () => {
  showSessions();
  loadSessions();
});
prevBtn?.addEventListener("click", () => {
  if (prevN != null) openPair(prevN);
});
nextBtn?.addEventListener("click", () => {
  if (nextN != null) openPair(nextN);
});

document.addEventListener("keydown", (e) => {
  if (viewerPanel.hidden) return;
  if (e.key === "ArrowRight" || e.key === "n") {
    e.preventDefault();
    if (nextN != null) openPair(nextN);
  } else if (e.key === "ArrowLeft" || e.key === "p") {
    e.preventDefault();
    if (prevN != null) openPair(prevN);
  } else if (e.key === "Escape") {
    showSessions();
  }
});

(async () => {
  await requireAdminSession();
  document.getElementById("logout-btn")?.addEventListener("click", logoutAdmin);
  await loadSessions();
})();
