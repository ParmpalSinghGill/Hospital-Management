const loginView = document.getElementById("login-view");
const adminView = document.getElementById("admin-view");
const loginForm = document.getElementById("login-form");
const loginError = document.getElementById("login-error");
const creditsEl = document.getElementById("credits");
const saveMsg = document.getElementById("save-msg");
const signInBtn = loginForm?.querySelector('button[type="submit"]');

async function api(path, opts = {}) {
  const res = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) {
    const detail = data?.detail || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function setView(which) {
  const showAdmin = which === "admin";
  loginView.classList.toggle("is-active", !showAdmin);
  adminView.classList.toggle("is-active", showAdmin);
  loginView.hidden = showAdmin;
  adminView.hidden = !showAdmin;
  // Clear any leftover inline display from older builds
  loginView.style.removeProperty("display");
  adminView.style.removeProperty("display");
}

function showAdmin(user) {
  setView("admin");
  const el = document.getElementById("admin-user");
  if (el) el.textContent = user || "Admin";
}

function showLogin() {
  setView("login");
}

function fillSelect(el, options, selected) {
  if (!el) return;
  el.innerHTML = "";
  let selectedOk = false;
  for (const opt of options || []) {
    const o = document.createElement("option");
    o.value = opt.id;
    o.textContent = opt.label;
    if (opt.enabled === false) {
      o.disabled = true;
    }
    if (opt.id === selected && opt.enabled !== false) {
      o.selected = true;
      selectedOk = true;
    }
    el.appendChild(o);
  }
  if (!selectedOk) {
    const first = [...el.options].find((o) => !o.disabled);
    if (first) first.selected = true;
  }
}

function readEnabledProviders() {
  const out = {};
  document.querySelectorAll("#enabled-providers input[data-provider]").forEach((input) => {
    out[input.dataset.provider] = Boolean(input.checked);
  });
  return out;
}

function normalizeEnabledMap(enabled) {
  if (!enabled || typeof enabled !== "object") return null;
  if (Array.isArray(enabled)) {
    const map = {};
    for (const item of enabled) {
      if (item && item.id != null) map[item.id] = item.enabled !== false;
    }
    return Object.keys(map).length ? map : null;
  }
  // Ignore maps that only contain undefined (flat fields missing).
  const map = {};
  let any = false;
  for (const [k, v] of Object.entries(enabled)) {
    if (v === undefined || v === null) continue;
    map[k] = Boolean(v);
    any = true;
  }
  return any ? map : null;
}

function writeEnabledProviders(enabled) {
  const map = normalizeEnabledMap(enabled);
  if (!map) return; // don't reset ticks when server omits the field
  document.querySelectorAll("#enabled-providers input[data-provider]").forEach((input) => {
    const id = input.dataset.provider;
    if (Object.prototype.hasOwnProperty.call(map, id)) {
      input.checked = Boolean(map[id]);
    }
  });
}

function syncRoutingAvailability() {
  const enabled = readEnabledProviders();
  const llmIds = ["deepseek", "glm", "groq", "openai"];
  for (const selectId of ["cascade_llm", "cli_llm"]) {
    const el = document.getElementById(selectId);
    if (!el) continue;
    let keep = null;
    for (const opt of el.options) {
      const on = enabled[opt.value] !== false;
      opt.disabled = !on;
      opt.textContent = opt.textContent.replace(/ \(disabled\)$/, "") + (on ? "" : " (disabled)");
      if (on && (opt.value === el.value || keep == null)) keep = opt.value;
    }
    if (keep != null) el.value = keep;
  }
  const pipe = document.getElementById("voice_pipeline_default");
  if (pipe) {
    for (const opt of pipe.options) {
      if (opt.value !== "realtime") continue;
      const on = enabled.realtime !== false;
      opt.disabled = !on;
      opt.textContent = "OpenAI Realtime" + (on ? "" : " (disabled)");
      if (!on && pipe.value === "realtime") pipe.value = "cascade";
    }
  }
  const realtimeFields = ["openai_realtime_model", "openai_realtime_voice"];
  for (const id of realtimeFields) {
    const el = document.getElementById(id);
    if (el) el.disabled = enabled.realtime === false;
  }
}

function fillVoices(el, voices, selected) {
  if (!el) return;
  el.innerHTML = "";
  for (const v of voices || []) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    if (v === selected) o.selected = true;
    el.appendChild(o);
  }
}

function setFieldValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value ?? "";
}

async function loadCredits() {
  creditsEl.innerHTML = "<p class='hint'>Checking providers…</p>";
  try {
    const data = await api("/admin/api/credits");
    creditsEl.innerHTML = "";
    for (const p of data.providers) {
      const card = document.createElement("div");
      card.className = "credit-card";
      const amount =
        p.credit == null || p.credit === ""
          ? "See vendor console"
          : `${p.credit}${p.unit ? ` ${p.unit}` : ""}`;
      card.innerHTML = `
        <div class="row">
          <div class="name">${p.provider}</div>
          <span class="badge ${p.ok ? "ok" : "bad"}">${p.ok ? "OK" : "Issue"}</span>
        </div>
        <p class="role">${p.role}</p>
        <div class="amount">${amount}</div>
        <p class="detail">${p.detail || ""}</p>
        <div class="key">Key: ${p.key_hint} · ${p.key_status}</div>
      `;
      creditsEl.appendChild(card);
    }
  } catch (err) {
    creditsEl.innerHTML = `<p class="error">${err.message}</p>`;
  }
}

function formatMsValue(ms) {
  if (ms == null || Number.isNaN(Number(ms))) return "—";
  const n = Number(ms);
  return n < 1000 ? `${Math.round(n)}ms` : `${(n / 1000).toFixed(2)}s`;
}

function formatAvgBucket(bucket) {
  if (!bucket || bucket.avg_ms == null || !bucket.count) return "—";
  return `${formatMsValue(bucket.avg_ms)} (${bucket.count})`;
}

async function loadResponseTimings() {
  const el = document.getElementById("response-timings");
  const meta = document.getElementById("response-timings-meta");
  if (!el) return;
  el.innerHTML = `<p class="hint">Scanning chats/sessions…</p>`;
  if (meta) meta.textContent = "";
  try {
    const data = await api("/admin/api/response-timings");
    const text = data.first_text || {};
    const speech = data.first_speech || {};
    if (meta) {
      meta.textContent = `${data.session_count || 0} sessions · ${data.turn_count || 0} turns · ${data.user_count || 0} users`;
    }
    el.innerHTML = "";
    const cards = [
      {
        name: "Avg first text",
        amount: formatAvgBucket(text),
        detail:
          text.count > 0
            ? `min ${formatMsValue(text.min_ms)} · max ${formatMsValue(text.max_ms)} · ${text.sessions || 0} sessions`
            : "No first-text samples yet. Enable Debug mode and chat/call so timings are written to sessions.",
      },
      {
        name: "Avg first speech",
        amount: formatAvgBucket(speech),
        detail:
          speech.count > 0
            ? `min ${formatMsValue(speech.min_ms)} · max ${formatMsValue(speech.max_ms)} · ${speech.sessions || 0} sessions`
            : "No first-speech samples yet. Place a voice call so audio latency is recorded.",
      },
    ];
    for (const c of cards) {
      const card = document.createElement("div");
      card.className = "credit-card";
      card.innerHTML = `
        <div class="row">
          <div class="name">${c.name}</div>
        </div>
        <div class="amount">${c.amount}</div>
        <p class="detail">${c.detail}</p>
      `;
      el.appendChild(card);
    }
  } catch (err) {
    el.innerHTML = `<p class="error">${err.message}</p>`;
  }
}

async function loadSettings() {
  const data = await api("/admin/api/settings");
  const s = data.settings || {};
  const o = data.options || {};
  const enabled =
    normalizeEnabledMap(s.enabled_providers) ||
    normalizeEnabledMap({
      deepseek: s.enable_deepseek,
      glm: s.enable_glm,
      groq: s.enable_groq,
      openai: s.enable_openai,
      realtime: s.enable_realtime,
    }) ||
    normalizeEnabledMap(o.enabled_providers);
  writeEnabledProviders(enabled);
  fillSelect(document.getElementById("stt"), o.stt, s.stt);
  fillSelect(document.getElementById("tts"), o.tts, s.tts);
  fillSelect(document.getElementById("cascade_llm"), o.cascade_llm, s.cascade_llm);
  fillSelect(document.getElementById("cli_llm"), o.cli_llm, s.cli_llm);
  fillSelect(
    document.getElementById("voice_pipeline_default"),
    o.voice_pipeline_default,
    s.voice_pipeline_default
  );
  fillVoices(
    document.getElementById("openai_realtime_voice"),
    o.openai_realtime_voices,
    s.openai_realtime_voice
  );
  setFieldValue("deepgram_voice", s.deepgram_voice);
  setFieldValue("glm_model", s.glm_model);
  setFieldValue("groq_model", s.groq_model);
  setFieldValue("deepseek_model", s.deepseek_model);
  setFieldValue("openai_realtime_model", s.openai_realtime_model);
  const debugEl = document.getElementById("debug_mode");
  if (debugEl) debugEl.checked = Boolean(s.debug_mode);
  const saveLlmEl = document.getElementById("save_llm_messages");
  if (saveLlmEl) saveLlmEl.checked = Boolean(s.save_llm_messages);
  const stopSecsEl = document.getElementById("vad_stop_secs");
  if (stopSecsEl) {
    const n = Number(s.vad_stop_secs);
    stopSecsEl.value = String(Number.isFinite(n) ? n : 0.2);
  }
  syncRoutingAvailability();
}

async function enterAdmin(user) {
  showAdmin(user);
  highlightNav("settings");
  try {
    await Promise.all([loadCredits(), loadSettings(), loadResponseTimings()]);
  } catch (err) {
    if (saveMsg) {
      saveMsg.textContent = err.message || "Failed to load admin data";
      saveMsg.className = "save-msg err";
      saveMsg.hidden = false;
    }
  }
}

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  loginError.hidden = true;
  loginError.textContent = "";
  if (signInBtn) {
    signInBtn.disabled = true;
    signInBtn.textContent = "Signing in…";
  }
  try {
    const data = await api("/admin/api/login", {
      method: "POST",
      body: JSON.stringify({
        username: document.getElementById("username").value.trim(),
        password: document.getElementById("password").value,
      }),
    });
    // Hard navigation so the cookie is applied cleanly (avoids stuck login UI).
    window.location.replace("/admin/?signed_in=1");
    return data;
  } catch (err) {
    loginError.textContent = err.message || "Login failed";
    loginError.hidden = false;
    showLogin();
  } finally {
    if (signInBtn) {
      signInBtn.disabled = false;
      signInBtn.textContent = "Sign in";
    }
  }
});

document.getElementById("logout-btn").addEventListener("click", async () => {
  try {
    await api("/admin/api/logout", { method: "POST" });
  } catch {
    /* still return to login */
  }
  window.location.replace("/admin/");
});

document.getElementById("refresh-credits").addEventListener("click", () => {
  loadCredits();
});
document.getElementById("refresh-response-timings")?.addEventListener("click", () => {
  loadResponseTimings();
});

async function saveSettings() {
  saveMsg.hidden = true;
  const enabled = readEnabledProviders();
  const llmOn = ["deepseek", "glm", "groq", "openai"].some((id) => enabled[id]);
  if (!llmOn) {
    saveMsg.textContent = "Keep at least one LLM provider enabled.";
    saveMsg.className = "save-msg err";
    saveMsg.hidden = false;
    return;
  }
  syncRoutingAvailability();
  const payload = {
    enabled_providers: enabled,
    enable_deepseek: Boolean(enabled.deepseek),
    enable_glm: Boolean(enabled.glm),
    enable_groq: Boolean(enabled.groq),
    enable_openai: Boolean(enabled.openai),
    enable_realtime: Boolean(enabled.realtime),
    stt: document.getElementById("stt")?.value,
    tts: document.getElementById("tts")?.value,
    cascade_llm: document.getElementById("cascade_llm")?.value,
    cli_llm: document.getElementById("cli_llm")?.value,
    voice_pipeline_default: document.getElementById("voice_pipeline_default")?.value,
    deepgram_voice: document.getElementById("deepgram_voice")?.value,
    glm_model: document.getElementById("glm_model")?.value,
    groq_model: document.getElementById("groq_model")?.value,
    deepseek_model: document.getElementById("deepseek_model")?.value,
    openai_realtime_model: document.getElementById("openai_realtime_model")?.value,
    openai_realtime_voice: document.getElementById("openai_realtime_voice")?.value,
    debug_mode: Boolean(document.getElementById("debug_mode")?.checked),
    save_llm_messages: Boolean(document.getElementById("save_llm_messages")?.checked),
    vad_stop_secs: Number(document.getElementById("vad_stop_secs")?.value || 0.2),
  };
  try {
    const res = await api("/admin/api/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    const savedEnabled =
      normalizeEnabledMap(res.settings?.enabled_providers) ||
      normalizeEnabledMap({
        deepseek: res.settings?.enable_deepseek,
        glm: res.settings?.enable_glm,
        groq: res.settings?.enable_groq,
        openai: res.settings?.enable_openai,
        realtime: res.settings?.enable_realtime,
      });
    if (!savedEnabled) {
      writeEnabledProviders(enabled);
      syncRoutingAvailability();
      saveMsg.textContent =
        "Routing saved, but cost controls did not persist. Restart bot.py, hard-refresh, then Save again.";
      saveMsg.className = "save-msg err";
      saveMsg.hidden = false;
      return;
    }
    const ignored = Object.keys(enabled).some(
      (k) => Boolean(savedEnabled[k]) !== Boolean(enabled[k])
    );
    if (ignored) {
      writeEnabledProviders(enabled);
      syncRoutingAvailability();
      saveMsg.textContent =
        "Cost controls did not stick. Restart bot.py, hard-refresh (Ctrl+Shift+R), then Save again.";
      saveMsg.className = "save-msg err";
      saveMsg.hidden = false;
      return;
    }
    await loadSettings();
    writeEnabledProviders(savedEnabled);
    syncRoutingAvailability();
    saveMsg.textContent = "Saved. Disabled providers cannot be used by chat or voice sessions.";
    saveMsg.className = "save-msg ok";
    saveMsg.hidden = false;
  } catch (err) {
    writeEnabledProviders(enabled);
    syncRoutingAvailability();
    saveMsg.textContent = err.message;
    saveMsg.className = "save-msg err";
    saveMsg.hidden = false;
  }
}

document.getElementById("enabled-providers")?.addEventListener("change", () => {
  syncRoutingAvailability();
});

document.getElementById("save-settings")?.addEventListener("click", saveSettings);
document.getElementById("save-settings-bottom")?.addEventListener("click", saveSettings);

(async function boot() {
  showLogin();
  try {
    const session = await api("/admin/api/session");
    if (session.authenticated) {
      await enterAdmin(session.user);
      return;
    }
  } catch {
    /* stay on login */
  }
  showLogin();
})();
