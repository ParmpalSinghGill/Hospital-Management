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
  for (const opt of options || []) {
    const o = document.createElement("option");
    o.value = opt.id;
    o.textContent = opt.label;
    if (opt.id === selected) o.selected = true;
    el.appendChild(o);
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

async function loadSettings() {
  const data = await api("/admin/api/settings");
  const s = data.settings || {};
  const o = data.options || {};
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
}

async function enterAdmin(user) {
  showAdmin(user);
  highlightNav("settings");
  try {
    await Promise.all([loadCredits(), loadSettings()]);
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

document.getElementById("save-settings").addEventListener("click", async () => {
  saveMsg.hidden = true;
  const payload = {
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
  };
  try {
    await api("/admin/api/settings", { method: "PUT", body: JSON.stringify(payload) });
    saveMsg.textContent = "Saved. New chat sessions pick up debug + provider settings.";
    saveMsg.className = "save-msg ok";
    saveMsg.hidden = false;
  } catch (err) {
    saveMsg.textContent = err.message;
    saveMsg.className = "save-msg err";
    saveMsg.hidden = false;
  }
});

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
