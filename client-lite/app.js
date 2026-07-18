/**
 * Lean voice UI — Connect opens the appointment line (audio only).
 * Uses WavMediaManager so connect works in Electron/Chromium.
 */
import { PipecatClient } from "../app/vendor/client-js.bundle.mjs";
import {
  SmallWebRTCTransport,
  WavMediaManager,
} from "../app/vendor/small-webrtc-transport.bundle.mjs";

const $ = (id) => document.getElementById(id);

const banner = $("banner");
const statusDot = $("status-dot");
const statusLabel = $("status-label");
const avgResponsePanel = $("avg-response-panel");
const avgTextEl = $("avg-text-value");
const avgVoiceEl = $("avg-voice-value");
const modeSelect = $("mode-select");
const micSelect = $("mic-select");
const userMeter = $("user-meter");
const botMeter = $("bot-meter");
const userMeterWrap = $("user-meter-wrap");
const botMeterWrap = $("bot-meter-wrap");
const userSpeakState = $("user-speak-state");
const botSpeakState = $("bot-speak-state");
const lineStatus = $("line-status");
const transcriptMessages = $("transcript-messages");
const btnConnect = $("btn-connect");
const btnDisconnect = $("btn-disconnect");
const btnMic = $("btn-mic");
const botAudio = $("bot-audio");

const USER_ID_KEY = "dbc_voice_desk_user_id";
const MIC_ACTIVE_THRESHOLD = 0.025;
const MIC_SOFT_THRESHOLD = 0.01;
const BAR_WEIGHTS = [0.35, 0.55, 1, 0.7, 0.45];
const CONNECT_TIMEOUT_MS = 45000;

let client = null;
let connected = false;
let connecting = false;
let micOn = true;
let botLineBody = null;
let botLineWrap = null;
let lastBotLineBody = null;
let lastBotLineWrap = null;
let bannerTimer = null;

let pendingUserAt = null;
let pendingUserWallAt = null;
let textLatencyRecorded = false;
let voiceLatencyRecorded = false;
let turnTextLatencyMs = null;
let turnVoiceLatencyMs = null;
let turnTextAt = null;
let turnVoiceAt = null;
let textResponseTimesMs = [];
let voiceResponseTimesMs = [];
let userTurnPending = false;
let botTextSource = null;
let preferredMicId = "";
let debugMode = false;
let callId = null;
let botSpeaking = false;
let userSpeaking = false;
let userVoiceStartAt = null;
let localMicLevel = 0;

function getOrCreateUserId() {
  try {
    let id = localStorage.getItem(USER_ID_KEY);
    if (id) return id;
    id =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? `usr_${crypto.randomUUID().slice(0, 8)}`
        : `usr_${Date.now().toString(16)}`;
    localStorage.setItem(USER_ID_KEY, id);
    return id;
  } catch {
    return "usr_anonymous";
  }
}

function newCallId() {
  const stamp = new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14);
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `sess_${stamp}_${crypto.randomUUID().slice(0, 6)}`;
  }
  return `sess_${stamp}_${Math.random().toString(16).slice(2, 8)}`;
}

async function postCallLog(path, payload) {
  try {
    await fetch(`/api/call-logs/${path}`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    console.warn("call log failed", path, err);
  }
}

function ensureCallId() {
  if (!callId) callId = newCallId();
  return callId;
}

async function startCallLog() {
  const id = ensureCallId();
  await postCallLog("start", {
    call_id: id,
    session_id: id,
    user_id: getOrCreateUserId(),
    pipeline_mode: modeSelect.value,
    channel: "voice_desk",
    audio_codec: "opus",
  });
  return id;
}

async function endCallLog() {
  if (!callId) return;
  const id = callId;
  callId = null;
  await postCallLog("end", { call_id: id });
}

function logClientEvent(type, extra = {}) {
  if (!callId) return;
  postCallLog("event", {
    call_id: callId,
    type,
    user_id: getOrCreateUserId(),
    pipeline_mode: modeSelect.value,
    channel: "voice_desk",
    ...extra,
  });
}

function debugLog(...args) {
  if (debugMode) console.info("[voice-desk]", ...args);
}

function showBanner(msg) {
  banner.textContent = msg;
  banner.hidden = false;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => {
    banner.hidden = true;
  }, 7000);
}

function setStatus(state) {
  statusDot.className = "status-dot" + (state ? ` ${state}` : "");
  const labels = {
    connecting: "connecting…",
    connected: "connected",
    error: "error",
  };
  statusLabel.textContent = labels[state] || "disconnected";
}

function setLineStatus(text, live = false) {
  if (!lineStatus) return;
  lineStatus.textContent = text;
  lineStatus.classList.toggle("is-live", live);
}

function formatTime(at = new Date()) {
  return at.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatLatency(ms) {
  if (ms == null || Number.isNaN(ms)) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatExactTime(dateLike) {
  if (!dateLike) return "";
  const d = dateLike instanceof Date ? dateLike : new Date(dateLike);
  const base = d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${base}.${ms}`;
}

function averageLatency(samples) {
  if (!samples.length) return "—";
  const avg = samples.reduce((sum, n) => sum + n, 0) / samples.length;
  return `${formatLatency(avg)} (${samples.length})`;
}

function updateAvgResponse() {
  avgTextEl.textContent = averageLatency(textResponseTimesMs);
  avgVoiceEl.textContent = averageLatency(voiceResponseTimesMs);
  avgResponsePanel.hidden = !debugMode;
  avgResponsePanel.classList.remove("updated");
  void avgResponsePanel.offsetWidth;
  avgResponsePanel.classList.add("updated");
}

function resetChatSession() {
  pendingUserAt = null;
  pendingUserWallAt = null;
  textLatencyRecorded = false;
  voiceLatencyRecorded = false;
  turnTextLatencyMs = null;
  turnVoiceLatencyMs = null;
  turnTextAt = null;
  turnVoiceAt = null;
  textResponseTimesMs = [];
  voiceResponseTimesMs = [];
  userTurnPending = false;
  botTextSource = null;
  botSpeaking = false;
  userSpeaking = false;
  userVoiceStartAt = null;
  localMicLevel = 0;
  setMeter(userMeter, userMeterWrap, userSpeakState, 0, "idle");
  setMeter(botMeter, botMeterWrap, botSpeakState, 0, "idle");
  setLineStatus("Disconnected");
  updateAvgResponse();
  transcriptMessages.innerHTML =
    '<p class="hint">Press <strong>Connect</strong> to open the appointment line. Audio only — no camera.</p>';
}

function setMeter(barsEl, wrapEl, stateEl, level, forcedState = null) {
  const v = Math.min(Math.max(Number(level) || 0, 0), 1);
  const spans = barsEl?.querySelectorAll("span") || [];
  spans.forEach((span, i) => {
    const weight = BAR_WEIGHTS[i] ?? 0.5;
    const h = 6 + v * 22 * weight;
    span.style.height = `${h.toFixed(1)}px`;
  });
  const active = v >= MIC_SOFT_THRESHOLD;
  const speaking = v >= MIC_ACTIVE_THRESHOLD;
  wrapEl?.classList.toggle("active", active);
  wrapEl?.classList.toggle("speaking", speaking);
  if (stateEl) {
    if (forcedState) stateEl.textContent = forcedState;
    else if (speaking) stateEl.textContent = "speaking";
    else if (active) stateEl.textContent = "quiet";
    else stateEl.textContent = "idle";
  }
}

function ensureDebugBox(wrap) {
  let box = wrap.querySelector(".msg-debug");
  if (!box) {
    box = document.createElement("div");
    box.className = "msg-debug";
    wrap.appendChild(box);
  }
  return box;
}

function renderUserDebug(wrap, sentAt) {
  if (!debugMode || !wrap) return;
  const box = ensureDebugBox(wrap);
  box.innerHTML = `<span class="dbg-line"><span class="dbg-label">Sent</span> ${formatExactTime(sentAt)}</span>`;
}

function renderBotDebug(wrap = botLineWrap ?? lastBotLineWrap) {
  if (!debugMode || !wrap) return;
  const box = ensureDebugBox(wrap);
  const lines = [];
  if (pendingUserWallAt) {
    lines.push(
      `<span class="dbg-line"><span class="dbg-label">User turn</span> ${formatExactTime(pendingUserWallAt)}</span>`
    );
  }
  if (textLatencyRecorded || turnTextAt) {
    lines.push(
      `<span class="dbg-line"><span class="dbg-label">First text</span> ${formatExactTime(turnTextAt)} · ${formatLatency(turnTextLatencyMs)} after send</span>`
    );
  }
  if (voiceLatencyRecorded || turnVoiceAt) {
    lines.push(
      `<span class="dbg-line"><span class="dbg-label">First audio</span> ${formatExactTime(turnVoiceAt)} · ${formatLatency(turnVoiceLatencyMs)} after send</span>`
    );
  }
  box.innerHTML =
    lines.join("") || `<span class="dbg-line">Waiting for response…</span>`;
}

function addMessage(who, text, { at = new Date(), pending = false, sentAt = null } = {}) {
  transcriptMessages.querySelector(".hint")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `line ${who}` + (pending ? " pending" : "");

  const meta = document.createElement("div");
  meta.className = "meta";

  const whoEl = document.createElement("span");
  whoEl.className = "who";
  whoEl.textContent = who === "user" ? "You" : "Desk";

  const timeEl = document.createElement("span");
  timeEl.className = "time";
  timeEl.textContent = formatTime(at);

  meta.append(whoEl, timeEl);

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text || "";

  wrap.append(meta, body);
  transcriptMessages.appendChild(wrap);
  if (who === "user") renderUserDebug(wrap, sentAt || at);
  transcriptMessages.scrollTop = transcriptMessages.scrollHeight;
  return { wrap, body, meta };
}

function ensureBotLine() {
  if (botLineBody) return;
  const { wrap, body } = addMessage("bot", "", { pending: true });
  botLineWrap = wrap;
  botLineBody = body;
  lastBotLineWrap = wrap;
  lastBotLineBody = body;
  renderBotDebug(wrap);
}

function beginTurnTiming() {
  pendingUserAt = performance.now();
  pendingUserWallAt = Date.now();
  textLatencyRecorded = false;
  voiceLatencyRecorded = false;
  turnTextLatencyMs = null;
  turnVoiceLatencyMs = null;
  turnTextAt = null;
  turnVoiceAt = null;
  userTurnPending = true;
  debugLog("user turn started", new Date(pendingUserWallAt).toISOString());
}

function recordTextLatency() {
  if (!userTurnPending || textLatencyRecorded || pendingUserAt == null) return;
  turnTextAt = Date.now();
  turnTextLatencyMs = performance.now() - pendingUserAt;
  textLatencyRecorded = true;
  textResponseTimesMs.push(turnTextLatencyMs);
  debugLog("first text", formatExactTime(turnTextAt), formatLatency(turnTextLatencyMs));
  renderBotDebug();
  updateAvgResponse();
  logClientEvent("bot_text_first_shown", {
    at_ms: turnTextAt,
    text: botLineBody?.textContent || "",
  });
}

function recordVoiceLatency() {
  if (!userTurnPending || voiceLatencyRecorded || pendingUserAt == null) return;
  turnVoiceAt = Date.now();
  turnVoiceLatencyMs = performance.now() - pendingUserAt;
  voiceLatencyRecorded = true;
  voiceResponseTimesMs.push(turnVoiceLatencyMs);
  debugLog("first audio", formatExactTime(turnVoiceAt), formatLatency(turnVoiceLatencyMs));
  renderBotDebug();
  updateAvgResponse();
  logClientEvent("bot_voice_first_heard", { at_ms: turnVoiceAt });
}

function appendBotText(chunk) {
  if (!chunk) return;
  ensureBotLine();
  const wasEmpty = !botLineBody.textContent;
  botLineBody.textContent += chunk;
  if (wasEmpty) recordTextLatency();
  transcriptMessages.scrollTop = transcriptMessages.scrollHeight;
}

function resetTurnTiming() {
  pendingUserAt = null;
  pendingUserWallAt = null;
  textLatencyRecorded = false;
  voiceLatencyRecorded = false;
  turnTextLatencyMs = null;
  turnVoiceLatencyMs = null;
  turnTextAt = null;
  turnVoiceAt = null;
  userTurnPending = false;
}

function endBotTurn() {
  const body = botLineBody ?? lastBotLineBody;
  const hadText = Boolean(body?.textContent?.trim());
  const wrap = botLineWrap ?? lastBotLineWrap;
  renderBotDebug(wrap);
  if (botLineWrap) lastBotLineWrap = botLineWrap;
  if (wrap && !hadText && !voiceLatencyRecorded) {
    wrap.remove();
    if (wrap === lastBotLineWrap) lastBotLineWrap = null;
    if (body === lastBotLineBody) lastBotLineBody = null;
  }
  botLineWrap?.classList.remove("pending");
  botLineWrap = null;
  botLineBody = null;
  if (hadText || voiceLatencyRecorded) {
    resetTurnTiming();
  }
}

function setUiConnected(on) {
  connected = on;
  connecting = false;
  btnConnect.disabled = on || connecting;
  btnDisconnect.disabled = !on;
  btnMic.disabled = !on;
  modeSelect.disabled = on;
  micSelect.disabled = !on;
}

function getLocalAudioTrack() {
  const tracks = client?._transport?.mediaManager?.tracks?.();
  return tracks?.local?.audio ?? null;
}

async function replaceLocalAudioTrack(track) {
  if (!track || !client?._transport) return;
  const transport = client._transport;
  try {
    const sender =
      transport.getAudioTransceiver?.()?.sender ??
      transport.pc?.getTransceivers?.()?.[0]?.sender;
    if (sender) await sender.replaceTrack(track);
  } catch (err) {
    console.warn("replaceLocalAudioTrack failed", err);
  }
}

async function startVoiceCapture() {
  await client.updateMic(preferredMicId || micSelect.value || "");
  await client.enableMic(true);
  const track = getLocalAudioTrack();
  if (track) await replaceLocalAudioTrack(track);
  micOn = true;
  btnMic.textContent = "Mic on";
  btnMic.classList.remove("muted");
}

function buildClient() {
  return new PipecatClient({
    transport: new SmallWebRTCTransport({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      mediaManager: new WavMediaManager(),
    }),
    enableMic: false,
    enableCam: false,
    callbacks: {
      onConnected: () => {
        setUiConnected(true);
        setStatus("connected");
        setLineStatus("Connected — speak when ready", true);
      },
      onBotReady: () => {
        setUiConnected(true);
        setStatus("connected");
        setLineStatus("Desk ready — listening", true);
      },
      onDisconnected: () => {
        client = null;
        setUiConnected(false);
        setStatus("");
        setMeter(userMeter, userMeterWrap, userSpeakState, 0, "idle");
        setMeter(botMeter, botMeterWrap, botSpeakState, 0, "idle");
        setLineStatus("Disconnected");
        endBotTurn();
        endCallLog();
      },
      onError: (msg) => {
        const text = msg?.data?.message ?? msg?.message ?? "Connection error";
        const fatal = msg?.data?.fatal ?? msg?.fatal ?? false;
        showBanner(text);
        if (fatal) {
          connecting = false;
          setStatus("error");
          setUiConnected(false);
          setLineStatus("Error");
        }
      },
      onAvailableMicsUpdated: (mics) => {
        const prev = micSelect.value || preferredMicId;
        micSelect.innerHTML = '<option value="">Default</option>';
        (mics || []).forEach((d) => {
          const o = document.createElement("option");
          o.value = d.deviceId;
          o.textContent = d.label || "Microphone";
          micSelect.appendChild(o);
        });
        if (prev && [...micSelect.options].some((o) => o.value === prev)) {
          micSelect.value = prev;
          preferredMicId = prev;
        }
      },
      onUserTranscript: (data) => {
        if (!data.final) return;
        const sentAt = Date.now();
        if (!userTurnPending) beginTurnTiming();
        addMessage("user", data.text, { at: new Date(sentAt), sentAt });
        debugLog("user transcript", data.text);
        logClientEvent("user_voice", {
          text: data.text,
          sent_at_ms: sentAt,
          voice_start_at_ms: userVoiceStartAt,
          voice_end_at_ms: sentAt,
        });
        userVoiceStartAt = null;
      },
      onUserStartedSpeaking: () => {
        userSpeaking = true;
        userVoiceStartAt = Date.now();
        setMeter(userMeter, userMeterWrap, userSpeakState, Math.max(localMicLevel, 0.2), "speaking");
        setLineStatus("You're speaking…", true);
      },
      onUserStoppedSpeaking: () => {
        userSpeaking = false;
        beginTurnTiming();
        setMeter(userMeter, userMeterWrap, userSpeakState, localMicLevel, "idle");
        setLineStatus("Thinking…", true);
      },
      onBotLlmStarted: () => {
        botTextSource = null;
        ensureBotLine();
        setLineStatus("Desk composing…", true);
      },
      onBotLlmText: (data) => {
        botTextSource = "llm";
        appendBotText(data?.text ?? "");
      },
      onBotOutput: (data) => {
        if (botTextSource === "llm") return;
        const text = data?.text ?? "";
        if (!text) return;
        botTextSource = "output";
        appendBotText(text);
      },
      onBotLlmStopped: () => {
        endBotTurn();
      },
      onBotStartedSpeaking: () => {
        botSpeaking = true;
        recordVoiceLatency();
        setMeter(botMeter, botMeterWrap, botSpeakState, 0.35, "speaking");
        setLineStatus("Desk speaking…", true);
      },
      onBotStoppedSpeaking: () => {
        botSpeaking = false;
        setMeter(botMeter, botMeterWrap, botSpeakState, 0, "idle");
        setLineStatus(connected ? "Listening…" : "Disconnected", connected);
        logClientEvent("bot_voice_complete", {
          at_ms: Date.now(),
          text: lastBotLineBody?.textContent || botLineBody?.textContent || "",
        });
      },
      onLocalAudioLevel: (level) => {
        localMicLevel = level;
        if (!micOn) {
          setMeter(userMeter, userMeterWrap, userSpeakState, 0, "muted");
          return;
        }
        const label = userSpeaking
          ? "speaking"
          : level >= MIC_ACTIVE_THRESHOLD
            ? "speaking"
            : level >= MIC_SOFT_THRESHOLD
              ? "quiet"
              : "idle";
        setMeter(userMeter, userMeterWrap, userSpeakState, level, label);
      },
      onRemoteAudioLevel: (level) => {
        if (!botSpeaking && level < MIC_SOFT_THRESHOLD) {
          setMeter(botMeter, botMeterWrap, botSpeakState, 0, "idle");
          return;
        }
        setMeter(
          botMeter,
          botMeterWrap,
          botSpeakState,
          Math.max(level, botSpeaking ? 0.2 : 0),
          botSpeaking || level >= MIC_ACTIVE_THRESHOLD ? "speaking" : "idle"
        );
      },
      onTrackStarted: async (track, participant) => {
        if (participant?.local && track.kind === "audio") {
          await replaceLocalAudioTrack(track);
          return;
        }
        if (participant?.local || track.kind !== "audio") return;
        botAudio.srcObject = new MediaStream([track]);
        botAudio.play().catch(() => {});
      },
    },
  });
}

async function requestMicPermission() {
  if (!navigator.mediaDevices?.getUserMedia) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((t) => t.stop());
  } catch (err) {
    console.warn("Mic permission failed", err);
    showBanner("Allow microphone access to use the voice line.");
    throw err;
  }
}

async function loadUiConfig() {
  try {
    const res = await fetch("/admin/api/ui-config", { credentials: "same-origin" });
    if (!res.ok) return;
    const data = await res.json();
    debugMode = Boolean(data.debug_mode);
    if (!connected && data.voice_pipeline_default) {
      modeSelect.value =
        data.voice_pipeline_default === "realtime" ? "realtime" : "cascade";
    }
    updateAvgResponse();
    debugLog("ui-config", { debugMode, pipeline: modeSelect.value });
  } catch (err) {
    console.warn("ui-config fetch failed", err);
  }
}

async function connect() {
  if (connected || connecting) return;
  connecting = true;
  setStatus("connecting");
  setLineStatus("Connecting…");
  btnConnect.disabled = true;
  resetChatSession();
  await loadUiConfig();

  try {
    await requestMicPermission();
  } catch {
    connecting = false;
    setStatus("error");
    btnConnect.disabled = false;
    setLineStatus("Mic permission needed");
    return;
  }

  callId = null;
  const id = await startCallLog();
  client = buildClient();
  try {
    const connectPromise = client.startBotAndConnect({
      endpoint: "/start",
      requestData: {
        transport: "webrtc",
        enableDefaultIceServers: true,
        body: {
          mode: modeSelect.value,
          call_id: id,
          user_id: getOrCreateUserId(),
        },
      },
    });
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(
        () => reject(new Error("Connection timed out after 45s. Check the server terminal.")),
        CONNECT_TIMEOUT_MS
      );
    });
    await Promise.race([connectPromise, timeoutPromise]);
    await startVoiceCapture();
    setUiConnected(true);
    setStatus("connected");
    setLineStatus("Connected — speak when ready", true);
    debugLog("connected", id);
  } catch (err) {
    connecting = false;
    setStatus("error");
    setUiConnected(false);
    setLineStatus("Connection failed");
    if (client) {
      try {
        await client.disconnect();
      } catch {
        /* ignore */
      }
    }
    client = null;
    await endCallLog();
    showBanner(String(err?.message || err));
  }
}

async function disconnect() {
  if (!client) {
    await endCallLog();
    return;
  }
  try {
    await client.enableMic(false);
  } catch {
    /* ignore */
  }
  try {
    await client.disconnect();
  } catch (err) {
    console.warn(err);
  }
  client = null;
  setUiConnected(false);
  setStatus("");
  setMeter(userMeter, userMeterWrap, userSpeakState, 0, "idle");
  setMeter(botMeter, botMeterWrap, botSpeakState, 0, "idle");
  setLineStatus("Disconnected");
  endBotTurn();
  await endCallLog();
}

btnConnect.addEventListener("click", () => connect());
btnDisconnect.addEventListener("click", () => disconnect());

btnMic.addEventListener("click", async () => {
  if (!client) return;
  micOn = !micOn;
  try {
    if (micOn) await startVoiceCapture();
    else await client.enableMic(false);
  } catch (err) {
    showBanner(String(err?.message || err));
    return;
  }
  btnMic.textContent = micOn ? "Mic on" : "Mic off";
  btnMic.classList.toggle("muted", !micOn);
  if (!micOn) setMeter(userMeter, userMeterWrap, userSpeakState, 0, "muted");
});

micSelect.addEventListener("change", async () => {
  preferredMicId = micSelect.value || "";
  if (!client || !connected) return;
  try {
    await startVoiceCapture();
  } catch (e) {
    showBanner(String(e?.message || e));
  }
});

updateAvgResponse();
loadUiConfig();
