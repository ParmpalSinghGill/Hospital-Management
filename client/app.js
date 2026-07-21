import { PipecatClient } from "./vendor/client-js.bundle.mjs";
import {
  SmallWebRTCTransport,
  WavMediaManager,
} from "./vendor/small-webrtc-transport.bundle.mjs";

const banner = document.getElementById("banner");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const backendLabel = document.getElementById("backend-label");
const newPatientBtn = document.getElementById("new-patient-btn");
const micSelect = document.getElementById("mic-select");
const voiceMicSelect = document.getElementById("voice-mic-select");
const chatEl = document.getElementById("chat");
const messagesEl = document.getElementById("messages");
const welcomeStrip = document.getElementById("welcome-strip");
const avgResponsePanel = document.getElementById("avg-response-panel");
const avgTextEl = document.getElementById("avg-text-value");
const avgVoiceEl = document.getElementById("avg-voice-value");
const composerForm = document.getElementById("composer-form");
const textInput = document.getElementById("text-input");
const micButton = document.getElementById("mic-button");
const micButtonLabel = document.getElementById("mic-button-label");
const vadButton = document.getElementById("vad-button");
const vadButtonLabel = document.getElementById("vad-button-label");

const callDock = document.getElementById("call-dock");
const voiceClose = document.getElementById("voice-close");
const orb = document.getElementById("orb");
const voiceMicIndicator = document.getElementById("voice-mic-indicator");
const voiceMicFeedback = document.getElementById("voice-mic-feedback");
const voiceStatus = document.getElementById("voice-status");
const voiceCaption = document.getElementById("voice-caption");
const voiceMute = document.getElementById("voice-mute");
const voiceVad = document.getElementById("voice-vad");
const botAudio = document.getElementById("bot-audio");

const VAD_MODE_KEY = "dbc_vad_mode";
/** @type {"sensitive" | "stable"} */
let vadMode = localStorage.getItem(VAD_MODE_KEY) === "stable" ? "stable" : "sensitive";

function vadLabel(mode = vadMode) {
  return mode === "stable" ? "VAD: Stable" : "VAD: Sensitive";
}

function syncVadButtons() {
  const label = vadLabel();
  const isStable = vadMode === "stable";
  if (vadButtonLabel) vadButtonLabel.textContent = label;
  if (vadButton) {
    vadButton.classList.toggle("stable", isStable);
    vadButton.title =
      modeHintTitle() +
      (voiceCallActive ? " (takes effect on next call)" : "");
  }
  if (voiceVad) {
    voiceVad.textContent = label;
    voiceVad.classList.toggle("stable", isStable);
    voiceVad.title =
      modeHintTitle() +
      (voiceCallActive ? " (takes effect on next call)" : "");
  }
}

function modeHintTitle() {
  return vadMode === "stable"
    ? "Stable: less sensitive voice detection (fewer echo cuts)"
    : "Sensitive: default voice detection (faster barge-in)";
}

function toggleVadMode() {
  vadMode = vadMode === "stable" ? "sensitive" : "stable";
  localStorage.setItem(VAD_MODE_KEY, vadMode);
  syncVadButtons();
  if (voiceCallActive) {
    showBanner("VAD set to " + (vadMode === "stable" ? "Stable" : "Sensitive") + " — end call and start again to apply.");
  }
}

let client = null;
let connected = false;
let connecting = false;
let voiceCallActive = false;
let micMuted = false;
let assistantEl = null;
let assistantWrap = null;
let lastAssistantEl = null;
let lastAssistantWrap = null;
let bannerTimeout = null;
let llmStreaming = false;
let connectingMessageEl = null;
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
let turnMode = null;
let userTurnPending = false;
let botTextSource = null;
/** Last bot reply shown in chat — survives bubble finalize so TTS can't re-add it. */
let lastShownBotText = "";
let debugMode = false;
let pipelineMode = "cascade";
let lastUserMsgEl = null;
let callId = null;
let userVoiceStartAt = null;
const USER_ID_KEY = "dbc_care_user_id";
const THREAD_ID_KEY = "dbc_care_thread_id";

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

function getOrCreateThreadId() {
  // Stable LangGraph memory for this browser window across calls.
  try {
    let id = localStorage.getItem(THREAD_ID_KEY);
    if (id) return id;
    id =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? `thread_${crypto.randomUUID().slice(0, 12)}`
        : `thread_${Date.now().toString(16)}`;
    localStorage.setItem(THREAD_ID_KEY, id);
    return id;
  } catch {
    return `thread_${getOrCreateUserId()}`;
  }
}

/** Wipe browser session so the next connect is a fresh LangGraph thread (new patient). */
function resetSessionIdentity() {
  try {
    localStorage.removeItem(USER_ID_KEY);
    localStorage.removeItem(THREAD_ID_KEY);
  } catch {
    /* ignore */
  }
  return {
    userId: getOrCreateUserId(),
    threadId: getOrCreateThreadId(),
  };
}

async function startNewPatientSession() {
  const ok = window.confirm(
    "Start as a new patient?\n\nThis clears the chat and agent memory in this browser. Use a different phone/name when registering."
  );
  if (!ok) return;
  try {
    newPatientBtn && (newPatientBtn.disabled = true);
    await disconnectAndReset();
    clearTranscript();
    const ids = resetSessionIdentity();
    callId = null;
    addMessage(
      "system",
      `New patient session ready (${ids.threadId}). Say hello and register with a phone number.`
    );
    if (welcomeStrip) welcomeStrip.hidden = true;
    showBanner("Started a new patient session — previous chat memory cleared.");
  } catch (err) {
    showBanner("Could not reset session: " + (err?.message || err));
  } finally {
    if (newPatientBtn) newPatientBtn.disabled = false;
  }
}

const CONNECT_TIMEOUT_MS = 45000;
const MIC_STORAGE_KEY = "pipecat-preferred-mic-id";
const MIC_ACTIVE_THRESHOLD = 0.025;
const MIC_SOFT_THRESHOLD = 0.01;

let availableMics = [];
let preferredMicId = (() => {
  const saved = localStorage.getItem(MIC_STORAGE_KEY) || "";
  // Chromium virtual ids often need a manual toggle; don't restore them.
  if (!saved || saved === "default" || saved === "communications") {
    localStorage.removeItem(MIC_STORAGE_KEY);
    return "";
  }
  return saved;
})();
let micPermissionRequested = false;
let botSpeaking = false;
let localMicLevel = 0;
let remoteAudioTrack = null;

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
    pipeline_mode: pipelineMode,
    channel: "web_app",
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
    pipeline_mode: pipelineMode,
    channel: "web_app",
    ...extra,
  });
}

function setStatus(state) {
  statusDot.classList.remove("connected", "connecting", "error");
  if (state === "connecting") {
    statusDot.classList.add("connecting");
    statusDot.title = "connecting";
    statusText.textContent = "connecting…";
  } else if (state === "connected") {
    statusDot.classList.add("connected");
    statusDot.title = "connected";
    statusText.textContent = "connected";
  } else if (state === "error") {
    statusDot.classList.add("error");
    statusDot.title = "error";
    statusText.textContent = "error";
  } else {
    statusDot.title = "disconnected";
    statusText.textContent = "disconnected";
  }
}

function hideWelcome() {
  if (welcomeStrip) welcomeStrip.hidden = true;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function formatLatency(ms) {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatExactTime(dateLike) {
  if (!dateLike) return "—";
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

function renderAssistantDebug(metaOrWrap) {
  const wrap =
    metaOrWrap?.classList?.contains("msg")
      ? metaOrWrap
      : metaOrWrap?.closest?.(".msg") || assistantWrap || lastAssistantWrap;
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
  box.innerHTML = lines.join("") || `<span class="dbg-line">Waiting for response…</span>`;
}

function recordTextLatency() {
  if (!userTurnPending || textLatencyRecorded || pendingUserAt == null) return;
  turnTextAt = Date.now();
  turnTextLatencyMs = performance.now() - pendingUserAt;
  textLatencyRecorded = true;
  textResponseTimesMs.push(turnTextLatencyMs);
  renderAssistantDebug();
  updateAvgResponse();
  logClientEvent("bot_text_first_shown", {
    at_ms: turnTextAt,
    text: assistantEl?.textContent || "",
  });
}

function recordVoiceLatency() {
  if (!userTurnPending || voiceLatencyRecorded || pendingUserAt == null) return;
  turnVoiceAt = Date.now();
  turnVoiceLatencyMs = performance.now() - pendingUserAt;
  voiceLatencyRecorded = true;
  voiceResponseTimesMs.push(turnVoiceLatencyMs);
  renderAssistantDebug();
  updateAvgResponse();
  logClientEvent("bot_voice_first_heard", { at_ms: turnVoiceAt });
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
  turnMode = null;
  userTurnPending = false;
}

function beginTurnTiming(mode) {
  pendingUserAt = performance.now();
  pendingUserWallAt = Date.now();
  textLatencyRecorded = false;
  voiceLatencyRecorded = false;
  turnTextLatencyMs = null;
  turnVoiceLatencyMs = null;
  turnTextAt = null;
  turnVoiceAt = null;
  turnMode = mode;
  userTurnPending = true;
  // Allow the next bot reply even if it matches the previous wording.
  lastShownBotText = "";
}

function addMessage(role, text, { sentAt = null } = {}) {
  hideWelcome();
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  if (role === "user") {
    lastUserMsgEl = wrap;
    if (debugMode) renderUserDebug(wrap, sentAt || Date.now());
  }
  scrollToBottom();
  return bubble;
}

function clearTranscript() {
  messagesEl.innerHTML = "";
  assistantEl = null;
  assistantWrap = null;
  lastAssistantEl = null;
  lastAssistantWrap = null;
  llmStreaming = false;
  connectingMessageEl = null;
  lastUserMsgEl = null;
  textResponseTimesMs = [];
  voiceResponseTimesMs = [];
  resetTurnTiming();
  botTextSource = null;
  lastShownBotText = "";
  if (welcomeStrip) welcomeStrip.hidden = false;
  updateAvgResponse();
}

function showConnectingMessage() {
  if (connectingMessageEl) return;
  const wrap = document.createElement("div");
  wrap.className = "msg system";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "Connecting…";
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  connectingMessageEl = wrap;
  scrollToBottom();
}

function clearConnectingMessage() {
  connectingMessageEl?.remove();
  connectingMessageEl = null;
}

function beginAssistantTurn() {
  if (assistantEl) return;
  hideWelcome();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble pending";
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  assistantWrap = wrap;
  lastAssistantWrap = wrap;
  assistantEl = bubble;
  lastAssistantEl = bubble;
  if (debugMode) renderAssistantDebug(wrap);
  scrollToBottom();
}

function appendAssistantText(chunk) {
  if (!chunk) return;
  if (!assistantEl) beginAssistantTurn();
  const current = assistantEl.textContent || "";
  // Same full reply delivered on LLM + TTS + server-message channels.
  if (current === chunk || current.endsWith(chunk)) return;
  if (chunk.startsWith(current) && chunk.length > current.length) {
    assistantEl.textContent = chunk;
    if (!current) recordTextLatency();
    updateVoiceCaption(assistantEl.textContent);
    scrollToBottom();
    return;
  }
  const wasEmpty = !current;
  assistantEl.textContent += chunk;
  if (wasEmpty) recordTextLatency();
  updateVoiceCaption(assistantEl.textContent);
  scrollToBottom();
}

function normalizeBotBubbleText(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim();
}

function isDuplicateBotText(incoming, existing) {
  if (!incoming || !existing) return false;
  if (incoming === existing) return true;
  // TTS often re-sends the same reply in chunks after the first bubble closed.
  if (existing.startsWith(incoming) || incoming.startsWith(existing)) return true;
  return false;
}

function finalizeAssistantTurn({ removeIfEmpty = false } = {}) {
  const hadText = Boolean(assistantEl?.textContent?.trim());
  const finalText = assistantEl?.textContent || "";
  if (hadText) {
    lastShownBotText = normalizeBotBubbleText(finalText);
    logClientEvent("bot_text_complete", {
      at_ms: Date.now(),
      text: finalText,
    });
  }
  if (debugMode) renderAssistantDebug(assistantWrap);
  if (assistantWrap) lastAssistantWrap = assistantWrap;
  if (assistantEl) {
    assistantEl.classList.remove("pending");
    const wrap = assistantEl.closest(".msg.assistant");
    // LLM stop often arrives before bot-llm-text / bot-tts-text. Keep the
    // empty bubble open until speech ends (or we explicitly clean up).
    if (!hadText && removeIfEmpty) {
      wrap?.remove();
      if (wrap === lastAssistantWrap) lastAssistantWrap = null;
      if (assistantEl === lastAssistantEl) lastAssistantEl = null;
    }
  }
  // If the bubble is still empty and we're not removing it, keep assistantEl
  // so late TTS/output text can fill the same bubble.
  if (hadText || removeIfEmpty || !assistantEl) {
    assistantEl = null;
    assistantWrap = null;
    botTextSource = null;
  }
  if (hadText || voiceLatencyRecorded) {
    resetTurnTiming();
  }
}

function ingestBotText(text, source) {
  if (!text) return;
  const normalized = normalizeBotBubbleText(text);
  if (!normalized) return;

  const active = normalizeBotBubbleText(assistantEl?.textContent || "");
  if (
    isDuplicateBotText(normalized, active) ||
    isDuplicateBotText(normalized, lastShownBotText)
  ) {
    if (!botTextSource) botTextSource = source;
    return;
  }
  // Prefer the first non-empty channel; ignore duplicates from later channels.
  if (botTextSource && botTextSource !== source && active) {
    return;
  }
  botTextSource = source;
  appendAssistantText(text);
  lastShownBotText = normalizeBotBubbleText(assistantEl?.textContent || normalized);
}

function setOrbState(state) {
  if (!voiceCallActive) return;
  orb.classList.remove("listening", "speaking");
  if (state === "listening") orb.classList.add("listening");
  if (state === "speaking") orb.classList.add("speaking");
}

function setVoiceStatus(text) {
  if (!voiceCallActive) return;
  voiceStatus.textContent = text;
}

function updateVoiceCaption(text) {
  if (!voiceCallActive) return;
  voiceCaption.textContent = text || "";
}

function updateMicVisual(level = localMicLevel) {
  localMicLevel = level;
  if (!voiceCallActive) return;
  const clamped = Math.min(Math.max(level, 0), 1);
  voiceMicIndicator.style.setProperty("--mic-level", clamped);
  if (micMuted) {
    voiceMicIndicator.classList.remove("active");
    voiceMicFeedback.textContent = "Microphone muted";
    return;
  }
  const active = clamped >= MIC_ACTIVE_THRESHOLD;
  voiceMicIndicator.classList.toggle("active", active);
  if (active) voiceMicFeedback.textContent = "You're being heard";
  else if (clamped >= MIC_SOFT_THRESHOLD) voiceMicFeedback.textContent = "Quiet — speak a little louder";
  else voiceMicFeedback.textContent = "Waiting for your voice…";
}

function setBotAudioMuted(muted) {
  botAudio.muted = muted;
  if (!muted && botAudio.srcObject) {
    botAudio.play().catch((err) => console.warn("Bot audio play failed", err));
  }
}

function attachBotAudioTrack(track) {
  if (!track || track.kind !== "audio") return;
  remoteAudioTrack = track;
  botAudio.srcObject = new MediaStream([track]);
  setBotAudioMuted(!voiceCallActive);
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

async function openCallDock() {
  voiceCallActive = true;
  document.body.classList.add("in-call");
  micMuted = false;
  botSpeaking = false;
  localMicLevel = 0;
  voiceMute.textContent = "Mute";
  voiceMute.classList.remove("muted");
  micButtonLabel.textContent = "In call";
  orb.style.setProperty("--level", 0);
  voiceMicIndicator.classList.remove("active");
  voiceMicFeedback.textContent = "Waiting for your voice…";
  setOrbState("listening");
  setVoiceStatus("Listening…");
  voiceCaption.textContent = "";
  callDock.hidden = false;
  setBotAudioMuted(false);
  syncVadButtons();
  // New voice leg after End call (WebRTC may still be up) → fresh call log,
  // same thread_id for agent memory.
  if (!callId) {
    try {
      await startCallLog();
    } catch (err) {
      console.warn("startCallLog failed", err);
    }
  }
}

function closeCallDock() {
  voiceCallActive = false;
  document.body.classList.remove("in-call");
  callDock.hidden = true;
  micButtonLabel.textContent = "Call";
  setBotAudioMuted(true);
  // Close this call's log, but keep thread_id + chat so the next call
  // in this window remembers the conversation (like continuing with the same desk).
  endCallLog();
}

function micLabel(device, index) {
  const name = (device.label || "").trim();
  if (!name) return `Microphone ${index + 1}`;
  // Chromium lists a system alias as "Default" / "Default - …" — keep it clear.
  if (device.deviceId === "default" && !/^default\b/i.test(name)) {
    return `Default — ${name}`;
  }
  return name;
}

function isVirtualMicId(deviceId) {
  const id = String(deviceId || "").trim().toLowerCase();
  return !id || id === "default" || id === "communications";
}

function isSystemDefaultMic(device) {
  if (!device) return false;
  if (isVirtualMicId(device.deviceId)) return true;
  const label = (device.label || "").trim().toLowerCase();
  return label === "default" || label.startsWith("default -") || label.startsWith("default –");
}

function isCommunicationsMic(device) {
  const id = (device?.deviceId || "").toLowerCase();
  const label = (device?.label || "").toLowerCase();
  return id === "communications" || label.startsWith("communications");
}

/**
 * Prefer a real hardware deviceId. Chromium's virtual "default" / "communications"
 * ids used with {exact: ...} often open a silent track until the user toggles the
 * dropdown — that is the bug we hit on /app/.
 */
function resolvePreferredMicId(mics, preferred = preferredMicId) {
  const list = (mics || []).filter((d) => d && d.deviceId);
  if (!list.length) return "";

  const hardware = list.filter((d) => !isVirtualMicId(d.deviceId));

  if (preferred && !isVirtualMicId(preferred) && list.some((d) => d.deviceId === preferred)) {
    return preferred;
  }

  if (hardware.length) return hardware[0].deviceId;

  // Last resort: allow virtual default only if nothing else exists.
  const systemDefault = list.find((d) => d.deviceId === "default") || list.find(isSystemDefaultMic);
  return (systemDefault || list[0]).deviceId;
}

async function startVoiceCapture() {
  await refreshMicList({ requestPermission: true });

  // Drop a stuck localStorage preference for Chromium's broken virtual "default".
  if (isVirtualMicId(preferredMicId)) {
    preferredMicId = "";
    localStorage.removeItem(MIC_STORAGE_KEY);
  }

  const micId = resolvePreferredMicId(availableMics, preferredMicId);
  preferredMicId = micId;
  for (const select of [micSelect, voiceMicSelect]) {
    if (micId && [...select.options].some((o) => o.value === micId)) {
      select.value = micId;
    }
  }
  if (micId && !isVirtualMicId(micId)) {
    localStorage.setItem(MIC_STORAGE_KEY, micId);
  } else {
    localStorage.removeItem(MIC_STORAGE_KEY);
  }

  // WavRecorder.begin(id) uses getUserMedia({ deviceId: { exact: id } }).
  // Opening once with unconstrained {audio:true}, then switching to the real
  // hardware id, mirrors the manual dropdown toggle that makes the mic work.
  try {
    await client.updateMic("");
  } catch (err) {
    console.warn("mic warm-up failed", err);
  }
  await client.enableMic(true);
  await new Promise((r) => setTimeout(r, 80));

  if (micId && !isVirtualMicId(micId)) {
    try {
      await client.updateMic(micId);
      if (!micMuted) await client.enableMic(true);
    } catch (err) {
      console.warn("hardware mic apply failed, staying on browser default stream", err);
    }
  }

  const track = getLocalAudioTrack();
  if (track) {
    track.enabled = !micMuted;
    await replaceLocalAudioTrack(track);
  }
}

function populateMicSelects(mics) {
  availableMics = (mics || []).filter((d) => d && d.kind !== "audiooutput");
  // Hide Chromium virtual aliases when real hardware exists — they cause the
  // "must toggle mic" failure with exact:deviceId constraints.
  const hardware = availableMics.filter((d) => !isVirtualMicId(d.deviceId));
  const listed = hardware.length ? hardware : availableMics;

  let preferred = preferredMicId || micSelect?.value || "";
  if (isVirtualMicId(preferred)) preferred = "";
  const chosen = resolvePreferredMicId(listed, preferred);

  for (const select of [micSelect, voiceMicSelect]) {
    select.innerHTML = "";
    if (!listed.length) {
      const fallback = document.createElement("option");
      fallback.value = "";
      fallback.textContent = "Browser default";
      select.appendChild(fallback);
    } else {
      listed.forEach((device, index) => {
        const opt = document.createElement("option");
        opt.value = device.deviceId;
        opt.textContent = micLabel(device, index);
        select.appendChild(opt);
      });
    }
    select.value = chosen;
    select.disabled = listed.length === 0;
  }
  preferredMicId = chosen;
  if (chosen && !isVirtualMicId(chosen)) localStorage.setItem(MIC_STORAGE_KEY, chosen);
  else localStorage.removeItem(MIC_STORAGE_KEY);
}

async function requestMicPermission() {
  if (micPermissionRequested || !navigator.mediaDevices?.getUserMedia) return;
  micPermissionRequested = true;
  try {
    // Unconstrained audio — never exact:"default" (often silent on Linux/Chrome).
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((track) => track.stop());
  } catch (err) {
    console.warn("Microphone permission request failed", err);
    showBanner("Microphone access denied. Allow mic permission to pick a device.");
  }
}

async function refreshMicList({ requestPermission = false } = {}) {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  if (requestPermission) await requestMicPermission();
  const devices = await navigator.mediaDevices.enumerateDevices();
  populateMicSelects(devices.filter((d) => d.kind === "audioinput"));
}

function primeMicPermission() {
  refreshMicList({ requestPermission: true }).catch((err) =>
    console.warn("Mic permission / list refresh failed", err)
  );
}

async function applyMicSelection(deviceId = preferredMicId) {
  let resolved = String(deviceId || "").trim();
  if (isVirtualMicId(resolved)) {
    resolved = resolvePreferredMicId(availableMics, "");
  } else {
    resolved = resolvePreferredMicId(availableMics, resolved);
  }
  preferredMicId = resolved;
  if (preferredMicId && !isVirtualMicId(preferredMicId)) {
    localStorage.setItem(MIC_STORAGE_KEY, preferredMicId);
  } else {
    localStorage.removeItem(MIC_STORAGE_KEY);
  }
  for (const select of [micSelect, voiceMicSelect]) {
    if ([...select.options].some((o) => o.value === preferredMicId)) {
      select.value = preferredMicId;
    }
  }
  if (!client || !voiceCallActive) return;
  try {
    // Same warm-up + switch pattern as call start.
    await client.updateMic("");
    await client.enableMic(true);
    await new Promise((r) => setTimeout(r, 50));
    if (preferredMicId && !isVirtualMicId(preferredMicId)) {
      await client.updateMic(preferredMicId);
    }
    if (!micMuted) {
      await client.enableMic(true);
      const track = getLocalAudioTrack();
      if (track) {
        track.enabled = true;
        await replaceLocalAudioTrack(track);
      }
    }
  } catch (err) {
    console.warn("updateMic failed", err);
    showBanner("Failed to switch microphone: " + (err?.message || err));
  }
}

function handleMicSelectChange(event) {
  applyMicSelection(event.target.value);
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
      onAvailableMicsUpdated: (mics) => populateMicSelects(mics ?? []),
      onMicUpdated: (device) => {
        // Ignore Chromium virtual "default" callbacks — keep the real hardware pick.
        if (!device?.deviceId || isVirtualMicId(device.deviceId)) return;
        preferredMicId = device.deviceId;
        localStorage.setItem(MIC_STORAGE_KEY, preferredMicId);
        for (const select of [micSelect, voiceMicSelect]) {
          if ([...select.options].some((o) => o.value === device.deviceId)) {
            select.value = device.deviceId;
          }
        }
      },
      onDeviceError: (err) => {
        console.error("Device error", err);
        showBanner(err?.message || "Microphone device error");
      },
      onConnected: () => {
        connected = true;
        connecting = false;
        setStatus("connected");
      },
      onBotReady: () => {
        connected = true;
        connecting = false;
        setStatus("connected");
      },
      onDisconnected: () => {
        connected = false;
        connecting = false;
        setStatus("idle");
        closeCallDock();
      },
      onError: (message) => {
        console.error("RTVI error", message);
        const fatal = message?.data?.fatal ?? message?.fatal ?? false;
        const text = message?.data?.message ?? message?.message ?? "Unknown error";
        showBanner(text);
        if (fatal) {
          connecting = false;
          setStatus("error");
        }
      },
      onMessageError: (message) => console.error("RTVI message error", message),
      onBotLlmStarted: () => {
        llmStreaming = true;
        // Do not open an empty bubble here — wait for real text
        // (bot_chat_text). Opening early + TTS replay caused duplicates.
      },
      onBotLlmText: (data) => {
        ingestBotText(data?.text ?? data?.content ?? "", "llm");
      },
      onBotOutput: (data) => {
        ingestBotText(data?.text ?? "", "output");
      },
      onBotTtsText: () => {
        // Cascade chat text comes from bot_chat_text. TTS text is the same
        // reply again (often after the first bubble was finalized) and was
        // creating a second identical message.
      },
      onServerMessage: (data) => {
        // Primary chat text from voice_bridge.
        if (data?.type === "bot_chat_text") {
          ingestBotText(data?.text ?? "", "server");
        }
      },
      onBotLlmStopped: () => {
        finalizeAssistantTurn({ removeIfEmpty: true });
        llmStreaming = false;
      },
      onUserTranscript: (data) => {
        if (!data.final) {
          updateVoiceCaption(data.text);
          return;
        }
        const sentAt = Date.now();
        addMessage("user", data.text, { sentAt });
        updateVoiceCaption("");
        logClientEvent("user_voice", {
          text: data.text,
          sent_at_ms: sentAt,
          voice_start_at_ms: userVoiceStartAt,
          voice_end_at_ms: sentAt,
        });
        userVoiceStartAt = null;
      },
      onUserStartedSpeaking: () => {
        userVoiceStartAt = Date.now();
        setOrbState("listening");
        setVoiceStatus("Listening…");
        if (botSpeaking) {
          logClientEvent("barge_in_while_bot_speaking", {
            at_ms: Date.now(),
            bot_speaking: true,
            bot_text: lastAssistantEl?.textContent || assistantEl?.textContent || "",
            user_text: "",
            phase: "bot_speaking",
          });
        } else {
          logClientEvent("user_started_speaking", {
            at_ms: Date.now(),
            bot_speaking: false,
            phase: "idle",
          });
        }
      },
      onUserStoppedSpeaking: () => {
        if (voiceCallActive) beginTurnTiming("voice");
        setVoiceStatus("Thinking…");
        logClientEvent("user_stopped_speaking", {
          at_ms: Date.now(),
          bot_speaking: botSpeaking,
        });
      },
      onBotStartedSpeaking: () => {
        recordVoiceLatency();
        botSpeaking = true;
        setOrbState("speaking");
        setVoiceStatus("Speaking…");
        logClientEvent("bot_started_speaking", {
          at_ms: Date.now(),
          bot_text: lastAssistantEl?.textContent || assistantEl?.textContent || "",
        });
      },
      onBotStoppedSpeaking: () => {
        botSpeaking = false;
        logClientEvent("bot_voice_complete", {
          at_ms: Date.now(),
          text: lastAssistantEl?.textContent || assistantEl?.textContent || "",
        });
        logClientEvent("bot_stopped_speaking", {
          at_ms: Date.now(),
          bot_text: lastAssistantEl?.textContent || assistantEl?.textContent || "",
        });
        // Close out any bubble still waiting for late LLM/TTS text.
        if (assistantEl) finalizeAssistantTurn({ removeIfEmpty: true });
        setOrbState("listening");
        setVoiceStatus("Listening…");
      },
      onLocalAudioLevel: (level) => {
        updateMicVisual(level);
        if (voiceCallActive && !botSpeaking) orb.style.setProperty("--level", level);
      },
      onRemoteAudioLevel: (level) => {
        if (voiceCallActive && botSpeaking) orb.style.setProperty("--level", level);
      },
      onTrackStarted: async (track, participant) => {
        if (participant?.local && track.kind === "audio") {
          await replaceLocalAudioTrack(track);
          return;
        }
        attachBotAudioTrack(track);
      },
      onTrackStopped: (track) => {
        if (track === remoteAudioTrack) {
          remoteAudioTrack = null;
          botAudio.srcObject = null;
        }
      },
    },
  });
}

async function ensureConnected() {
  if (connected || connecting) return;
  connecting = true;
  setStatus("connecting");
  showConnectingMessage();
  // Refresh Admin backend selection before each connect
  await loadUiConfig();
  const id = await startCallLog();
  client = buildClient();
  try {
    const connectPromise = client.startBotAndConnect({
      endpoint: "/start",
      requestData: {
        transport: "webrtc",
        enableDefaultIceServers: true,
        body: {
          mode: pipelineMode,
          call_id: id,
          thread_id: getOrCreateThreadId(),
          user_id: getOrCreateUserId(),
          vad_mode: vadMode,
        },
      },
    });
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(
        () => reject(new Error("Connection timed out after 45s. Check the server terminal for errors.")),
        CONNECT_TIMEOUT_MS
      );
    });
    await Promise.race([connectPromise, timeoutPromise]);
    connected = true;
    connecting = false;
    clearConnectingMessage();
    setStatus("connected");
  } catch (err) {
    connecting = false;
    clearConnectingMessage();
    setStatus("error");
    if (client) {
      try {
        await client.disconnect();
      } catch {
        /* ignore */
      }
    }
    client = null;
    await endCallLog();
    showBanner("Failed to connect: " + (err?.message || err));
    throw err;
  }
}

async function disconnectAndReset() {
  if (client) {
    try {
      await client.disconnect();
    } catch (err) {
      console.warn("Error during disconnect", err);
    }
  }
  client = null;
  connected = false;
  connecting = false;
  await endCallLog();
  // Avoid double endCallLog from closeCallDock
  voiceCallActive = false;
  document.body.classList.remove("in-call");
  if (callDock) callDock.hidden = true;
  if (micButtonLabel) micButtonLabel.textContent = "Call";
  setBotAudioMuted(true);
  // Keep transcript + thread_id so a reconnect in this window continues the chat.
}

composerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  const sentAt = Date.now();
  addMessage("user", text, { sentAt });
  try {
    await ensureConnected();
    beginTurnTiming("text");
    logClientEvent("user_text", { text, sent_at_ms: sentAt });
    await client.sendText(text, { audio_response: voiceCallActive });
  } catch (err) {
    showBanner("Failed to send message: " + (err?.message || err));
  }
});

micButton.addEventListener("click", async () => {
  if (voiceCallActive) {
    client?.enableMic(false)?.catch((err) => console.warn("enableMic(false) failed", err));
    closeCallDock();
    return;
  }
  try {
    await ensureConnected();
  } catch {
    return;
  }
  try {
    await refreshMicList({ requestPermission: true });
    await startVoiceCapture();
    openCallDock();
  } catch (err) {
    console.warn("Voice mode start failed", err);
    showBanner("Could not start microphone: " + (err?.message || err));
  }
});

voiceClose.addEventListener("click", () => {
  client?.enableMic(false)?.catch((err) => console.warn("enableMic(false) failed", err));
  closeCallDock();
});

voiceMute.addEventListener("click", () => {
  micMuted = !micMuted;
  client?.enableMic(!micMuted)?.catch((err) => console.warn("enableMic toggle failed", err));
  voiceMute.classList.toggle("muted", micMuted);
  voiceMute.textContent = micMuted ? "Unmute" : "Mute";
  updateMicVisual(micMuted ? 0 : localMicLevel);
});

vadButton?.addEventListener("click", toggleVadMode);
voiceVad?.addEventListener("click", toggleVadMode);
syncVadButtons();

micSelect.addEventListener("change", handleMicSelectChange);
voiceMicSelect.addEventListener("change", handleMicSelectChange);
for (const select of [micSelect, voiceMicSelect]) {
  select.addEventListener("pointerdown", primeMicPermission);
}

navigator.mediaDevices?.addEventListener?.("devicechange", () => {
  refreshMicList().catch((err) => console.warn("devicechange refresh failed", err));
});

function formatBackendLabel(cfg) {
  const pipeline = cfg.voice_pipeline_default || "cascade";
  if (pipeline === "realtime") {
    return "Realtime · OpenAI";
  }
  const llm = (cfg.cascade_llm || "deepseek").toUpperCase();
  const stt = cfg.stt || "deepgram";
  const tts = cfg.tts || "deepgram";
  return `Cascade · ${stt} / ${llm} / ${tts}`;
}

async function loadUiConfig() {
  try {
    const res = await fetch("/admin/api/ui-config", { credentials: "same-origin" });
    if (!res.ok) return;
    const data = await res.json();
    debugMode = Boolean(data.debug_mode);
    pipelineMode = data.voice_pipeline_default === "realtime" ? "realtime" : "cascade";
    if (backendLabel) backendLabel.textContent = formatBackendLabel(data);
    updateAvgResponse();
  } catch (err) {
    console.warn("ui-config fetch failed", err);
    if (backendLabel) backendLabel.textContent = "Cascade (default)";
  }
}

setBotAudioMuted(true);
updateAvgResponse();
loadUiConfig();
refreshMicList().catch((err) => console.warn("Initial mic list failed", err));
newPatientBtn?.addEventListener("click", () => {
  startNewPatientSession().catch((err) => console.warn("new patient failed", err));
});
