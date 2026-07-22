/** Pure display helpers shared by the chat UI. */

export function formatLatency(ms) {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatExactTime(dateLike) {
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

export function averageLatency(samples) {
  if (!samples.length) return "—";
  const avg = samples.reduce((sum, n) => sum + n, 0) / samples.length;
  return `${formatLatency(avg)} (${samples.length})`;
}

export function normalizeBotBubbleText(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim();
}

export function isDuplicateBotText(incoming, existing) {
  if (!incoming || !existing) return false;
  if (incoming === existing) return true;
  // TTS often re-sends the same reply in chunks after the first bubble closed.
  if (existing.startsWith(incoming) || incoming.startsWith(existing)) return true;
  return false;
}
