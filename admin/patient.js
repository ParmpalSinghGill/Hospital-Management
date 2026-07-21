(async function () {
  highlightNav("patients");
  bindLogout();
  if (!(await requireAdminSession())) return;

  const pathParts = window.location.pathname.split("/").filter(Boolean);
  let patientId = "";
  const idx = pathParts.indexOf("patient");
  if (idx >= 0 && pathParts[idx + 1]) {
    patientId = decodeURIComponent(pathParts[idx + 1]);
  }
  if (!patientId) {
    patientId = new URLSearchParams(window.location.search).get("id") || "";
  }

  const errEl = document.getElementById("patient-error");
  const chatsPanel = document.getElementById("chats-panel");
  const chatsList = document.getElementById("chats-list");
  const toggleChatsBtn = document.getElementById("toggle-chats-btn");
  const closeChatsBtn = document.getElementById("close-chats-btn");

  if (!patientId) {
    errEl.textContent = "Missing patient id.";
    errEl.hidden = false;
    return;
  }

  function updateChatCount(n) {
    toggleChatsBtn.textContent = chatsPanel.hidden
      ? `Previous chats (${n})`
      : `Hide chats`;
    const jump = document.getElementById("jump-chats");
    if (jump) jump.textContent = String(n);
  }

  function renderChats(chats) {
    if (!chats || !chats.length) {
      chatsList.innerHTML = `<p class="hint">No previous chats for this patient yet.</p>`;
      updateChatCount(0);
      return;
    }
    chatsList.innerHTML = "";
    for (const chat of chats) {
      const row = document.createElement("div");
      row.className = "chat-card-row";

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "chat-delete";
      delBtn.setAttribute("aria-label", "Delete chat");
      delBtn.title = "Delete chat";
      delBtn.textContent = "×";

      const card = document.createElement("details");
      card.className = "chat-card";
      const kind = chat.kind || chat.pipeline_mode || "live";
      const topic = chat.topic ? ` · ${chat.topic}` : "";
      const sessionKey = chat.session_id || chat.file;
      const turnsHtml = (chat.turns || [])
        .map((t) => {
          const agent = (t.agent || "").trim();
          const assistantLabel = agent ? `Assistant (${agent})` : "Assistant";
          const assistantTitle = t.agent_inferred
            ? ' title="Agent inferred for an older chat; the original agent was not recorded"'
            : "";
          return `
          <div class="chat-turn">
            <div class="chat-user"><span class="chat-speaker">Patient:</span> ${escapeHtml(t.user || "")}</div>
            <div class="chat-bot"><span class="chat-speaker"${assistantTitle}>${escapeHtml(assistantLabel)}:</span> ${escapeHtml(t.assistant || "")}</div>
          </div>`;
        })
        .join("");
      const timeline = chat.timeline || [];
      const timelineHtml = timeline.length
        ? `<div class="chat-timeline">
            <p class="hint">Interrupt / speak timeline</p>
            ${timeline
              .map((ev) => {
                const when = String(ev.at || "").replace("T", " ").replace("Z", " UTC");
                const bits = [
                  ev.type || "event",
                  ev.phase ? `phase=${ev.phase}` : "",
                  ev.strategy ? `via ${ev.strategy}` : "",
                ].filter(Boolean);
                const texts = [
                  ev.user_text ? `user: ${ev.user_text}` : "",
                  ev.bot_text ? `bot: ${ev.bot_text}` : "",
                ].filter(Boolean);
                return `<div class="timeline-row"><span class="timeline-time">${escapeHtml(when)}</span> <strong>${escapeHtml(bits.join(" · "))}</strong>${texts.length ? `<div class="timeline-text">${escapeHtml(texts.join(" | "))}</div>` : ""}</div>`;
              })
              .join("")}
          </div>`
        : "";
      card.innerHTML = `
        <summary>
          <strong>${escapeHtml(sessionKey)}</strong>
          <span class="chat-meta">
            <span class="kind-tag">${escapeHtml(kind)}${escapeHtml(topic)}</span> ·
            ${escapeHtml(chat.appointment_id || "no appointment")} ·
            ${escapeHtml(chat.session_start_time || "")} ·
            ${escapeHtml(String(chat.interaction_count || 0))} turns${timeline.length ? ` · ${timeline.length} interrupts` : ""}
          </span>
        </summary>
        <div class="chat-body">${turnsHtml || '<p class="hint">Empty transcript.</p>'}${timelineHtml}</div>
      `;

      delBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!sessionKey) return;
        if (!confirm(`Delete chat “${sessionKey}”? This cannot be undone.`)) return;
        delBtn.disabled = true;
        try {
          await api(`/admin/api/chats/${encodeURIComponent(sessionKey)}`, {
            method: "DELETE",
          });
          row.remove();
          const remaining = chatsList.querySelectorAll(".chat-card-row").length;
          if (!remaining) {
            chatsList.innerHTML = `<p class="hint">No previous chats for this patient yet.</p>`;
          }
          updateChatCount(remaining);
        } catch (err) {
          delBtn.disabled = false;
          alert(err.message || "Could not delete chat.");
        }
      });

      row.appendChild(delBtn);
      row.appendChild(card);
      chatsList.appendChild(row);
    }
    updateChatCount(chats.length);
  }

  function showChats(show) {
    chatsPanel.hidden = !show;
    const n = chatsList.querySelectorAll(".chat-card-row").length;
    toggleChatsBtn.textContent = show ? "Hide chats" : `Previous chats (${n})`;
    if (show) {
      chatsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  toggleChatsBtn.addEventListener("click", () => {
    showChats(chatsPanel.hidden);
  });
  closeChatsBtn.addEventListener("click", () => showChats(false));

  try {
    const data = await api(`/admin/api/patients/${encodeURIComponent(patientId)}`);
    const p = data.patient || {};
    document.title = `DBC Care — ${p.name || patientId}`;
    document.getElementById("patient-name").textContent = p.name || "Patient";
    const idLink = document.getElementById("patient-id-link");
    idLink.textContent = p.patient_id || patientId;
    idLink.href = patientUrl(p.patient_id || patientId);

    const chatCount = (data.chats || []).length;

    document.getElementById("patient-info").innerHTML = `
      <div><dt>Patient ID</dt><dd><a class="inline-link" href="${patientUrl(p.patient_id)}">${escapeHtml(p.patient_id)}</a></dd></div>
      <div><dt>Name</dt><dd>${escapeHtml(p.name)}</dd></div>
      <div><dt>Phone</dt><dd>${escapeHtml(p.phone)}</dd></div>
      <div><dt>Address</dt><dd>${escapeHtml(p.address)}</dd></div>
      <div><dt>Visits</dt><dd>${escapeHtml(data.visit_count)}</dd></div>
      <div><dt>Prescriptions</dt><dd>${escapeHtml(data.prescription_count)}</dd></div>
      <div><dt>Chats</dt><dd><a class="inline-link" href="#chats-panel" id="jump-chats">${escapeHtml(chatCount)}</a></dd></div>
    `;
    document.getElementById("jump-chats")?.addEventListener("click", (e) => {
      e.preventDefault();
      showChats(true);
    });

    renderChats(data.chats || []);

    // Deep-link: /admin/patient/PAT-0001?chats=1
    if (new URLSearchParams(window.location.search).get("chats") === "1") {
      showChats(true);
    }

    const visitsBody = document.getElementById("visits-tbody");
    const visits = data.visits || [];
    if (!visits.length) {
      visitsBody.innerHTML = `<tr><td colspan="5" class="hint">No visits on file.</td></tr>`;
    } else {
      visitsBody.innerHTML = visits
        .map((v) => {
          const depQ = encodeURIComponent(v.department || "");
          const docId = v.doctor_id || "";
          const docLabel = escapeHtml(v.doctor || v.doctor_id || "—");
          const docCell = docId
            ? `<a class="inline-link" href="${doctorUrl(docId)}">${docLabel}</a>`
            : docLabel;
          return `
            <tr>
              <td>${escapeHtml(v.appointment_id)}</td>
              <td>${escapeHtml(v.time)}</td>
              <td>${docCell}</td>
              <td><a class="inline-link" href="/admin/appointments?department=${depQ}">${escapeHtml(v.department || "—")}</a></td>
              <td><span class="badge ${v.status === "CANCELLED" ? "bad" : "ok"}">${escapeHtml(v.status || "")}</span></td>
            </tr>
          `;
        })
        .join("");
    }

    const rxBody = document.getElementById("rx-tbody");
    const rxs = data.prescriptions || [];
    if (!rxs.length) {
      rxBody.innerHTML = `<tr><td colspan="5" class="hint">No prescriptions on file.</td></tr>`;
    } else {
      rxBody.innerHTML = rxs
        .map((r) => {
          const depQ = encodeURIComponent(r.department || "");
          const docId = r.doctor_id || "";
          const docLabel = escapeHtml(r.doctor_name || r.doctor_id || "—");
          const docCell = docId
            ? `<a class="inline-link" href="${doctorUrl(docId)}">${docLabel}</a>`
            : docLabel;
          return `
            <tr>
              <td>${escapeHtml(r.prescription_id)}</td>
              <td>${escapeHtml(r.medicine_name)}</td>
              <td>${escapeHtml(r.timing)}</td>
              <td>${docCell}</td>
              <td><a class="inline-link" href="/admin/appointments?department=${depQ}">${escapeHtml(r.department || "—")}</a></td>
            </tr>
          `;
        })
        .join("");
    }
  } catch (err) {
    errEl.textContent = err.message || "Failed to load patient";
    errEl.hidden = false;
  }
})();
