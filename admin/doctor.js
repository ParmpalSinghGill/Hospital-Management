(async function () {
  highlightNav("doctors");
  bindLogout();
  if (!(await requireAdminSession())) return;

  const pathParts = window.location.pathname.split("/").filter(Boolean);
  let doctorId = "";
  const idx = pathParts.indexOf("doctor");
  if (idx >= 0 && pathParts[idx + 1]) {
    doctorId = decodeURIComponent(pathParts[idx + 1]);
  }
  if (!doctorId) {
    doctorId = new URLSearchParams(window.location.search).get("id") || "";
  }

  const errEl = document.getElementById("doctor-error");
  const dayEl = document.getElementById("schedule-day");
  const gridEl = document.getElementById("schedule-grid");
  const statsEl = document.getElementById("schedule-stats");
  const blocksEl = document.getElementById("blocks-list");
  const nameEl = document.getElementById("doctor-name");

  if (!doctorId) {
    errEl.textContent = "Missing doctor id.";
    errEl.hidden = false;
    return;
  }

  const today = new Date();
  const todayStr = [
    today.getFullYear(),
    String(today.getMonth() + 1).padStart(2, "0"),
    String(today.getDate()).padStart(2, "0"),
  ].join("-");
  // Always open on today. Stale ?day= from a previous visit (replaceState)
  // used to reopen yesterday/older dates when clicking a doctor link again.
  dayEl.value = todayStr;
  if (window.location.search) {
    history.replaceState(null, "", `/admin/doctor/${encodeURIComponent(doctorId)}`);
  }

  function shiftDay(delta) {
    const raw = dayEl.value || todayStr;
    const parts = raw.split("-").map(Number);
    if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return;
    const d = new Date(parts[0], parts[1] - 1, parts[2]);
    d.setDate(d.getDate() + delta);
    dayEl.value = [
      d.getFullYear(),
      String(d.getMonth() + 1).padStart(2, "0"),
      String(d.getDate()).padStart(2, "0"),
    ].join("-");
    loadSchedule().catch((err) => {
      errEl.textContent = err.message || "Failed to load";
      errEl.hidden = false;
    });
  }

  function addMinutes(hm, minutes) {
    const [h, m] = hm.split(":").map(Number);
    const total = h * 60 + m + minutes;
    const nh = Math.floor(total / 60);
    const nm = total % 60;
    return `${String(nh).padStart(2, "0")}:${String(nm).padStart(2, "0")}`;
  }

  function renderBlocks(blocks, day) {
    const list = blocks || [];
    if (!list.length) {
      blocksEl.innerHTML = `<p class="hint">No unavailable blocks.</p>`;
      return;
    }
    blocksEl.innerHTML = `
      <ul class="block-ul">
        ${list
          .map((b) => {
            const scope = b.day ? `on ${escapeHtml(b.day)}` : "every day";
            return `<li>
              <strong>${escapeHtml(b.start_hm)}–${escapeHtml(b.end_hm)}</strong>
              · ${escapeHtml(b.reason || "unavailable")} · ${scope}
              <button type="button" class="ghost tiny" data-block="${escapeHtml(b.block_id)}">Clear</button>
            </li>`;
          })
          .join("")}
      </ul>
      <p class="hint tight">Clearing the recurring lunch block removes the default 2–3 PM break for this doctor.</p>
    `;
    blocksEl.querySelectorAll("button[data-block]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const blockId = btn.getAttribute("data-block");
        if (!confirm(`Clear unavailable block ${blockId}?`)) return;
        try {
          await api(
            `/admin/api/doctors/${encodeURIComponent(doctorId)}/unavailable/${encodeURIComponent(blockId)}`,
            { method: "DELETE" }
          );
          await loadSchedule();
        } catch (err) {
          alert(err.message || "Could not clear block");
        }
      });
    });
  }

  function renderGrid(data) {
    const hours = data.hours || [];
    const minutes = data.minutes || [];
    const grid = data.grid || [];
    let html = `<div class="sched-row sched-head"><div class="sched-label">Hour</div>`;
    for (const m of minutes) {
      html += `<div class="sched-colhead">:${String(m).padStart(2, "0")}</div>`;
    }
    html += `</div>`;

    for (let r = 0; r < hours.length; r++) {
      html += `<div class="sched-row"><div class="sched-label">${hours[r]}:00</div>`;
      const row = grid[r] || [];
      for (let c = 0; c < minutes.length; c++) {
        const cell = row[c];
        if (!cell) {
          html += `<div class="sched-cell empty">—</div>`;
          continue;
        }
        const status = cell.status || "free";
        const past = !!cell.past;
        let label = cell.hm || "";
        let title = `${cell.time} · ${status}${past ? " · passed" : ""}`;
        if (status === "booked" && cell.appointment) {
          label = (cell.appointment.patient_name || cell.appointment.patient_id || "Booked").split(" ")[0];
          title = `${cell.time} · ${cell.appointment.patient_name || ""} · ${cell.appointment.appointment_id || ""}${past ? " · passed" : ""}`;
        } else if (status === "unavailable") {
          label = (cell.unavailable && cell.unavailable.reason) || "off";
          title = `${cell.time} · unavailable (${(cell.unavailable && cell.unavailable.reason) || ""})${past ? " · passed" : ""}`;
        } else if (past) {
          label = "passed";
        }
        const pid = cell.appointment && cell.appointment.patient_id
          ? encodeURIComponent(cell.appointment.patient_id)
          : "";
        const blockId = cell.unavailable && cell.unavailable.block_id
          ? encodeURIComponent(cell.unavailable.block_id)
          : "";
        const classes = ["sched-cell", status];
        if (past) classes.push("past");
        const disabled = past && status !== "booked";
        html += `<button type="button" class="${escapeHtml(classes.join(" "))}"
          data-time="${escapeHtml(cell.time || "")}"
          data-hm="${escapeHtml(cell.hm || "")}"
          data-status="${escapeHtml(status)}"
          data-past="${past ? "1" : "0"}"
          data-patient="${pid}"
          data-block="${blockId}"
          ${disabled ? "disabled" : ""}
          title="${escapeHtml(title)}">${escapeHtml(label)}</button>`;
      }
      html += `</div>`;
    }
    gridEl.innerHTML = html;

    gridEl.querySelectorAll(".sched-cell").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (btn.disabled) return;
        const status = btn.getAttribute("data-status");
        const past = btn.getAttribute("data-past") === "1";
        const hm = btn.getAttribute("data-hm");
        const patientId = btn.getAttribute("data-patient");
        const blockId = btn.getAttribute("data-block");
        if (status === "booked" && patientId) {
          openPatient(decodeURIComponent(patientId));
          return;
        }
        if (past) return;
        if (status === "free") {
          const end = addMinutes(hm, 10);
          if (!confirm(`Mark ${hm}–${end} unavailable on ${dayEl.value}?`)) return;
          try {
            await api(`/admin/api/doctors/${encodeURIComponent(doctorId)}/unavailable`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                day: dayEl.value,
                start_hm: hm,
                end_hm: end,
                reason: "unavailable",
              }),
            });
            await loadSchedule();
          } catch (err) {
            alert(err.message || "Could not mark unavailable");
          }
          return;
        }
        if (status === "unavailable" && blockId) {
          const decoded = decodeURIComponent(blockId);
          if (!confirm(`Clear unavailable block ${decoded}?`)) return;
          try {
            await api(
              `/admin/api/doctors/${encodeURIComponent(doctorId)}/unavailable/${encodeURIComponent(decoded)}`,
              { method: "DELETE" }
            );
            await loadSchedule();
          } catch (err) {
            alert(err.message || "Could not clear block");
          }
        }
      });
    });
  }

  async function loadSchedule() {
    errEl.hidden = true;
    const day = dayEl.value || todayStr;
    const path = `/admin/doctor/${encodeURIComponent(doctorId)}?day=${encodeURIComponent(day)}`;
    if (`${window.location.pathname}${window.location.search}` !== path) {
      history.replaceState(null, "", path);
    }
    const data = await api(
      `/admin/api/doctors/${encodeURIComponent(doctorId)}/schedule?day=${encodeURIComponent(day)}`
    );
    const doc = data.doctor || {};
    nameEl.textContent = doc.name || doctorId;
    document.getElementById("doctor-title").textContent = doc.name || "Doctor schedule";
    document.title = `DBC Care — ${doc.name || doctorId}`;
    const st = data.stats || {};
    statsEl.textContent =
      `${st.booked || 0} booked · ${st.free || 0} free · ${st.unavailable || 0} unavailable · ` +
      `${st.past || 0} passed · fill ${st.fill_pct || 0}% of remaining bookable · ${doc.department || ""}`;
    renderGrid(data);
    renderBlocks(data.blocks || [], day);
  }

  document.getElementById("refresh-schedule").addEventListener("click", () => {
    loadSchedule().catch((err) => {
      errEl.textContent = err.message || "Failed to load";
      errEl.hidden = false;
    });
  });
  document.getElementById("day-prev").addEventListener("click", () => shiftDay(-1));
  document.getElementById("day-next").addEventListener("click", () => shiftDay(1));
  dayEl.addEventListener("change", () => {
    loadSchedule().catch((err) => {
      errEl.textContent = err.message || "Failed to load";
      errEl.hidden = false;
    });
  });

  document.getElementById("unavailable-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const start = document.getElementById("ua-start").value;
    const end = document.getElementById("ua-end").value;
    const reason = document.getElementById("ua-reason").value || "unavailable";
    try {
      await api(`/admin/api/doctors/${encodeURIComponent(doctorId)}/unavailable`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          day: dayEl.value,
          start_hm: start,
          end_hm: end,
          reason,
        }),
      });
      await loadSchedule();
    } catch (err) {
      alert(err.message || "Could not set unavailable");
    }
  });

  try {
    await loadSchedule();
  } catch (err) {
    errEl.textContent = err.message || "Failed to load schedule";
    errEl.hidden = false;
  }
})();
