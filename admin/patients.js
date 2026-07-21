(async function () {
  highlightNav("patients");
  bindLogout();
  if (!(await requireAdminSession())) return;

  const searchEl = document.getElementById("patient-search");
  const tbody = document.getElementById("patient-tbody");
  const metaEl = document.getElementById("patient-meta");
  let timer = null;

  async function loadPatients() {
    tbody.innerHTML = `<tr><td colspan="4" class="hint">Loading…</td></tr>`;
    const q = (searchEl.value || "").trim();
    const params = q ? `?q=${encodeURIComponent(q)}` : "";
    try {
      const data = await api(`/admin/api/patients${params}`);
      const rows = data.patients || [];
      metaEl.textContent = `${data.count || 0} patient${data.count === 1 ? "" : "s"}`;
      if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="hint">No patients match.</td></tr>`;
        return;
      }
      tbody.innerHTML = "";
      for (const p of rows) {
        const tr = document.createElement("tr");
        tr.className = "click-row";
        tr.innerHTML = `
          <td><a class="inline-link" href="${patientUrl(p.patient_id)}" target="_blank" rel="noopener">${escapeHtml(p.patient_id)}</a></td>
          <td>${escapeHtml(p.name)}</td>
          <td>${escapeHtml(p.phone)}</td>
          <td>${escapeHtml(p.address)}</td>
        `;
        tr.addEventListener("click", (e) => {
          if (e.target.closest("a")) return;
          openPatient(p.patient_id);
        });
        tbody.appendChild(tr);
      }
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="4" class="error">${escapeHtml(err.message)}</td></tr>`;
    }
  }

  searchEl.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(loadPatients, 220);
  });
  document.getElementById("refresh-patients").addEventListener("click", loadPatients);

  await loadPatients();
})();
