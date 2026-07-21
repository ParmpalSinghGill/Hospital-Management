(async function () {
  highlightNav("doctors");
  bindLogout();
  if (!(await requireAdminSession())) return;

  const depEl = document.getElementById("filter-department");
  const tbody = document.getElementById("doctor-tbody");
  const metaEl = document.getElementById("doctor-meta");
  let doctorsCache = [];

  async function loadFilters() {
    const deps = await api("/admin/api/departments");
    depEl.innerHTML = '<option value="">All departments</option>';
    for (const name of deps.departments || []) {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      depEl.appendChild(o);
    }
  }

  async function loadDoctors() {
    const params = new URLSearchParams();
    if (depEl.value) params.set("department", depEl.value);
    const data = await api(`/admin/api/doctors?${params}`);
    doctorsCache = data.doctors || [];
    metaEl.textContent = `${data.count || 0} doctors · clinic 9:00 AM–5:00 PM · lunch 2:00–3:00 PM`;
    if (!doctorsCache.length) {
      tbody.innerHTML = `<tr><td colspan="3" class="hint">No doctors match.</td></tr>`;
      return;
    }
    tbody.innerHTML = "";
    for (const d of doctorsCache) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><a class="inline-link" href="${doctorUrl(d.doctor_id)}">${escapeHtml(d.doctor_id)}</a></td>
        <td>${escapeHtml(d.name)}</td>
        <td>${escapeHtml(d.department || "—")}</td>
      `;
      tr.addEventListener("click", (e) => {
        if (e.target.closest("a")) return;
        window.location.href = doctorUrl(d.doctor_id);
      });
      tbody.appendChild(tr);
    }
  }

  depEl.addEventListener("change", () => loadDoctors().catch((err) => {
    metaEl.textContent = err.message || "Failed to load";
  }));
  document.getElementById("refresh-doctors").addEventListener("click", () => loadDoctors());

  try {
    await loadFilters();
    await loadDoctors();
  } catch (err) {
    metaEl.textContent = err.message || "Failed to load doctors";
  }
})();
