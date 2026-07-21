(async function () {
  highlightNav("appointments");
  bindLogout();
  if (!(await requireAdminSession())) return;

  const depEl = document.getElementById("filter-department");
  const docEl = document.getElementById("filter-doctor");
  const listEl = document.getElementById("appt-list");
  const metaEl = document.getElementById("appt-meta");
  let doctorsCache = [];
  const urlParams = new URLSearchParams(window.location.search);

  function fillDoctors(department) {
    const dep = (department || "").toLowerCase();
    const filtered = dep
      ? doctorsCache.filter((d) => (d.department || "").toLowerCase().includes(dep))
      : doctorsCache;
    const prev = docEl.value;
    docEl.innerHTML = '<option value="">All doctors</option>';
    for (const d of filtered) {
      const o = document.createElement("option");
      o.value = d.doctor_id || "";
      o.textContent = `${d.name} · ${d.department || ""}`;
      docEl.appendChild(o);
    }
    if ([...docEl.options].some((o) => o.value === prev)) docEl.value = prev;
  }

  async function loadFilters() {
    const [deps, docs] = await Promise.all([
      api("/admin/api/departments"),
      api("/admin/api/doctors"),
    ]);
    doctorsCache = docs.doctors || [];
    depEl.innerHTML = '<option value="">All departments</option>';
    for (const name of deps.departments || []) {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      depEl.appendChild(o);
    }
    const preDep = urlParams.get("department") || "";
    if (preDep && [...depEl.options].some((o) => o.value === preDep)) {
      depEl.value = preDep;
    }
    fillDoctors(depEl.value);
    const preDoc = urlParams.get("doctor_id") || "";
    if (preDoc && [...docEl.options].some((o) => o.value === preDoc)) {
      docEl.value = preDoc;
    }
  }

  async function loadAppointments() {
    listEl.innerHTML = "<p class='hint'>Loading…</p>";
    const params = new URLSearchParams({ from_now: "true" });
    if (depEl.value) params.set("department", depEl.value);
    if (docEl.value) params.set("doctor_id", docEl.value);
    const next = params.toString();
    const path = `/admin/appointments${next ? `?${next}` : ""}`;
    if (`${window.location.pathname}${window.location.search}` !== path) {
      history.replaceState(null, "", path);
    }
    try {
      const data = await api(`/admin/api/appointments?${params}`);
      const rows = data.appointments || [];
      metaEl.textContent = `${data.count || 0} upcoming · as of ${data.as_of || "now"}`;
      if (!rows.length) {
        listEl.innerHTML = "<p class='hint'>No upcoming appointments for this filter.</p>";
        return;
      }
      listEl.innerHTML = "";
      for (const a of rows) {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "appt-card";
        card.innerHTML = `
          <div class="appt-time">${escapeHtml(a.time)}</div>
          <div class="appt-id">${escapeHtml(a.appointment_id)}</div>
          <div class="appt-patient">
            <a class="inline-link" href="${patientUrl(a.patient_id)}" target="_blank" rel="noopener">
              ${escapeHtml(a.patient_name || a.patient_id)}
            </a>
          </div>
          <div class="appt-meta-line">${
            a.doctor_id
              ? `<a class="inline-link" href="${doctorUrl(a.doctor_id)}">${escapeHtml(a.doctor || a.doctor_id)}</a>`
              : escapeHtml(a.doctor || "")
          }</div>
          <div class="appt-meta-line">${escapeHtml(a.department || "")} · ${escapeHtml(a.status || "")}</div>
        `;
        card.addEventListener("click", (e) => {
          if (e.target.closest("a")) return;
          openPatient(a.patient_id);
        });
        listEl.appendChild(card);
      }
    } catch (err) {
      listEl.innerHTML = `<p class="error">${escapeHtml(err.message)}</p>`;
    }
  }

  depEl.addEventListener("change", () => {
    fillDoctors(depEl.value);
    loadAppointments();
  });
  docEl.addEventListener("change", loadAppointments);
  document.getElementById("refresh-appts").addEventListener("click", loadAppointments);

  await loadFilters();
  await loadAppointments();
})();
