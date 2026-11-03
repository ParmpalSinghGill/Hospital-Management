(async function () {
  highlightNav("appointments");
  bindLogout();
  if (!(await requireAdminSession())) return;

  const depEl = document.getElementById("filter-department");
  const docEl = document.getElementById("filter-doctor");
  const listEl = document.getElementById("appt-list");
  const metaEl = document.getElementById("appt-meta");
  const rangeEl = document.getElementById("filter-range");
  const sortBtn = document.getElementById("sort-time");
  const pagerEl = document.getElementById("appt-pager");
  const pageInfoEl = document.getElementById("page-info");
  const prevBtn = document.getElementById("page-prev");
  const nextBtn = document.getElementById("page-next");
  const delDateEl = document.getElementById("delete-before-date");
  const delBtn = document.getElementById("delete-before-btn");
  const PAGE_SIZE = 50;
  let doctorsCache = [];
  const urlParams = new URLSearchParams(window.location.search);
  let sortDir = urlParams.get("sort") === "desc" ? "desc" : "asc";
  let page = Math.max(1, parseInt(urlParams.get("page") || "1", 10) || 1);
  let pages = 1;
  if (urlParams.get("range") === "all") rangeEl.value = "all";

  function todayLocal() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  }
  delDateEl.value = todayLocal();

  function renderSortButton() {
    sortBtn.textContent = sortDir === "asc" ? "Time ↑ oldest first" : "Time ↓ newest first";
  }
  renderSortButton();

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
    const upcoming = rangeEl.value !== "all";
    // Page URL keeps only the user's choices; the API also gets paging/sort.
    const view = new URLSearchParams();
    if (depEl.value) view.set("department", depEl.value);
    if (docEl.value) view.set("doctor_id", docEl.value);
    if (!upcoming) view.set("range", "all");
    if (sortDir !== "asc") view.set("sort", sortDir);
    if (page > 1) view.set("page", String(page));
    const next = view.toString();
    const path = `/admin/appointments${next ? `?${next}` : ""}`;
    if (`${window.location.pathname}${window.location.search}` !== path) {
      history.replaceState(null, "", path);
    }
    const params = new URLSearchParams({
      from_now: upcoming ? "true" : "false",
      sort: sortDir,
      page: String(page),
      page_size: String(PAGE_SIZE),
    });
    if (depEl.value) params.set("department", depEl.value);
    if (docEl.value) params.set("doctor_id", docEl.value);
    try {
      const data = await api(`/admin/api/appointments?${params}`);
      const rows = data.appointments || [];
      page = data.page || 1;
      pages = data.pages || 1;
      const total = data.count || 0;
      const first = total ? (page - 1) * PAGE_SIZE + 1 : 0;
      const last = Math.min(total, page * PAGE_SIZE);
      metaEl.textContent =
        `${total} ${upcoming ? "upcoming" : "total"} · showing ${first}–${last}` +
        ` · as of ${data.as_of || "now"}`;
      pagerEl.hidden = pages <= 1;
      pageInfoEl.textContent = `Page ${page} of ${pages}`;
      prevBtn.disabled = page <= 1;
      nextBtn.disabled = page >= pages;
      if (!rows.length) {
        listEl.innerHTML = `<p class='hint'>No ${upcoming ? "upcoming " : ""}appointments for this filter.</p>`;
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

  // Any filter change starts again from page 1.
  function reload() {
    page = 1;
    loadAppointments();
  }

  depEl.addEventListener("change", () => {
    fillDoctors(depEl.value);
    reload();
  });
  docEl.addEventListener("change", reload);
  rangeEl.addEventListener("change", reload);
  document.getElementById("refresh-appts").addEventListener("click", loadAppointments);

  sortBtn.addEventListener("click", () => {
    sortDir = sortDir === "asc" ? "desc" : "asc";
    renderSortButton();
    reload();
  });
  prevBtn.addEventListener("click", () => {
    if (page > 1) { page -= 1; loadAppointments(); listEl.scrollIntoView({ block: "start" }); }
  });
  nextBtn.addEventListener("click", () => {
    if (page < pages) { page += 1; loadAppointments(); listEl.scrollIntoView({ block: "start" }); }
  });

  delBtn.addEventListener("click", async () => {
    const before = delDateEl.value;
    if (!before) {
      alert("Pick a date first.");
      return;
    }
    delBtn.disabled = true;
    try {
      // Count first so the confirm says exactly what will go.
      const preview = await api("/admin/api/appointments/delete-before", {
        method: "POST",
        body: JSON.stringify({ before, dry_run: true }),
      });
      if (!preview.count) {
        alert(`No appointments before ${before}.`);
        return;
      }
      const ok = confirm(
        `Delete ${preview.count} appointment(s) dated before ${before}?\n` +
          `Appointments on ${before} and later are kept. This cannot be undone.`
      );
      if (!ok) return;
      const res = await api("/admin/api/appointments/delete-before", {
        method: "POST",
        body: JSON.stringify({ before }),
      });
      page = 1;
      await loadAppointments();
      metaEl.textContent = `Deleted ${res.count} appointment(s) before ${before} · ${metaEl.textContent}`;
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    } finally {
      delBtn.disabled = false;
    }
  });

  await loadFilters();
  await loadAppointments();
})();
