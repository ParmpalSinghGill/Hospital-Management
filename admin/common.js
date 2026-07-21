/** Shared admin helpers: API, auth guard, top nav. */

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

function patientUrl(patientId) {
  return `/admin/patient/${encodeURIComponent(patientId)}`;
}

function openPatient(patientId) {
  if (!patientId) return;
  window.open(patientUrl(patientId), "_blank", "noopener,noreferrer");
}

function doctorUrl(doctorId) {
  if (!doctorId) return "/admin/doctors";
  // Never embed a day — doctor schedule always opens on today.
  return `/admin/doctor/${encodeURIComponent(doctorId)}`;
}

function openDoctor(doctorId) {
  if (!doctorId) return;
  window.open(doctorUrl(doctorId), "_blank", "noopener,noreferrer");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlightNav(active) {
  document.querySelectorAll("[data-nav]").forEach((el) => {
    el.classList.toggle("is-active", el.getAttribute("data-nav") === active);
  });
}

async function requireAdminSession() {
  try {
    const session = await api("/admin/api/session");
    if (!session.authenticated) {
      window.location.replace("/admin/");
      return null;
    }
    const el = document.getElementById("admin-user");
    if (el) el.textContent = session.user || "Admin";
    return session;
  } catch {
    window.location.replace("/admin/");
    return null;
  }
}

async function logoutAdmin() {
  try {
    await api("/admin/api/logout", { method: "POST" });
  } catch {
    /* still leave */
  }
  window.location.replace("/admin/");
}

function bindLogout() {
  const btn = document.getElementById("logout-btn");
  if (btn) btn.addEventListener("click", logoutAdmin);
}
