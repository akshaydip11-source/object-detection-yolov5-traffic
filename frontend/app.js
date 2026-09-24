/* SafeCityAI shared frontend utilities */
const API = "/api";

const SafeCity = {
  escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  },
  token() {
    return sessionStorage.getItem("sc_token") || "";
  },
  user() {
    try {
      return JSON.parse(sessionStorage.getItem("sc_user") || "null");
    } catch {
      return null;
    }
  },
  setAuth(token, user) {
    sessionStorage.setItem("sc_token", token);
    sessionStorage.setItem("sc_user", JSON.stringify(user));
  },
  logout() {
    sessionStorage.removeItem("sc_token");
    sessionStorage.removeItem("sc_user");
    window.location.href = "/login";
  },
  headers(json = true) {
    const h = {};
    if (json) h["Content-Type"] = "application/json";
    const t = this.token();
    if (t) h["Authorization"] = `Bearer ${t}`;
    return h;
  },
  async api(path, opts = {}) {
    const res = await fetch(`${API}${path}`, {
      ...opts,
      headers: { ...this.headers(!(opts.body instanceof FormData)), ...(opts.headers || {}) },
    });
    if (res.status === 401) {
      if (path !== "/auth/login") this.logout();
      const err = await res.json().catch(() => ({ detail: "Unauthorized" }));
      throw Object.assign(new Error(err.detail || "Unauthorized"), { status: 401 });
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw Object.assign(new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)), {
        status: res.status,
      });
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res;
  },
  toast(msg, type = "ok") {
    let wrap = document.querySelector(".toast-wrap");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "toast-wrap";
      document.body.appendChild(wrap);
    }
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    wrap.appendChild(el);
    setTimeout(() => el.remove(), 3800);
  },
  badgeStatus(s) {
    return `<span class="badge badge-${this.escape(s)}">${this.escape(s)}</span>`;
  },
  badgeSev(s) {
    return `<span class="badge badge-${this.escape(s)}">${this.escape(s)}</span>`;
  },
  fmtMs(ms) {
    if (ms == null) return "—";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  },
  fmtDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  },
  money(n) {
    return `₹${Number(n || 0).toLocaleString("en-IN")}`;
  },
  violationLabel(t) {
    return this.escape(t === "no_helmet" ? "No Helmet — Review" : t);
  },
  requireAuth(redirect = true) {
    if (!this.token()) {
      if (redirect) window.location.href = "/login";
      return false;
    }
    return true;
  },
  mountUserChip(el) {
    if (!el) return;
    const u = this.user();
    if (!u) {
      el.innerHTML = `<a class="btn btn-sm btn-secondary" href="/login">Sign in</a>`;
      return;
    }
    const initials = (u.full_name || u.email || "U")
      .split(/\s+/)
      .map((p) => p[0])
      .join("")
      .slice(0, 2)
      .toUpperCase();
    el.innerHTML = `
      <div class="user-chip">
        <div class="user-avatar">${this.escape(initials)}</div>
        <div>
          <div style="color:var(--text);font-weight:600;font-size:.85rem">${this.escape(u.full_name || "")}</div>
          <div style="font-size:.7rem">${this.escape(u.role)}</div>
        </div>
        <button class="btn btn-sm btn-ghost" id="logoutBtn" title="Logout">⎋</button>
      </div>`;
    el.querySelector("#logoutBtn")?.addEventListener("click", () => this.logout());
  },
  async download(path, filename) {
    const response = await this.api(path);
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
  async refreshMe() {
    if (!this.token()) return null;
    try {
      const me = await this.api("/auth/me");
      this.setAuth(this.token(), me);
      if (!["admin", "officer"].includes(me.role)) {
        document.querySelectorAll('a[href="/dashboard"], a[href="/violations"]').forEach(link => { link.hidden = true; });
      }
      return me;
    } catch {
      return null;
    }
  },
};

// Mobile nav
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-nav-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelector(".nav-links")?.classList.toggle("open");
      document.querySelector(".sidebar")?.classList.toggle("open");
    });
  });
  // active link highlight
  const path = location.pathname.replace(/\.html$/, "") || "/";
  document.querySelectorAll(".nav-links a, .side-nav a").forEach((a) => {
    const href = a.getAttribute("href")?.replace(/\.html$/, "") || "";
    if (href === path || (href !== "/" && path.startsWith(href))) {
      a.classList.add("active");
    }
  });
});

window.SafeCity = SafeCity;

// Display readiness on operational pages without promising a working detector.
document.addEventListener("DOMContentLoaded", async () => {
  if (location.pathname.replace(/\.html$/, "") !== "/live") return;
  try {
    const health = await SafeCity.api("/health");
    if (!health.model_loaded) SafeCity.toast("Model unavailable: provide trusted models/best.pt and restart the server.", "err");
  } catch { SafeCity.toast("API unavailable", "err"); }
});

// Server authorization is authoritative; this guard only improves navigation.
const protectedPage = location.pathname.replace(/\.html$/, "");
if (["/app", "/live", "/dashboard", "/violations"].includes(protectedPage)) {
  SafeCity.requireAuth();
  SafeCity.refreshMe().then(user => {
    if (!user) return;
    if (["/dashboard", "/violations"].includes(protectedPage) && !["officer", "admin"].includes(user.role)) {
      window.location.replace("/app");
    }
  });
}
