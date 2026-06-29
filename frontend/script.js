/* Craival — client logic v5 (custom dropdowns, mobile-optimized, seamless nav) */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const API = window.API_BASE || "";

  const splash = $("splash-screen");
  const homeScreen = $("home-screen");
  const resultsScreen = $("results-screen");
  const detailScreen = $("detail-screen");
  const detailContent = $("detail-content");
  const emptyScreen = $("empty-screen");
  const grid = $("results-grid");
  const refineBar = $("refine-bar");
  const homeError = $("home-error");

  let sessionId = null;
  let currentRecs = [];

  // Filter state
  const filters = { location: "", budget: "", cuisine: "", rating: "0" };

  // ---- Cycling placeholder ----
  const placeholders = [
    "Try: cheap Italian near Koramangala...",
    "Try: rooftop dining for a date night...",
    "Try: family-friendly under ₹500 in BTM...",
    "Try: best rated Chinese in Indiranagar...",
  ];
  let phIdx = 0;
  setInterval(() => {
    phIdx = (phIdx + 1) % placeholders.length;
    const el = $("search-input");
    if (el && document.activeElement !== el) el.placeholder = placeholders[phIdx];
  }, 3500);

  // ---- Helpers ----
  const CUISINE_EMOJI = [
    ["pizza|italian", "🍕"], ["chinese|asian|thai|momos", "🥡"],
    ["biryani|north indian|mughlai|hyderabadi", "🍛"], ["south indian|dosa|idli|kerala", "🥘"],
    ["burger|american|fast food", "🍔"], ["dessert|bakery|cafe|ice cream", "🧁"],
    ["sushi|japanese", "🍣"], ["seafood|mangalorean|coastal", "🦐"],
    ["bbq|grill|kebab|steak", "🍢"], ["beverages|juice|tea|coffee", "☕"],
    ["pasta|continental|european|mediterranean", "🍝"], ["healthy|salad", "🥗"],
  ];
  const emojiFor = (c) => {
    const s = (c || "").toLowerCase();
    for (const [k, e] of CUISINE_EMOJI) if (k.split("|").some((x) => s.includes(x))) return e;
    return "🍽️";
  };
  const stars = (rate) => {
    if (rate == null) return '<span class="empty">Unrated</span>';
    const full = Math.floor(rate), half = rate - full >= 0.5;
    let s = "★".repeat(full); if (half) s += "⯨";
    const empty = 5 - full - (half ? 1 : 0);
    return `${s}<span class="empty">${"★".repeat(Math.max(0, empty))}</span>`;
  };
  const esc = (t) => String(t == null ? "" : t).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const titleCase = (t) => String(t || "").replace(/\w\S*/g, (w) => w.charAt(0).toUpperCase() + w.slice(1));

  // ============================================================
  //  CUSTOM DROPDOWN COMPONENT
  // ============================================================
  const openDropdowns = [];
  function closeAllDropdowns(except) {
    openDropdowns.forEach((d) => { if (d !== except) d.close(); });
  }

  function buildDropdown(cfg) {
    // cfg: { el, label, icon, iconColor, placeholder, options:[{value,label}], searchable, onChange }
    const el = cfg.el;
    let options = cfg.options || [];
    let value = "";

    el.innerHTML = `
      <span class="cdrop-label">${esc(cfg.label)}</span>
      <button type="button" class="cdrop-btn placeholder">
        <span class="material-symbols-outlined text-[18px] ${cfg.iconColor || "text-brand"}">${cfg.icon}</span>
        <span class="cdrop-value">${esc(cfg.placeholder)}</span>
        <span class="material-symbols-outlined cdrop-chevron">expand_more</span>
      </button>
      <div class="cdrop-panel hidden">
        ${cfg.searchable ? '<input class="cdrop-search" type="text" placeholder="Search..."/>' : ""}
        <ul class="cdrop-list"></ul>
      </div>`;

    const btn = el.querySelector(".cdrop-btn");
    const valueEl = el.querySelector(".cdrop-value");
    const panel = el.querySelector(".cdrop-panel");
    const list = el.querySelector(".cdrop-list");
    const search = el.querySelector(".cdrop-search");

    function renderList(filterText) {
      const q = (filterText || "").toLowerCase();
      const shown = options.filter((o) => !q || o.label.toLowerCase().includes(q));
      if (!shown.length) { list.innerHTML = '<li class="cdrop-empty">No matches</li>'; return; }
      list.innerHTML = shown.map((o) =>
        `<li class="cdrop-opt ${o.value === value ? "active" : ""}" data-value="${esc(o.value)}">${esc(o.label)}</li>`).join("");
      list.querySelectorAll(".cdrop-opt").forEach((li) =>
        li.addEventListener("click", () => select(li.dataset.value)));
    }

    function select(v) {
      value = v;
      const opt = options.find((o) => o.value === v);
      valueEl.textContent = opt ? opt.label : cfg.placeholder;
      btn.classList.toggle("placeholder", !v);
      api.close();
      if (cfg.onChange) cfg.onChange(v);
    }

    const api = {
      open() {
        closeAllDropdowns(api);
        panel.classList.remove("hidden");
        btn.classList.add("open");
        renderList("");
        if (search) { search.value = ""; setTimeout(() => search.focus(), 30); }
      },
      close() { panel.classList.add("hidden"); btn.classList.remove("open"); },
      isOpen() { return !panel.classList.contains("hidden"); },
      setOptions(opts) { options = opts; },
      reset() { select(""); },
      get value() { return value; },
    };

    btn.addEventListener("click", (e) => { e.stopPropagation(); api.isOpen() ? api.close() : api.open(); });
    if (search) {
      search.addEventListener("input", () => renderList(search.value));
      search.addEventListener("click", (e) => e.stopPropagation());
      search.addEventListener("keydown", (e) => { if (e.key === "Enter") { const first = list.querySelector(".cdrop-opt"); if (first) select(first.dataset.value); } });
    }
    panel.addEventListener("click", (e) => e.stopPropagation());

    openDropdowns.push(api);
    return api;
  }

  document.addEventListener("click", () => closeAllDropdowns(null));

  // Build the four filters
  const ddLocation = buildDropdown({
    el: $("cd-location"), label: "Location", icon: "location_on", placeholder: "– Select –",
    searchable: true, options: [], onChange: (v) => { filters.location = v; },
  });
  const ddBudget = buildDropdown({
    el: $("cd-budget"), label: "Budget", icon: "payments", placeholder: "Any budget",
    options: [{ value: "", label: "Any budget" }, { value: "low", label: "Low (≤₹300)" },
      { value: "medium", label: "Medium (≤₹800)" }, { value: "high", label: "High (₹800+)" }],
    onChange: (v) => { filters.budget = v; },
  });
  const ddCuisine = buildDropdown({
    el: $("cd-cuisine"), label: "Cuisine", icon: "restaurant", placeholder: "Any cuisine",
    searchable: true, options: [], onChange: (v) => { filters.cuisine = v; },
  });
  const ddRating = buildDropdown({
    el: $("cd-rating"), label: "Min rating", icon: "star", iconColor: "text-gold", placeholder: "Any rating",
    options: [{ value: "0", label: "Any rating" }, { value: "3.5", label: "3.5+ ★" },
      { value: "4", label: "4.0+ ★" }, { value: "4.5", label: "4.5+ ★" }],
    onChange: (v) => { filters.rating = v; },
  });

  // ============================================================
  //  VIEW SWITCHING
  // ============================================================
  function showBase(screen) {
    [homeScreen, resultsScreen, emptyScreen].forEach((s) => s.classList.add("hidden"));
    detailScreen.classList.add("hidden");
    screen.classList.remove("hidden");
    window.scrollTo(0, 0);
  }
  const showHome = () => { showBase(homeScreen); refineBar.classList.add("hidden"); };
  const showResults = () => { showBase(resultsScreen); refineBar.classList.remove("hidden"); };
  const showEmpty = () => { showBase(emptyScreen); refineBar.classList.remove("hidden"); };

  function renderSkeletons(n = 3) {
    showResults();
    $("result-meta").innerHTML = '<span class="pulse-text">✨ Our AI is curating your best matches...</span>';
    $("relaxed-badge").classList.add("hidden");
    grid.innerHTML = "";
    for (let i = 0; i < n; i++) {
      const c = document.createElement("div");
      c.className = "glass-panel rounded-xl overflow-hidden flex flex-col h-[360px]";
      c.innerHTML = `<div class="h-28 w-full skeleton"></div>
        <div class="p-4 flex-grow flex flex-col gap-3">
          <div class="h-6 w-2/3 rounded skeleton"></div>
          <div class="flex gap-2"><div class="h-5 w-16 rounded-full skeleton"></div><div class="h-5 w-20 rounded-full skeleton"></div></div>
          <div class="mt-auto h-16 w-full rounded skeleton"></div>
        </div>`;
      grid.appendChild(c);
    }
  }

  // ============================================================
  //  RESULTS
  // ============================================================
  function cardHtml(r, i) {
    const cuisines = titleCase(r.cuisines) || "Restaurant";
    const loc = titleCase(r.location);
    const cost = r.approx_cost != null ? `₹${r.approx_cost} for two` : "Cost N/A";
    const badges = [];
    if (r.online_order) badges.push("Online Order");
    if (r.book_table) badges.push("Book Table");
    const badgeHtml = badges.map((b) => `<span class="feat-badge">${b}</span>`).join("");
    return `
      <article class="rcard glass-panel rounded-xl overflow-hidden flex flex-col group" data-idx="${i}" style="animation-delay:${i * 0.08}s">
        <div class="card-banner" style="background:radial-gradient(circle at 50% 30%, rgba(226,55,68,0.28), rgba(19,19,19,0.85));">
          <span style="position:relative;z-index:1;">${emojiFor(r.cuisines)}</span>
          <span class="glass-panel" style="position:absolute;top:.75rem;left:.75rem;z-index:2;padding:.1rem .6rem;border-radius:9999px;font-size:.7rem;font-weight:700;">#${r.rank}</span>
        </div>
        <div class="p-4 flex flex-col flex-grow gap-3">
          <div>
            <h3 class="font-head text-xl font-semibold group-hover:text-brand transition">${esc(r.name)}</h3>
            <p class="text-sm muted">${esc(cuisines)}${loc ? " • " + esc(loc) : ""}</p>
          </div>
          <div class="flex items-center justify-between text-sm">
            <span class="stars">${stars(r.rate)}</span>
            <span class="muted">${esc(cost)}</span>
          </div>
          ${badgeHtml ? `<div class="flex flex-wrap gap-2">${badgeHtml}</div>` : ""}
          <div class="ai-insight rounded-lg p-3 mt-auto relative">
            <span class="material-symbols-outlined text-gold" style="position:absolute;top:.5rem;right:.5rem;font-size:20px;opacity:.4;">auto_awesome</span>
            <p class="text-sm italic">${esc(r.explanation)}</p>
          </div>
          <div class="flex justify-between items-center pt-2 divider">
            <span class="text-xs muted">Tap card for details</span>
            <div class="flex gap-2">
              <button class="fb-btn p-1.5 rounded-full hover:bg-black/10 muted hover:text-success transition" data-name="${esc(r.name)}" data-up="1"><span class="material-symbols-outlined text-[20px]">thumb_up</span></button>
              <button class="fb-btn p-1.5 rounded-full hover:bg-black/10 muted hover:text-brand transition" data-name="${esc(r.name)}" data-up="0"><span class="material-symbols-outlined text-[20px]">thumb_down</span></button>
            </div>
          </div>
        </div>
      </article>`;
  }

  const SOURCE = {
    llm: ["AI Ranked", "auto_awesome", "text-gold"],
    fallback: ["Popularity Based", "trending_up", "text-success"],
    cache: ["Cached", "bolt", "text-brand"], cached: ["Cached", "bolt", "text-brand"],
  };

  function renderResults(data) {
    currentRecs = data.recommendations || [];
    if (currentRecs.length === 0) { showEmpty(); return; }
    showResults();
    grid.innerHTML = currentRecs.map((r, i) => cardHtml(r, i)).join("");
    $("result-meta").textContent = `${data.count} result${data.count === 1 ? "" : "s"} · ${data.processing_time_ms} ms`;
    const [label, icon, color] = SOURCE[data.source] || SOURCE.llm;
    $("source-text").textContent = label;
    $("source-icon").textContent = icon;
    $("source-badge").className = "glass-panel px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1 " + color;
    const relaxed = data.filters_relaxed || [];
    const rb = $("relaxed-badge");
    if (relaxed.length) {
      $("relaxed-text").textContent = "We expanded your search — relaxed: " + relaxed.join(", ");
      rb.classList.remove("hidden"); rb.classList.add("flex");
    } else { rb.classList.add("hidden"); rb.classList.remove("flex"); }
    grid.querySelectorAll(".rcard").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.closest(".fb-btn")) return;
        openDetail(currentRecs[parseInt(card.dataset.idx, 10)]);
      });
    });
    bindFeedback();
  }

  // ============================================================
  //  DETAIL (overlay → instant back)
  // ============================================================
  function openDetail(r) {
    const cuisines = titleCase(r.cuisines) || "Restaurant";
    const loc = titleCase(r.location);
    const cost = r.approx_cost != null ? `₹${r.approx_cost} for two` : "Cost not available";
    const dishes = (r.dish_liked || "").split(",").map((d) => d.trim()).filter(Boolean).slice(0, 8);
    const badges = [];
    if (r.online_order) badges.push(["delivery_dining", "Online Order"]);
    if (r.book_table) badges.push(["restaurant_menu", "Book Table"]);
    if (r.rest_type) badges.push(["storefront", titleCase(r.rest_type)]);
    detailContent.innerHTML = `
      <div class="slide-in-right max-w-4xl mx-auto min-h-screen">
        <div class="detail-hero p-6 md:p-10">
          <span class="hero-emoji">${emojiFor(r.cuisines)}</span>
          <div class="relative z-10">
            <h1 class="font-head font-bold text-3xl md:text-5xl mb-2">${esc(r.name)}</h1>
            <p class="muted text-base md:text-lg">${esc(cuisines)}${loc ? " • " + esc(loc) : ""}</p>
            <div class="flex items-center gap-3 mt-3 flex-wrap text-sm md:text-base">
              <span class="stars">${stars(r.rate)} <span style="color:#fff;" class="text-sm">${r.rate != null ? r.rate : ""}</span></span>
              <span class="muted">·</span><span class="muted">${esc(cost)}</span>
              ${r.votes ? `<span class="muted">· 👍 ${r.votes} votes</span>` : ""}
            </div>
          </div>
        </div>
        <div class="p-6 md:p-10 flex flex-col gap-8 pb-24">
          ${badges.length ? `<div class="flex flex-wrap gap-3">${badges.map(([ic, t]) =>
            `<div class="glass-panel rounded-full px-4 py-2 flex items-center gap-2"><span class="material-symbols-outlined text-brand text-[18px]">${ic}</span><span class="text-sm">${esc(t)}</span></div>`).join("")}</div>` : ""}
          <section class="glass-panel rounded-xl p-6 relative overflow-hidden">
            <div style="position:absolute;top:-2rem;right:-2rem;width:8rem;height:8rem;background:rgba(226,55,68,.2);border-radius:9999px;filter:blur(40px);"></div>
            <div class="flex items-center gap-2 mb-3 relative z-10"><span class="material-symbols-outlined text-gold">auto_awesome</span><h2 class="font-head font-semibold text-xl md:text-2xl">Why it's a match for you</h2></div>
            <p class="muted leading-relaxed relative z-10">${esc(r.explanation)}</p>
          </section>
          ${dishes.length ? `<section><h3 class="font-head font-semibold text-lg md:text-xl mb-3">Popular dishes</h3>
            <div class="flex flex-wrap gap-3">${dishes.map((d) => `<span class="dish-chip">🍴 ${esc(titleCase(d))}</span>`).join("")}</div></section>` : ""}
          <section class="glass-panel rounded-xl p-6 flex flex-col items-center text-center gap-3">
            <h3 class="font-head font-semibold text-lg">How's this pick?</h3>
            <div class="flex gap-5">
              <button class="fb-btn w-12 h-12 rounded-full glass-panel flex items-center justify-center hover:text-success transition" data-name="${esc(r.name)}" data-up="1"><span class="material-symbols-outlined">thumb_up</span></button>
              <button class="fb-btn w-12 h-12 rounded-full glass-panel flex items-center justify-center hover:text-brand transition" data-name="${esc(r.name)}" data-up="0"><span class="material-symbols-outlined">thumb_down</span></button>
            </div>
          </section>
        </div>
      </div>`;
    detailScreen.classList.remove("hidden");
    document.body.classList.add("detail-open");
    detailScreen.scrollTop = 0;
    window.scrollTo(0, 0);
    bindFeedback();
  }
  function closeDetail() { detailScreen.classList.add("hidden"); document.body.classList.remove("detail-open"); window.scrollTo(0, 0); }

  // ============================================================
  //  API + SEARCH
  // ============================================================
  async function recommend(payload) {
    renderSkeletons();
    try {
      const res = await fetch(`${API}/api/recommend`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, session_id: sessionId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "request failed");
      sessionId = data.session_id || sessionId;
      renderResults(data);
    } catch (err) { console.error(err); showEmpty(); }
  }
  function runSearch() {
    homeError.textContent = "";
    const query = $("search-input").value.trim();
    if (query) { recommend({ query }); return; }
    if (!filters.location) { homeError.textContent = "Type a search above, or pick a Location."; return; }
    recommend({
      location: filters.location, cuisine: filters.cuisine || null,
      budget_tier: filters.budget || null, min_rating: parseFloat(filters.rating) || 0,
    });
  }

  // ---- Feedback ----
  function bindFeedback() {
    document.querySelectorAll(".fb-btn").forEach((btn) => {
      if (btn._bound) return; btn._bound = true;
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const up = btn.dataset.up === "1";
        btn.classList.add(up ? "text-success" : "text-brand");
        try {
          await fetch(`${API}/api/feedback`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ restaurant: btn.dataset.name, thumbs_up: up, session_id: sessionId }),
          });
        } catch (_) {}
        showToast();
      });
    });
  }
  function showToast() {
    const t = $("toast"); t.classList.remove("hidden");
    clearTimeout(showToast._t); showToast._t = setTimeout(() => t.classList.add("hidden"), 1600);
  }

  // ---- Refine sheet ----
  const overlay = $("refine-overlay"), sheet = $("refine-sheet");
  function openRefine() { overlay.classList.remove("hidden"); requestAnimationFrame(() => sheet.classList.remove("translate-y-full")); setTimeout(() => $("refine-input").focus(), 250); }
  function closeRefine() { sheet.classList.add("translate-y-full"); setTimeout(() => overlay.classList.add("hidden"), 300); }

  // ---- Theme ----
  function applyTheme(dark) {
    document.documentElement.classList.toggle("dark", dark);
    $("theme-icon").textContent = dark ? "light_mode" : "dark_mode";
    try { localStorage.setItem("craival-theme", dark ? "dark" : "light"); } catch (_) {}
  }

  // ---- Events ----
  function enterApp() {
    splash.classList.add("hide");
    setTimeout(() => { splash.style.display = "none"; }, 500);
    showHome();
    try { sessionStorage.setItem("craival-splash-seen", "1"); } catch (_) {}
  }
  $("get-started").addEventListener("click", enterApp);
  $("search-btn").addEventListener("click", runSearch);
  $("search-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
  document.querySelectorAll(".example-chip").forEach((chip) =>
    chip.addEventListener("click", () => { $("search-input").value = chip.textContent.trim(); runSearch(); }));
  $("back-home").addEventListener("click", showHome);
  $("detail-back").addEventListener("click", closeDetail);
  $("nav-home").addEventListener("click", () => { if (splash.style.display === "none") showHome(); });
  $("empty-retry").addEventListener("click", showHome);
  $("open-refine").addEventListener("click", openRefine);
  $("close-refine").addEventListener("click", closeRefine);
  $("refine-scrim").addEventListener("click", closeRefine);
  document.querySelectorAll(".refine-chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      document.querySelectorAll(".refine-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active"); $("refine-input").value = chip.textContent.trim();
    }));
  $("update-recs").addEventListener("click", () => {
    const text = $("refine-input").value.trim(); if (!text) return;
    closeRefine(); recommend({ query: text }); $("refine-input").value = "";
    document.querySelectorAll(".refine-chip").forEach((c) => c.classList.remove("active"));
  });
  $("theme-toggle").addEventListener("click", () => applyTheme(!document.documentElement.classList.contains("dark")));
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (openDropdowns.some((d) => d.isOpen())) { closeAllDropdowns(null); return; }
    if (!overlay.classList.contains("hidden")) closeRefine();
    else if (!detailScreen.classList.contains("hidden")) closeDetail();
  });

  // ---- Init ----
  try { const t = localStorage.getItem("craival-theme"); if (t) applyTheme(t === "dark"); } catch (_) {}
  try {
    if (sessionStorage.getItem("craival-splash-seen")) { splash.style.display = "none"; splash.classList.add("hide"); showHome(); }
  } catch (_) {}

  // Fetch stats to populate header
  fetch(`${API}/api/stats`)
    .then(r => r.json())
    .then(s => {
      const r = $("stat-restaurants"); if(r) r.textContent = s.restaurant_count;
      const l = $("stat-locations"); if(l) l.textContent = s.location_count;
    })
    .catch(e => console.error("Error loading stats:", e));

  async function loadOptions(url, key, dropdown, withAny, anyLabel) {
    try {
      const res = await fetch(url); const data = await res.json();
      const opts = (data[key] || []).map((v) => ({ value: v, label: titleCase(v) }));
      dropdown.setOptions(withAny ? [{ value: "", label: anyLabel }, ...opts] : opts);
    } catch (e) { console.error("load " + key, e); }
  }
  loadOptions(`${API}/api/locations`, "locations", ddLocation, false);
  loadOptions(`${API}/api/cuisines`, "cuisines", ddCuisine, true, "Any cuisine");
})();
