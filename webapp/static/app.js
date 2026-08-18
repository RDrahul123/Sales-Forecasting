"use strict";

const state = {
  datasets: [],
  ctx: { dataset: null, store: null },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

function setStatus(el, msg, kind = "") {
  el.textContent = msg;
  el.className = "status" + (kind ? " " + kind : "");
}

function toast(msg, kind = "error") {
  const stack = $("#toast-stack");
  const el = document.createElement("div");
  el.className = "toast" + (kind === "ok" ? " ok" : "");
  el.textContent = msg;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

/* ============================================================
   Tabs
   ============================================================ */
function initTabs() {
  $$(".nav-item[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".nav-item[data-tab]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      $("#tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "dashboard") refreshDashboard();
    });
  });
}

/* ============================================================
   Shared sidebar context (dataset / store)
   ============================================================ */
function fillDatasetSelect(select, withUpload = false) {
  const prev = select.value;
  select.innerHTML = "";
  for (const ds of state.datasets) {
    if (!withUpload && ds.dataset.startsWith("upload:")) continue;
    const opt = document.createElement("option");
    opt.value = ds.dataset;
    opt.textContent = ds.label + " (" + ds.dataset + ")";
    select.appendChild(opt);
  }
  if (select.options.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No datasets available";
    select.appendChild(opt);
  } else if (prev) {
    setSelectValue(select, prev);
  }
}

function fillStoreSelect(select, dataset) {
  const prev = select.value;
  const ds = state.datasets.find((d) => d.dataset === dataset);
  select.innerHTML = "";
  if (!ds) return;
  for (const st of ds.stores) {
    const opt = document.createElement("option");
    opt.value = st.store_id;
    opt.textContent = "Store " + st.store_id + (st.trained ? "" : " · untrained");
    select.appendChild(opt);
  }
  if (prev) setSelectValue(select, prev);
}

function setSelectValue(select, value) {
  for (const opt of select.options) {
    if (opt.value === String(value)) {
      select.value = String(value);
      return true;
    }
  }
  return false;
}

async function loadDatasets() {
  try {
    const data = await api("/api/datasets");
    state.datasets = data.datasets;
  } catch (err) {
    state.datasets = [];
    toast("Failed to load datasets: " + err.message);
  }
  fillDatasetSelect($("#ctx-dataset"));
  fillDatasetSelect($("#up-dataset"), true);
}

function syncContext() {
  state.ctx.dataset = $("#ctx-dataset").value || null;
  fillStoreSelect($("#ctx-store"), state.ctx.dataset);
  state.ctx.store = $("#ctx-store").value ? Number($("#ctx-store").value) : null;
  updateStoreReadout();
  refreshActiveTab();
}

function updateStoreReadout() {
  const nameEl = $("#ctx-store-name");
  const subEl = $("#ctx-store-sub");
  if (state.ctx.store == null) {
    nameEl.textContent = "Select a store";
    subEl.textContent = "";
    $("#ctx-glance").innerHTML = "";
    return;
  }
  const ds = state.datasets.find((d) => d.dataset === state.ctx.dataset);
  nameEl.textContent = "Store " + state.ctx.store;
  subEl.textContent = ds ? ds.label : state.ctx.dataset;
}

function refreshActiveTab() {
  const active = $(".tab-panel.active");
  if (!active) return;
  if (active.id === "tab-dashboard") refreshDashboard();
}

/* ============================================================
   Dashboard
   ============================================================ */
async function refreshDashboard() {
  const { dataset, store } = state.ctx;
  const content = $("#dash-content");
  if (!dataset || store == null) {
    content.innerHTML = emptyState("◈", "Choose a dataset and store from the sidebar to load its forecast readings.");
    return;
  }
  const ds = state.datasets.find((d) => d.dataset === dataset);
  const st = ds && ds.stores.find((s) => s.store_id === store);
  if (st && !st.trained) {
    content.innerHTML = emptyState(
      "⇪",
      `Store ${store} hasn't been trained yet. Head to <b>Upload &amp; Train</b> to train SARIMA, LightGBM and LSTM for it.`
    );
    return;
  }
  content.innerHTML = '<p class="placeholder">Reading store history&hellip;</p>';
  try {
    const ov = await api(`/api/stores/${store}/overview?dataset=${encodeURIComponent(dataset)}`);
    renderOverview(ov);
    updateGlance(ov);
  } catch (err) {
    content.innerHTML = emptyState("!", esc(err.message));
  }
}

function emptyState(glyph, html) {
  return `<div class="empty-state"><div class="empty-glyph">${glyph}</div><p>${html}</p></div>`;
}

function updateGlance(ov) {
  const s = ov.summary;
  const m = s.metrics[s.best_model] || {};
  $("#ctx-glance").innerHTML = `
    <div class="glance-item"><span class="glance-label">Best model</span><span class="glance-value">${esc(s.best_model)}</span></div>
    <div class="glance-item jade"><span class="glance-label">MAPE</span><span class="glance-value">${m.mape != null ? Number(m.mape).toFixed(1) + "%" : "–"}</span></div>
  `;
}

function renderOverview(ov) {
  const s = ov.summary;
  const best = s.best_model;
  const m = s.metrics[best] || {};
  const ins = s.insights;

  const kpis = [
    ["Best model", esc(best), "lightgbm / lstm / sarima", "hero"],
    ["MAPE", m.mape != null ? Number(m.mape).toFixed(2) + "%" : "–", "mean abs. pct error", "hero"],
    ["RMSE", m.rmse != null ? Number(m.rmse).toLocaleString() : "–", "recursive multi-step", ""],
    ["MAE", m.mae != null ? Number(m.mae).toLocaleString() : "–", "avg daily error", ""],
    ["Avg sales", Number(ins.avg_sales || 0).toLocaleString(), "daily mean", ""],
    ["Peak / trough", esc(ins.peak_dow) + " / " + esc(ins.trough_dow), "day of week", ""],
    ["Promo lift", (Number(ins.promo_lift_pct || 0) >= 0 ? "+" : "") + Number(ins.promo_lift_pct || 0).toFixed(1) + "%", "sales on promo days", "good"],
    ["Holiday delta", (Number(ins.holiday_delta_pct || 0) >= 0 ? "+" : "") + Number(ins.holiday_delta_pct || 0).toFixed(1) + "%", "sales on holidays", ""],
  ];

  $("#dash-content").innerHTML =
    '<div class="kpi-strip">' +
    kpis.map((c) =>
      `<div class="kpi-cell ${c[3]}"><div class="kpi-label">${c[0]}</div><div class="kpi-value">${c[1]}</div><div class="kpi-sub">${c[2]}</div></div>`
    ).join("") +
    "</div>" +

    '<div class="section-head"><h3>Forecast, at a glance</h3></div>' +
    '<div class="grid-2" id="dash-forecast-charts"></div>' +

    '<div class="section-head"><h3>Model comparison</h3><span class="muted" style="font-size:12px;">hold-out test &middot; recursive</span></div>' +
    "<div class='table-wrap'><table class='data-table'><thead><tr>" +
    "<th>Model</th><th class='num'>RMSE</th><th class='num'>MAE</th><th class='num'>MAPE</th><th class='num'>MASE</th></tr></thead><tbody>" +
    Object.keys(s.metrics).filter((k) => !k.endsWith("_recursive")).map((k) => {
      const mm = s.metrics[k];
      return "<tr" + (k === best ? " class='is-best'" : "") + ">" +
        "<td>" + esc(k) + "</td>" +
        "<td class='num'>" + Number(mm.rmse).toLocaleString() + "</td>" +
        "<td class='num'>" + Number(mm.mae).toLocaleString() + "</td>" +
        "<td class='num'>" + Number(mm.mape).toFixed(2) + "%</td>" +
        "<td class='num'>" + Number(mm.mase).toFixed(3) + "</td></tr>";
    }).join("") + "</tbody></table></div>" +

    '<div class="grid-2" style="margin-top:14px;align-items:stretch;">' +
      '<div class="card"><div class="eyebrow">By segment</div><h3 class="panel-title" style="font-size:14px;">Error, ' + esc(best) + '</h3><div style="margin-top:12px;">' +
        renderSegmentation(s.segmentation) +
      "</div></div>" +
      '<div class="card"><div class="eyebrow">Signal</div><h3 class="panel-title" style="font-size:14px;">Feature importance</h3><div id="dash-importance" style="margin-top:14px;"></div></div>' +
    "</div>" +

    '<div class="section-head"><h3>Exploratory analysis</h3></div>' +
    '<div class="fig-grid">' +
    ov.figures.map((f) =>
      `<figure class="fig-card" data-img="${esc(f.url)}"><img src="${esc(f.url)}" alt="${esc(f.name)}" loading="lazy"><figcaption class="fig-title">${esc(f.name)}</figcaption></figure>`
    ).join("") +
    "</div>";

  drawImportance($("#dash-importance"), s.feature_importance);
  loadForecastCharts(ov);
  bindFigureLightbox();
}

function renderSegmentation(seg) {
  const rows = [];
  for (const [key, groups] of Object.entries(seg || {})) {
    for (const g of groups) {
      rows.push("<tr><td>" + esc(key) + "</td><td>" + esc(g.group) + "</td>" +
        "<td class='num'>" + Number(g.count).toLocaleString() + "</td>" +
        "<td class='num'>" + Number(g.mape).toFixed(2) + "%</td>" +
        "<td class='num'>" + Number(g.mae).toLocaleString() + "</td></tr>");
    }
  }
  return "<div class='table-wrap'><table class='data-table'><thead><tr>" +
    "<th>Dimension</th><th>Group</th><th class='num'>Days</th><th class='num'>MAPE</th><th class='num'>MAE</th></tr></thead>" +
    "<tbody>" + rows.join("") + "</tbody></table></div>";
}

function drawImportance(el, imp) {
  const entries = Object.entries(imp || {}).slice(0, 10);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  el.innerHTML = entries.map(([k, v]) =>
    `<div class="imp-row"><div class="imp-label"><span>${esc(k)}</span><span>${(v).toFixed(3)}</span></div>` +
    `<div class="imp-track"><div class="imp-fill" style="width:${((v / max) * 100).toFixed(1)}%"></div></div></div>`
  ).join("");
}

function bindFigureLightbox() {
  $$(".fig-card").forEach((card) => {
    card.addEventListener("click", () => {
      $("#lightbox-img").src = card.dataset.img;
      $("#lightbox-img").alt = card.querySelector("img").alt;
      $("#lightbox").classList.remove("hidden");
    });
  });
}

async function loadForecastCharts(ov) {
  const chartsEl = $("#dash-forecast-charts");
  chartsEl.innerHTML = "";
  const horizons = [...new Set(ov.forecasts.map((f) => f.horizon))];
  let histData = null;
  try {
    histData = await api(`/api/stores/${ov.summary.store_id || state.ctx.store}/history?dataset=${encodeURIComponent(state.ctx.dataset)}&days=90`);
  } catch (_) { /* charts still render without history */ }

  for (const h of horizons) {
    const fcs = ov.forecasts.filter((f) => f.horizon === h);
    const best = fcs.find((f) => f.model.toLowerCase() === ov.summary.best_model.toLowerCase()) || fcs[0];
    if (!best) continue;
    const wrap = document.createElement("div");
    wrap.className = "chart-card";
    const div = document.createElement("div");
    div.className = "chart";
    wrap.appendChild(div);
    chartsEl.appendChild(wrap);
    drawBandChart(div, best, h + "-day forecast · " + best.model, histData ? histData.history : []);
  }
}

function makeBandTraces(hist, fc) {
  const traces = [];
  if (hist && hist.length) {
    traces.push({
      x: hist.map((h) => h.date), y: hist.map((h) => h.sales),
      name: "History", mode: "lines",
      line: { color: "#9a8ca4", width: 1.4 },
    });
  }
  if (fc) {
    traces.push(
      {
        x: fc.dates, y: fc.lower_95, name: "95% band", type: "scatter",
        mode: "lines", line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.upper_95, name: "95% band", type: "scatter",
        mode: "lines", fill: "tonexty", fillcolor: "rgba(232,163,61,0.08)",
        line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.lower_80, name: "80% band", type: "scatter",
        mode: "lines", line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.upper_80, name: "80% band", type: "scatter",
        mode: "lines", fill: "tonexty", fillcolor: "rgba(79,178,134,0.18)",
        line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.predicted_sales, name: "Forecast", mode: "lines",
        line: { color: "#e8a33d", width: 2.4 },
      }
    );
  }
  return traces;
}

function drawBandChart(el, fc, title, hist) {
  const traces = makeBandTraces(hist || [], fc);
  const layout = {
    title: { text: title || (fc.model ? fc.model + " forecast" : "Forecast"), font: { family: "IBM Plex Mono, monospace", size: 12.5, color: "#cabdc6" } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#f3ead9", family: "Inter, sans-serif", size: 11.5 },
    margin: { l: 56, r: 16, t: 44, b: 40 },
    xaxis: { gridcolor: "#3a2c44", zerolinecolor: "#3a2c44" },
    yaxis: { gridcolor: "#3a2c44", title: "Sales", zerolinecolor: "#3a2c44" },
    legend: { orientation: "h", y: 1.18, font: { size: 10.5 } },
    colorway: ["#e8a33d", "#4fb286"],
  };
  Plotly.newPlot(el, traces, layout, { responsive: true, displaylogo: false });
}

/* ============================================================
   Forecast tab
   ============================================================ */
async function runForecast() {
  const { dataset, store } = state.ctx;
  if (!dataset || store == null) {
    toast("Pick a dataset and store from the sidebar first.");
    return;
  }
  const payload = {
    dataset, store_id: store,
    horizon: Number($("#fc-horizon").value),
    model: $("#fc-model").value,
    promo_mode: $("#fc-promo").value,
  };
  const btn = $("#fc-run");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const fc = await api("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("#fc-result").classList.remove("hidden");
    $("#fc-title").textContent =
      "Store " + fc.store_id + " · " + fc.horizon + "-day · " + fc.model +
      " · promo: " + fc.promo_mode;
    $("#fc-download").onclick = () => window.open(fc.download_url, "_blank");

    const histData = await api(
      `/api/stores/${fc.store_id}/history?dataset=${encodeURIComponent(fc.dataset)}&days=90`
    );
    drawBandChart($("#fc-chart"), fc, null, histData.history);
    renderForecastTable(fc);
  } catch (err) {
    toast("Forecast failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run forecast";
  }
}

function renderForecastTable(fc) {
  const rows = fc.dates.map((d, i) =>
    "<tr><td>" + esc(d) + "</td><td class='num'>" + fc.predicted_sales[i].toLocaleString() + "</td>" +
    "<td class='num'>" + fc.lower_80[i].toLocaleString() + "</td><td class='num'>" + fc.upper_80[i].toLocaleString() + "</td>" +
    "<td class='num'>" + fc.lower_95[i].toLocaleString() + "</td><td class='num'>" + fc.upper_95[i].toLocaleString() + "</td></tr>"
  ).join("");
  $("#fc-table").innerHTML =
    "<thead><tr><th>Date</th><th class='num'>Predicted</th><th class='num'>80% low</th><th class='num'>80% high</th><th class='num'>95% low</th><th class='num'>95% high</th></tr></thead>" +
    "<tbody>" + rows + "</tbody>";
}

/* ============================================================
   Upload tab
   ============================================================ */
function fileLabel(inputEl, labelEl, dropEl, text) {
  inputEl.addEventListener("change", () => {
    const has = inputEl.files.length > 0;
    labelEl.textContent = has ? inputEl.files[0].name : text;
    dropEl.classList.toggle("filled", has);
    $("#upload-btn").disabled = !$("#train-file").files.length;
  });
}

async function uploadFiles() {
  const trainFile = $("#train-file").files[0];
  if (!trainFile) return;
  const fd = new FormData();
  fd.append("train_file", trainFile);
  const storeFile = $("#store-file").files[0];
  if (storeFile) fd.append("store_file", storeFile);

  setStatus($("#upload-status"), "Uploading…", "");
  $("#upload-btn").disabled = true;
  try {
    const info = await api("/api/upload", { method: "POST", body: fd });
    setStatus(
      $("#upload-status"),
      "Uploaded " + esc(info.original_filename) + " · " + info.stores.length +
      " store(s) found. Continue to step 2.",
      "ok"
    );
    await loadDatasets();
    const target = "upload:" + info.upload_id;
    if (setSelectValue($("#up-dataset"), target)) {
      $("#up-dataset").dispatchEvent(new Event("change", { bubbles: true }));
    }
    $("#train-btn").disabled = false;
  } catch (err) {
    setStatus($("#upload-status"), "Upload failed: " + err.message, "error");
    $("#upload-btn").disabled = false;
  }
}

async function startTraining() {
  const dataset = $("#up-dataset").value;
  const store = $("#up-store").value;
  if (!dataset || store == null) return;
  $("#train-btn").disabled = true;
  setStatus($("#train-status"), "Queued…", "");
  $("#train-progress").classList.remove("hidden");
  $("#train-progress-bar").style.width = "0%";
  $("#train-progress-label").textContent = "0%";
  try {
    const job = await api("/api/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, store_id: Number(store) }),
    });
    pollJob(job.job_id, dataset, store);
  } catch (err) {
    setStatus($("#train-status"), "Failed to start: " + err.message, "error");
    $("#train-btn").disabled = false;
  }
}

function pollJob(jobId, dataset, store) {
  const tick = async () => {
    try {
      const job = await api("/api/jobs/" + jobId);
      $("#train-progress-bar").style.width = job.progress + "%";
      $("#train-progress-label").textContent =
        job.progress + "% · " + esc(job.message);
      if (job.status === "succeeded") {
        setStatus(
          $("#train-status"),
          "Training complete for store " + store + " — ready to forecast.",
          "ok"
        );
        toast("Store " + store + " finished training.", "ok");
        $("#train-btn").disabled = false;
        await loadDatasets();
        if (setSelectValue($("#ctx-dataset"), dataset)) {
          $("#ctx-dataset").dispatchEvent(new Event("change", { bubbles: true }));
          setSelectValue($("#ctx-store"), store);
          $("#ctx-store").dispatchEvent(new Event("change", { bubbles: true }));
        }
      } else if (job.status === "failed") {
        setStatus($("#train-status"), "Training failed: " + esc(job.message), "error");
        toast("Training failed for store " + store + ".");
        $("#train-btn").disabled = false;
      } else {
        setTimeout(tick, 1500);
      }
    } catch (err) {
      setStatus($("#train-status"), "Error checking job: " + err.message, "error");
      $("#train-btn").disabled = false;
    }
  };
  tick();
}

function updateTrainBtn() {
  const ds = state.datasets.find((d) => d.dataset === $("#up-dataset").value);
  const st = ds && ds.stores.find((s) => s.store_id === Number($("#up-store").value));
  $("#train-btn").disabled = !(ds && st && st.store_id != null);
}

/* ============================================================
   Wiring
   ============================================================ */
function bindEvents() {
  $("#ctx-dataset").addEventListener("change", syncContext);
  $("#ctx-store").addEventListener("change", () => {
    state.ctx.store = $("#ctx-store").value ? Number($("#ctx-store").value) : null;
    updateStoreReadout();
    refreshActiveTab();
  });

  $("#fc-run").addEventListener("click", runForecast);

  $("#up-dataset").addEventListener("change", () => {
    fillStoreSelect($("#up-store"), $("#up-dataset").value);
    updateTrainBtn();
  });
  $("#up-store").addEventListener("change", updateTrainBtn);
  $("#upload-btn").addEventListener("click", uploadFiles);
  $("#train-btn").addEventListener("click", startTraining);

  $("#lightbox").addEventListener("click", () => $("#lightbox").classList.add("hidden"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#lightbox").classList.add("hidden");
  });
}

(async function init() {
  initTabs();
  fileLabel($("#train-file"), $("#train-file-label"), $("#train-drop"), "Choose train.csv");
  fileLabel($("#store-file"), $("#store-file-label"), $("#store-drop"), "Optional: choose store.csv");
  bindEvents();
  await loadDatasets();
  syncContext();
})();
