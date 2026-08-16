"use strict";

const state = {
  datasets: [],
  overviewCache: new Map(),
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

function initTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.classList.contains("tab-link")) return;
      $$(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      $("#tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "dashboard") refreshDashboard();
      if (btn.dataset.tab === "forecast") refreshForecastForm();
    });
  });
}

function fillDatasetSelect(select, withUpload = false) {
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
  }
}

function fillStoreSelect(select, dataset) {
  const ds = state.datasets.find((d) => d.dataset === dataset);
  select.innerHTML = "";
  if (!ds) return;
  for (const st of ds.stores) {
    const opt = document.createElement("option");
    opt.value = st.store_id;
    opt.textContent = "Store " + st.store_id + (st.trained ? "" : " (untrained)");
    select.appendChild(opt);
  }
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
    setStatus($("#dash-status"), "Failed to load datasets: " + err.message, "error");
  }
  fillDatasetSelect($("#dash-dataset"));
  fillDatasetSelect($("#fc-dataset"), true);
  fillDatasetSelect($("#up-dataset"), true);
}

function currentStore(datasetKey) {
  const sel = datasetKey === "dash" ? $("#dash-store") : $("#fc-store");
  return sel.value ? Number(sel.value) : null;
}

async function refreshDashboard() {
  const dataset = $("#dash-dataset").value;
  const store = currentStore("dash");
  if (!dataset || store == null) {
    $("#dash-content").innerHTML = '<p class="placeholder">Select a dataset and store.</p>';
    return;
  }
  const ds = state.datasets.find((d) => d.dataset === dataset);
  const st = ds && ds.stores.find((s) => s.store_id === store);
  if (st && !st.trained) {
    $("#dash-content").innerHTML =
      '<div class="card"><p class="muted">Store ' + store +
      ' has not been trained yet. Go to the <b>Upload &amp; Train</b> tab to train it.</p></div>';
    return;
  }
  $("#dash-content").innerHTML = '<p class="placeholder">Loading dashboard&hellip;</p>';
  try {
    const ov = await api(`/api/stores/${store}/overview?dataset=${encodeURIComponent(dataset)}`);
    renderOverview(ov);
  } catch (err) {
    $("#dash-content").innerHTML =
      '<div class="card"><p class="status error">' + esc(err.message) + "</p></div>";
  }
}

function renderOverview(ov) {
  const s = ov.summary;
  const best = s.best_model;
  const m = s.metrics[best] || {};
  const ins = s.insights;

  const cards = [
    ["Best model", esc(best), "lightgbm / lstm / sarima"],
    ["RMSE", m.rmse != null ? Number(m.rmse).toLocaleString() : "&ndash;", "recursive multi-step"],
    ["MAPE", m.mape != null ? Number(m.mape).toFixed(2) + "%" : "&ndash;", "mean abs. pct error"],
    ["MAE", m.mae != null ? Number(m.mae).toLocaleString() : "&ndash;", "avg daily error"],
    ["Avg sales", Number(ins.avg_sales || 0).toLocaleString(), "daily mean"],
    ["Peak / trough day", esc(ins.peak_dow) + " / " + esc(ins.trough_dow), "day of week"],
    ["Promo lift", Number(ins.promo_lift_pct || 0).toFixed(1) + "%", "sales on promo days"],
    ["Holiday delta", Number(ins.holiday_delta_pct || 0).toFixed(1) + "%", "sales on holidays"],
  ];
  $("#dash-content").innerHTML =
    '<div class="grid-2">' +
    '<div class="card"><h2>Model comparison <span class="muted">(hold-out test, recursive)</span></h2>' +
    "<div class='table-wrap'><table class='data-table'><thead><tr>" +
    "<th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>MASE</th></tr></thead><tbody>" +
    Object.keys(s.metrics).filter((k) => !k.endsWith("_recursive")).map((k) => {
      const mm = s.metrics[k];
      return "<tr" + (k === best ? " style='background:rgba(79,140,255,0.12)'" : "") + ">" +
        "<td>" + esc(k) + "</td><td>" + Number(mm.rmse).toLocaleString() + "</td>" +
        "<td>" + Number(mm.mae).toLocaleString() + "</td>" +
        "<td>" + Number(mm.mape).toFixed(2) + "%</td>" +
        "<td>" + Number(mm.mase).toFixed(3) + "</td></tr>";
    }).join("") + "</tbody></table></div>" +
    "<h3 class='section-title'>Feature importance (top 10)</h3>" +
    "<div id='dash-importance'></div></div>" +

    '<div class="card"><h2>Key insights</h2>' +
    '<div class="grid-3">' +
    cards.map((c) =>
      "<div class='stat-card'><div class='stat-label'>" + c[0] + "</div>" +
      "<div class='stat-value'>" + c[1] + "</div>" +
      "<div class='stat-sub'>" + c[2] + "</div></div>"
    ).join("") +
    "</div>" +
    "<h3 class='section-title'>Error by segment (LightGBM)</h3>" +
    renderSegmentation(s.segmentation) +
    "</div></div>" +

    "<h3 class='section-title'>Forecast charts</h3>" +
    '<div class="grid-2" id="dash-forecast-charts"></div>' +

    "<h3 class='section-title'>Exploratory analysis</h3>" +
    '<div class="fig-grid">' +
    ov.figures.map((f) =>
      "<figure class='fig-card'><img src='" + esc(f.url) + "' alt='" + esc(f.name) + "'>" +
      "<figcaption class='fig-title'>" + esc(f.name) + "</figcaption></figure>"
    ).join("") +
    "</div>";

  drawImportance($("#dash-importance"), s.feature_importance);
  loadForecastCharts(ov);
}

function renderSegmentation(seg) {
  const rows = [];
  for (const [key, groups] of Object.entries(seg || {})) {
    for (const g of groups) {
      rows.push("<tr><td>" + esc(key) + "</td><td>" + esc(g.group) + "</td>" +
        "<td>" + Number(g.count).toLocaleString() + "</td>" +
        "<td>" + Number(g.mape).toFixed(2) + "%</td>" +
        "<td>" + Number(g.mae).toLocaleString() + "</td></tr>");
    }
  }
  return "<div class='table-wrap'><table class='data-table'><thead><tr>" +
    "<th>Dimension</th><th>Group</th><th>Days</th><th>MAPE</th><th>MAE</th></tr></thead>" +
    "<tbody>" + rows.join("") + "</tbody></table></div>";
}

function drawImportance(el, imp) {
  const entries = Object.entries(imp || {}).slice(0, 10);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  el.innerHTML = entries.map(([k, v]) =>
    "<div style='margin-bottom:5px'><span style='font-size:12px'>" + esc(k) + "</span>" +
    "<div style='background:var(--panel-2);border-radius:4px;height:10px;margin-top:3px'>" +
    "<div style='width:" + ((v / max) * 100).toFixed(1) +
    "%;height:100%;background:var(--accent);border-radius:4px'></div></div></div>"
  ).join("");
}

async function loadForecastCharts(ov) {
  const chartsEl = $("#dash-forecast-charts");
  chartsEl.innerHTML = "";
  const horizons = [...new Set(ov.forecasts.map((f) => f.horizon))];
  for (const h of horizons) {
    const fcs = ov.forecasts.filter((f) => f.horizon === h);
    const best = fcs.find((f) => f.model.toLowerCase() === ov.summary.best_model.toLowerCase()) || fcs[0];
    if (!best) continue;
    const div = document.createElement("div");
    div.className = "chart";
    div.style.padding = "10px";
    div.innerHTML = '<h3 style="font-size:14px;margin:6px 10px">' + h +
      "-day forecast (" + esc(best.model) + ")</h3>";
    chartsEl.appendChild(div);
    try {
      const fc = await fetchCsv(best.url);
      drawBandChart(div, fc, ov.summary.best_model);
    } catch (e) {
      div.innerHTML += '<p class="muted" style="padding:0 10px">' + esc(e.message) + "</p>";
    }
  }
}

async function fetchCsv(url) {
  const text = await (await fetch(url)).text();
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",").map((s) => s.trim());
  const fc = { dates: [], predicted_sales: [], lower_80: [], upper_80: [], lower_95: [], upper_95: [] };
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(",").map((s) => s.trim());
    if (cells.length < header.length) continue;
    const row = {};
    header.forEach((h, j) => (row[h] = cells[j]));
    fc.dates.push(row.date);
    fc.predicted_sales.push(Number(row.predicted_sales));
    fc.lower_80.push(Number(row.lower_80));
    fc.upper_80.push(Number(row.upper_80));
    fc.lower_95.push(Number(row.lower_95));
    fc.upper_95.push(Number(row.upper_95));
  }
  return fc;
}

function makeBandTraces(hist, fc) {
  const traces = [];
  if (hist && hist.length) {
    traces.push({
      x: hist.map((r) => r.date), y: hist.map((r) => r.sales),
      name: "History", mode: "lines", line: { color: "#8b96ad", width: 1 },
    });
  }
  if (fc && fc.dates) {
    traces.push(
      {
        x: fc.dates, y: fc.lower_95, name: "95% band", type: "scatter",
        mode: "lines", line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.upper_95, name: "95% band", type: "scatter",
        mode: "lines", fill: "tonexty", fillcolor: "rgba(79,140,255,0.15)",
        line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.lower_80, name: "80% band", type: "scatter",
        mode: "lines", line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.upper_80, name: "80% band", type: "scatter",
        mode: "lines", fill: "tonexty", fillcolor: "rgba(110,231,183,0.15)",
        line: { width: 0 }, hoverinfo: "skip",
      },
      {
        x: fc.dates, y: fc.predicted_sales, name: "Forecast", mode: "lines",
        line: { color: "#4f8cff", width: 2 },
      }
    );
  }
  return traces;
}

function drawBandChart(el, fc, title, hist) {
  const traces = makeBandTraces(hist || [], fc);
  const layout = {
    title: title || (fc.model ? fc.model + " forecast" : "Forecast"),
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e6ebf5" },
    margin: { l: 60, r: 20, t: 40, b: 40 },
    xaxis: { gridcolor: "#2a3550" }, yaxis: { gridcolor: "#2a3550", title: "Sales" },
    legend: { orientation: "h", y: 1.12 },
  };
  Plotly.newPlot(el, traces, layout, { responsive: true, displaylogo: false });
}

async function refreshForecastForm() {
  fillStoreSelect($("#fc-store"), $("#fc-dataset").value);
}

async function runForecast() {
  const payload = {
    dataset: $("#fc-dataset").value,
    store_id: Number($("#fc-store").value),
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
    alert("Forecast failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run forecast";
  }
}

function renderForecastTable(fc) {
  const rows = fc.dates.map((d, i) =>
    "<tr><td>" + esc(d) + "</td><td>" + fc.predicted_sales[i].toLocaleString() + "</td>" +
    "<td>" + fc.lower_80[i].toLocaleString() + "</td><td>" + fc.upper_80[i].toLocaleString() + "</td>" +
    "<td>" + fc.lower_95[i].toLocaleString() + "</td><td>" + fc.upper_95[i].toLocaleString() + "</td></tr>"
  ).join("");
  $("#fc-table").innerHTML =
    "<thead><tr><th>Date</th><th>Predicted</th><th>80% low</th><th>80% high</th><th>95% low</th><th>95% high</th></tr></thead>" +
    "<tbody>" + rows + "</tbody>";
}

function fileLabel(inputEl, labelEl, text) {
  inputEl.addEventListener("change", () => {
    labelEl.textContent = inputEl.files.length
      ? inputEl.files[0].name
      : text;
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
      " store(s) found. Go to step 2 to train.",
      "ok"
    );
    await loadDatasets();
    fillDatasetSelect($("#up-dataset"), true);
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
          "Training complete for store " + store + ". You can now forecast it.",
          "ok"
        );
        $("#train-btn").disabled = false;
        await loadDatasets();
        fillDatasetSelect($("#fc-dataset"), true);
        if (setSelectValue($("#fc-dataset"), dataset)) {
          $("#fc-dataset").dispatchEvent(new Event("change", { bubbles: true }));
          setSelectValue($("#fc-store"), store);
        }
        setStatus($("#dash-status"), "", "");
      } else if (job.status === "failed") {
        setStatus($("#train-status"), "Training failed: " + esc(job.message), "error");
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

function bindEvents() {
  $("#dash-dataset").addEventListener("change", () => {
    fillStoreSelect($("#dash-store"), $("#dash-dataset").value);
    refreshDashboard();
  });
  $("#dash-store").addEventListener("change", refreshDashboard);

  $("#fc-dataset").addEventListener("change", () =>
    fillStoreSelect($("#fc-store"), $("#fc-dataset").value)
  );
  $("#fc-run").addEventListener("click", runForecast);

  $("#up-dataset").addEventListener("change", () => {
    fillStoreSelect($("#up-store"), $("#up-dataset").value);
    updateTrainBtn();
  });
  $("#up-store").addEventListener("change", updateTrainBtn);
  $("#upload-btn").addEventListener("click", uploadFiles);
  $("#train-btn").addEventListener("click", startTraining);
}

(async function init() {
  initTabs();
  fileLabel($("#train-file"), $("#train-file-label"), "Choose train.csv");
  fileLabel($("#store-file"), $("#store-file-label"), "Optional: choose store.csv");
  bindEvents();
  await loadDatasets();
  fillDatasetSelect($("#dash-dataset"));
  $("#dash-dataset").dispatchEvent(new Event("change", { bubbles: true }));
})();
