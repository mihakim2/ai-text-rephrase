const $ = (id) => document.getElementById(id);

// --- "How it works" modal ---
let mermaidLoaded = false;
function loadMermaidOnce() {
  return new Promise((resolve) => {
    if (mermaidLoaded) return resolve();
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
    s.onload = () => {
      window.mermaid?.initialize({ startOnLoad: false, securityLevel: "loose" });
      mermaidLoaded = true;
      resolve();
    };
    s.onerror = () => resolve(); // fail open; diagram just won't render
    document.head.appendChild(s);
  });
}
async function openHowItWorks() {
  $("howit-modal").classList.remove("hidden");
  await loadMermaidOnce();
  if (window.mermaid) {
    try { await window.mermaid.run({ querySelector: "#howit-modal .mermaid" }); } catch {}
  }
}
function closeHowItWorks() {
  $("howit-modal").classList.add("hidden");
}
document.addEventListener("DOMContentLoaded", () => {
  $("howit-btn").addEventListener("click", openHowItWorks);
  $("howit-close").addEventListener("click", closeHowItWorks);
  $("howit-modal").querySelector(".modal-backdrop").addEventListener("click", closeHowItWorks);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("howit-modal").classList.contains("hidden")) closeHowItWorks();
  });
});

// --- Word-level diff (LCS) ---
function tokenize(s) {
  // Split into words + whitespace + punctuation, preserving runs
  return s.match(/(\s+|[A-Za-z0-9_'’]+|[^\s\w])/g) || [];
}
function diffTokens(a, b) {
  const A = tokenize(a), B = tokenize(b);
  const m = A.length, n = B.length;
  // Build LCS lengths
  const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out = [];
  let i = 0, j = 0;
  while (i < m && j < n) {
    if (A[i] === B[j]) { out.push({ t: "same", s: A[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ t: "del", s: A[i] }); i++; }
    else { out.push({ t: "add", s: B[j] }); j++; }
  }
  while (i < m) out.push({ t: "del", s: A[i++] });
  while (j < n) out.push({ t: "add", s: B[j++] });
  // Collapse consecutive same-type runs and drop pure-whitespace dels/adds (they're noise)
  const merged = [];
  for (const tok of out) {
    if (tok.t !== "same" && /^\s+$/.test(tok.s)) tok.t = "same"; // treat ws changes as same
    const last = merged[merged.length - 1];
    if (last && last.t === tok.t) last.s += tok.s;
    else merged.push({ ...tok });
  }
  return merged;
}
function renderDiff(original, finalText) {
  if (original === finalText) {
    const span = document.createElement("span");
    span.className = "same";
    span.textContent = finalText;
    return [span];
  }
  return diffTokens(original, finalText).map((tok) => {
    const span = document.createElement("span");
    span.className = tok.t;
    span.textContent = tok.s;
    return span;
  });
}

const state = {
  jobId: null,
  source: "file",   // "file" or "paste"
  file: null,
  pasted: "",
  mode: "deep",
  intensity: "balanced",
  unitCount: 0,
  unitLabel: "blocks",
  blockTypes: [],
  rowMap: {},
};

const dropZone = $("drop-zone");
const fileInput = $("file-input");
const dropLabel = $("drop-label");
const startBtn = $("start-btn");
const uploadMsg = $("upload-msg");
const samplesWrap = $("samples-wrap");

// --- Mode selector ---
document.querySelectorAll('input[name="mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    state.mode = document.querySelector('input[name="mode"]:checked').value;
    samplesWrap.classList.toggle("hidden-soft", state.mode !== "deep");
  });
});
// Initial state
state.mode = document.querySelector('input[name="mode"]:checked').value;
samplesWrap.classList.toggle("hidden-soft", state.mode !== "deep");

// --- Intensity selector ---
function refreshIntensityLabel() {
  const lbl = { light: "light", balanced: "balanced (default)", aggressive: "aggressive" }[state.intensity] || state.intensity;
  const el = document.getElementById("intensity-current");
  if (el) el.textContent = lbl;
}
document.querySelectorAll('input[name="intensity"]').forEach((r) => {
  r.addEventListener("change", () => {
    state.intensity = document.querySelector('input[name="intensity"]:checked').value;
    refreshIntensityLabel();
  });
});
state.intensity = document.querySelector('input[name="intensity"]:checked').value;
refreshIntensityLabel();

// --- Source tabs (file vs paste) ---
function refreshStartEnabled() {
  if (state.source === "file") {
    startBtn.disabled = !state.file;
  } else {
    startBtn.disabled = state.pasted.trim().length < 20;
  }
}
document.querySelectorAll(".source-tabs .tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".source-tabs .tab").forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    document.querySelectorAll(".source-pane").forEach((p) => p.classList.add("hidden-soft"));
    $(tab.dataset.target).classList.remove("hidden-soft");
    state.source = tab.id === "tab-paste" ? "paste" : "file";
    refreshStartEnabled();
    uploadMsg.textContent = "";
  });
});
$("paste-input").addEventListener("input", (e) => {
  state.pasted = e.target.value;
  $("paste-count").textContent = `${state.pasted.length.toLocaleString()} characters`;
  refreshStartEnabled();
});

// --- File picker / drag-drop ---
dropZone.addEventListener("click", (e) => {
  e.preventDefault();
  fileInput.click();
});
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    onFile(e.dataTransfer.files[0]);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) onFile(fileInput.files[0]);
});

function onFile(f) {
  if (!f.name.toLowerCase().endsWith(".txt")) {
    uploadMsg.textContent = "Please choose a .txt file.";
    return;
  }
  state.file = f;
  dropLabel.textContent = `Selected: ${f.name} (${Math.round(f.size / 1024)} KB)`;
  refreshStartEnabled();
  uploadMsg.textContent = "";
}

// --- Start job ---
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  uploadMsg.textContent = "Uploading...";
  const fd = new FormData();
  if (state.source === "paste") {
    if (state.pasted.trim().length < 20) {
      uploadMsg.textContent = "Please paste at least a few sentences.";
      refreshStartEnabled();
      return;
    }
    fd.append("text", state.pasted);
  } else {
    if (!state.file) {
      uploadMsg.textContent = "Please choose a .txt file.";
      refreshStartEnabled();
      return;
    }
    fd.append("file", state.file);
  }
  fd.append("mode", state.mode);
  fd.append("intensity", state.intensity);
  if (state.mode === "deep") {
    const samples = $("samples").value;
    if (samples.trim()) fd.append("samples", samples);
  }
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "upload failed");
    state.jobId = j.job_id;
    state.mode = j.mode;
    state.unitCount = j.unit_count;
    state.unitLabel = j.unit_label;
    state.blockTypes = j.block_types || [];
    uploadMsg.textContent = `Uploaded. ${j.unit_count} ${state.unitLabel}. Starting...`;
    await fetch(`/api/jobs/${state.jobId}/start`, { method: "POST" });
    showProgress();
    openStream();
  } catch (e) {
    showError(e.message);
    startBtn.disabled = false;
  }
});

// --- Progress UI ---
function showProgress() {
  $("upload-section").classList.add("hidden");
  $("progress-section").classList.remove("hidden");
  const total = state.mode === "deep"
    ? state.blockTypes.filter((t) => t === "prose").length
    : state.unitCount;
  $("progress-text").textContent = `0 / ${total}`;
  const container = $("units");
  container.innerHTML = "";
  state.rowMap = {};
  if (state.mode === "deep") {
    state.blockTypes.forEach((t, i) => {
      const row = document.createElement("div");
      row.className = "sent block";
      row.id = `unit-${i}`;
      row.innerHTML = `
        <span class="idx">#${i}</span>
        <span class="type">${t}</span>
        <span class="pill ${t === "prose" ? "pending" : "passthrough"}">${t === "prose" ? "pending" : "passthrough"}</span>
        <span class="score">—</span>
        <span class="preview"></span>`;
      container.appendChild(row);
      state.rowMap[i] = row;
    });
  } else {
    for (let i = 0; i < state.unitCount; i++) {
      const row = document.createElement("div");
      row.className = "sent";
      row.id = `unit-${i}`;
      row.innerHTML = `
        <span class="idx">#${i}</span>
        <span class="pill pending">pending</span>
        <span class="score">—</span>
        <span class="preview"></span>`;
      container.appendChild(row);
      state.rowMap[i] = row;
    }
  }
}

function updateUnit(d) {
  const idx = state.mode === "deep" ? d.block_idx : d.sentence_idx;
  const row = state.rowMap[idx];
  if (!row) return;
  const pill = row.querySelector(".pill");
  const score = row.querySelector(".score");
  const preview = row.querySelector(".preview");
  if (d.status) {
    pill.textContent = d.status;
    pill.className = `pill ${d.status}`;
  }
  if (state.mode === "deep") {
    // Show det / ai / sim if present (compact form)
    const det = d.detector_score, ai = d.critic_score, sim = d.similarity_score;
    if (det != null || ai != null || sim != null) {
      const parts = [];
      if (det != null) parts.push(`d${det}`);
      if (ai != null) parts.push(`a${ai}`);
      if (sim != null) parts.push(`s${sim}`);
      score.textContent = parts.join(" ");
    }
  } else {
    if (d.score != null) score.textContent = d.score;
  }
  if (d.preview) preview.textContent = d.preview;
}

function appendLog(level, msg) {
  const panel = $("log-panel");
  const line = document.createElement("div");
  line.className = "log-line" + (level === "ERROR" ? " error" : level === "WARNING" ? " warn" : "");
  line.textContent = msg;
  panel.appendChild(line);
  while (panel.children.length > 500) panel.removeChild(panel.firstChild);
  if ($("autoscroll").checked) panel.scrollTop = panel.scrollHeight;
}

function openStream() {
  const es = new EventSource(`/api/jobs/${state.jobId}/stream`);
  es.addEventListener("log", (e) => {
    try {
      const d = JSON.parse(e.data);
      appendLog(d.level || "INFO", d.msg || "");
    } catch {}
  });
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    if (d.stage === "voice_profile") {
      const vp = d.voice_profile || {};
      $("voice-line").textContent = `Voice: field=${vp.field_guess || "?"} • ${vp.tone_notes || ""}${vp.source === "user_samples" ? " (from your samples)" : ""}`;
    } else if (d.stage === "triage") {
      $("stage-line").textContent = `Triage: blind detector scoring ${d.total} prose blocks...`;
    } else if (d.stage === "intent") {
      $("stage-line").textContent = `Extracting intents on ${d.total} flagged blocks...`;
    } else if (d.stage === "final_coherence") {
      $("stage-line").textContent = `Final coherence pass...`;
    } else if (d.block_idx != null || d.sentence_idx != null) {
      updateUnit(d);
      // Progress count: accepted + passthrough
      let done = 0;
      let total = 0;
      Object.values(state.rowMap).forEach((r) => {
        const t = r.querySelector(".pill").textContent;
        if (state.mode === "deep") {
          // Count only prose blocks for the bar (others are passthrough by definition)
          const type = r.querySelector(".type")?.textContent;
          if (type === "prose") {
            total++;
            if (t === "accepted" || t === "passthrough") done++;
          }
        } else {
          total++;
          if (t === "accepted" || t === "skipped") done++;
        }
      });
      const pct = total ? Math.round((done / total) * 100) : 0;
      $("bar-fill").style.width = pct + "%";
      $("progress-text").textContent = `${done} / ${total}`;
    }
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    es.close();
    showDone(d);
  });
  es.addEventListener("error", (e) => {
    try {
      const d = JSON.parse(e.data || "{}");
      showError(d.message || "stream error");
    } catch {
      // Network blip; EventSource auto-reconnects.
    }
  });
}

function fmt(n) {
  return (n ?? 0).toLocaleString();
}

function renderUsage(u) {
  if (!u || !u.per_model) return;
  const body = $("usage-body");
  const foot = $("usage-foot");
  // Invert roles map: model -> [roles]
  const roleByModel = {};
  for (const [role, model] of Object.entries(u.roles || {})) {
    (roleByModel[model] = roleByModel[model] || []).push(role);
  }
  body.innerHTML = "";
  // Sort by total tokens desc
  const rows = Object.entries(u.per_model).sort((a, b) => (b[1].total || 0) - (a[1].total || 0));
  for (const [model, m] of rows) {
    const roles = (roleByModel[model] || []).join(", ");
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${model}</td>
      <td class="role">${roles}</td>
      <td class="num">${fmt(m.calls)}</td>
      <td class="num">${fmt(m.input)}</td>
      <td class="num">${fmt(m.output)}</td>
      <td class="num">${fmt(m.cache_read)}</td>
      <td class="num">${fmt(m.total)}</td>`;
    body.appendChild(tr);
  }
  foot.innerHTML = `
    <tr>
      <td colspan="2">Total</td>
      <td class="num">${fmt(u.totals.calls)}</td>
      <td class="num">${fmt(u.totals.input)}</td>
      <td class="num">${fmt(u.totals.output)}</td>
      <td class="num">${fmt(u.totals.cache_read)}</td>
      <td class="num">${fmt(u.totals.total)}</td>
    </tr>`;
  $("usage-wrap").classList.remove("hidden");
}

function showDone(d) {
  $("progress-section").classList.add("hidden");
  $("done-section").classList.remove("hidden");
  if (state.mode === "deep") {
    $("done-summary").innerHTML = `
      <div>Blocks: <strong>${d.blocks_total}</strong> total — <strong>${d.prose_total}</strong> prose</div>
      <div>Rewritten: <strong>${d.rewritten}</strong> · Passthrough (already human): <strong>${d.passthrough}</strong></div>
      <div>Avg blind-detector score (rewritten): <strong>${d.avg_detector_score ?? "—"}</strong> / 10</div>
      <div>Duration: <strong>${d.duration_sec}s</strong></div>`;
  } else {
    $("done-summary").innerHTML = `
      <div>Sentences: <strong>${d.sentences}</strong></div>
      <div>Average critic score: <strong>${d.avg_score}</strong> / 10</div>
      <div>Duration: <strong>${d.duration_sec}s</strong></div>`;
  }
  if (d.usage) renderUsage(d.usage);
  $("download-btn").href = `/api/jobs/${state.jobId}/download`;
}

function showError(msg) {
  $("upload-section").classList.add("hidden");
  $("progress-section").classList.add("hidden");
  $("done-section").classList.add("hidden");
  $("error-section").classList.remove("hidden");
  $("error-msg").textContent = msg;
}

$("reset-btn").addEventListener("click", () => location.reload());
$("error-reset-btn").addEventListener("click", () => location.reload());

// --- View changes (diff view) ---
let changesLoaded = false;
$("changes-btn").addEventListener("click", async () => {
  const section = $("changes-section");
  if (!section.classList.contains("hidden")) {
    section.classList.add("hidden");
    $("changes-btn").textContent = "View changes";
    return;
  }
  section.classList.remove("hidden");
  $("changes-btn").textContent = "Hide changes";
  if (changesLoaded) return;
  try {
    const r = await fetch(`/api/jobs/${state.jobId}/changes`);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "failed");
    renderChanges(data);
    changesLoaded = true;
  } catch (e) {
    $("changes-body").innerHTML = `<div class="error">${e.message}</div>`;
  }
});

function renderChanges(data) {
  const body = $("changes-body");
  body.innerHTML = "";
  const items = data.items || [];
  if (data.mode === "deep") {
    for (const it of items) {
      if (it.type !== "prose") {
        // Non-prose blocks shown as plain passthrough
        body.appendChild(buildChangeBlock({
          label: it.type,
          status: "passthrough",
          original: it.original,
          final: it.original,
          scores: "",
        }));
        continue;
      }
      const isRewritten = it.status === "accepted";
      const label = isRewritten ? `paragraph #${it.id}` : it.type;
      const status = isRewritten ? "rewritten" : (it.status === "passthrough" ? "passthrough" : it.status);
      const scoreBits = [];
      if (it.detector_score != null) scoreBits.push(`det:${it.detector_score}`);
      if (it.critic_score != null) scoreBits.push(`ai:${it.critic_score}`);
      if (it.similarity_score != null) scoreBits.push(`sim:${it.similarity_score}`);
      if (it.iterations) scoreBits.push(`iter:${it.iterations}`);
      body.appendChild(buildChangeBlock({
        label,
        status,
        original: it.original,
        final: it.final,
        scores: scoreBits.join("  "),
      }));
    }
  } else {
    // Quick mode: one row per sentence
    for (const it of items) {
      const isRewritten = it.status === "accepted";
      const status = isRewritten ? "rewritten" : (it.status === "skipped" ? "skipped" : it.status);
      const scoreBits = [];
      if (it.critic_score != null) scoreBits.push(`ai:${it.critic_score}`);
      if (it.iterations) scoreBits.push(`iter:${it.iterations}`);
      body.appendChild(buildChangeBlock({
        label: `sentence #${it.id}`,
        status,
        original: it.original,
        final: it.final,
        scores: scoreBits.join("  "),
      }));
    }
  }
}

function buildChangeBlock({ label, status, original, final, scores }) {
  const div = document.createElement("div");
  div.className = `change-block ${status}`;
  const meta = document.createElement("div");
  meta.className = "change-meta";
  meta.innerHTML = `<span>${label}</span><span class="badge ${status}">${status}</span><span class="scores">${scores}</span>`;
  div.appendChild(meta);
  const txt = document.createElement("div");
  txt.className = "diff-text";
  if (status === "rewritten") {
    const nodes = renderDiff(original, final);
    nodes.forEach((n) => txt.appendChild(n));
  } else {
    txt.textContent = final || original;
  }
  div.appendChild(txt);
  return div;
}
