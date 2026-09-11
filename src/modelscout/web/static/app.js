// ModelScout Web Frontend Application

// Escape catalogue and hardware text before putting it in HTML templates.
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

function safeHref(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? escapeHtml(url.href) : "#";
  } catch {
    return "#";
  }
}

let currentHardware = null;
let currentReport = null;
let selectedCompareIds = new Set();

document.addEventListener("DOMContentLoaded", () => {
  initNav();
  initAutocomplete();
  initHardwareDetection();
  initFormActions();
  initBenchmarkAction();
  initFilters();
});

// Navigation handling
function initNav() {
  const btnHw = document.getElementById("nav-btn-hw");
  const btnCatalog = document.getElementById("nav-btn-catalog");
  const btnCompare = document.getElementById("nav-btn-compare");

  const secResults = document.getElementById("results-section");
  const secHw = document.querySelector(".hardware-card");
  const secCatalog = document.getElementById("catalog-section");
  const secCompare = document.getElementById("compare-section");

  btnHw.addEventListener("click", () => {
    setActiveNav(btnHw);
    secHw.style.display = "block";
    secCatalog.style.display = "none";
    secCompare.style.display = "none";
    if (currentReport) secResults.style.display = "block";
  });

  btnCatalog.addEventListener("click", () => {
    setActiveNav(btnCatalog);
    secHw.style.display = "none";
    secResults.style.display = "none";
    secCompare.style.display = "none";
    secCatalog.style.display = "block";
    loadCatalogTable();
  });

  btnCompare.addEventListener("click", () => {
    setActiveNav(btnCompare);
    secHw.style.display = "none";
    secResults.style.display = "none";
    secCatalog.style.display = "none";
    secCompare.style.display = "block";
    renderComparisonTable();
  });

  document.getElementById("btn-modal-close").addEventListener("click", () => {
    document.getElementById("modal-backdrop").style.display = "none";
  });
  document.getElementById("modal-backdrop").addEventListener("click", (e) => {
    if (e.target.id === "modal-backdrop") {
      document.getElementById("modal-backdrop").style.display = "none";
    }
  });

  document.getElementById("btn-clear-compare").addEventListener("click", () => {
    selectedCompareIds.clear();
    updateCompareCount();
    renderComparisonTable();
  });
}

function setActiveNav(btn) {
  document.querySelectorAll(".nav-links button").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
}

// Autocomplete for Processors and GPUs with instant in-memory filtering & clock speeds
let cachedProcessors = [];
let cachedGpus = [];

async function preloadHardwareCatalog() {
  try {
    const [cpuRes, gpuRes] = await Promise.all([
      fetch("/api/catalog/processors?limit=500"),
      fetch("/api/catalog/gpus?limit=500")
    ]);
    if (cpuRes.ok) cachedProcessors = await cpuRes.json();
    if (gpuRes.ok) cachedGpus = await gpuRes.json();
  } catch (e) {
    console.warn("Catalog preload error:", e);
  }
}

function getVendorBadge(vendor, architecture, family) {
  const v = (vendor || "").toLowerCase();
  const fam = (family || "").toLowerCase();
  const arch = (architecture || "").toLowerCase();

  if (v === "apple") {
    return `<span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">Apple</span>`;
  }
  if (v === "intel") {
    if (fam.includes("data center") || fam.includes("gaudi") || fam.includes("xeon")) {
      return `<span style="background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">Intel AI / Server</span>`;
    }
    return `<span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">Intel</span>`;
  }
  if (v === "amd") {
    if (fam.includes("instinct") || fam.includes("epyc")) {
      return `<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">AMD Enterprise</span>`;
    }
    return `<span style="background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">AMD</span>`;
  }
  if (v === "nvidia") {
    if (fam.includes("data center") || fam.includes("tesla") || arch.includes("hopper") || arch.includes("grace") || fam.includes("grace hopper")) {
      return `<span style="background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">NVIDIA AI / Data Center</span>`;
    }
    if (fam.includes("workstation") || fam.includes("quadro")) {
      return `<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">NVIDIA Workstation</span>`;
    }
    return `<span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">NVIDIA GeForce</span>`;
  }
  return `<span style="background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">${vendor || "Generic"}</span>`;
}

function initAutocomplete() {
  const cpuInput = document.getElementById("input-cpu");
  const cpuDrop = document.getElementById("cpu-dropdown");
  const gpuInput = document.getElementById("input-gpu");
  const gpuDrop = document.getElementById("gpu-dropdown");

  let activeCpuFilter = "all";
  let activeGpuFilter = "all";

  preloadHardwareCatalog();

  function renderCpuItems(items) {
    if (!items || items.length === 0) {
      cpuDrop.innerHTML = `
        <div class="dropdown-filter-bar">
          <span class="filter-pill ${activeCpuFilter === 'all' ? 'active' : ''}" data-cpufilter="all">All (${cachedProcessors.length})</span>
          <span class="filter-pill ${activeCpuFilter === 'apple' ? 'active' : ''}" data-cpufilter="apple">Apple Silicon (${cachedProcessors.filter(p=>p.vendor==='Apple').length})</span>
          <span class="filter-pill ${activeCpuFilter === 'intel' ? 'active' : ''}" data-cpufilter="intel">Intel (${cachedProcessors.filter(p=>p.vendor==='Intel').length})</span>
          <span class="filter-pill ${activeCpuFilter === 'amd' ? 'active' : ''}" data-cpufilter="amd">AMD (${cachedProcessors.filter(p=>p.vendor==='AMD').length})</span>
        </div>
        <div style="padding: 12px; color: var(--text-muted); font-size: 0.85rem; text-align: center;">No matching processors found.</div>
      `;
      cpuDrop.style.display = "block";
      return;
    }

    const pillsHtml = `
      <div class="dropdown-filter-bar">
        <span class="filter-pill ${activeCpuFilter === 'all' ? 'active' : ''}" data-cpufilter="all">All (${cachedProcessors.length})</span>
        <span class="filter-pill ${activeCpuFilter === 'apple' ? 'active' : ''}" data-cpufilter="apple">Apple Silicon (${cachedProcessors.filter(p=>p.vendor==='Apple').length})</span>
        <span class="filter-pill ${activeCpuFilter === 'intel' ? 'active' : ''}" data-cpufilter="intel">Intel (${cachedProcessors.filter(p=>p.vendor==='Intel').length})</span>
        <span class="filter-pill ${activeCpuFilter === 'amd' ? 'active' : ''}" data-cpufilter="amd">AMD (${cachedProcessors.filter(p=>p.vendor==='AMD').length})</span>
      </div>
    `;

    const itemsHtml = items
      .slice(0, 40)
      .map(
        (item) => `
      <div class="dropdown-item" data-val="${escapeHtml(item.name)}">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
          <span style="font-weight: 600; color: #fff;">${escapeHtml(item.name)}</span>
          ${getVendorBadge(item.vendor, item.architecture, item.family)}
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; gap: 8px; flex-wrap: wrap; margin-top: 3px;">
          ${item.frequency_ghz ? `<span style="color: #60a5fa; font-weight: 500;">⚡ ${escapeHtml(item.frequency_ghz)}</span>` : ""}
          <span>• ${item.cores || 8} Cores / ${item.threads || item.cores || 8} Threads</span>
          ${item.family ? `<span>• ${escapeHtml(item.family)}</span>` : ""}
          ${item.architecture ? `<span>(${escapeHtml(item.architecture)})</span>` : ""}
        </div>
      </div>`
      )
      .join("");

    cpuDrop.innerHTML = pillsHtml + itemsHtml;
    cpuDrop.style.display = "block";
  }

  async function filterProcessors(query) {
    if (cachedProcessors.length === 0) {
      await preloadHardwareCatalog();
    }
    let baseList = cachedProcessors;
    if (activeCpuFilter !== "all") {
      baseList = cachedProcessors.filter((p) => (p.vendor || "").toLowerCase() === activeCpuFilter);
    }
    const q = (query || "").trim().toLowerCase();
    if (!q) {
      if (activeCpuFilter === "all") {
        const apple = cachedProcessors.filter((p) => (p.vendor || "").toLowerCase() === "apple").slice(0, 8);
        const intel = cachedProcessors.filter((p) => (p.vendor || "").toLowerCase() === "intel").slice(0, 10);
        const amd = cachedProcessors.filter((p) => (p.vendor || "").toLowerCase() === "amd").slice(0, 10);
        renderCpuItems([...apple, ...intel, ...amd]);
      } else {
        renderCpuItems(baseList);
      }
      return;
    }
    const words = q.split(/\s+/);
    const matches = baseList.filter((p) => {
      const full = `${p.vendor || ""} ${p.name} ${p.family || ""} ${p.frequency_ghz || ""} ${p.architecture || ""}`.toLowerCase();
      return words.every((w) => full.includes(w));
    });
    renderCpuItems(matches);
  }

  cpuInput.addEventListener("focus", () => {
    cpuInput.select();
    filterProcessors(cpuInput.value);
  });
  cpuInput.addEventListener("input", () => filterProcessors(cpuInput.value));

  cpuDrop.addEventListener("click", (e) => {
    const pill = e.target.closest("[data-cpufilter]");
    if (pill) {
      e.stopPropagation();
      activeCpuFilter = pill.dataset.cpufilter;
      filterProcessors(cpuInput.value);
      return;
    }
    const item = e.target.closest(".dropdown-item");
    if (item) {
      cpuInput.value = item.dataset.val;
      cpuDrop.style.display = "none";
    }
  });

  function renderGpuItems(items) {
    if (!items || items.length === 0) {
      gpuDrop.innerHTML = `
        <div class="dropdown-filter-bar">
          <span class="filter-pill ${activeGpuFilter === 'all' ? 'active' : ''}" data-gpufilter="all">All (${cachedGpus.length})</span>
          <span class="filter-pill ${activeGpuFilter === 'datacenter' ? 'active' : ''}" data-gpufilter="datacenter">Data Center & AI</span>
          <span class="filter-pill ${activeGpuFilter === 'geforce' ? 'active' : ''}" data-gpufilter="geforce">GeForce & RTX</span>
          <span class="filter-pill ${activeGpuFilter === 'workstation' ? 'active' : ''}" data-gpufilter="workstation">Workstation RTX</span>
          <span class="filter-pill ${activeGpuFilter === 'amd' ? 'active' : ''}" data-gpufilter="amd">AMD</span>
          <span class="filter-pill ${activeGpuFilter === 'intel' ? 'active' : ''}" data-gpufilter="intel">Intel</span>
        </div>
        <div style="padding: 12px; color: var(--text-muted); font-size: 0.85rem; text-align: center;">No matching GPUs found.</div>
      `;
      gpuDrop.style.display = "block";
      return;
    }

    const pillsHtml = `
      <div class="dropdown-filter-bar">
        <span class="filter-pill ${activeGpuFilter === 'all' ? 'active' : ''}" data-gpufilter="all">All (${cachedGpus.length})</span>
        <span class="filter-pill ${activeGpuFilter === 'datacenter' ? 'active' : ''}" data-gpufilter="datacenter">Data Center & AI</span>
        <span class="filter-pill ${activeGpuFilter === 'geforce' ? 'active' : ''}" data-gpufilter="geforce">GeForce & RTX</span>
        <span class="filter-pill ${activeGpuFilter === 'workstation' ? 'active' : ''}" data-gpufilter="workstation">Workstation RTX</span>
        <span class="filter-pill ${activeGpuFilter === 'amd' ? 'active' : ''}" data-gpufilter="amd">AMD</span>
        <span class="filter-pill ${activeGpuFilter === 'intel' ? 'active' : ''}" data-gpufilter="intel">Intel</span>
      </div>
    `;

    const itemsHtml = items
      .slice(0, 40)
      .map(
        (item) => `
      <div class="dropdown-item" data-val="${escapeHtml(item.name)}" data-vram="${item.vram_gb || 0}">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
          <span style="font-weight: 600; color: #fff;">${escapeHtml(item.name)}</span>
          ${getVendorBadge(item.vendor, item.architecture, item.family)}
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; gap: 8px; flex-wrap: wrap; margin-top: 3px;">
          <span style="color: #34d399; font-weight: 500;">💾 ${item.vram_gb || 0} GB VRAM</span>
          ${item.memory_bandwidth_gbps ? `<span>• ${item.memory_bandwidth_gbps} GB/s bandwidth</span>` : ""}
          ${item.memory_type ? `<span>• ${escapeHtml(item.memory_type)}</span>` : ""}
          ${item.architecture ? `<span>(${escapeHtml(item.architecture)})</span>` : ""}
        </div>
      </div>`
      )
      .join("");

    gpuDrop.innerHTML = pillsHtml + itemsHtml;
    gpuDrop.style.display = "block";
  }

  async function filterGpus(query) {
    if (cachedGpus.length === 0) {
      await preloadHardwareCatalog();
    }
    let baseList = cachedGpus;
    if (activeGpuFilter === "datacenter") {
      baseList = cachedGpus.filter((g) => {
        const f = (g.family || "").toLowerCase();
        const a = (g.architecture || "").toLowerCase();
        return f.includes("data center") || f.includes("tesla") || f.includes("instinct") || f.includes("gaudi") || a.includes("hopper") || a.includes("grace") || (g.vram_gb && g.vram_gb >= 40);
      });
    } else if (activeGpuFilter === "geforce") {
      baseList = cachedGpus.filter((g) => (g.vendor || "").toLowerCase() === "nvidia" && (g.family || "").toLowerCase().includes("geforce"));
    } else if (activeGpuFilter === "workstation") {
      baseList = cachedGpus.filter((g) => {
        const f = (g.family || "").toLowerCase();
        return f.includes("workstation") || f.includes("quadro") || f.includes("radeon pro");
      });
    } else if (activeGpuFilter === "amd") {
      baseList = cachedGpus.filter((g) => (g.vendor || "").toLowerCase() === "amd");
    } else if (activeGpuFilter === "intel") {
      baseList = cachedGpus.filter((g) => (g.vendor || "").toLowerCase() === "intel");
    }

    const q = (query || "").trim().toLowerCase();
    if (!q) {
      if (activeGpuFilter === "all") {
        const dc = cachedGpus.filter((g) => (g.vram_gb && g.vram_gb >= 40) || (g.family || "").toLowerCase().includes("data center")).slice(0, 10);
        const nv = cachedGpus.filter((g) => (g.vendor || "").toLowerCase() === "nvidia" && (g.family || "").toLowerCase().includes("geforce")).slice(0, 10);
        const amd = cachedGpus.filter((g) => (g.vendor || "").toLowerCase() === "amd").slice(0, 6);
        const intel = cachedGpus.filter((g) => (g.vendor || "").toLowerCase() === "intel").slice(0, 4);
        renderGpuItems([...dc, ...nv, ...amd, ...intel]);
      } else {
        renderGpuItems(baseList);
      }
      return;
    }
    const words = q.split(/\s+/);
    const matches = baseList.filter((g) => {
      const full = `${g.vendor || ""} ${g.name} ${g.family || ""} ${g.architecture || ""} ${g.vram_gb || ""}gb`.toLowerCase();
      return words.every((w) => full.includes(w));
    });
    renderGpuItems(matches);
  }

  gpuInput.addEventListener("focus", () => {
    gpuInput.select();
    filterGpus(gpuInput.value);
  });
  gpuInput.addEventListener("input", () => filterGpus(gpuInput.value));

  gpuDrop.addEventListener("click", (e) => {
    const pill = e.target.closest("[data-gpufilter]");
    if (pill) {
      e.stopPropagation();
      activeGpuFilter = pill.dataset.gpufilter;
      filterGpus(gpuInput.value);
      return;
    }
    const item = e.target.closest(".dropdown-item");
    if (item) {
      gpuInput.value = item.dataset.val;
      gpuDrop.style.display = "none";
    }
  });

  document.addEventListener("click", (e) => {
    if (!cpuInput.contains(e.target) && !cpuDrop.contains(e.target)) cpuDrop.style.display = "none";
    if (!gpuInput.contains(e.target) && !gpuDrop.contains(e.target)) gpuDrop.style.display = "none";
  });
}

// Hardware detection on load
async function initHardwareDetection() {
  try {
    const res = await fetch("/api/hardware");
    if (res.ok) {
      currentHardware = await res.json();
      populateHardwareForm(currentHardware);
      // Auto run analysis on first load
      runAnalysis();
    }
  } catch (e) {
    console.warn("Could not fetch hardware on init:", e);
  }
}

function populateHardwareForm(hw) {
  document.getElementById("input-cpu").value = hw.cpu.name || "";
  if (hw.gpus && hw.gpus.length > 0) {
    document.getElementById("input-gpu").value = hw.gpus[0].name || "";
  } else {
    document.getElementById("input-gpu").value = "Integrated Graphics";
  }
  document.getElementById("input-ram").value = Math.round(hw.memory.total_ram_gb);
  document.getElementById("input-storage").value = Math.round(hw.storage.available_gb);
}

function initFormActions() {
  document.getElementById("btn-reset-hw").addEventListener("click", () => {
    if (currentHardware) {
      populateHardwareForm(currentHardware);
      runAnalysis();
    }
  });

  document.getElementById("btn-find-out").addEventListener("click", () => {
    runAnalysis();
  });

  document.getElementById("btn-toggle-excluded").addEventListener("click", () => {
    const content = document.getElementById("excluded-content");
    const icon = document.getElementById("excluded-toggle-icon");
    if (content.style.display === "none") {
      content.style.display = "block";
      icon.textContent = "▲";
    } else {
      content.style.display = "none";
      icon.textContent = "▼";
    }
  });
}

function initBenchmarkAction() {
  const btn = document.getElementById("btn-run-benchmark");
  const banner = document.getElementById("benchmark-feedback");

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Running 5s Test...";
    banner.style.display = "block";
    banner.innerHTML = "<em>Executing memory throughput & CPU FLOPS measurement...</em>";

    try {
      const res = await fetch("/api/benchmark", { method: "POST" });
      const data = await res.json();
      banner.innerHTML = `<strong>✓ Measured Performance:</strong> Memory Bandwidth: <strong>${data.measured_ram_bandwidth_gbps} GB/s</strong> | CPU Vector Compute: <strong>${data.measured_cpu_gflops} GFLOPS</strong> (Duration: ${data.duration_seconds}s)`;
    } catch (e) {
      banner.innerHTML = `<span style="color: var(--danger)">Error running benchmark: ${escapeHtml(e.message)}</span>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Benchmark System";
    }
  });
}

// Run Analysis
async function runAnalysis() {
  const cpu = document.getElementById("input-cpu").value.trim();
  const gpu = document.getElementById("input-gpu").value.trim();
  const ram = parseFloat(document.getElementById("input-ram").value) || 16;
  const storage = parseFloat(document.getElementById("input-storage").value) || 256;
  const profile = document.getElementById("select-profile").value;
  const quant = document.getElementById("select-quant").value;

  const btn = document.getElementById("btn-find-out");
  const btnIcon = document.getElementById("btn-find-out-icon");
  const btnText = document.getElementById("btn-find-out-text");
  const statusText = document.getElementById("analysis-status-text");

  btn.disabled = true;
  if (btnIcon) btnIcon.innerHTML = '<span class="spinner"></span>';
  if (btnText) btnText.textContent = "Scanning your setup...";

  const stages = [
    "🔍 Checking your CPU & GPU compute architecture...",
    "🦙 Looking for local Ollama and llama.cpp runtimes...",
    "🤗 Cross-referencing Hugging Face quantized models...",
    "💾 Calculating memory footprints & KV cache requirements...",
    "⚡ Estimating real-world tokens/second and fit..."
  ];
  let stageIdx = 0;
  statusText.style.display = "flex";
  statusText.innerHTML = stages[0];

  const stageInterval = setInterval(() => {
    stageIdx = (stageIdx + 1) % stages.length;
    statusText.innerHTML = stages[stageIdx];
  }, 400);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cpu: cpu,
        gpu: gpu,
        ram_gb: ram,
        storage_gb: storage,
        profile: profile,
        quantization: quant,
        top_n: 20,
      }),
    });

    if (!res.ok) throw new Error("Failed to run analysis");
    currentReport = await res.json();
    renderReport(currentReport);
    statusText.innerHTML = `✓ Found ${currentReport.recommendations.length} models ready for your machine!`;
    setTimeout(() => {
      statusText.style.display = "none";
    }, 3200);
  } catch (e) {
    statusText.innerHTML = `<span style="color: var(--danger)">Error: ${escapeHtml(e.message)}</span>`;
    alert("Analysis error: " + e.message);
  } finally {
    clearInterval(stageInterval);
    btn.disabled = false;
    if (btnIcon) btnIcon.textContent = "⚡";
    if (btnText) btnText.textContent = "Find Models That Fit";
  }
}

// Render Results
function renderReport(report) {
  document.getElementById("results-section").style.display = "block";

  // 1. Hardware Score
  const hs = report.hardware.hardware_score || { overall_score: 75, gpu_score: 75, memory_capacity_score: 75, memory_bandwidth_score: 75, cpu_score: 75, ai_readiness_score: 75, summary: "" };
  document.getElementById("score-val").textContent = hs.overall_score;
  document.getElementById("score-summary").textContent = hs.summary || "Good configuration for local inference.";
  document.getElementById("sub-gpu").textContent = hs.gpu_score + "/100";
  document.getElementById("sub-mem").textContent = hs.memory_capacity_score + "/100";
  document.getElementById("sub-bw").textContent = hs.memory_bandwidth_score + "/100";
  document.getElementById("sub-cpu").textContent = hs.cpu_score + "/100";
  document.getElementById("sub-ready").textContent = hs.ai_readiness_score + "/100";

  const hw = report.hardware;
  document.getElementById("hw-details-snippet").innerHTML = `
    <span>${escapeHtml(hw.cpu.name)}</span> •
    <span>${hw.memory.total_ram_gb} GB ${hw.memory.unified_memory ? "Unified" : "RAM"}</span> •
    <span>${escapeHtml(hw.gpus.length > 0 ? hw.gpus[0].name : "CPU")}</span>
  `;

  // 2. Highlights
  const hlContainer = document.getElementById("highlights-container");
  const highlights = [];
  if (report.best_overall) highlights.push({ tag: "★ Best Overall", m: report.best_overall, desc: `Score: ${report.best_overall.recommendation_score} — Best intelligence, speed, and fit balance.` });
  if (report.best_quality) highlights.push({ tag: "✦ Highest Quality", m: report.best_quality, desc: `Benchmark: ${report.best_quality.benchmark_score} (${report.best_quality.benchmark_source})` });
  if (report.fastest) highlights.push({ tag: "⚡ Fastest Speed", m: report.fastest, desc: `${report.fastest.speed_display} for ultra-responsive generation.` });
  if (report.best_coding) highlights.push({ tag: "⌨ Best for Coding", m: report.best_coding, desc: "Trained and benchmarked for code generation & agentic workflows." });

  hlContainer.innerHTML = highlights
    .map(
      (h) => `
    <div class="highlight-card">
      <div class="highlight-tag">${escapeHtml(h.tag)}</div>
      <div class="highlight-model">${escapeHtml(h.m.display_name)}</div>
      <div class="highlight-desc">${escapeHtml(h.desc)}</div>
    </div>
  `
    )
    .join("");

  // 3. Render Recommendation Cards
  renderRecommendationCards(report.recommendations);

  // 4. Excluded
  const excContainer = document.getElementById("excluded-content");
  document.getElementById("excluded-count").textContent = report.excluded.length;
  excContainer.innerHTML = report.excluded
    .map(
      (ex) => `
    <div style="padding: 10px 0; border-bottom: 1px solid var(--card-border);">
      <strong style="color: #f87171;">✗ ${escapeHtml(ex.display_name)}</strong> (${(ex.parameters / 1e9).toFixed(1)}B params)
      <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 2px;">${escapeHtml(ex.reason)}</p>
    </div>
  `
    )
    .join("");
}

function renderRecommendationCards(recs) {
  const container = document.getElementById("recs-container");
  if (recs.length === 0) {
    container.innerHTML = `<div class="card"><p class="dim">No models match current filters.</p></div>`;
    return;
  }

  container.innerHTML = recs
    .map((r, i) => {
      let fitClass = "tag-fit-full";
      if (r.fit_type === "PARTIAL_OFFLOAD") fitClass = "tag-fit-partial";
      else if (r.fit_type === "CPU_ONLY") fitClass = "tag-fit-cpu";

      const isChecked = selectedCompareIds.has(r.model_id);

      const isDirect = (r.benchmark_evidence || "").toLowerCase() === "direct";
      const evidenceHtml = isDirect
        ? `<span class="meta-value" style="color: #34d399; font-weight: 600;">Direct <small style="font-size: 0.72rem; color: #a7f3d0; font-weight: 400;">(Stored record)</small> <span class="info-tooltip" title="Direct is the stored evidence category. Check the original benchmark result; ModelScout does not verify it.">ℹ️</span></span>`
        : `<span class="meta-value" style="color: #f59e0b; font-weight: 600;">Transferred <small style="font-size: 0.72rem; color: #fde68a; font-weight: 400;">(Base)</small> <span class="info-tooltip" title="Transferred Evidence: Benchmark score measured on unquantized base/instruct model and mathematically adjusted for ${escapeHtml(r.quantization)} quantization loss.">ℹ️</span></span>`;

      // Source Hub Badges
      const sourceLinks = [];
      if (r.huggingface_url) {
        sourceLinks.push(`<a href="${safeHref(r.huggingface_url)}" target="_blank" rel="noopener" class="source-link-badge hf-link" title="Open ${escapeHtml(r.huggingface_id)} on Hugging Face">🤗 Hugging Face</a>`);
      }
      if (r.ollama_url) {
        sourceLinks.push(`<a href="${safeHref(r.ollama_url)}" target="_blank" rel="noopener" class="source-link-badge ollama-link" title="View ${escapeHtml(r.ollama_name)} on Ollama Library">🦙 Ollama</a>`);
      }
      if (r.nvidia_build_url) {
        sourceLinks.push(`<a href="${safeHref(r.nvidia_build_url)}" target="_blank" rel="noopener" class="source-link-badge nvidia-link" title="Open NVIDIA Build NIM Registry">⚡ NVIDIA Build</a>`);
      }

      return `
      <div class="rec-card" data-id="${escapeHtml(r.model_id)}">
        <div class="rec-left">
          <div class="rec-title-row">
            <span style="font-weight: bold; color: var(--text-muted); font-size: 1.1rem;">#${i + 1}</span>
            <span class="rec-title">${escapeHtml(r.display_name)}</span>
            <span class="tag ${fitClass}">${escapeHtml(r.fit_label)}</span>
          </div>

          <div class="rec-tags">
            <span class="tag tag-neutral" style="color: #38bdf8; font-weight: 600;">💾 ${r.file_size_gb.toFixed(1)} GB File</span>
            <span class="tag tag-neutral">${(r.parameters / 1e9).toFixed(1)}B Parameters</span>
            <span class="tag tag-neutral">${escapeHtml(r.quantization)}</span>
            <span class="tag tag-neutral">${Math.round(r.context_length / 1024)}k Context</span>
            ${r.ollama_name ? `<span class="tag" style="background: rgba(59,130,246,0.15); color: #60a5fa;">Ollama: ${escapeHtml(r.ollama_name)}</span>` : ""}
            ${r.capabilities.vision ? '<span class="tag" style="background: rgba(168,85,247,0.15); color: #c084fc;">👁 Vision</span>' : ""}
            ${r.capabilities.coding ? '<span class="tag" style="background: rgba(16,185,129,0.15); color: #34d399;">⌨ Coding</span>' : ""}
          </div>

          <div class="rec-meta-grid">
            <div class="meta-item">
              <span class="meta-label">Model Size</span>
              <span class="meta-value" style="color: #38bdf8; font-weight: bold;">${r.file_size_gb.toFixed(1)} GB</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Est. Speed</span>
              <span class="meta-value" style="color: #34d399;">${escapeHtml(r.speed_display)}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Required RAM/VRAM</span>
              <span class="meta-value">${r.vram_required_gb.toFixed(1)} GB</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Benchmark</span>
              <span class="meta-value">${r.benchmark_score.toFixed(1)} <small style="color:var(--text-muted);">(${escapeHtml(r.benchmark_source)})</small></span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Evidence</span>
              ${evidenceHtml}
            </div>
          </div>

          <ul class="rec-why">
            ${r.why_recommended.slice(0, 3).map((w) => `<li>${escapeHtml(w)}</li>`).join("")}
          </ul>

          ${sourceLinks.length > 0 ? `
            <div class="rec-sources-row">
              <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 500;">Verify & Explore:</span>
              ${sourceLinks.join("")}
            </div>
          ` : ""}
        </div>

        <div class="rec-right">
          <div class="rec-score-box">
            <div class="rec-score-num">${r.recommendation_score.toFixed(1)}</div>
            <div class="rec-score-sub">Decision Score / 100</div>
          </div>

          <div class="rec-actions">
            <button class="btn-secondary btn-details" data-id="${escapeHtml(r.model_id)}">Details</button>
            <label style="font-size: 0.8rem; display: flex; align-items: center; gap: 4px; cursor: pointer;">
              <input type="checkbox" class="chk-compare" data-id="${escapeHtml(r.model_id)}" ${isChecked ? "checked" : ""}> Compare
            </label>
          </div>
        </div>
      </div>
    `;
    })
    .join("");

  // Attach card event listeners
  container.querySelectorAll(".btn-details").forEach((b) => {
    b.addEventListener("click", () => showModelDetails(b.dataset.id));
  });

  container.querySelectorAll(".chk-compare").forEach((chk) => {
    chk.addEventListener("change", (e) => {
      const mid = e.target.dataset.id;
      if (e.target.checked) selectedCompareIds.add(mid);
      else selectedCompareIds.delete(mid);
      updateCompareCount();
    });
  });
}

function updateCompareCount() {
  document.getElementById("compare-count").textContent = selectedCompareIds.size;
}

// Model details modal
function showModelDetails(modelId) {
  if (!currentReport) return;
  const m = currentReport.recommendations.find((r) => r.model_id === modelId);
  if (!m) return;

  const isDirect = (m.benchmark_evidence || "").toLowerCase() === "direct";
  const evidenceExplanation = isDirect
    ? `<span style="color: #34d399; font-weight: 600;">Direct Evidence</span> — The catalogue labels this as a direct result for the model. It is a bundled record; check the original result before relying on the score.`
    : `<span style="color: #f59e0b; font-weight: 600;">Transferred (${escapeHtml(m.benchmark_evidence)})</span> — Evaluated against the unquantized base/instruct release, with mathematically modeled degradation penalties applied for ${escapeHtml(m.quantization)} quantization.`;

  // Source Hub Badges
  const hubLinks = [];
  if (m.huggingface_url) {
    hubLinks.push(`<a href="${safeHref(m.huggingface_url)}" target="_blank" rel="noopener" class="source-link-badge hf-link">🤗 Hugging Face (${escapeHtml(m.huggingface_id || "Weights")})</a>`);
  }
  if (m.ollama_url) {
    hubLinks.push(`<a href="${safeHref(m.ollama_url)}" target="_blank" rel="noopener" class="source-link-badge ollama-link">🦙 Ollama Library (${escapeHtml(m.ollama_name)})</a>`);
  }
  if (m.nvidia_build_url) {
    hubLinks.push(`<a href="${safeHref(m.nvidia_build_url)}" target="_blank" rel="noopener" class="source-link-badge nvidia-link">⚡ NVIDIA Build Registry</a>`);
  }

  const modalBody = document.getElementById("modal-body");
  modalBody.innerHTML = `
    <h2 style="margin-bottom: 8px;">${escapeHtml(m.display_name)}</h2>
    <div style="display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;">
      <span class="tag tag-fit-full">${escapeHtml(m.fit_label)}</span>
      <span class="tag tag-neutral">Score: ${m.recommendation_score.toFixed(1)} / 100</span>
      <span class="tag tag-neutral">${(m.parameters / 1e9).toFixed(1)}B Parameters</span>
      <span class="tag tag-neutral">Size: ${m.file_size_gb.toFixed(1)} GB</span>
    </div>

    <!-- Model source links -->
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid var(--card-border); padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
      <h4 style="color: #60a5fa; margin-bottom: 8px; font-size: 0.9rem;">Model Sources & Hub Registries</h4>
      <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: center;">
        ${hubLinks.join("") || "<span style='color: var(--text-muted); font-size: 0.85rem;'>Standard open-weights GGUF artifact.</span>"}
      </div>
    </div>

    <div style="background: #12141c; padding: 14px; border-radius: 8px; margin-bottom: 16px;">
      <h4 style="color: #60a5fa; margin-bottom: 6px;">Decision Engine Breakdown</h4>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.85rem;">
        <div>Benchmark Quality: <strong>${m.score_breakdown.benchmark_quality.toFixed(1)}</strong></div>
        <div>Hardware Fit: <strong>${m.score_breakdown.hardware_fit.toFixed(1)}</strong></div>
        <div>Speed Score: <strong>${m.score_breakdown.speed_score.toFixed(1)}</strong></div>
        <div>Model Capability: <strong>${m.score_breakdown.model_capability.toFixed(1)}</strong></div>
        <div>Evidence Level: <strong>${m.score_breakdown.evidence_score.toFixed(1)}</strong></div>
        <div>Runtime Support: <strong>${m.score_breakdown.runtime_score.toFixed(1)}</strong></div>
      </div>
    </div>

    <div style="background: #12141c; padding: 14px; border-radius: 8px; margin-bottom: 16px;">
      <h4 style="color: #34d399; margin-bottom: 6px;">Benchmark & Evidence Provenance</h4>
      <p style="font-size: 0.85rem; color: #e2e8f0; margin-bottom: 8px;">
        <strong>Score:</strong> ${m.benchmark_score.toFixed(1)}/100 (Primary: ${escapeHtml(m.benchmark_source)})<br>
        <strong>Confidence:</strong> ${escapeHtml(m.benchmark_confidence)}
      </p>
      <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.4; background: rgba(0,0,0,0.3); padding: 8px 10px; border-radius: 4px;">
        ${evidenceExplanation}
      </div>
    </div>

    <h4 style="margin-bottom: 6px;">Execution Details</h4>
    <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 12px;">
      <strong>Artifact:</strong> ${escapeHtml(m.artifact)} (~${m.file_size_gb.toFixed(1)} GB)<br>
      <strong>Context Length:</strong> ${m.context_length.toLocaleString()} tokens<br>
      <strong>Speed Estimation:</strong> ${escapeHtml(m.speed_display)} (${escapeHtml(m.speed_confidence)} confidence) • ${escapeHtml(m.speed_notes)}
    </p>

    ${m.ollama_name ? `
      <div style="background: #1a2236; border: 1px solid var(--primary); padding: 10px 14px; border-radius: 6px; margin-bottom: 16px;">
        <span style="font-size: 0.8rem; color: #93c5fd; text-transform: uppercase; font-weight: bold;">Quick Run (Ollama)</span>
        <code style="display: block; font-size: 0.95rem; color: #fff; margin-top: 4px;">ollama run ${escapeHtml(m.ollama_name)}</code>
      </div>
    ` : ""}

    <h4 style="margin-bottom: 6px;">Why This Model?</h4>
    <ul class="rec-why" style="margin-bottom: 16px;">
      ${m.why_recommended.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}
    </ul>
  `;

  document.getElementById("modal-backdrop").style.display = "flex";
}

// Comparison Table View
function renderComparisonTable() {
  const container = document.getElementById("compare-table-container");
  if (!currentReport || selectedCompareIds.size === 0) {
    container.innerHTML = `<p class="dim" style="padding: 24px; text-align: center;">Select models using the 'Compare' checkboxes on recommendation cards.</p>`;
    return;
  }

  const models = currentReport.recommendations.filter((r) => selectedCompareIds.has(r.model_id));
  if (models.length === 0) {
    container.innerHTML = `<p class="dim" style="padding: 24px;">Selected models not in current recommendations.</p>`;
    return;
  }

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Attribute</th>
          ${models.map((m) => `<th>${escapeHtml(m.display_name)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        <tr><td><strong>Recommendation Score</strong></td>${models.map((m) => `<td style="color: #60a5fa; font-weight: bold;">${m.recommendation_score.toFixed(1)} / 100</td>`).join("")}</tr>
        <tr><td><strong>Parameters</strong></td>${models.map((m) => `<td>${(m.parameters / 1e9).toFixed(1)}B</td>`).join("")}</tr>
        <tr><td><strong>Hardware Fit</strong></td>${models.map((m) => `<td>${escapeHtml(m.fit_label)}</td>`).join("")}</tr>
        <tr><td><strong>Estimated Speed</strong></td>${models.map((m) => `<td style="color: #34d399;">${escapeHtml(m.speed_display)}</td>`).join("")}</tr>
        <tr><td><strong>Required Memory</strong></td>${models.map((m) => `<td>${m.vram_required_gb.toFixed(1)} GB</td>`).join("")}</tr>
        <tr><td><strong>Artifact File Size</strong></td>${models.map((m) => `<td>${m.file_size_gb.toFixed(1)} GB</td>`).join("")}</tr>
        <tr><td><strong>Benchmark Score</strong></td>${models.map((m) => `<td>${m.benchmark_score.toFixed(1)} (${escapeHtml(m.benchmark_source)})</td>`).join("")}</tr>
        <tr><td><strong>Evidence Level</strong></td>${models.map((m) => `<td style="text-transform: capitalize;">${escapeHtml(m.benchmark_evidence)}</td>`).join("")}</tr>
        <tr><td><strong>Coding</strong></td>${models.map((m) => `<td>${m.capabilities.coding ? "✓ Yes" : "✗ No"}</td>`).join("")}</tr>
        <tr><td><strong>Reasoning</strong></td>${models.map((m) => `<td>${m.capabilities.reasoning ? "✓ Yes" : "✗ No"}</td>`).join("")}</tr>
        <tr><td><strong>Vision</strong></td>${models.map((m) => `<td>${m.capabilities.vision ? "✓ Yes" : "✗ No"}</td>`).join("")}</tr>
        <tr><td><strong>Ollama Tag</strong></td>${models.map((m) => `<td>${escapeHtml(m.ollama_name || "None")}</td>`).join("")}</tr>
      </tbody>
    </table>
  `;
}

// Full Catalogue Table View
async function loadCatalogTable() {
  const container = document.getElementById("catalog-table-container");
  container.innerHTML = "<p class='dim'>Loading catalog...</p>";
  try {
    const res = await fetch("/api/models");
    const models = await res.json();
    container.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Canonical Name</th>
            <th>Family</th>
            <th>Parameters</th>
            <th>Context</th>
            <th>Capabilities</th>
            <th>Ollama Tag</th>
            <th>Model Sources</th>
          </tr>
        </thead>
        <tbody>
          ${models
            .map((m) => {
              const caps = [];
              if (m.has_coding) caps.push("Code");
              if (m.has_reasoning) caps.push("Reason");
              if (m.has_vision) caps.push("Vision");

              const nameHtml = m.huggingface_id
                ? `<a href="https://huggingface.co/${escapeHtml(m.huggingface_id)}" target="_blank" rel="noopener" style="color: #60a5fa; text-decoration: none; font-weight: 600;" title="View on Hugging Face">${escapeHtml(m.canonical_name)}</a>`
                : `<strong>${escapeHtml(m.canonical_name)}</strong>`;

              const ollamaHtml = m.ollama_name
                ? `<a href="https://ollama.com/library/${escapeHtml(encodeURIComponent(m.ollama_name.split(':')[0]))}" target="_blank" rel="noopener" style="color: #cbd5e1; text-decoration: none;" title="View on Ollama"><code>${escapeHtml(m.ollama_name)}</code></a>`
                : `<span style="color: var(--text-muted);">-</span>`;

              const hubHtml = m.huggingface_id
                ? `<a href="https://huggingface.co/${escapeHtml(m.huggingface_id)}" target="_blank" rel="noopener" class="source-link-badge hf-link" style="font-size: 0.7rem;">🤗 Hugging Face</a>`
                : `<span style="color: var(--text-muted); font-size: 0.75rem;">Seed Catalogue</span>`;

              return `
              <tr>
                <td>${nameHtml}</td>
                <td>${escapeHtml(m.family_id)}</td>
                <td>${(m.total_parameters / 1e9).toFixed(1)}B</td>
                <td>${Math.round(m.context_length / 1024)}k</td>
                <td>${caps.join(", ") || "Text"}</td>
                <td>${ollamaHtml}</td>
                <td>${hubHtml}</td>
              </tr>
            `;
            })
            .join("")}
        </tbody>
      </table>
    `;
  } catch (e) {
    container.innerHTML = `<p style="color: var(--danger);">Failed to load catalogue: ${escapeHtml(e.message)}</p>`;
  }
}

// Filter listeners
function initFilters() {
  const sortSelect = document.getElementById("filter-sort");
  const fitSelect = document.getElementById("filter-fit");
  const speedSelect = document.getElementById("filter-speed");
  const searchInput = document.getElementById("filter-search");

  function applyFilters() {
    if (!currentReport) return;
    const sortVal = sortSelect ? sortSelect.value : "score";
    const fitVal = fitSelect.value;
    const speedVal = speedSelect.value;
    const searchVal = searchInput.value.toLowerCase().trim();

    const filtered = currentReport.recommendations.filter((r) => {
      if (fitVal === "full_gpu" && !r.fit_type.includes("FULL_GPU") && !r.fit_type.includes("UNIFIED")) return false;
      if (fitVal === "partial" && r.fit_type !== "PARTIAL_OFFLOAD") return false;
      if (fitVal === "cpu_only" && r.fit_type !== "CPU_ONLY") return false;

      if (speedVal === "fast" && r.estimated_tok_per_sec < 20.0) return false;
      if (speedVal === "usable" && r.estimated_tok_per_sec < 5.0) return false;

      if (searchVal) {
        const words = searchVal.split(/\s+/);
        const combined = `${r.display_name} ${r.quantization} ${r.ollama_name || ""} ${(r.parameters / 1e9).toFixed(1)}b`.toLowerCase();
        if (!words.every((w) => combined.includes(w))) return false;
      }

      return true;
    });

    // Apply sorting
    filtered.sort((a, b) => {
      if (sortVal === "speed_desc") {
        return b.estimated_tok_per_sec - a.estimated_tok_per_sec;
      } else if (sortVal === "size_asc") {
        return a.file_size_gb - b.file_size_gb;
      } else if (sortVal === "size_desc") {
        return b.file_size_gb - a.file_size_gb;
      } else if (sortVal === "benchmark_desc") {
        return b.benchmark_score - a.benchmark_score;
      } else {
        // default: recommendation_score descending
        return b.recommendation_score - a.recommendation_score;
      }
    });

    renderRecommendationCards(filtered);
  }

  if (sortSelect) sortSelect.addEventListener("change", applyFilters);
  fitSelect.addEventListener("change", applyFilters);
  speedSelect.addEventListener("change", applyFilters);
  searchInput.addEventListener("input", applyFilters);
}
