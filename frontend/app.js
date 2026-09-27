// Race Readiness Check frontend (spec/05-ui.md).
// The verdict is computed by the backend (POST /api/check); this file only collects
// input, validates it for fast feedback, and renders the result it is given.

const FIELDS = {
  voltage_v: { label: "Battery voltage", limit: "voltage_v" },
  current_a: { label: "Battery current", limit: "current_a" },
  temperature_c: { label: "Battery temperature", limit: "temperature_c" },
  duration_min: { label: "Charging duration", limit: "duration_min" },
  soc_percent: { label: "State of charge", limit: "soc_percent" },
};

const form = document.getElementById("check-form");
const submitBtn = document.getElementById("submit");
const durationHint = document.getElementById("duration-hint");
const progress = document.getElementById("progress");
const progressLabel = document.getElementById("progress-label");
const spinner = document.getElementById("spinner");
const stagesList = document.getElementById("stages");
const resultEl = document.getElementById("result");
const failPopup = document.getElementById("fail-popup");
const failPopupBody = document.getElementById("fail-popup-body");
const failPopupClose = document.getElementById("fail-popup-close");

let limits = null; // input_limits from spec/thresholds.yaml, served by /api/config
let secondsPerMinute = 60;
const touched = new Set();

// ---------------------------------------------------------------------------
// Validation (spec/01-requirements.md §1.1). The server validates again.
// ---------------------------------------------------------------------------

function validateField(name, raw) {
  const { label, limit } = FIELDS[name];
  if (raw.trim() === "") return `${label} is required.`;
  const value = Number(raw);
  if (!Number.isFinite(value)) return `${label} must be a finite number.`;
  if (!limits) return "";
  const lim = limits[limit];
  if (name === "duration_min" && value <= lim.min) {
    return "Charging duration must be greater than 0 minutes.";
  }
  if (value < lim.min || value > lim.max) {
    return `${label} must be between ${lim.min} and ${lim.max}.`;
  }
  return "";
}

function setFieldError(name, message) {
  const input = document.getElementById(name);
  document.getElementById(`${name}-error`).textContent = message;
  input.classList.toggle("invalid", Boolean(message));
  input.setAttribute("aria-invalid", message ? "true" : "false");
}

function validateForm() {
  let valid = true;
  for (const name of Object.keys(FIELDS)) {
    const message = validateField(name, document.getElementById(name).value);
    if (message) valid = false;
    setFieldError(name, touched.has(name) ? message : "");
  }
  submitBtn.disabled = !valid || !limits;
  return valid;
}

function updateDurationHint() {
  const minutes = Number(document.getElementById("duration_min").value);
  const raw = document.getElementById("duration_min").value.trim();
  if (raw === "" || !Number.isFinite(minutes)) {
    durationHint.textContent = "= — s";
    return;
  }
  const seconds = minutes * secondsPerMinute;
  durationHint.textContent = `= ${seconds.toLocaleString("en-US", { maximumFractionDigits: 3 })} s`;
}

for (const name of Object.keys(FIELDS)) {
  const input = document.getElementById(name);
  input.addEventListener("input", () => {
    touched.add(name);
    if (name === "duration_min") updateDurationHint();
    validateForm();
  });
  input.addEventListener("blur", () => {
    touched.add(name);
    validateForm();
  });
}

// ---------------------------------------------------------------------------
// Stage progress: "Overcharge ✓ → Thermal … → Prediction"
// ---------------------------------------------------------------------------

function setStages(statuses) {
  for (const li of stagesList.children) {
    li.className = statuses[li.dataset.stage] || "";
  }
}

function showRunning() {
  progress.classList.remove("hidden");
  spinner.classList.remove("done");
  progressLabel.textContent = "Running safety checks…";
  setStages({ 1: "running" });
}

function showStages(stages) {
  spinner.classList.add("done");
  progressLabel.textContent = "Safety checks complete";
  const statuses = {};
  for (const s of stages) statuses[s.stage] = s.status;
  setStages(statuses);
}

// ---------------------------------------------------------------------------
// Result rendering (spec/05-ui.md §2–§3)
// ---------------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const child of children) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function failureCard(f) {
  const dl = el("dl", {},
    el("dt", {}, `Recorded (${f.recorded_kind}):`),
    el("dd", {}, f.recorded.join(", ")),
    el("dt", {}, `${f.threshold_label}:`),
    el("dd", {}, f.thresholds.join("; ")),
  );
  return el("div", { class: "failure" }, el("p", { class: "reason" }, f.reason), dl);
}

function renderResult(data) {
  resultEl.replaceChildren();
  resultEl.classList.remove("hidden");

  if (data.verdict === "CAN_PROCEED") {
    const banner = el("div", { class: "banner ok" },
      el("h2", {}, "✅ CAN PROCEED"),
      el("p", {}, "All safety checks passed."),
    );
    if (data.predicted_efficiency_pct != null) {
      banner.append(el("p", {}, `Predicted efficiency: ${data.predicted_efficiency_pct.toFixed(1)} %`));
    }
    resultEl.append(banner);
    return;
  }

  const banner = el("div", { class: "banner fail" }, el("h2", { id: "fail-popup-title" }, "⛔ DO NOT PROCEED"));

  // Group failure cards by the stage that failed. Only one stage can fail, because
  // the pipeline stops at the first failed stage; every failure in that stage is listed.
  const byStage = new Map();
  for (const f of data.failures) {
    if (!byStage.has(f.stage)) byStage.set(f.stage, { name: f.stage_name, items: [] });
    byStage.get(f.stage).items.push(f);
  }
  for (const [stage, group] of byStage) {
    const block = el("div", { class: "stage-block" }, el("h3", {}, `Stage ${stage}: ${group.name}`));
    for (const f of group.items) block.append(failureCard(f));
    banner.append(block);
  }

  if (data.comment) banner.append(el("p", { class: "comment" }, `Comment: ${data.comment}`));

  // Show the failure in the burning-car popup; keep a copy on the page for when it is closed.
  resultEl.append(banner.cloneNode(true));
  resultEl.querySelector("h2").removeAttribute("id");
  failPopupBody.replaceChildren(banner);
  failPopup.classList.remove("hidden");
  failPopupClose.focus();
}

function closeFailPopup() {
  failPopup.classList.add("hidden");
  submitBtn.focus();
}

failPopupClose.addEventListener("click", closeFailPopup);
failPopup.addEventListener("click", (event) => {
  if (event.target === failPopup) closeFailPopup();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !failPopup.classList.contains("hidden")) closeFailPopup();
});

function renderRequestError(message) {
  resultEl.classList.remove("hidden");
  resultEl.replaceChildren(el("div", { class: "card request-error" }, message));
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  for (const name of Object.keys(FIELDS)) touched.add(name);
  if (!validateForm()) return;

  const payload = {};
  for (const name of Object.keys(FIELDS)) payload[name] = Number(document.getElementById(name).value);

  submitBtn.disabled = true;
  resultEl.classList.add("hidden");
  failPopup.classList.add("hidden");
  showRunning();

  try {
    const response = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (response.status === 422 && data.errors) {
      // IN-4: invalid input never produces a verdict; show the field errors only.
      progress.classList.add("hidden");
      for (const [name, message] of Object.entries(data.errors)) setFieldError(name, message);
      return;
    }
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    showStages(data.stages);
    renderResult(data);
  } catch (err) {
    progress.classList.add("hidden");
    renderRequestError(`Could not run the check: ${err.message}. Is the backend running?`);
  } finally {
    submitBtn.disabled = !validateForm();
  }
});

// ---------------------------------------------------------------------------
// Load input limits from the backend (spec/thresholds.yaml → input_limits)
// ---------------------------------------------------------------------------

fetch("/api/config")
  .then((r) => {
    if (!r.ok) throw new Error(`Server returned ${r.status}`);
    return r.json();
  })
  .then((cfg) => {
    limits = cfg.input_limits;
    secondsPerMinute = cfg.seconds_per_minute;
    updateDurationHint();
    validateForm();
  })
  .catch((err) => renderRequestError(`Could not load settings from the backend: ${err.message}.`));
