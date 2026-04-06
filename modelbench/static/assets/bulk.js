const state = {
  models: [],
  datasets: [],
  selectedModelIds: [],
  runId: null,
  run: null,
  pollTimer: null,
  settingsTimer: null,
  settingsRequestToken: 0,
};

const bulkModelList = document.getElementById("bulkModelList");
const bulkModelDescription = document.getElementById("bulkModelDescription");
const babyMaxSlider = document.getElementById("babyMaxSlider");
const adultMaxSlider = document.getElementById("adultMaxSlider");
const babyMaxValue = document.getElementById("babyMaxValue");
const adultMaxValue = document.getElementById("adultMaxValue");
const ageClassSummary = document.getElementById("ageClassSummary");
const startBulkRunButton = document.getElementById("startBulkRunButton");
const bulkRunMessage = document.getElementById("bulkRunMessage");
const bulkRunStatusBadge = document.getElementById("bulkRunStatusBadge");
const statusCurrentDataset = document.getElementById("statusCurrentDataset");
const statusCurrentModel = document.getElementById("statusCurrentModel");
const statusTestedImages = document.getElementById("statusTestedImages");
const statusRemainingImages = document.getElementById("statusRemainingImages");
const bulkResultsPanels = document.getElementById("bulkResultsPanels");

function getSelectedModels() {
  return state.models.filter((model) => state.selectedModelIds.includes(model.id));
}

function getActiveRunStatus() {
  return state.run ? state.run.status : "idle";
}

function isRunActive() {
  return ["queued", "running"].includes(getActiveRunStatus());
}

function formatPercent(value, testedCount) {
  if (!testedCount) {
    return "--";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function getStatusLabel(status) {
  if (status === "queued") return "Queued";
  if (status === "running") return "Running";
  if (status === "done") return "Done";
  if (status === "error") return "Error";
  return "Idle";
}

function normalizeSliderValues(changedSlider) {
  let babyMax = Number(babyMaxSlider.value);
  let adultMax = Number(adultMaxSlider.value);

  if (babyMax >= adultMax) {
    if (changedSlider === "baby") {
      adultMax = Math.min(116, babyMax + 1);
      if (adultMax <= babyMax) {
        babyMax = adultMax - 1;
      }
    } else {
      babyMax = Math.max(0, adultMax - 1);
    }
  }

  babyMaxSlider.value = String(babyMax);
  adultMaxSlider.value = String(adultMax);
  return { babyMax, adultMax };
}

function renderAgeClassSummary() {
  const { babyMax, adultMax } = normalizeSliderValues();
  babyMaxValue.textContent = `Baby ages: 0-${babyMax}`;
  adultMaxValue.textContent = `Man/Woman ages: ${babyMax + 1}-${adultMax}`;
  ageClassSummary.innerHTML = `
    <div class="age-class-pill"><span>Baby</span><strong>0-${babyMax}</strong></div>
    <div class="age-class-pill"><span>Man/Woman</span><strong>${babyMax + 1}-${adultMax}</strong></div>
    <div class="age-class-pill"><span>Old</span><strong>${adultMax + 1}+</strong></div>
  `;
}

function hasRunSnapshot() {
  return Boolean(state.runId && state.run);
}

function renderModelList() {
  bulkModelList.innerHTML = "";

  state.models.forEach((model) => {
    const option = document.createElement("label");
    option.className = "model-option";
    option.dataset.modelId = model.id;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = model.id;
    checkbox.checked = state.selectedModelIds.includes(model.id);
    checkbox.disabled = isRunActive();
    checkbox.addEventListener("change", () => {
      if (!checkbox.checked && state.selectedModelIds.length === 1) {
        checkbox.checked = true;
        return;
      }

      if (checkbox.checked) {
        state.selectedModelIds = [...state.selectedModelIds, model.id];
      } else {
        state.selectedModelIds = state.selectedModelIds.filter((id) => id !== model.id);
      }
      renderModelList();
      renderModelDescription();
      renderResultsPanels();
    });

    const content = document.createElement("div");
    content.className = "model-option-copy";

    const title = document.createElement("div");
    title.className = "model-option-title";
    title.textContent = model.label;

    const meta = document.createElement("div");
    meta.className = "model-option-meta";
    meta.textContent = model.description;

    content.append(title, meta);
    option.append(checkbox, content);
    option.classList.toggle("active", checkbox.checked);
    bulkModelList.appendChild(option);
  });
}

function renderModelDescription() {
  const selectedModels = getSelectedModels();
  if (!selectedModels.length) {
    bulkModelDescription.textContent = "Select at least one model.";
    return;
  }

  bulkModelDescription.textContent = `${selectedModels.length} model${selectedModels.length === 1 ? "" : "s"} selected: ${selectedModels.map((model) => model.label).join(", ")}.`;
}

function renderRunStatus() {
  const run = state.run;
  const status = getActiveRunStatus();
  bulkRunStatusBadge.textContent = getStatusLabel(status);
  bulkRunStatusBadge.classList.toggle("busy", status === "running" || status === "queued");

  if (!run) {
    statusCurrentDataset.textContent = "-";
    statusCurrentModel.textContent = "-";
    statusTestedImages.textContent = "0";
    statusRemainingImages.textContent = "0";
    return;
  }

  const progress = run.progress;
  const datasetMap = Object.fromEntries(run.datasets.map((item) => [item.id, item.name]));
  const modelMap = Object.fromEntries(run.selected_models.map((item) => [item.id, item.label]));

  statusCurrentDataset.textContent = datasetMap[progress.current_dataset_id] || "-";
  statusCurrentModel.textContent = modelMap[progress.current_model_id] || "-";
  statusTestedImages.textContent = String(progress.tested_images);
  statusRemainingImages.textContent = String(Math.max(progress.total_images - progress.tested_images, 0));
}

function renderResultsPanels() {
  bulkResultsPanels.innerHTML = "";

  const selectedModels = getSelectedModels();
  const run = state.run;
  const datasets = run ? run.datasets : state.datasets;
  const models = run ? run.selected_models : selectedModels;

  datasets.forEach((dataset) => {
    const panel = document.createElement("section");
    panel.className = "panel bulk-dataset-panel";

    const heading = document.createElement("div");
    heading.className = "panel-header";
    heading.innerHTML = `
      <div>
        <h2>${dataset.name}</h2>
        <p class="help-text">${dataset.image_count} local images</p>
      </div>
    `;

    const table = document.createElement("table");
    table.className = "bulk-results-table";
    table.innerHTML = `
      <thead>
        <tr>
          <th>Model</th>
          <th>Tested</th>
          <th>Gender Accuracy</th>
          <th>Age-Class Accuracy</th>
          <th>Missed Detections</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;

    const tbody = table.querySelector("tbody");
    models.forEach((model) => {
      const rowData = run ? run.results?.[dataset.id]?.models?.[model.id] : null;
      const testedCount = rowData ? rowData.tested_count : 0;
      const totalCount = rowData ? rowData.total_count : dataset.image_count;
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>
          <div class="metric-value">${model.label}</div>
          <div class="metric-subtext">${model.provider}</div>
        </td>
        <td>
          <div class="metric-value">${testedCount} / ${totalCount}</div>
        </td>
        <td>
          <div class="metric-value">${rowData ? formatPercent(rowData.gender_accuracy, rowData.tested_count) : "--"}</div>
          <div class="metric-subtext">${rowData ? `${rowData.gender_correct_count} correct` : "Waiting to run"}</div>
        </td>
        <td>
          <div class="metric-value">${rowData ? formatPercent(rowData.age_class_accuracy, rowData.tested_count) : "--"}</div>
          <div class="metric-subtext">${rowData ? `${rowData.age_class_correct_count} correct` : "Waiting to run"}</div>
          ${
            rowData
              ? Object.values(rowData.age_class_breakdown || {})
                  .map(
                    (entry) =>
                      `<div class="metric-subtext">${entry.label}: ${entry.correct_count} / ${entry.total_count}</div>`,
                  )
                  .join("")
              : ""
          }
        </td>
        <td>
          <div class="metric-value">${rowData ? rowData.missed_detection_count : 0}</div>
        </td>
        <td>
          <span class="bulk-row-status ${rowData ? rowData.status : "queued"}">${getStatusLabel(rowData ? rowData.status : "queued")}</span>
          ${rowData && rowData.last_error ? `<div class="metric-subtext">${rowData.last_error}</div>` : ""}
        </td>
      `;
      tbody.appendChild(row);
    });

    panel.append(heading, table);
    bulkResultsPanels.appendChild(panel);
  });
}

function renderRunMessage(message, isError = false) {
  bulkRunMessage.textContent = message;
  bulkRunMessage.style.color = isError ? "#b91c1c" : "";
}

function setControlsDisabled(disabled) {
  startBulkRunButton.disabled = disabled;
  Array.from(bulkModelList.querySelectorAll("input[type='checkbox']")).forEach((input) => {
    input.disabled = disabled;
  });
}

function setSliderState(disabled) {
  babyMaxSlider.disabled = disabled;
  adultMaxSlider.disabled = disabled;
}

async function loadInitialData() {
  const [modelsResponse, datasetsResponse] = await Promise.all([
    fetch("/api/models"),
    fetch("/api/datasets"),
  ]);

  if (!modelsResponse.ok) {
    throw new Error("Unable to load models.");
  }
  if (!datasetsResponse.ok) {
    throw new Error("Unable to load datasets.");
  }

  state.models = (await modelsResponse.json()).models;
  state.datasets = (await datasetsResponse.json()).datasets;
  state.selectedModelIds = state.models.map((model) => model.id);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollRun() {
  if (!state.runId) {
    return;
  }

  try {
    const response = await fetch(`/api/bulk-runs/${state.runId}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to fetch bulk run progress.");
    }

    state.run = payload.run;
    renderRunStatus();
    renderResultsPanels();
    setSliderState(false);
    const active = isRunActive();
    setControlsDisabled(active);
    if (active) {
      renderRunMessage("Bulk inference is running. Metrics update as each image finishes.");
      state.pollTimer = window.setTimeout(pollRun, 1000);
    } else {
      renderRunMessage("Bulk inference finished. Adjust age classes to recalculate saved results or run again.");
      stopPolling();
    }
  } catch (error) {
    renderRunMessage(error.message, true);
    setControlsDisabled(false);
    setSliderState(false);
    stopPolling();
  }
}

async function startBulkRun() {
  const selectedModels = getSelectedModels();
  if (!selectedModels.length) {
    renderRunMessage("Select at least one model before starting.", true);
    return;
  }

  const { babyMax, adultMax } = normalizeSliderValues();
  renderAgeClassSummary();
  setControlsDisabled(true);
  setSliderState(true);
  renderRunMessage("Starting bulk inference run...");

  try {
    const response = await fetch("/api/bulk-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_ids: selectedModels.map((model) => model.id),
        baby_max: babyMax,
        adult_max: adultMax,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to start bulk inference.");
    }

    stopPolling();
    state.runId = payload.run_id;
    state.run = payload.run;
    renderRunStatus();
    renderResultsPanels();
    setSliderState(false);
    if (isRunActive()) {
      state.pollTimer = window.setTimeout(pollRun, 1000);
    } else {
      setControlsDisabled(false);
    }
  } catch (error) {
    renderRunMessage(error.message, true);
    setControlsDisabled(false);
    setSliderState(false);
  }
}

async function pushRunSettings() {
  if (!hasRunSnapshot()) {
    return;
  }

  const { babyMax, adultMax } = normalizeSliderValues();
  const token = ++state.settingsRequestToken;

  try {
    const response = await fetch(`/api/bulk-runs/${state.runId}/settings`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        baby_max: babyMax,
        adult_max: adultMax,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to update age classes.");
    }

    if (token !== state.settingsRequestToken) {
      return;
    }

    state.run = payload.run;
    renderRunStatus();
    renderResultsPanels();
    if (isRunActive()) {
      renderRunMessage("Age classes updated. Saved results are being recomputed while the run continues.");
    } else {
      renderRunMessage("Age classes updated. Age-class metrics were recalculated from saved predictions.");
    }
  } catch (error) {
    renderRunMessage(error.message, true);
  }
}

function scheduleSettingsUpdate() {
  if (!hasRunSnapshot()) {
    return;
  }
  if (state.settingsTimer) {
    window.clearTimeout(state.settingsTimer);
  }
  state.settingsTimer = window.setTimeout(() => {
    state.settingsTimer = null;
    pushRunSettings();
  }, 200);
}

babyMaxSlider.addEventListener("input", () => {
  normalizeSliderValues("baby");
  renderAgeClassSummary();
  scheduleSettingsUpdate();
});

adultMaxSlider.addEventListener("input", () => {
  normalizeSliderValues("adult");
  renderAgeClassSummary();
  scheduleSettingsUpdate();
});

startBulkRunButton.addEventListener("click", startBulkRun);

async function init() {
  try {
    await loadInitialData();
    renderAgeClassSummary();
    renderModelList();
    renderModelDescription();
    renderRunStatus();
    renderResultsPanels();
    renderRunMessage("Choose models and age classes, then start a run.");
    setSliderState(false);
  } catch (error) {
    renderRunMessage(error.message, true);
  }
}

init();
