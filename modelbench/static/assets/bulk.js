const state = {
  models: [],
  datasets: [],
  selectedModelIds: [],
  runId: null,
  run: null,
  presets: null,
  pollTimer: null,
  settingsTimer: null,
  settingsRequestToken: 0,
  activePreview: null,
};

const bulkModelList = document.getElementById("bulkModelList");
const bulkModelDescription = document.getElementById("bulkModelDescription");
const babyMaxSlider = document.getElementById("babyMaxSlider");
const adultMaxSlider = document.getElementById("adultMaxSlider");
const babyMaxValue = document.getElementById("babyMaxValue");
const adultMaxValue = document.getElementById("adultMaxValue");
const ageClassSummary = document.getElementById("ageClassSummary");
const presetGroups = document.getElementById("presetGroups");
const startBulkRunButton = document.getElementById("startBulkRunButton");
const bulkRunMessage = document.getElementById("bulkRunMessage");
const bulkRunStatusBadge = document.getElementById("bulkRunStatusBadge");
const statusCurrentDataset = document.getElementById("statusCurrentDataset");
const statusCurrentModel = document.getElementById("statusCurrentModel");
const statusTestedImages = document.getElementById("statusTestedImages");
const statusRemainingImages = document.getElementById("statusRemainingImages");
const bulkResultsPanels = document.getElementById("bulkResultsPanels");
const classPreviewModal = document.getElementById("classPreviewModal");
const classPreviewBackdrop = document.getElementById("classPreviewBackdrop");
const closeClassPreviewButton = document.getElementById("closeClassPreviewButton");
const classPreviewTitle = document.getElementById("classPreviewTitle");
const classPreviewSummary = document.getElementById("classPreviewSummary");
const classPreviewContent = document.getElementById("classPreviewContent");

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

function getCurrentSliderSettings() {
  return {
    babyMax: Number(babyMaxSlider.value),
    adultMax: Number(adultMaxSlider.value),
  };
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

function formatPresetRange(preset) {
  return `Baby 0-${preset.baby_max} · Man/Woman ${preset.baby_max + 1}-${preset.adult_max} · Old ${preset.adult_max + 1}+`;
}

function renderPresetSection(title, presets) {
  if (!presets.length) {
    return "";
  }

  const { babyMax, adultMax } = getCurrentSliderSettings();
  const cards = presets
    .map((preset) => {
      const active = preset.baby_max === babyMax && preset.adult_max === adultMax;
      return `
        <button
          class="preset-card ${active ? "active" : ""}"
          type="button"
          data-action="apply-preset"
          data-baby-max="${preset.baby_max}"
          data-adult-max="${preset.adult_max}"
        >
          <div class="preset-card-title">${preset.label}</div>
          <div class="preset-card-range">${formatPresetRange(preset)}</div>
          <div class="preset-card-meta">Accuracy ${(preset.score_accuracy * 100).toFixed(1)}%</div>
          <div class="preset-card-meta">${preset.tested_images} tested images</div>
        </button>
      `;
    })
    .join("");

  return `
    <section class="preset-section">
      <h3>${title}</h3>
      <div class="preset-card-list">${cards}</div>
    </section>
  `;
}

function renderPresetGroups() {
  if (!hasRunSnapshot()) {
    presetGroups.innerHTML = `<div class="preset-empty">Run bulk inference to compute live preset recommendations.</div>`;
    return;
  }

  if (!state.presets) {
    presetGroups.innerHTML = `<div class="preset-empty">Computing preset recommendations from saved results…</div>`;
    return;
  }

  const sections = [
    renderPresetSection("By dataset", state.presets.dataset_presets || []),
    renderPresetSection("By model", state.presets.model_presets || []),
    renderPresetSection("By model + dataset", state.presets.combination_presets || []),
  ].filter(Boolean);

  if (!sections.length) {
    presetGroups.innerHTML = `<div class="preset-empty">Preset recommendations will appear as soon as images finish processing.</div>`;
    return;
  }

  presetGroups.innerHTML = sections.join("");
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
      const breakdownMarkup = rowData
        ? Object.entries(rowData.age_class_breakdown || {})
            .map(
              ([classId, entry]) => `
                <button
                  class="bulk-breakdown-button"
                  type="button"
                  data-action="open-class-preview"
                  data-dataset-id="${dataset.id}"
                  data-model-id="${model.id}"
                  data-class-id="${classId}"
                >
                  ${entry.label}: ${entry.correct_count} / ${entry.total_count}
                </button>
              `,
            )
            .join("")
        : "";
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
          ${breakdownMarkup}
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

async function loadPresets() {
  if (!state.runId) {
    state.presets = null;
    renderPresetGroups();
    return;
  }

  try {
    const response = await fetch(`/api/bulk-runs/${state.runId}/presets`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load presets.");
    }
    state.presets = payload.presets;
    renderPresetGroups();
  } catch (error) {
    presetGroups.innerHTML = `<div class="preset-empty">${error.message}</div>`;
  }
}

function closeClassPreview() {
  state.activePreview = null;
  classPreviewModal.classList.add("hidden");
  classPreviewModal.setAttribute("aria-hidden", "true");
}

async function loadClassPreview() {
  if (!state.activePreview) {
    return;
  }

  const params = new URLSearchParams({
    dataset_id: state.activePreview.datasetId,
    model_id: state.activePreview.modelId,
    class_id: state.activePreview.classId,
  });

  classPreviewSummary.textContent = "Loading preview…";
  classPreviewContent.innerHTML = "";

  try {
    const response = await fetch(`/api/bulk-runs/${state.runId}/class-preview?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load class preview.");
    }

    const preview = payload.preview;
    state.activePreview = {
      datasetId: preview.dataset.id,
      modelId: preview.model.id,
      classId: preview.class_id,
    };
    classPreviewTitle.textContent = `${preview.dataset.name} · ${preview.model.label} · ${preview.class_label}`;
    classPreviewSummary.textContent = `Showing ${preview.summary.total_count} actual ${preview.class_label} images. ${preview.summary.correct_count} predicted as ${preview.class_label}.`;

    if (!preview.items.length) {
      classPreviewContent.innerHTML = `<div class="bulk-preview-empty">No completed images are available for this class yet.</div>`;
      return;
    }

    classPreviewContent.innerHTML = preview.items
      .map(
        (item) => `
          <article class="bulk-preview-card ${item.correct_for_clicked_class ? "correct" : "incorrect"}">
            <img class="bulk-preview-image" src="${item.image_url}" alt="${item.dataset_image_id}" loading="lazy" />
            <div class="bulk-preview-copy">
              <div class="bulk-preview-id">${item.dataset_image_id}</div>
              <div class="bulk-preview-line">Actual: ${item.actual_class_labels.join(", ")}</div>
              <div class="bulk-preview-line">Actual age: ${item.actual_age_display ?? "n/a"}</div>
              <div class="bulk-preview-line">Predicted: ${item.predicted_class_label}</div>
              <div class="bulk-preview-line">Predicted age: ${item.predicted_age_years ?? "n/a"}</div>
              ${item.missed_detection ? `<div class="bulk-preview-badge">Missed detection</div>` : ""}
            </div>
          </article>
        `,
      )
      .join("");
  } catch (error) {
    classPreviewSummary.textContent = error.message;
    classPreviewContent.innerHTML = `<div class="bulk-preview-empty">Preview unavailable.</div>`;
  }
}

async function openClassPreview(datasetId, modelId, classId) {
  if (!state.runId) {
    renderRunMessage("Start a bulk run before opening class previews.", true);
    return;
  }

  state.activePreview = { datasetId, modelId, classId };
  classPreviewModal.classList.remove("hidden");
  classPreviewModal.setAttribute("aria-hidden", "false");
  await loadClassPreview();
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
    loadPresets();
    if (state.activePreview) {
      loadClassPreview();
    }
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
    state.presets = null;
    renderRunStatus();
    renderResultsPanels();
    renderPresetGroups();
    loadPresets();
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
    renderPresetGroups();
    if (state.activePreview) {
      loadClassPreview();
    }
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

function applyPreset(babyMax, adultMax) {
  babyMaxSlider.value = String(babyMax);
  adultMaxSlider.value = String(adultMax);
  renderAgeClassSummary();
  renderPresetGroups();
  scheduleSettingsUpdate();
}

babyMaxSlider.addEventListener("input", () => {
  normalizeSliderValues("baby");
  renderAgeClassSummary();
  renderPresetGroups();
  scheduleSettingsUpdate();
});

adultMaxSlider.addEventListener("input", () => {
  normalizeSliderValues("adult");
  renderAgeClassSummary();
  renderPresetGroups();
  scheduleSettingsUpdate();
});

startBulkRunButton.addEventListener("click", startBulkRun);
closeClassPreviewButton.addEventListener("click", closeClassPreview);
classPreviewBackdrop.addEventListener("click", closeClassPreview);
document.addEventListener("click", (event) => {
  const trigger = event.target.closest("[data-action='open-class-preview']");
  if (trigger) {
    openClassPreview(
      trigger.dataset.datasetId,
      trigger.dataset.modelId,
      trigger.dataset.classId,
    );
    return;
  }

  const presetTrigger = event.target.closest("[data-action='apply-preset']");
  if (presetTrigger) {
    applyPreset(
      Number(presetTrigger.dataset.babyMax),
      Number(presetTrigger.dataset.adultMax),
    );
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.activePreview) {
    closeClassPreview();
  }
});

async function init() {
  try {
    await loadInitialData();
    renderAgeClassSummary();
    renderModelList();
    renderModelDescription();
    renderRunStatus();
    renderResultsPanels();
    renderPresetGroups();
    renderRunMessage("Choose models and age classes, then start a run.");
    setSliderState(false);
  } catch (error) {
    renderRunMessage(error.message, true);
  }
}

init();
