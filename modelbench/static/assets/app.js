const state = {
  models: [],
  selectedModelIds: [],
  datasets: [],
  datasetImages: [],
  resultsByModel: {},
  imageSize: null,
  activeDatasetId: null,
  activeImageId: null,
  activeDisplayModelId: null,
  activeDetectionIndex: 0,
  activePreviewUrl: null,
  groundTruth: null,
};

const modelList = document.getElementById("modelList");
const modelDescription = document.getElementById("modelDescription");
const datasetCards = document.getElementById("datasetCards");
const datasetStatus = document.getElementById("datasetStatus");
const imageBrowser = document.getElementById("imageBrowser");
const previewImage = document.getElementById("previewImage");
const overlayCanvas = document.getElementById("overlayCanvas");
const emptyState = document.getElementById("emptyState");
const previewTitle = document.getElementById("previewTitle");
const previewSubtitle = document.getElementById("previewSubtitle");
const warningBox = document.getElementById("warningBox");
const statusPill = document.getElementById("statusPill");
const comparisonView = document.getElementById("comparisonView");
const comparisonNote = document.getElementById("comparisonNote");
const detectionList = document.getElementById("detectionList");
const detectionsSummary = document.getElementById("detectionsSummary");
const reloadDatasets = document.getElementById("reloadDatasets");
const datasetCardTemplate = document.getElementById("datasetCardTemplate");
const imageTileTemplate = document.getElementById("imageTileTemplate");
const detectionCardTemplate = document.getElementById("detectionCardTemplate");

function getSelectedModels() {
  return state.models.filter((model) => state.selectedModelIds.includes(model.id));
}

function getActiveDisplayModel() {
  if (state.activeDisplayModelId && state.selectedModelIds.includes(state.activeDisplayModelId)) {
    return state.models.find((model) => model.id === state.activeDisplayModelId) || null;
  }
  return getSelectedModels()[0] || null;
}

function getActiveImageRecord() {
  return state.datasetImages.find((item) => item.id === state.activeImageId) || null;
}

function getResultForModel(modelId) {
  return state.resultsByModel[modelId] || null;
}

function getDetectionForModel(modelId) {
  const result = getResultForModel(modelId);
  if (!result || !result.detections || !result.detections.length) {
    return null;
  }
  return result.detections[state.activeDetectionIndex] || result.detections[0];
}

function getDisplayResult() {
  const activeModel = getActiveDisplayModel();
  if (!activeModel) {
    return null;
  }
  return getResultForModel(activeModel.id);
}

function setStatus(label, busy = false) {
  statusPill.textContent = label;
  statusPill.classList.toggle("busy", busy);
}

function setWarnings(warnings = []) {
  if (!warnings.length) {
    warningBox.hidden = true;
    warningBox.textContent = "";
    return;
  }

  warningBox.hidden = false;
  warningBox.textContent = warnings.join(" ");
}

function setPreview(url, title, subtitle) {
  if (state.activePreviewUrl && state.activePreviewUrl.startsWith("blob:")) {
    URL.revokeObjectURL(state.activePreviewUrl);
  }

  state.activePreviewUrl = url;
  previewImage.src = url;
  previewImage.style.display = "block";
  emptyState.hidden = true;
  previewTitle.textContent = title;
  previewSubtitle.textContent = subtitle;
}

function syncPreviewSubtitle() {
  const imageRecord = getActiveImageRecord();
  const activeModel = getActiveDisplayModel();
  const selectedModels = getSelectedModels();
  if (!imageRecord) {
    previewSubtitle.textContent = "Select a dataset image to run inference.";
    return;
  }

  const parts = [
    state.activeDatasetId,
    `${selectedModels.length} model${selectedModels.length === 1 ? "" : "s"} selected`,
  ];
  if (activeModel) {
    parts.push(`overlay: ${activeModel.label}`);
  }
  previewSubtitle.textContent = parts.join(" • ");
}

function syncModelDescription() {
  const selectedModels = getSelectedModels();
  const activeModel = getActiveDisplayModel();

  if (!selectedModels.length) {
    modelDescription.textContent = "Select at least one model.";
    return;
  }

  const selectedSummary = selectedModels.map((model) => model.label).join(", ");
  const overlaySummary = activeModel ? `Viewing inference: ${activeModel.label}.` : "";
  modelDescription.textContent = `${selectedSummary}. ${overlaySummary} Click any inference card to switch the preview overlay and active model focus.`;
}

function formatPredictionAge(detection, ageKind) {
  if (!detection) {
    return "No detected face";
  }
  if (ageKind === "bucket") {
    return `${detection.age_years} years (${detection.age_bucket})`;
  }
  return `${detection.age_years} years`;
}

function renderModelList() {
  modelList.innerHTML = "";

  state.models.forEach((model) => {
    const option = document.createElement("label");
    option.className = "model-option";
    option.dataset.modelId = model.id;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = model.id;
    checkbox.checked = state.selectedModelIds.includes(model.id);
    checkbox.addEventListener("change", async (event) => {
      const { checked } = event.target;
      if (!checked && state.selectedModelIds.length === 1) {
        event.target.checked = true;
        return;
      }

      if (checked) {
        state.selectedModelIds = [...state.selectedModelIds, model.id];
      } else {
        state.selectedModelIds = state.selectedModelIds.filter((id) => id !== model.id);
        delete state.resultsByModel[model.id];
        if (state.activeDisplayModelId === model.id) {
          state.activeDisplayModelId = state.selectedModelIds[0] || null;
          state.activeDetectionIndex = 0;
        }
      }

      renderModelList();
      syncModelDescription();
      syncPreviewSubtitle();
      renderDetectionList();
      renderComparison();
      drawOverlay();

      const imageRecord = getActiveImageRecord();
      if (imageRecord) {
        await analyzeDatasetImage(imageRecord);
      }
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
    modelList.appendChild(option);
  });
}

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error("Unable to load model presets.");
  }

  const payload = await response.json();
  state.models = payload.models;
  state.selectedModelIds = state.models.slice(0, 1).map((model) => model.id);
  state.activeDisplayModelId = state.selectedModelIds[0] || null;
  renderModelList();
  syncModelDescription();
}

async function loadDatasets() {
  const response = await fetch("/api/datasets");
  if (!response.ok) {
    throw new Error("Unable to load datasets.");
  }

  const payload = await response.json();
  state.datasets = payload.datasets;
  renderDatasetCards();

  if (!state.activeDatasetId && state.datasets.length) {
    await selectDataset(state.datasets[0].id);
  }
}

function renderDatasetCards() {
  datasetCards.innerHTML = "";

  state.datasets.forEach((dataset) => {
    const node = datasetCardTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.datasetId = dataset.id;
    node.querySelector(".dataset-name").textContent = dataset.name;
    node.querySelector(".dataset-description").textContent = dataset.description;
    node.querySelector(".dataset-meta").textContent = `${dataset.image_count} local images`;
    node.classList.toggle("active", dataset.id === state.activeDatasetId);
    node.addEventListener("click", () => selectDataset(dataset.id));
    datasetCards.appendChild(node);
  });
}

function resetAnalysisState() {
  state.activeImageId = null;
  state.activeDetectionIndex = 0;
  state.resultsByModel = {};
  state.groundTruth = null;
  state.imageSize = null;
}

async function selectDataset(datasetId) {
  state.activeDatasetId = datasetId;
  resetAnalysisState();
  renderDatasetCards();
  renderDetectionList();
  renderComparison();
  drawOverlay();

  const response = await fetch(`/api/datasets/${datasetId}/images`);
  if (!response.ok) {
    throw new Error("Unable to load dataset images.");
  }

  const payload = await response.json();
  state.datasetImages = payload.images;
  datasetStatus.textContent = `${state.datasetImages.length} local images loaded. Scroll and select one to test.`;
  renderImageBrowser();
  syncPreviewSubtitle();
}

function renderImageBrowser() {
  imageBrowser.innerHTML = "";

  if (!state.datasetImages.length) {
    imageBrowser.innerHTML = `<p class="empty-inline">No local dataset images found.</p>`;
    return;
  }

  state.datasetImages.forEach((imageRecord) => {
    const node = imageTileTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.imageId = imageRecord.id;
    node.querySelector(".image-thumb").src = imageRecord.thumbnail_url;
    node.querySelector(".image-thumb").alt = imageRecord.label_summary;
    node.querySelector(".image-label").textContent = imageRecord.label_summary;
    node.classList.toggle("active", imageRecord.id === state.activeImageId);
    node.addEventListener("click", () => analyzeDatasetImage(imageRecord));
    imageBrowser.appendChild(node);
  });
}

async function fetchModelAnalysis(imageRecord, model) {
  const formData = new FormData();
  formData.append("dataset_image_id", imageRecord.id);
  formData.append("model_id", model.id);

  const response = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Inference failed.");
  }
  return payload;
}

async function analyzeDatasetImage(imageRecord) {
  const selectedModels = getSelectedModels();
  if (!selectedModels.length) {
    setWarnings(["Select at least one model."]);
    return;
  }

  state.activeImageId = imageRecord.id;
  state.activeDetectionIndex = 0;
  renderImageBrowser();

  setStatus("Running", true);
  setWarnings([]);
  setPreview(imageRecord.image_url, imageRecord.label_summary, "");
  syncPreviewSubtitle();

  const resultEntries = await Promise.all(
    selectedModels.map(async (model) => {
      try {
        const payload = await fetchModelAnalysis(imageRecord, model);
        return { model, payload, error: null };
      } catch (error) {
        return { model, payload: null, error };
      }
    })
  );

  const nextResults = {};
  const warnings = [];
  let firstPayload = null;

  resultEntries.forEach(({ model, payload, error }) => {
    if (error) {
      nextResults[model.id] = {
        model,
        detections: [],
        warnings: [error.message],
        error: error.message,
      };
      warnings.push(`${model.label}: ${error.message}`);
      return;
    }

    if (!firstPayload) {
      firstPayload = payload;
    }

    nextResults[model.id] = {
      model: payload.model,
      detections: payload.detections,
      warnings: payload.warnings,
      error: null,
    };

    payload.warnings.forEach((warning) => {
      warnings.push(`${payload.model.label}: ${warning}`);
    });
  });

  state.resultsByModel = nextResults;
  state.groundTruth = firstPayload ? firstPayload.ground_truth : null;
  state.imageSize = firstPayload ? firstPayload.image : null;

  if (!state.activeDisplayModelId || !state.selectedModelIds.includes(state.activeDisplayModelId)) {
    state.activeDisplayModelId = selectedModels[0].id;
  }

  const activeResult = getDisplayResult();
  if (!activeResult || !activeResult.detections.length) {
    const firstWithDetection = selectedModels.find((model) => {
      const result = nextResults[model.id];
      return result && result.detections && result.detections.length;
    });
    if (firstWithDetection) {
      state.activeDisplayModelId = firstWithDetection.id;
    }
  }

  setWarnings(warnings);
  renderDetectionList();
  renderComparison();
  drawOverlay();
  setStatus("Ready");
  syncModelDescription();
  syncPreviewSubtitle();
}

function renderDetectionList() {
  detectionList.innerHTML = "";

  const activeModel = getActiveDisplayModel();
  const selectedModels = getSelectedModels();

  if (!selectedModels.length) {
    detectionsSummary.textContent = "No model selected.";
    detectionList.innerHTML = `<p class="empty-inline">Select at least one model to inspect detections.</p>`;
    return;
  }

  const detectionCards = [];
  const emptyModels = [];

  selectedModels.forEach((model) => {
    const result = getResultForModel(model.id);
    if (!result || result.error || !result.detections || !result.detections.length) {
      emptyModels.push(model.label);
      return;
    }

    result.detections.forEach((detection, index) => {
      detectionCards.push({
        model,
        detection,
        index,
        isActive: state.activeDisplayModelId === model.id && state.activeDetectionIndex === index,
      });
    });
  });

  if (!detectionCards.length) {
    detectionsSummary.textContent = activeModel ? `Showing ${activeModel.label}` : "No detections yet.";
    detectionList.innerHTML = `<p class="empty-inline">No detected faces available for the selected models.</p>`;
    return;
  }

  const summaryParts = [
    `${detectionCards.length} face card${detectionCards.length === 1 ? "" : "s"} across ${selectedModels.length} model${selectedModels.length === 1 ? "" : "s"}`,
  ];
  if (activeModel) {
    summaryParts.push(`showing ${activeModel.label}`);
  }
  if (emptyModels.length) {
    summaryParts.push(`no faces: ${emptyModels.join(", ")}`);
  }
  detectionsSummary.textContent = summaryParts.join(" • ");

  detectionCards.forEach(({ model, detection, index, isActive }) => {
    const node = detectionCardTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.modelId = model.id;
    node.dataset.detectionIndex = String(index);
    node.querySelector(".detection-thumb").src = detection.face_thumbnail_url;
    node.querySelector(".detection-thumb").alt = `${model.label} ${detection.label}`;
    node.querySelector(".detection-name").textContent = `${model.label} | ${detection.label}`;
    node.querySelector(".confidence-chip").textContent =
      detection.face_confidence == null ? "n/a" : `${Math.round(detection.face_confidence * 100)}%`;
    node.querySelector(".detection-meta").textContent = `${detection.age_years} years • ${detection.gender_label}`;
    node.classList.toggle("active", isActive);
    node.addEventListener("click", () => {
      state.activeDisplayModelId = model.id;
      state.activeDetectionIndex = index;
      syncModelDescription();
      syncPreviewSubtitle();
      renderDetectionList();
      renderComparison();
      drawOverlay();
    });
    detectionList.appendChild(node);
  });
}

function renderComparison(errorMessage = null) {
  const selectedModels = getSelectedModels();
  const activeDisplayModel = getActiveDisplayModel();

  if (errorMessage) {
    comparisonNote.textContent = "Comparison unavailable.";
    comparisonView.className = "comparison-view empty-detail";
    comparisonView.textContent = errorMessage;
    return;
  }

  if (!state.groundTruth) {
    comparisonNote.textContent = "Ground truth and selected model predictions appear here.";
    comparisonView.className = "comparison-view empty-detail";
    comparisonView.textContent = "No dataset image selected.";
    return;
  }

  comparisonNote.textContent =
    state.groundTruth.age_kind === "bucket"
      ? "FairFace uses age buckets. Each selected model is compared against that bucketed label."
      : "UTKFace uses exact ages. Each selected model is compared against the numeric dataset age.";

  const predictionSections = selectedModels
    .map((model) => {
      const result = getResultForModel(model.id);
      const detection = getDetectionForModel(model.id);
      const isActive = activeDisplayModel && activeDisplayModel.id === model.id;

      if (!result) {
        return `
          <section class="comparison-section model-section${isActive ? " active-model" : ""}" data-model-id="${model.id}">
            <h3>${model.label}</h3>
            <p class="empty-inline">No result loaded yet.</p>
          </section>
        `;
      }

      if (result.error) {
        return `
          <section class="comparison-section model-section${isActive ? " active-model" : ""}" data-model-id="${model.id}">
            <h3>${model.label}</h3>
            <p class="empty-inline">${result.error}</p>
          </section>
        `;
      }

      return `
        <section class="comparison-section model-section${isActive ? " active-model" : ""}" data-model-id="${model.id}">
          <h3>${model.label}</h3>
          <dl class="comparison-list">
            <div><dt>Model</dt><dd>${model.label}</dd></div>
            <div><dt>Age</dt><dd>${formatPredictionAge(detection, state.groundTruth.age_kind)}</dd></div>
            <div><dt>Gender</dt><dd>${detection ? detection.gender_label : "No detected face"}</dd></div>
            <div><dt>Face confidence</dt><dd>${detection && detection.face_confidence != null ? `${Math.round(detection.face_confidence * 100)}%` : "n/a"}</dd></div>
            <div><dt>Faces detected</dt><dd>${result.detections.length}</dd></div>
          </dl>
        </section>
      `;
    })
    .join("");

  comparisonView.className = "comparison-view";
  comparisonView.innerHTML = `
    <div class="comparison-grid multi-model-grid">
      <section class="comparison-section">
        <h3>Dataset ground truth</h3>
        <dl class="comparison-list">
          <div><dt>Age</dt><dd>${state.groundTruth.age_display}</dd></div>
          <div><dt>Gender</dt><dd>${state.groundTruth.gender_display}</dd></div>
          <div><dt>Metadata</dt><dd>${state.groundTruth.demographic_label || "n/a"}</dd></div>
        </dl>
      </section>
      ${predictionSections}
    </div>
  `;
}

function drawOverlay() {
  const context = overlayCanvas.getContext("2d");
  const rect = previewImage.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const parentRect = overlayCanvas.parentElement.getBoundingClientRect();
  const displayResult = getDisplayResult();
  const detections = displayResult && displayResult.detections ? displayResult.detections : [];

  overlayCanvas.style.width = `${rect.width}px`;
  overlayCanvas.style.height = `${rect.height}px`;
  overlayCanvas.style.left = `${rect.left - parentRect.left}px`;
  overlayCanvas.style.top = `${rect.top - parentRect.top}px`;

  overlayCanvas.width = rect.width * pixelRatio;
  overlayCanvas.height = rect.height * pixelRatio;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  if (!state.imageSize || !detections.length || !rect.width || !rect.height) {
    return;
  }

  const scaleX = rect.width / state.imageSize.width;
  const scaleY = rect.height / state.imageSize.height;

  detections.forEach((detection, index) => {
    const isActive = index === state.activeDetectionIndex;
    const { x, y, width, height } = detection.bbox;
    const left = x * scaleX;
    const top = y * scaleY;
    const boxWidth = width * scaleX;
    const boxHeight = height * scaleY;

    context.strokeStyle = isActive ? "#0f172a" : "#64748b";
    context.lineWidth = isActive ? 3 : 2;
    context.fillStyle = isActive ? "rgba(15, 23, 42, 0.08)" : "rgba(100, 116, 139, 0.06)";
    context.fillRect(left, top, boxWidth, boxHeight);
    context.strokeRect(left, top, boxWidth, boxHeight);

    context.font = '600 12px "Arial", sans-serif';
    context.fillStyle = isActive ? "#0f172a" : "#334155";
    context.fillRect(left, Math.max(0, top - 18), 24, 18);
    context.fillStyle = "#ffffff";
    context.fillText(String(index + 1), left + 8, Math.max(13, top - 5));
  });
}

function hitTestDetection(offsetX, offsetY) {
  const rect = overlayCanvas.getBoundingClientRect();
  const displayResult = getDisplayResult();
  const detections = displayResult && displayResult.detections ? displayResult.detections : [];

  if (!state.imageSize || !detections.length) {
    return null;
  }

  const scaleX = rect.width / state.imageSize.width;
  const scaleY = rect.height / state.imageSize.height;

  const index = detections.findIndex((detection) => {
    const box = detection.bbox;
    const left = box.x * scaleX;
    const top = box.y * scaleY;
    const width = box.width * scaleX;
    const height = box.height * scaleY;
    return (
      offsetX >= left &&
      offsetX <= left + width &&
      offsetY >= top &&
      offsetY <= top + height
    );
  });
  return index >= 0 ? index : null;
}

comparisonView.addEventListener("click", (event) => {
  const section = event.target.closest("[data-model-id]");
  if (!section) {
    return;
  }

  const modelId = section.dataset.modelId;
  if (!modelId || !state.selectedModelIds.includes(modelId)) {
    return;
  }

  state.activeDisplayModelId = modelId;
  state.activeDetectionIndex = 0;
  syncModelDescription();
  syncPreviewSubtitle();
  renderDetectionList();
  renderComparison();
  drawOverlay();
});

overlayCanvas.addEventListener("click", (event) => {
  const detectionIndex = hitTestDetection(event.offsetX, event.offsetY);
  if (detectionIndex == null || detectionIndex < 0) {
    return;
  }
  state.activeDetectionIndex = detectionIndex;
  renderDetectionList();
  renderComparison();
  drawOverlay();
});

reloadDatasets.addEventListener("click", async () => {
  try {
    await loadDatasets();
  } catch (error) {
    setWarnings([error.message]);
  }
});

previewImage.addEventListener("load", drawOverlay);
window.addEventListener("resize", drawOverlay);

async function init() {
  setStatus("Loading", true);
  try {
    await loadModels();
    await loadDatasets();
    setStatus("Ready");
  } catch (error) {
    setWarnings([error.message]);
    setStatus("Error");
  }
}

init();
