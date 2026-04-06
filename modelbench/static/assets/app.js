const state = {
  models: [],
  datasets: [],
  datasetImages: [],
  detections: [],
  imageSize: null,
  activeDatasetId: null,
  activeImageId: null,
  activeDetectionId: null,
  activePreviewUrl: null,
  groundTruth: null,
};

const modelSelect = document.getElementById("modelSelect");
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

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error("Unable to load model presets.");
  }
  const payload = await response.json();
  state.models = payload.models;

  modelSelect.innerHTML = "";
  state.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    if (index === 0) {
      option.selected = true;
    }
    modelSelect.appendChild(option);
  });

  syncModelDescription();
}

function syncModelDescription() {
  const active = state.models.find((model) => model.id === modelSelect.value);
  modelDescription.textContent = active ? active.description : "";
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

async function selectDataset(datasetId) {
  state.activeDatasetId = datasetId;
  state.activeImageId = null;
  state.activeDetectionId = null;
  state.detections = [];
  state.groundTruth = null;
  state.imageSize = null;
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

async function analyzeDatasetImage(imageRecord) {
  state.activeImageId = imageRecord.id;
  state.activeDetectionId = null;
  renderImageBrowser();

  setStatus("Running", true);
  setWarnings([]);
  setPreview(imageRecord.image_url, imageRecord.label_summary, `${state.activeDatasetId} • ${modelSelect.value}`);

  const formData = new FormData();
  formData.append("dataset_image_id", imageRecord.id);
  formData.append("model_id", modelSelect.value);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Inference failed.");
    }

    state.imageSize = payload.image;
    state.detections = payload.detections;
    state.groundTruth = payload.ground_truth;
    state.activeDetectionId = payload.detections.length ? payload.detections[0].id : null;
    setWarnings(payload.warnings);
    renderDetectionList();
    renderComparison();
    drawOverlay();
    setStatus("Ready");
  } catch (error) {
    state.detections = [];
    state.groundTruth = null;
    state.imageSize = null;
    state.activeDetectionId = null;
    renderDetectionList();
    renderComparison(error.message);
    drawOverlay();
    setWarnings([error.message]);
    setStatus("Error");
  }
}

function getActiveDetection() {
  return state.detections.find((item) => item.id === state.activeDetectionId) || null;
}

function renderDetectionList() {
  detectionList.innerHTML = "";

  if (!state.detections.length) {
    detectionsSummary.textContent = "No detections yet.";
    detectionList.innerHTML = `<p class="empty-inline">Select an image to inspect detected faces.</p>`;
    return;
  }

  detectionsSummary.textContent = `${state.detections.length} face${state.detections.length === 1 ? "" : "s"} detected`;
  state.detections.forEach((detection) => {
    const node = detectionCardTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.detectionId = detection.id;
    node.querySelector(".detection-thumb").src = detection.face_thumbnail_url;
    node.querySelector(".detection-thumb").alt = detection.label;
    node.querySelector(".detection-name").textContent = detection.label;
    node.querySelector(".confidence-chip").textContent = `${Math.round(detection.face_confidence * 100)}%`;
    node.querySelector(".detection-meta").textContent = `${detection.age_years} years • ${detection.gender_label}`;
    node.classList.toggle("active", detection.id === state.activeDetectionId);
    node.addEventListener("click", () => {
      state.activeDetectionId = detection.id;
      renderDetectionList();
      renderComparison();
      drawOverlay();
    });
    detectionList.appendChild(node);
  });
}

function renderComparison(errorMessage = null) {
  const detection = getActiveDetection();

  if (errorMessage) {
    comparisonNote.textContent = "Comparison unavailable.";
    comparisonView.className = "comparison-view empty-detail";
    comparisonView.textContent = errorMessage;
    return;
  }

  if (!state.groundTruth) {
    comparisonNote.textContent = "Ground truth and predictions appear here.";
    comparisonView.className = "comparison-view empty-detail";
    comparisonView.textContent = "No dataset image selected.";
    return;
  }

  comparisonNote.textContent =
    state.groundTruth.age_kind === "bucket"
      ? "FairFace uses age buckets, so the predicted age bucket is shown beside the dataset label."
      : "UTKFace uses exact ages, so the numeric prediction is shown beside the dataset age.";

  const predictedAgeBlock = detection
    ? state.groundTruth.age_kind === "bucket"
      ? `${detection.age_years} years (${detection.age_bucket})`
      : `${detection.age_years} years`
    : "No detected face selected";

  const predictedGenderBlock = detection ? detection.gender_label : "No detected face selected";

  comparisonView.className = "comparison-view";
  comparisonView.innerHTML = `
    <div class="comparison-grid">
      <section class="comparison-section">
        <h3>Dataset ground truth</h3>
        <dl class="comparison-list">
          <div><dt>Age</dt><dd>${state.groundTruth.age_display}</dd></div>
          <div><dt>Gender</dt><dd>${state.groundTruth.gender_display}</dd></div>
          <div><dt>Metadata</dt><dd>${state.groundTruth.demographic_label || "n/a"}</dd></div>
        </dl>
      </section>
      <section class="comparison-section">
        <h3>Selected prediction</h3>
        <dl class="comparison-list">
          <div><dt>Age</dt><dd>${predictedAgeBlock}</dd></div>
          <div><dt>Gender</dt><dd>${predictedGenderBlock}</dd></div>
          <div><dt>Face confidence</dt><dd>${detection ? `${Math.round(detection.face_confidence * 100)}%` : "n/a"}</dd></div>
        </dl>
      </section>
    </div>
  `;
}

function drawOverlay() {
  const context = overlayCanvas.getContext("2d");
  const rect = previewImage.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const parentRect = overlayCanvas.parentElement.getBoundingClientRect();

  overlayCanvas.style.width = `${rect.width}px`;
  overlayCanvas.style.height = `${rect.height}px`;
  overlayCanvas.style.left = `${rect.left - parentRect.left}px`;
  overlayCanvas.style.top = `${rect.top - parentRect.top}px`;

  overlayCanvas.width = rect.width * pixelRatio;
  overlayCanvas.height = rect.height * pixelRatio;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  if (!state.imageSize || !state.detections.length || !rect.width || !rect.height) {
    return;
  }

  const scaleX = rect.width / state.imageSize.width;
  const scaleY = rect.height / state.imageSize.height;

  state.detections.forEach((detection, index) => {
    const isActive = detection.id === state.activeDetectionId;
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
  if (!state.imageSize || !state.detections.length) {
    return null;
  }

  const rect = overlayCanvas.getBoundingClientRect();
  const scaleX = rect.width / state.imageSize.width;
  const scaleY = rect.height / state.imageSize.height;

  return (
    state.detections.find((detection) => {
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
    }) || null
  );
}

modelSelect.addEventListener("change", async () => {
  syncModelDescription();
  if (!state.activeImageId) {
    return;
  }
  const imageRecord = state.datasetImages.find((item) => item.id === state.activeImageId);
  if (imageRecord) {
    await analyzeDatasetImage(imageRecord);
  }
});

overlayCanvas.addEventListener("click", (event) => {
  const detection = hitTestDetection(event.offsetX, event.offsetY);
  if (detection) {
    state.activeDetectionId = detection.id;
    renderDetectionList();
    renderComparison();
    drawOverlay();
  }
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
