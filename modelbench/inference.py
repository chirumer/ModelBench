from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from mtcnn import MTCNN
import tensorflow as tf

from modelbench.catalog import DATASET_DEFINITIONS, MODEL_PRESETS
from modelbench.ssrnet_model import SSRNet, SSRNetGeneral


IMAGE_SIZE = 64
STAGE_NUM = [3, 3, 3]
LAMBDA_LOCAL = 1.0
LAMBDA_D = 1.0
FACE_PADDING = 0.4
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class InferenceInputError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _encode_image_data_url(image_array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_array.astype(np.uint8)).save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _fairface_age_bucket(age_years: int) -> str:
    if age_years <= 2:
        return "0-2"
    if age_years <= 9:
        return "3-9"
    if age_years <= 19:
        return "10-19"
    if age_years <= 29:
        return "20-29"
    if age_years <= 39:
        return "30-39"
    if age_years <= 49:
        return "40-49"
    if age_years <= 59:
        return "50-59"
    if age_years <= 69:
        return "60-69"
    return "more than 70"


class InferenceService:
    def __init__(self) -> None:
        tf.get_logger().setLevel("ERROR")
        self.detector = MTCNN()
        self._model_cache: dict[str, dict] = {}
        self._dataset_manifest_cache: dict[str, dict] = {}
        self._dataset_image_index: dict[str, tuple[str, dict]] | None = None

    def list_models(self) -> list[dict]:
        return [
            {
                "id": preset.id,
                "label": preset.label,
                "description": preset.description,
            }
            for preset in MODEL_PRESETS.values()
        ]

    def list_datasets(self) -> list[dict]:
        items = []
        for dataset_id, definition in DATASET_DEFINITIONS.items():
            manifest = self._load_manifest(dataset_id)
            items.append(
                {
                    "id": dataset_id,
                    "name": definition.name,
                    "description": definition.description,
                    "image_count": len(manifest["images"]),
                }
            )
        return items

    def list_dataset_images(self, dataset_id: str) -> list[dict]:
        definition = DATASET_DEFINITIONS.get(dataset_id)
        if definition is None:
            raise InferenceInputError(f"Unknown dataset '{dataset_id}'.", status_code=404)
        manifest = self._load_manifest(dataset_id)
        return [
            {
                "id": entry["id"],
                "image_url": entry["image_url"],
                "thumbnail_url": entry["thumbnail_url"],
                "label_summary": entry["label_summary"],
            }
            for entry in manifest["images"]
        ]

    def analyze_dataset_image(self, dataset_image_id: str, model_id: str) -> dict:
        dataset_id, record = self._lookup_dataset_image(dataset_image_id)
        manifest = self._load_manifest(dataset_id)
        image_path = Path(record["image_path"])
        image = self._load_image_from_path(image_path)
        definition = DATASET_DEFINITIONS[dataset_id]
        return self._analyze_image(
            image,
            model_id=model_id,
            source="dataset",
            dataset={
                "id": dataset_id,
                "name": definition.name,
                "description": definition.description,
                "image_id": record["id"],
                "image_url": record["image_url"],
                "thumbnail_url": record["thumbnail_url"],
            },
            ground_truth=record["ground_truth"],
            manifest_label_summary=record["label_summary"],
            manifest=manifest,
        )

    def analyze_upload(self, filename: str, data: bytes, model_id: str) -> dict:
        suffix = Path(filename or "upload").suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise InferenceInputError("Unsupported image type. Use JPG, PNG, or WebP.")
        image = self._load_image_from_bytes(data)
        return self._analyze_image(image, model_id=model_id, source="upload")

    def _load_manifest(self, dataset_id: str) -> dict:
        if dataset_id in self._dataset_manifest_cache:
            return self._dataset_manifest_cache[dataset_id]

        definition = DATASET_DEFINITIONS.get(dataset_id)
        if definition is None:
            raise InferenceInputError(f"Unknown dataset '{dataset_id}'.", status_code=404)

        if definition.manifest_path.exists():
            payload = json.loads(definition.manifest_path.read_text())
        else:
            payload = {"dataset": {"id": dataset_id, "name": definition.name}, "images": []}

        self._dataset_manifest_cache[dataset_id] = payload
        self._dataset_image_index = None
        return payload

    def _lookup_dataset_image(self, dataset_image_id: str) -> tuple[str, dict]:
        if self._dataset_image_index is None:
            index = {}
            for dataset_id in DATASET_DEFINITIONS:
                for entry in self._load_manifest(dataset_id)["images"]:
                    index[entry["id"]] = (dataset_id, entry)
            self._dataset_image_index = index

        record = self._dataset_image_index.get(dataset_image_id)
        if record is None:
            raise InferenceInputError(
                f"Unknown dataset_image_id '{dataset_image_id}'.",
                status_code=404,
            )
        return record

    def _get_models_for_preset(self, model_id: str) -> dict:
        if model_id not in MODEL_PRESETS:
            raise InferenceInputError(f"Unknown model_id '{model_id}'.", status_code=400)

        if model_id not in self._model_cache:
            preset = MODEL_PRESETS[model_id]
            age_model = SSRNet(IMAGE_SIZE, STAGE_NUM, LAMBDA_LOCAL, LAMBDA_D)()
            age_model.load_weights(preset.age_weight_path)
            gender_model = SSRNetGeneral(IMAGE_SIZE, STAGE_NUM, LAMBDA_LOCAL, LAMBDA_D)()
            gender_model.load_weights(preset.gender_weight_path)
            self._model_cache[model_id] = {
                "age_model": age_model,
                "gender_model": gender_model,
            }

        return self._model_cache[model_id]

    def _load_image_from_path(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _load_image_from_bytes(self, data: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(data)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except UnidentifiedImageError as exc:
            raise InferenceInputError("Unsupported image content. Use JPG, PNG, or WebP.") from exc

    def _analyze_image(
        self,
        image: np.ndarray,
        *,
        model_id: str,
        source: str,
        dataset: dict | None = None,
        ground_truth: dict | None = None,
        manifest_label_summary: str | None = None,
        manifest: dict | None = None,
    ) -> dict:
        bundle = self._get_models_for_preset(model_id)
        height, width = image.shape[:2]
        raw_detections = self.detector.detect_faces(image)
        valid_entries = []

        for detection in raw_detections:
            confidence = float(detection.get("confidence", 0.0))
            x, y, box_width, box_height = detection.get("box", [0, 0, 0, 0])
            x = max(int(x), 0)
            y = max(int(y), 0)
            box_width = max(int(box_width), 0)
            box_height = max(int(box_height), 0)

            if box_width == 0 or box_height == 0:
                continue

            x2 = min(x + box_width, width)
            y2 = min(y + box_height, height)
            if x2 <= x or y2 <= y:
                continue

            padded_crop = self._extract_face(image, x, y, x2, y2, width, height)
            face_crop = np.asarray(
                Image.fromarray(image[y:y2, x:x2]).resize((120, 120), Image.BILINEAR),
                dtype=np.uint8,
            )
            valid_entries.append(
                {
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": x2 - x,
                        "height": y2 - y,
                    },
                    "face_confidence": confidence,
                    "face": padded_crop,
                    "face_thumb": face_crop,
                }
            )

        valid_entries.sort(key=lambda entry: (entry["bbox"]["x"], entry["bbox"]["y"]))

        warnings = []
        if not valid_entries:
            warnings.append("No faces detected in this image.")

        detections = []
        if valid_entries:
            face_batch = np.stack([entry["face"] for entry in valid_entries]).astype(np.float32)
            age_predictions = bundle["age_model"].predict(face_batch, verbose=0).reshape(-1)
            gender_predictions = bundle["gender_model"].predict(face_batch, verbose=0).reshape(-1)

            for index, (entry, age_prediction, gender_prediction) in enumerate(
                zip(valid_entries, age_predictions, gender_predictions),
                start=1,
            ):
                age_years = max(0, int(np.rint(float(age_prediction))))
                gender_score = float(gender_prediction)
                detections.append(
                    {
                        "id": f"face-{index}",
                        "label": f"Face {index}",
                        "bbox": entry["bbox"],
                        "face_confidence": round(entry["face_confidence"], 4),
                        "age_years": age_years,
                        "age_bucket": _fairface_age_bucket(age_years),
                        "gender_label": "female" if gender_score < 0.5 else "male",
                        "gender_score": round(gender_score, 4),
                        "face_thumbnail_url": _encode_image_data_url(entry["face_thumb"]),
                    }
                )

        return {
            "source": source,
            "model": {
                "id": model_id,
                "label": MODEL_PRESETS[model_id].label,
                "description": MODEL_PRESETS[model_id].description,
            },
            "dataset": dataset,
            "image": {"width": width, "height": height},
            "ground_truth": ground_truth,
            "label_summary": manifest_label_summary,
            "dataset_image_count": len(manifest["images"]) if manifest else None,
            "detections": detections,
            "warnings": warnings,
        }

    def _extract_face(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, width: int, height: int
    ) -> np.ndarray:
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * FACE_PADDING)
        pad_y = int(box_height * FACE_PADDING)

        left = max(x1 - pad_x, 0)
        top = max(y1 - pad_y, 0)
        right = min(x2 + pad_x, width)
        bottom = min(y2 + pad_y, height)

        crop = image[top:bottom, left:right]
        resized = Image.fromarray(crop).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32)
