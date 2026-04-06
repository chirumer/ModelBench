from __future__ import annotations

from pathlib import Path

import numpy as np
from mtcnn import MTCNN
from PIL import Image
import tensorflow as tf

from modelbench.catalog import MODEL_PRESETS
from modelbench.ssrnet_model import SSRNet, SSRNetGeneral
from services.shared import (
    ALLOWED_EXTENSIONS,
    ServiceInputError,
    encode_image_data_url,
    extract_face_thumbnail,
    fairface_age_bucket,
    load_rgb_image_from_bytes,
)


IMAGE_SIZE = 64
STAGE_NUM = [3, 3, 3]
LAMBDA_LOCAL = 1.0
LAMBDA_D = 1.0
FACE_PADDING = 0.4


class SSRNetInferenceService:
    def __init__(self) -> None:
        tf.get_logger().setLevel("ERROR")
        self.detector = MTCNN()
        self._model_cache: dict[str, dict] = {}

    def list_models(self) -> list[dict]:
        return [
            {
                "id": preset.id,
                "label": preset.label,
                "description": preset.description,
                "provider": preset.provider,
                "family": preset.family,
            }
            for preset in MODEL_PRESETS.values()
            if preset.provider == "ssrnet"
        ]

    def analyze(self, filename: str, data: bytes, model_id: str) -> dict:
        suffix = Path(filename or "upload").suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise ServiceInputError("Unsupported image type. Use JPG, PNG, or WebP.")

        image = load_rgb_image_from_bytes(data)
        detections, warnings = self._analyze_with_ssrnet(image, model_id=model_id)
        preset = self._get_model_definition(model_id)
        return {
            "model": {
                "id": preset.id,
                "label": preset.label,
                "description": preset.description,
                "provider": preset.provider,
                "family": preset.family,
            },
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "detections": detections,
            "warnings": warnings,
        }

    def _get_model_definition(self, model_id: str):
        definition = MODEL_PRESETS.get(model_id)
        if definition is None or definition.provider != "ssrnet":
            raise ServiceInputError(f"Unknown model_id '{model_id}'.", status_code=400)
        return definition

    def _get_models_for_preset(self, model_id: str) -> dict:
        if model_id not in self._model_cache:
            preset = self._get_model_definition(model_id)
            age_model = SSRNet(IMAGE_SIZE, STAGE_NUM, LAMBDA_LOCAL, LAMBDA_D)()
            age_model.load_weights(preset.age_weight_path)
            gender_model = SSRNetGeneral(IMAGE_SIZE, STAGE_NUM, LAMBDA_LOCAL, LAMBDA_D)()
            gender_model.load_weights(preset.gender_weight_path)
            self._model_cache[model_id] = {
                "age_model": age_model,
                "gender_model": gender_model,
            }
        return self._model_cache[model_id]

    def _analyze_with_ssrnet(self, image: np.ndarray, *, model_id: str) -> tuple[list[dict], list[str]]:
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
            face_bbox = {
                "x": x,
                "y": y,
                "width": x2 - x,
                "height": y2 - y,
            }
            valid_entries.append(
                {
                    "bbox": face_bbox,
                    "face_confidence": confidence,
                    "face": padded_crop,
                    "face_thumb": extract_face_thumbnail(image, face_bbox),
                }
            )

        valid_entries.sort(key=lambda entry: (entry["bbox"]["x"], entry["bbox"]["y"]))

        if not valid_entries:
            return [], ["No faces detected in this image."]

        face_batch = np.stack([entry["face"] for entry in valid_entries]).astype(np.float32)
        age_predictions = bundle["age_model"].predict(face_batch, verbose=0).reshape(-1)
        gender_predictions = bundle["gender_model"].predict(face_batch, verbose=0).reshape(-1)

        detections = []
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
                    "age_bucket": fairface_age_bucket(age_years),
                    "gender_label": "female" if gender_score < 0.5 else "male",
                    "gender_score": round(gender_score, 4),
                    "face_thumbnail_url": encode_image_data_url(entry["face_thumb"]),
                }
            )

        return detections, []

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
