from __future__ import annotations

from pathlib import Path
import sys

from modelbench.catalog import DEEPFACE_DIR
from services.shared import (
    ALLOWED_EXTENSIONS,
    ServiceInputError,
    as_probability,
    encode_image_data_url,
    extract_face_thumbnail,
    fairface_age_bucket,
    load_rgb_image_from_bytes,
    normalize_bbox,
    normalize_gender_label,
)


class DeepFaceInferenceService:
    def __init__(self) -> None:
        deepface_source = str(DEEPFACE_DIR)
        if deepface_source not in sys.path:
            sys.path.insert(0, deepface_source)

        try:
            from deepface import DeepFace
        except Exception as exc:  # pragma: no cover - runtime dependency failure
            raise ServiceInputError(
                "DeepFace is unavailable in this service environment. "
                "Install the deepface service dependencies and `pip install --no-deps -e ./deepface`.",
                status_code=500,
            ) from exc

        self._deepface = DeepFace

    def list_models(self) -> list[dict]:
        return [
            {
                "id": "deepface",
                "label": "DeepFace",
                "description": "DeepFace age and gender analysis with MTCNN face detection.",
                "provider": "deepface",
                "family": "demography",
            }
        ]

    def analyze(self, filename: str, data: bytes, model_id: str) -> dict:
        if model_id != "deepface":
            raise ServiceInputError(f"Unknown model_id '{model_id}'.", status_code=400)

        suffix = Path(filename or "upload").suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise ServiceInputError("Unsupported image type. Use JPG, PNG, or WebP.")

        image = load_rgb_image_from_bytes(data)

        try:
            results = self._deepface.analyze(
                img_path=image[:, :, ::-1],
                actions=("age", "gender"),
                detector_backend="mtcnn",
                enforce_detection=True,
                align=True,
                silent=True,
            )
        except Exception as exc:
            message = str(exc)
            if "Face could not be detected" in message:
                return {
                    "model": self.list_models()[0],
                    "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
                    "detections": [],
                    "warnings": ["No faces detected in this image."],
                }
            raise ServiceInputError(f"DeepFace inference failed. {message}", status_code=500) from exc

        detections = []
        for index, result in enumerate(results, start=1):
            region = result.get("region") or {}
            bbox = normalize_bbox(
                region.get("x"),
                region.get("y"),
                region.get("w"),
                region.get("h"),
                int(image.shape[1]),
                int(image.shape[0]),
            )
            if bbox is None:
                continue

            gender_label = normalize_gender_label(result.get("dominant_gender"))
            gender_scores = result.get("gender") or {}
            score_value = None
            if gender_label == "male":
                score_value = gender_scores.get("Man") or gender_scores.get("Male")
            elif gender_label == "female":
                score_value = gender_scores.get("Woman") or gender_scores.get("Female")

            age_years = max(0, int(round(float(result.get("age", 0)))))
            face_thumb = extract_face_thumbnail(image, bbox)
            detections.append(
                {
                    "id": f"face-{index}",
                    "label": f"Face {index}",
                    "bbox": bbox,
                    "face_confidence": as_probability(result.get("face_confidence")),
                    "age_years": age_years,
                    "age_bucket": fairface_age_bucket(age_years),
                    "gender_label": gender_label or "unknown",
                    "gender_score": as_probability(score_value),
                    "face_thumbnail_url": encode_image_data_url(face_thumb),
                }
            )

        warnings = [] if detections else ["No faces detected in this image."]
        return {
            "model": self.list_models()[0],
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "detections": detections,
            "warnings": warnings,
        }
