from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ServiceInputError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def encode_image_data_url(image_array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_array.astype(np.uint8)).save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def fairface_age_bucket(age_years: int) -> str:
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


def as_probability(value: float | int | None) -> float | None:
    if value is None:
        return None
    score = float(value)
    if score > 1.0:
        score = score / 100.0
    if score < 0:
        return None
    return round(score, 4)


def normalize_gender_label(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = str(label).strip().lower()
    if normalized in {"man", "male"}:
        return "male"
    if normalized in {"woman", "female"}:
        return "female"
    return None


def load_rgb_image_from_bytes(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ServiceInputError("Unsupported image content. Use JPG, PNG, or WebP.") from exc


def normalize_bbox(
    x: int | float | None,
    y: int | float | None,
    width: int | float | None,
    height: int | float | None,
    image_width: int,
    image_height: int,
) -> dict | None:
    if x is None or y is None or width is None or height is None:
        return None

    left = max(int(x), 0)
    top = max(int(y), 0)
    box_width = max(int(width), 0)
    box_height = max(int(height), 0)
    if box_width == 0 or box_height == 0:
        return None

    right = min(left + box_width, image_width)
    bottom = min(top + box_height, image_height)
    if right <= left or bottom <= top:
        return None

    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
    }


def extract_face_thumbnail(image: np.ndarray, bbox: dict) -> np.ndarray:
    x1 = bbox["x"]
    y1 = bbox["y"]
    x2 = x1 + bbox["width"]
    y2 = y1 + bbox["height"]
    return np.asarray(
        Image.fromarray(image[y1:y2, x1:x2]).resize((120, 120), Image.BILINEAR),
        dtype=np.uint8,
    )
