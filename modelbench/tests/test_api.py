from fastapi.testclient import TestClient

from modelbench.main import create_app


class FakeService:
    def __init__(self):
        self._bulk_runs = {
            "run-123": {
                "run_id": "run-123",
                "status": "running",
                "created_at": "2026-04-06T00:00:00Z",
                "started_at": "2026-04-06T00:00:01Z",
                "finished_at": None,
                "selected_models": [
                    {
                        "id": "wiki",
                        "label": "SSR-Net (WIKI)",
                        "description": "WIKI preset.",
                        "provider": "ssrnet",
                        "family": "ssrnet",
                    }
                ],
                "datasets": [
                    {
                        "id": "utkface",
                        "name": "UTKFace",
                        "description": "Exact-age face crops.",
                        "image_count": 100,
                    }
                ],
                "settings": {"baby_max": 12, "adult_max": 59},
                "progress": {
                    "tested_images": 4,
                    "total_images": 100,
                    "current_dataset_id": "utkface",
                    "current_model_id": "wiki",
                },
                "results": {
                    "utkface": {
                        "dataset": {
                            "id": "utkface",
                            "name": "UTKFace",
                            "description": "Exact-age face crops.",
                            "image_count": 100,
                        },
                        "models": {
                            "wiki": {
                                "tested_count": 4,
                                "total_count": 100,
                                "gender_correct_count": 3,
                                "gender_accuracy": 0.75,
                                "age_class_correct_count": 2,
                                "age_class_accuracy": 0.5,
                                "age_class_breakdown": {
                                    "baby": {
                                        "label": "Baby",
                                        "correct_count": 1,
                                        "total_count": 2,
                                        "accuracy": 0.5,
                                    },
                                    "adult": {
                                        "label": "Man/Woman",
                                        "correct_count": 1,
                                        "total_count": 1,
                                        "accuracy": 1.0,
                                    },
                                    "old": {
                                        "label": "Old",
                                        "correct_count": 0,
                                        "total_count": 1,
                                        "accuracy": 0.0,
                                    },
                                },
                                "missed_detection_count": 1,
                                "status": "running",
                                "last_error": None,
                            }
                        },
                    }
                },
            }
        }

    def list_models(self):
        return [
            {
                "id": "wiki",
                "label": "SSR-Net (WIKI)",
                "description": "SSR-Net using WIKI age weights with WIKI gender weights.",
                "provider": "ssrnet",
                "family": "ssrnet",
            },
            {
                "id": "deepface",
                "label": "DeepFace",
                "description": "DeepFace preset.",
                "provider": "deepface",
                "family": "demography",
            },
        ]

    def list_datasets(self):
        return [
            {
                "id": "utkface",
                "name": "UTKFace",
                "description": "Exact-age face crops.",
                "image_count": 100,
            }
        ]

    def list_dataset_images(self, dataset_id):
        if dataset_id != "utkface":
            raise FakeInputError("Unknown dataset", status_code=404)
        return [
            {
                "id": "utkface-001",
                "image_url": "/static/datasets/utkface/images/utkface-001.jpg",
                "thumbnail_url": "/static/datasets/utkface/thumbs/utkface-001.jpg",
                "label_summary": "31 • Male • White",
            }
        ]

    def analyze_dataset_image(self, dataset_image_id, model_id):
        if dataset_image_id != "utkface-001":
            raise FakeInputError("Unknown dataset image", status_code=404)
        if model_id not in {"wiki", "deepface"}:
            raise FakeInputError("Unknown model_id", status_code=400)
        return {
            "source": "dataset",
            "model": {
                "id": model_id,
                "label": "DeepFace" if model_id == "deepface" else "SSR-Net (WIKI)",
                "description": (
                    "DeepFace preset."
                    if model_id == "deepface"
                    else "SSR-Net using WIKI age weights with WIKI gender weights."
                ),
                "provider": "deepface" if model_id == "deepface" else "ssrnet",
                "family": "demography" if model_id == "deepface" else "ssrnet",
            },
            "dataset": {
                "id": "utkface",
                "name": "UTKFace",
                "description": "Exact-age face crops.",
                "image_id": "utkface-001",
                "image_url": "/static/datasets/utkface/images/utkface-001.jpg",
                "thumbnail_url": "/static/datasets/utkface/thumbs/utkface-001.jpg",
            },
            "image": {"width": 640, "height": 480},
            "ground_truth": {
                "dataset_type": "utkface",
                "age_kind": "exact",
                "age_display": "31",
                "age_value": 31,
                "gender_display": "Male",
                "gender_value": "male",
                "demographic_label": "White",
                "comparison_age_display": "31",
            },
            "detections": (
                [
                    {
                        "id": "face-1",
                        "label": "Face 1",
                        "bbox": {"x": 10, "y": 20, "width": 100, "height": 120},
                        "face_confidence": None,
                        "age_years": 31,
                        "age_bucket": "30-39",
                        "gender_label": "female",
                        "gender_score": None,
                        "face_thumbnail_url": "data:image/jpeg;base64,abc",
                    }
                ]
                if model_id == "deepface"
                else []
            ),
            "warnings": [] if model_id == "deepface" else ["No faces detected in this image."],
        }

    def analyze_upload(self, filename, data, model_id):
        if filename.endswith(".txt"):
            raise FakeInputError("Unsupported image type.", status_code=400)
        if model_id not in {"wiki", "deepface"}:
            raise FakeInputError("Unknown model_id", status_code=400)
        return {
            "source": "upload",
            "model": {
                "id": model_id,
                "label": "DeepFace" if model_id == "deepface" else "SSR-Net (WIKI)",
                "description": (
                    "DeepFace preset."
                    if model_id == "deepface"
                    else "SSR-Net using WIKI age weights with WIKI gender weights."
                ),
                "provider": "deepface" if model_id == "deepface" else "ssrnet",
                "family": "demography" if model_id == "deepface" else "ssrnet",
            },
            "dataset": None,
            "image": {"width": 640, "height": 480},
            "ground_truth": None,
            "detections": [
                {
                    "id": "person-1",
                    "label": "Face 1",
                    "bbox": {"x": 10, "y": 20, "width": 100, "height": 120},
                    "face_confidence": None if model_id == "deepface" else 0.991,
                    "age_years": 31,
                    "age_bucket": "30-39",
                    "gender_label": "male",
                    "gender_score": None if model_id == "deepface" else 0.74,
                    "face_thumbnail_url": "data:image/jpeg;base64,abc",
                }
            ],
            "warnings": [],
        }

    def start_bulk_run(self, model_ids, baby_max, adult_max):
        if not model_ids:
            raise FakeInputError("model_ids must include at least one model.", status_code=400)
        if baby_max < 0 or baby_max >= adult_max or adult_max > 116:
            raise FakeInputError("Use slider values where 0 <= baby_max < adult_max <= 116.", status_code=400)
        return self._bulk_runs["run-123"]

    def get_bulk_run(self, run_id):
        snapshot = self._bulk_runs.get(run_id)
        if snapshot is None:
            raise FakeInputError(f"Unknown bulk run '{run_id}'.", status_code=404)
        return snapshot

    def update_bulk_run_settings(self, run_id, baby_max, adult_max):
        snapshot = self._bulk_runs.get(run_id)
        if snapshot is None:
            raise FakeInputError(f"Unknown bulk run '{run_id}'.", status_code=404)
        if baby_max < 0 or baby_max >= adult_max or adult_max > 116:
            raise FakeInputError("Use slider values where 0 <= baby_max < adult_max <= 116.", status_code=400)
        snapshot["settings"] = {"baby_max": baby_max, "adult_max": adult_max}
        snapshot["results"]["utkface"]["models"]["wiki"]["age_class_accuracy"] = 0.25
        snapshot["results"]["utkface"]["models"]["wiki"]["age_class_breakdown"]["adult"]["total_count"] = 2
        return snapshot


class FakeInputError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def build_client():
    app = create_app(service_factory=FakeService)
    return TestClient(app)


def test_models_endpoint_returns_catalog():
    with build_client() as client:
        response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 2
    assert payload["models"][0]["id"] == "wiki"
    assert payload["models"][0]["label"] == "SSR-Net (WIKI)"
    assert payload["models"][1]["provider"] == "deepface"


def test_bulk_inference_page_serves_html():
    with build_client() as client:
        response = client.get("/bulk-inference")

    assert response.status_code == 200
    assert "Bulk Inference" in response.text


def test_datasets_endpoint_returns_catalog():
    with build_client() as client:
        response = client.get("/api/datasets")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["datasets"]) == 1
    assert payload["datasets"][0]["image_count"] == 100


def test_dataset_images_endpoint_returns_images():
    with build_client() as client:
        response = client.get("/api/datasets/utkface/images")

    assert response.status_code == 200
    payload = response.json()
    assert payload["images"][0]["id"] == "utkface-001"


def test_analyze_requires_model_id():
    with build_client() as client:
        response = client.post("/api/analyze", data={"dataset_image_id": "utkface-001"})

    assert response.status_code == 400
    assert response.json()["detail"] == "model_id is required."


def test_analyze_accepts_dataset_image_id():
    with build_client() as client:
        response = client.post(
            "/api/analyze",
            data={"dataset_image_id": "utkface-001", "model_id": "wiki"},
        )

    assert response.status_code == 200
    assert response.json()["source"] == "dataset"


def test_analyze_accepts_deepface_dataset_image():
    with build_client() as client:
        response = client.post(
            "/api/analyze",
            data={"dataset_image_id": "utkface-001", "model_id": "deepface"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]["provider"] == "deepface"
    assert payload["detections"][0]["face_confidence"] is None


def test_analyze_accepts_upload():
    with build_client() as client:
        response = client.post(
            "/api/analyze",
            data={"model_id": "wiki"},
            files={"file": ("face.png", b"fake-image-bytes", "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "upload"
    assert payload["detections"][0]["gender_label"] == "male"
    assert payload["detections"][0]["age_bucket"] == "30-39"


def test_upload_rejects_unsupported_extension():
    with build_client() as client:
        response = client.post(
            "/api/analyze",
            data={"model_id": "wiki"},
            files={"file": ("notes.txt", b"plain text", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported image type."


def test_create_bulk_run_rejects_empty_models():
    with build_client() as client:
        response = client.post(
            "/api/bulk-runs",
            json={"model_ids": [], "baby_max": 12, "adult_max": 59},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "model_ids must include at least one model."


def test_create_bulk_run_rejects_invalid_slider_range():
    with build_client() as client:
        response = client.post(
            "/api/bulk-runs",
            json={"model_ids": ["wiki"], "baby_max": 59, "adult_max": 59},
        )

    assert response.status_code == 400
    assert "0 <= baby_max < adult_max <= 116" in response.json()["detail"]


def test_create_bulk_run_returns_snapshot():
    with build_client() as client:
        response = client.post(
            "/api/bulk-runs",
            json={"model_ids": ["wiki"], "baby_max": 12, "adult_max": 59},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-123"
    assert payload["run"]["progress"]["tested_images"] == 4


def test_get_bulk_run_returns_404_for_unknown_id():
    with build_client() as client:
        response = client.get("/api/bulk-runs/missing-run")

    assert response.status_code == 404


def test_patch_bulk_run_settings_updates_snapshot():
    with build_client() as client:
        response = client.patch(
            "/api/bulk-runs/run-123/settings",
            json={"baby_max": 10, "adult_max": 50},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["settings"] == {"baby_max": 10, "adult_max": 50}
    assert payload["run"]["results"]["utkface"]["models"]["wiki"]["age_class_accuracy"] == 0.25


def test_patch_bulk_run_settings_rejects_invalid_range():
    with build_client() as client:
        response = client.patch(
            "/api/bulk-runs/run-123/settings",
            json={"baby_max": 60, "adult_max": 50},
        )

    assert response.status_code == 400
    assert "0 <= baby_max < adult_max <= 116" in response.json()["detail"]


def test_patch_bulk_run_settings_returns_404_for_unknown_id():
    with build_client() as client:
        response = client.patch(
            "/api/bulk-runs/missing-run/settings",
            json={"baby_max": 10, "adult_max": 50},
        )

    assert response.status_code == 404
