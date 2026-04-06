from fastapi.testclient import TestClient

from modelbench.main import create_app


class FakeService:
    def list_models(self):
        return [
            {
                "id": "wiki",
                "label": "WIKI",
                "description": "WIKI preset.",
            }
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
        if model_id != "wiki":
            raise FakeInputError("Unknown model_id", status_code=400)
        return {
            "source": "dataset",
            "model": {"id": "wiki", "label": "WIKI", "description": "WIKI preset."},
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
            "detections": [],
            "warnings": ["No faces detected in this image."],
        }

    def analyze_upload(self, filename, data, model_id):
        if filename.endswith(".txt"):
            raise FakeInputError("Unsupported image type.", status_code=400)
        if model_id != "wiki":
            raise FakeInputError("Unknown model_id", status_code=400)
        return {
            "source": "upload",
            "model": {"id": "wiki", "label": "WIKI", "description": "WIKI preset."},
            "dataset": None,
            "image": {"width": 640, "height": 480},
            "ground_truth": None,
            "detections": [
                {
                    "id": "person-1",
                    "label": "Face 1",
                    "bbox": {"x": 10, "y": 20, "width": 100, "height": 120},
                    "face_confidence": 0.991,
                    "age_years": 31,
                    "age_bucket": "30-39",
                    "gender_label": "male",
                    "gender_score": 0.74,
                    "face_thumbnail_url": "data:image/jpeg;base64,abc",
                }
            ],
            "warnings": [],
        }


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
    assert len(payload["models"]) == 1
    assert payload["models"][0]["id"] == "wiki"


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
