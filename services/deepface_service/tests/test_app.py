from fastapi.testclient import TestClient

from services.deepface_service.app import create_app


class FakeDeepFaceService:
    def list_models(self):
        return [
            {
                "id": "deepface",
                "label": "DeepFace",
                "description": "DeepFace preset.",
                "provider": "deepface",
                "family": "demography",
            }
        ]

    def analyze(self, filename, data, model_id):
        if model_id != "deepface":
            raise ValueError("unknown model")
        return {
            "model": self.list_models()[0],
            "image": {"width": 640, "height": 480},
            "detections": [
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
            ],
            "warnings": [],
        }


def build_client():
    app = create_app(service_factory=FakeDeepFaceService)
    return TestClient(app)


def test_health_endpoint_returns_ok():
    with build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["service"] == "deepface"


def test_models_endpoint_returns_deepface_model():
    with build_client() as client:
        response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["models"][0]["id"] == "deepface"


def test_analyze_returns_normalized_detection_shape():
    with build_client() as client:
        response = client.post(
            "/analyze",
            data={"model_id": "deepface"},
            files={"file": ("face.jpg", b"fake-bytes", "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["detections"][0]["bbox"]["width"] == 100
    assert payload["detections"][0]["face_confidence"] is None
