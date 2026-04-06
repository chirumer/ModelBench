from fastapi.testclient import TestClient

from services.ssrnet_service.app import create_app


class FakeSSRNetService:
    def list_models(self):
        return [
            {
                "id": "wiki",
                "label": "WIKI",
                "description": "WIKI preset.",
                "provider": "ssrnet",
                "family": "ssrnet",
            }
        ]

    def analyze(self, filename, data, model_id):
        if model_id != "wiki":
            raise ValueError("unknown model")
        return {
            "model": self.list_models()[0],
            "image": {"width": 640, "height": 480},
            "detections": [],
            "warnings": ["No faces detected in this image."],
        }


def build_client():
    app = create_app(service_factory=FakeSSRNetService)
    return TestClient(app)


def test_health_endpoint_returns_ok():
    with build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["service"] == "ssrnet"


def test_models_endpoint_returns_ssrnet_models():
    with build_client() as client:
        response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["models"][0]["provider"] == "ssrnet"


def test_analyze_requires_file():
    with build_client() as client:
        response = client.post("/analyze", data={"model_id": "wiki"})

    assert response.status_code == 400
    assert response.json()["detail"] == "file is required."
