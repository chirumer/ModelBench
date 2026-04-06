from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

from modelbench.catalog import DATASET_DEFINITIONS
from services.runtime import RUNTIME_CONFIGS


SERVICE_ORDER = ("ssrnet", "deepface")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class InferenceInputError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class BackendServiceClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.runtime = RUNTIME_CONFIGS[name]
        self.base_url = f"http://127.0.0.1:{self.runtime.port}"
        self._client = httpx.Client(base_url=self.base_url, timeout=90.0)

    def list_models(self) -> list[dict]:
        try:
            response = self._client.get("/models")
        except httpx.HTTPError as exc:
            raise InferenceInputError(
                f"{self.name} service is unavailable while loading models.",
                status_code=502,
            ) from exc
        if response.status_code != 200:
            raise InferenceInputError(
                f"{self.name} service is unavailable while loading models.",
                status_code=502,
            )
        return response.json()["models"]

    def health(self) -> dict:
        try:
            response = self._client.get("/health")
        except httpx.HTTPError as exc:
            raise InferenceInputError(
                f"{self.name} service health check failed.",
                status_code=502,
            ) from exc
        if response.status_code != 200:
            raise InferenceInputError(
                f"{self.name} service health check failed.",
                status_code=502,
            )
        return response.json()

    def analyze(self, filename: str, data: bytes, model_id: str) -> dict:
        try:
            response = self._client.post(
                "/analyze",
                data={"model_id": model_id},
                files={"file": (filename, data, "application/octet-stream")},
            )
        except httpx.HTTPError as exc:
            raise InferenceInputError(
                f"{self.name} service is unavailable while running inference.",
                status_code=502,
            ) from exc

        payload = response.json()
        if response.status_code != 200:
            detail = payload.get("detail", "Inference failed.")
            raise InferenceInputError(detail, status_code=response.status_code)
        return payload

    def close(self) -> None:
        self._client.close()


class InferenceService:
    def __init__(self) -> None:
        self._dataset_manifest_cache: dict[str, dict] = {}
        self._dataset_image_index: dict[str, tuple[str, dict]] | None = None
        self._service_clients = {
            name: BackendServiceClient(name) for name in SERVICE_ORDER if name in RUNTIME_CONFIGS
        }
        self._model_index: dict[str, dict] | None = None

    def close(self) -> None:
        for client in self._service_clients.values():
            client.close()

    def health(self) -> dict:
        services = {}
        for name, client in self._service_clients.items():
            services[name] = client.health()
        return {"status": "ok", "service": "modelbench", "services": services}

    def list_models(self) -> list[dict]:
        self._refresh_model_index()
        assert self._model_index is not None
        return list(self._model_index.values())

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
        image_path = Path(record["image_path"])
        data = image_path.read_bytes()
        response = self._forward_analysis(image_path.name, data, model_id)
        definition = DATASET_DEFINITIONS[dataset_id]
        manifest = self._load_manifest(dataset_id)
        response.update(
            {
                "source": "dataset",
                "dataset": {
                    "id": dataset_id,
                    "name": definition.name,
                    "description": definition.description,
                    "image_id": record["id"],
                    "image_url": record["image_url"],
                    "thumbnail_url": record["thumbnail_url"],
                },
                "ground_truth": record["ground_truth"],
                "label_summary": record["label_summary"],
                "dataset_image_count": len(manifest["images"]),
            }
        )
        return response

    def analyze_upload(self, filename: str, data: bytes, model_id: str) -> dict:
        suffix = Path(filename or "upload").suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise InferenceInputError("Unsupported image type. Use JPG, PNG, or WebP.")
        self._validate_image_bytes(data)
        response = self._forward_analysis(filename or "upload", data, model_id)
        response.update(
            {
                "source": "upload",
                "dataset": None,
                "ground_truth": None,
                "label_summary": None,
                "dataset_image_count": None,
            }
        )
        return response

    def _refresh_model_index(self) -> None:
        index: dict[str, dict] = {}
        for name in SERVICE_ORDER:
            client = self._service_clients[name]
            for model in client.list_models():
                entry = dict(model)
                entry["service"] = name
                index[entry["id"]] = entry
        self._model_index = index

    def _get_model_entry(self, model_id: str) -> dict:
        if self._model_index is None or model_id not in self._model_index:
            self._refresh_model_index()
        assert self._model_index is not None
        entry = self._model_index.get(model_id)
        if entry is None:
            raise InferenceInputError(f"Unknown model_id '{model_id}'.", status_code=400)
        return entry

    def _forward_analysis(self, filename: str, data: bytes, model_id: str) -> dict:
        model = self._get_model_entry(model_id)
        client = self._service_clients[model["service"]]
        return client.analyze(filename, data, model_id)

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

    def _validate_image_bytes(self, data: bytes) -> None:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except UnidentifiedImageError as exc:
            raise InferenceInputError("Unsupported image content. Use JPG, PNG, or WebP.") from exc
