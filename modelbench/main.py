from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = BASE_DIR / "templates" / "index.html"


def create_app(service_factory: Callable[[], object] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        factory = service_factory
        if factory is None:
            from modelbench.inference import InferenceService

            factory = InferenceService
        app.state.service = factory()
        yield
        close = getattr(app.state.service, "close", None)
        if callable(close):
            close()

    app = FastAPI(
        title="ModelBench",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        return request.app.state.service.health()

    @app.get("/api/models")
    async def list_models(request: Request) -> dict:
        return {"models": request.app.state.service.list_models()}

    @app.get("/api/datasets")
    async def list_datasets(request: Request) -> dict:
        return {"datasets": request.app.state.service.list_datasets()}

    @app.get("/api/datasets/{dataset_id}/images")
    async def list_dataset_images(request: Request, dataset_id: str) -> dict:
        return {"images": request.app.state.service.list_dataset_images(dataset_id)}

    @app.post("/api/analyze")
    async def analyze(
        request: Request,
        model_id: str | None = Form(default=None),
        file: UploadFile | None = File(default=None),
        dataset_image_id: str | None = Form(default=None),
    ) -> dict:
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required.")

        if bool(file) == bool(dataset_image_id):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of file or dataset_image_id.",
            )

        service = request.app.state.service

        try:
            if dataset_image_id:
                return service.analyze_dataset_image(dataset_image_id, model_id)

            assert file is not None
            data = await file.read()
            return service.analyze_upload(file.filename or "upload", data, model_id)
        except Exception as exc:
            status_code = getattr(exc, "status_code", 500)
            detail = getattr(exc, "message", "Inference failed.")
            raise HTTPException(status_code=status_code, detail=detail) from exc

    return app


app = create_app()
