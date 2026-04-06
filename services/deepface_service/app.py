from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from services.shared import ServiceInputError


def create_app(service_factory: Callable[[], object] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        factory = service_factory
        if factory is None:
            from services.deepface_service.inference import DeepFaceInferenceService

            factory = DeepFaceInferenceService
        app.state.service = factory()
        yield

    app = FastAPI(title="DeepFace Service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "deepface"}

    @app.get("/models")
    async def list_models(request: Request) -> dict:
        return {"models": request.app.state.service.list_models()}

    @app.post("/analyze")
    async def analyze(
        request: Request,
        model_id: str | None = Form(default=None),
        file: UploadFile | None = File(default=None),
    ) -> dict:
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required.")
        if file is None:
            raise HTTPException(status_code=400, detail="file is required.")

        try:
            data = await file.read()
            return request.app.state.service.analyze(file.filename or "upload", data, model_id)
        except ServiceInputError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return app


app = create_app()
