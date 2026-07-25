from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from api.contracts import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)
from api.runtime import (
    EngineLoader,
    RetrievalService,
    load_production_service,
)


LOGGER = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def create_app(
    *,
    engine_loader: EngineLoader = load_production_service,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retrieval_service = None
        app.state.startup_error = None
        try:
            app.state.retrieval_service = engine_loader()
        except Exception as exc:
            app.state.startup_error = exc
            LOGGER.exception("The retrieval service failed to initialize.")
        yield
        app.state.retrieval_service = None

    app = FastAPI(
        title="Moogle Retrieval API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        _: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            code="VALIDATION_ERROR",
            message="The retrieval request is invalid.",
        )

    @app.exception_handler(ApiError)
    async def api_error_handler(
        request: Request,
        exc: ApiError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = _request_id(request)
        LOGGER.exception(
            "Unhandled API error for request %s.",
            request_id,
            exc_info=exc,
        )
        return _error_response(
            request,
            status_code=500,
            code="INTERNAL_ERROR",
            message="The retrieval service encountered an unexpected error.",
        )

    @app.get(
        "/api/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
    )
    def health(request: Request) -> HealthResponse | JSONResponse:
        ready = request.app.state.retrieval_service is not None
        response = HealthResponse(
            status="ready" if ready else "not_ready",
            model_loaded=ready,
            catalog_loaded=ready,
            index_loaded=ready,
        )
        if not ready:
            return JSONResponse(
                status_code=503,
                content=response.model_dump(mode="json"),
            )
        return response

    @app.post(
        "/api/retrieval",
        response_model=RetrievalResponse,
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def retrieve(
        payload: RetrievalRequest,
        request: Request,
    ) -> RetrievalResponse:
        service = _ready_service(request)
        started = perf_counter()
        domain_results = service.search(payload.query, top_k=payload.top_k)
        elapsed_ms = max(0, int((perf_counter() - started) * 1000))

        return RetrievalResponse(
            schema_version=1,
            query=payload.query,
            model_id=service.model_id,
            index_size=service.index_size,
            elapsed_ms=elapsed_ms,
            results=[
                RetrievalResult(
                    rank=result.rank,
                    patch_id=result.patch_id,
                    similarity=result.similarity,
                    description=result.description,
                    source_version=result.source_version,
                    prompt_style=result.prompt_style,
                    wac_image_url=f"/api/patches/{result.patch_id}/wac",
                    latitude=result.latitude,
                    longitude=result.longitude,
                )
                for result in domain_results
            ],
        )

    @app.get(
        "/api/patches/{patch_id}/wac",
        response_class=FileResponse,
        responses={
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def get_wac_image(patch_id: int, request: Request) -> FileResponse:
        service = _ready_service(request)
        try:
            image_path = service.wac_image_path(patch_id)
        except KeyError as exc:
            raise _patch_not_found() from exc
        if not image_path.is_file():
            raise _patch_not_found()
        return FileResponse(image_path, media_type="image/webp")

    return app


def _ready_service(request: Request) -> RetrievalService:
    service = request.app.state.retrieval_service
    if service is None:
        raise ApiError(
            status_code=503,
            code="MODEL_NOT_READY",
            message="The retrieval model is not ready.",
        )
    return service


def _patch_not_found() -> ApiError:
    return ApiError(
        status_code=404,
        code="PATCH_NOT_FOUND",
        message="The requested lunar patch was not found.",
    )


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        request_id = uuid4().hex
        request.state.request_id = request_id
    return request_id


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    request_id = _request_id(request)
    content = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=content.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


app = create_app()
