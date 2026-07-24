from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from moogle_inference import RetrievalResult

from api.application import create_app


@dataclass
class FakeRetrievalService:
    image_paths: dict[int, Path]
    model_id: str = "bpe_geo"
    index_size: int = 22_578
    searches: list[tuple[str, int]] = field(default_factory=list)
    search_error: Exception | None = None

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        self.searches.append((query, top_k))
        if self.search_error is not None:
            raise self.search_error
        results = [
            RetrievalResult(
                rank=1,
                patch_id=7,
                similarity=0.312,
                description="A young crater surrounded by bright ejecta.",
                source_version="v3.0",
                prompt_style="Geologist to Non-Geologist",
                wac_image_path=self.image_paths[7],
                latitude=-12.34,
                longitude=45.67,
            ),
            RetrievalResult(
                rank=2,
                patch_id=11,
                similarity=-0.125,
                description="A subdued crater on a smooth plain.",
                source_version="v3.0",
                prompt_style="Geologist to Non-Geologist",
                wac_image_path=self.image_paths[11],
                latitude=2.5,
                longitude=-30.0,
            ),
        ]
        return results[:top_k]

    def wac_image_path(self, patch_id: int) -> Path:
        try:
            return self.image_paths[patch_id]
        except KeyError as exc:
            raise KeyError(patch_id) from exc


@pytest.fixture
def service(tmp_path: Path) -> FakeRetrievalService:
    image_paths: dict[int, Path] = {}
    for patch_id in (7, 11):
        path = tmp_path / f"{patch_id}.webp"
        path.write_bytes(f"webp-{patch_id}".encode())
        image_paths[patch_id] = path
    return FakeRetrievalService(image_paths=image_paths)


@pytest.fixture
def client(
    service: FakeRetrievalService,
) -> Iterator[TestClient]:
    app = create_app(engine_loader=lambda: service)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_retrieval_returns_ranked_contract_response(
    client: TestClient,
    service: FakeRetrievalService,
) -> None:
    response = client.post(
        "/api/retrieval",
        json={"query": "  young crater with bright ejecta  ", "top_k": 2},
    )

    assert response.status_code == 200
    assert service.searches == [("young crater with bright ejecta", 2)]
    assert response.headers["x-request-id"]
    assert response.json() == {
        "schema_version": 1,
        "query": "young crater with bright ejecta",
        "model_id": "bpe_geo",
        "index_size": 22_578,
        "elapsed_ms": response.json()["elapsed_ms"],
        "results": [
            {
                "rank": 1,
                "patch_id": 7,
                "similarity": 0.312,
                "description": "A young crater surrounded by bright ejecta.",
                "source_version": "v3.0",
                "prompt_style": "Geologist to Non-Geologist",
                "wac_image_url": "/api/patches/7/wac",
                "latitude": -12.34,
                "longitude": 45.67,
            },
            {
                "rank": 2,
                "patch_id": 11,
                "similarity": -0.125,
                "description": "A subdued crater on a smooth plain.",
                "source_version": "v3.0",
                "prompt_style": "Geologist to Non-Geologist",
                "wac_image_url": "/api/patches/11/wac",
                "latitude": 2.5,
                "longitude": -30.0,
            },
        ],
    }
    assert isinstance(response.json()["elapsed_ms"], int)
    assert response.json()["elapsed_ms"] >= 0


def test_retrieval_defaults_to_five_results(
    client: TestClient,
    service: FakeRetrievalService,
) -> None:
    response = client.post("/api/retrieval", json={"query": "crater"})

    assert response.status_code == 200
    assert service.searches == [("crater", 5)]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "terrain", "top_k": 0},
        {"query": "terrain", "top_k": 11},
        {"query": "terrain", "top_k": "5"},
        {"query": "terrain", "extra": True},
    ],
)
def test_invalid_retrieval_returns_422_without_inference(
    client: TestClient,
    service: FakeRetrievalService,
    payload: dict[str, object],
) -> None:
    response = client.post("/api/retrieval", json=payload)

    assert response.status_code == 422
    assert service.searches == []
    _assert_error(response, code="VALIDATION_ERROR")


def test_unavailable_model_returns_503() -> None:
    def unavailable_loader() -> FakeRetrievalService:
        raise RuntimeError("index unavailable")

    app = create_app(engine_loader=unavailable_loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/retrieval", json={"query": "terrain"})

    assert response.status_code == 503
    _assert_error(response, code="MODEL_NOT_READY")


def test_invalid_request_stays_422_when_model_is_unavailable() -> None:
    def unavailable_loader() -> FakeRetrievalService:
        raise RuntimeError("index unavailable")

    app = create_app(engine_loader=unavailable_loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/retrieval", json={"query": "   "})

    assert response.status_code == 422
    _assert_error(response, code="VALIDATION_ERROR")


def test_unexpected_inference_failure_returns_500(
    service: FakeRetrievalService,
) -> None:
    service.search_error = RuntimeError("GPU failure")
    app = create_app(engine_loader=lambda: service)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/retrieval", json={"query": "terrain"})

    assert response.status_code == 500
    _assert_error(response, code="INTERNAL_ERROR")


def test_wac_endpoint_serves_corresponding_webp(
    client: TestClient,
) -> None:
    response = client.get("/api/patches/7/wac")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content == b"webp-7"


def test_unknown_wac_patch_returns_404(
    client: TestClient,
) -> None:
    response = client.get("/api/patches/999/wac")

    assert response.status_code == 404
    _assert_error(response, code="PATCH_NOT_FOUND")


def _assert_error(response, *, code: str) -> None:
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "request_id"}
    assert payload["error"]["code"] == code
    assert payload["error"]["request_id"] == response.headers["x-request-id"]
