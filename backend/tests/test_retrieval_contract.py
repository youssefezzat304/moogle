from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from api.contracts import (
    ErrorDetail,
    ErrorResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "docs/api/retrieval-v1.yaml"
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/retrieval-response.json"


@pytest.fixture
def response_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text())


def test_shared_response_fixture_matches_backend_contract(
    response_fixture: dict[str, object],
) -> None:
    response = RetrievalResponse.model_validate(response_fixture)

    assert response.model_id == "bpe_geo"
    assert response.results[0].similarity == pytest.approx(0.312)
    assert response.results[0].source_version == "v3.0"
    assert response.results[0].prompt_style == "Geologist to Non-Geologist"
    assert response.results[0].wac_image_url == "/api/patches/123/wac"


def test_request_trims_query_and_defaults_to_five_results() -> None:
    request = RetrievalRequest.model_validate({"query": "  cratered highlands  "})

    assert request.query == "cratered highlands"
    assert request.top_k == 5


def test_error_envelope_matches_contract() -> None:
    response = ErrorResponse.model_validate(
        {
            "error": {
                "code": "MODEL_NOT_READY",
                "message": "The retrieval model is not ready.",
                "request_id": "abc123",
            }
        }
    )

    assert response.error.code == "MODEL_NOT_READY"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "terrain", "top_k": 0},
        {"query": "terrain", "top_k": 11},
        {"query": "terrain", "top_k": "5"},
        {"query": "terrain", "unexpected": True},
    ],
)
def test_invalid_requests_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate(payload)


def test_missing_response_fields_are_rejected(
    response_fixture: dict[str, object],
) -> None:
    response_fixture.pop("model_id")

    with pytest.raises(ValidationError):
        RetrievalResponse.model_validate(response_fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latitude", -90.01),
        ("latitude", 90.01),
        ("longitude", -180.01),
        ("longitude", 180),
    ],
)
def test_invalid_coordinates_are_rejected(
    response_fixture: dict[str, object],
    field: str,
    value: float,
) -> None:
    response_fixture["results"][0][field] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        RetrievalResponse.model_validate(response_fixture)


def test_empty_result_response_is_valid(
    response_fixture: dict[str, object],
) -> None:
    response_fixture["results"] = []

    response = RetrievalResponse.model_validate(response_fixture)

    assert response.results == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda results: results[0].update(rank=2),
        lambda results: results.append(dict(results[0])),
        lambda results: results.append(
            {
                **results[0],
                "rank": 2,
                "patch_id": 456,
                "similarity": results[0]["similarity"] + 1,
            }
        ),
    ],
)
def test_invalid_rankings_are_rejected(
    response_fixture: dict[str, object],
    mutation,
) -> None:
    mutation(response_fixture["results"])

    with pytest.raises(ValidationError):
        RetrievalResponse.model_validate(response_fixture)


def test_unsupported_response_fields_are_rejected(
    response_fixture: dict[str, object],
) -> None:
    response_fixture["confidence"] = 0.99

    with pytest.raises(ValidationError):
        RetrievalResponse.model_validate(response_fixture)


@pytest.mark.parametrize(
    ("schema_name", "model"),
    [
        ("RetrievalRequest", RetrievalRequest),
        ("RetrievalResult", RetrievalResult),
        ("RetrievalResponse", RetrievalResponse),
        ("ErrorDetail", ErrorDetail),
        ("ErrorResponse", ErrorResponse),
    ],
)
def test_openapi_required_fields_match_pydantic_models(
    schema_name: str,
    model: type[BaseModel],
) -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text())
    schema = contract["components"]["schemas"][schema_name]
    required_by_openapi = set(schema.get("required", []))
    required_by_pydantic = {
        name for name, field in model.model_fields.items() if field.is_required()
    }

    assert set(schema["properties"]) == set(model.model_fields)
    assert required_by_openapi == required_by_pydantic


def test_openapi_example_is_the_shared_fixture(
    response_fixture: dict[str, object],
) -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text())
    example = contract["paths"]["/api/retrieval"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["example"]

    assert example == response_fixture


def test_openapi_request_default_matches_pydantic_default() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text())
    top_k_schema = contract["components"]["schemas"]["RetrievalRequest"]["properties"][
        "top_k"
    ]

    assert top_k_schema["default"] == RetrievalRequest(query="terrain").top_k == 5
