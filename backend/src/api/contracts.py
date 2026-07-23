"""Strict wire models for the Moogle retrieval API."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RetrievalRequest(ContractModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query", mode="before")
    @classmethod
    def trim_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class RetrievalResult(ContractModel):
    rank: int = Field(ge=1)
    patch_id: int
    similarity: FiniteFloat
    description: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    prompt_style: str = Field(min_length=1)
    wac_image_url: str = Field(
        min_length=1,
        pattern=r"^(?:/|https?://)",
    )
    latitude: FiniteFloat = Field(ge=-90, le=90)
    longitude: FiniteFloat = Field(ge=-180, lt=180)

    @field_validator(
        "description",
        "source_version",
        "prompt_style",
        "wac_image_url",
        mode="before",
    )
    @classmethod
    def trim_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class RetrievalResponse(ContractModel):
    schema_version: Literal[1]
    query: str = Field(min_length=1, max_length=500)
    model_id: str = Field(min_length=1)
    index_size: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)
    results: list[RetrievalResult] = Field(max_length=10)

    @field_validator("query", "model_id", mode="before")
    @classmethod
    def trim_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_ranking(self) -> Self:
        expected_ranks = list(range(1, len(self.results) + 1))
        ranks = [result.rank for result in self.results]
        if ranks != expected_ranks:
            raise ValueError("result ranks must be contiguous and match array order")

        patch_ids = [result.patch_id for result in self.results]
        if len(patch_ids) != len(set(patch_ids)):
            raise ValueError("result patch_id values must be unique")

        scores = [result.similarity for result in self.results]
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ValueError("results must be sorted by descending similarity")

        return self


class ErrorDetail(ContractModel):
    code: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    message: str = Field(min_length=1)
    request_id: str = Field(min_length=1)

    @field_validator("code", "message", "request_id", mode="before")
    @classmethod
    def trim_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ErrorResponse(ContractModel):
    error: ErrorDetail
