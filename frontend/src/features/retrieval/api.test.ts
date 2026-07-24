/// <reference types="node" />

import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  formatSimilarity,
  parseRetrievalResponse,
  RetrievalApiError,
  retrieveLunarPatches,
} from "./api";

const fixturePath = new URL(
  "../../../../tests/fixtures/retrieval-response.json",
  import.meta.url,
);

function responseFixture(): Record<string, unknown> {
  return JSON.parse(readFileSync(fixturePath, "utf8"));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("retrieval API contract", () => {
  it("parses the shared ranked-response fixture", () => {
    const response = parseRetrievalResponse(responseFixture());

    expect(response).toMatchObject({
      schemaVersion: 1,
      query: "young crater with bright ejecta",
      modelId: "bpe_geo",
      indexSize: 22578,
      elapsedMs: 84,
    });
    expect(response.results).toEqual([
      expect.objectContaining({
        id: "123",
        rank: 1,
        patchId: 123,
        similarity: 0.312,
        sourceVersion: "v3.0",
        promptStyle: "Geologist to Non-Geologist",
        wacImageUrl: "/api/patches/123/wac",
        lat: -12.34,
        lng: 45.67,
      }),
    ]);
  });

  it("accepts a successful empty response", () => {
    const fixture = responseFixture();
    fixture.results = [];

    expect(parseRetrievalResponse(fixture).results).toEqual([]);
  });

  it.each([
    [
      "missing required field",
      (fixture: Record<string, unknown>) => {
        delete fixture.model_id;
      },
    ],
    [
      "unsupported field",
      (fixture: Record<string, unknown>) => {
        fixture.confidence = 0.99;
      },
    ],
    [
      "invalid latitude",
      (fixture: Record<string, unknown>) => {
        (fixture.results as Record<string, unknown>[])[0].latitude = 91;
      },
    ],
    [
      "invalid longitude",
      (fixture: Record<string, unknown>) => {
        (fixture.results as Record<string, unknown>[])[0].longitude = 180;
      },
    ],
    [
      "incorrect rank",
      (fixture: Record<string, unknown>) => {
        (fixture.results as Record<string, unknown>[])[0].rank = 2;
      },
    ],
  ])("rejects a response with a %s", (_, mutate) => {
    const fixture = responseFixture();
    mutate(fixture);

    expect(() => parseRetrievalResponse(fixture)).toThrow(RetrievalApiError);
  });

  it("formats similarity as a raw score", () => {
    expect(formatSimilarity(0.31249)).toBe("0.312");
    expect(formatSimilarity(-0.1256)).toBe("-0.126");
  });

  it("rejects empty queries before calling the backend", async () => {
    await expect(retrieveLunarPatches("   ")).rejects.toThrow(
      "Enter a retrieval query.",
    );
  });

  it("rejects an out-of-range result limit", async () => {
    await expect(
      retrieveLunarPatches("cratered terrain", { limit: 11 }),
    ).rejects.toThrow("integer from 1 to 10");
  });

  it("sends the default top-k and preserves the error envelope", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      json: async () => ({
        error: {
          code: "MODEL_NOT_READY",
          message: "The retrieval model is not ready.",
          request_id: "abc123",
        },
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      retrieveLunarPatches("cratered terrain"),
    ).rejects.toMatchObject({
      status: 503,
      code: "MODEL_NOT_READY",
      requestId: "abc123",
    });

    const [, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(request.body))).toEqual({
      query: "cratered terrain",
      top_k: 5,
    });
  });

  it("sends the selected top-k", async () => {
    const fixture = responseFixture();
    (fixture.results as Record<string, unknown>[])[0].wac_image_url =
      "https://example.test/patch.webp";
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => fixture,
    });
    vi.stubGlobal("fetch", fetchMock);

    await retrieveLunarPatches("young crater with bright ejecta", { limit: 8 });

    const [, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(request.body))).toEqual({
      query: "young crater with bright ejecta",
      top_k: 8,
    });
  });
});
