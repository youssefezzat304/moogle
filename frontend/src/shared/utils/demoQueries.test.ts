import { describe, expect, it } from "vitest";
import {
  parseDemoQueryCatalog,
  pickFeatureDemoQuery,
  pickRandomDemoSuggestions,
} from "./demoQueries";

const catalog = parseDemoQueryCatalog({
  source_version: "v2.0",
  prompt_style: "llm_description",
  features: {
    Cc: [
      { patch_id: 12, query: "Copernican crater terrain." },
      { patch_id: 18, query: "A second Copernican example." },
    ],
    Em: [{ patch_id: 25, query: "Eratosthenian mare terrain." }],
    Ig: [],
  },
});

describe("demo query catalog", () => {
  it("parses feature queries and their provenance", () => {
    expect(catalog).toEqual({
      sourceVersion: "v2.0",
      promptStyle: "llm_description",
      features: {
        Cc: [
          { patchId: 12, query: "Copernican crater terrain." },
          { patchId: 18, query: "A second Copernican example." },
        ],
        Em: [{ patchId: 25, query: "Eratosthenian mare terrain." }],
        Ig: [],
      },
    });
  });

  it("rejects queries that exceed the retrieval contract", () => {
    expect(() =>
      parseDemoQueryCatalog({
        source_version: "v2.0",
        prompt_style: "llm_description",
        features: {
          Cc: [{ patch_id: 1, query: "x".repeat(501) }],
        },
      }),
    ).toThrow("Demo queries for feature Cc are invalid.");
  });

  it("selects a feature query with injectable randomness", () => {
    expect(pickFeatureDemoQuery(catalog, "Cc", () => 0.99)).toBe(
      "A second Copernican example.",
    );
    expect(pickFeatureDemoQuery(catalog, "Ig")).toBeNull();
  });

  it("returns distinct random suggestions", () => {
    expect(pickRandomDemoSuggestions(catalog, 2, () => 0)).toEqual([
      "Eratosthenian mare terrain.",
      "Copernican crater terrain.",
    ]);
  });
});
