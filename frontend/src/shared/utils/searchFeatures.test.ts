import { describe, expect, it } from "vitest";
import { parseSearchFeatures } from "./searchFeatures";

describe("parseSearchFeatures", () => {
  it("maps legend entries into searchable features", () => {
    expect(
      parseSearchFeatures({
        Cc: {
          color: "#FCDC0A",
          description: "Crater Unit",
          long_description: "Copernican Crater Unit",
        },
      }),
    ).toEqual([
      {
        code: "Cc",
        color: "#FCDC0A",
        description: "Crater Unit",
        longDescription: "Copernican Crater Unit",
      },
    ]);
  });

  it("rejects malformed legend entries", () => {
    expect(() =>
      parseSearchFeatures({
        Cc: {
          color: "yellow",
          description: "Crater Unit",
          long_description: "Copernican Crater Unit",
        },
      }),
    ).toThrow("Legend entry Cc is invalid.");
  });
});
