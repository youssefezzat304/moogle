import { describe, expect, it } from "vitest";
import { latLngToSpherical } from "./sphericalCoordinates";

describe("latLngToSpherical", () => {
  it.each([
    { latitude: 90, longitude: 0, expected: [0, 2, 0] },
    { latitude: -90, longitude: 0, expected: [0, -2, 0] },
    { latitude: 0, longitude: -90, expected: [0, 0, 2] },
    { latitude: 0, longitude: 0, expected: [2, 0, 0] },
  ])(
    "maps latitude $latitude and longitude $longitude to the expected axis",
    ({ latitude, longitude, expected }) => {
      const result = latLngToSpherical(latitude, longitude, 2);

      expect(result.x).toBeCloseTo(expected[0]);
      expect(result.y).toBeCloseTo(expected[1]);
      expect(result.z).toBeCloseTo(expected[2]);
    },
  );

  it("returns a new vector at the requested radius", () => {
    const first = latLngToSpherical(23.4, -12.8, 6.4);
    const second = latLngToSpherical(23.4, -12.8, 6.4);

    expect(first).not.toBe(second);
    expect(first.length()).toBeCloseTo(6.4);
    expect(second.length()).toBeCloseTo(6.4);
  });
});
