import { Vector3 } from "three";

const DEGREES_TO_RADIANS = Math.PI / 180;

export function latLngToSpherical(
  latitude: number,
  longitude: number,
  radius: number,
): Vector3 {
  const polarAngle = (90 - latitude) * DEGREES_TO_RADIANS;
  const azimuthalAngle = (longitude + 90) * DEGREES_TO_RADIANS;

  return new Vector3().setFromSphericalCoords(
    radius,
    polarAngle,
    azimuthalAngle,
  );
}
