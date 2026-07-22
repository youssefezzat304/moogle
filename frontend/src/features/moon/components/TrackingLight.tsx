import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";
import type { MoonTarget } from "../types";
import { latLngToSpherical } from "../lib/sphericalCoordinates";

interface TrackingLightProps {
  activeResult: MoonTarget;
}

function TrackingLight({ activeResult }: TrackingLightProps) {
  const keyLight = useRef<THREE.DirectionalLight>(null);
  const glowLight = useRef<THREE.PointLight>(null);
  const lightPosition = useRef(new THREE.Vector3(6, 3, 4));

  useFrame(() => {
    if (!activeResult) return;

    const surfaceDirection = latLngToSpherical(
      activeResult.lat,
      activeResult.lng,
      1,
    ).normalize();
    const targetLightPosition = surfaceDirection
      .multiplyScalar(6.7)
      .add(new THREE.Vector3(0.75, 1.15, 0.55));

    lightPosition.current.lerp(targetLightPosition, 0.045);

    if (keyLight.current) {
      keyLight.current.position.copy(lightPosition.current);
      keyLight.current.target.position.set(0, 0, 0);
      keyLight.current.target.updateMatrixWorld();
    }

    if (glowLight.current) {
      glowLight.current.position.copy(
        lightPosition.current.clone().multiplyScalar(0.72),
      );
    }
  });

  return (
    <>
      <ambientLight intensity={0.025} />
      <directionalLight
        ref={keyLight}
        position={[6, 3, 4]}
        intensity={1.55}
        color="#f8fbff"
      />
      <pointLight
        ref={glowLight}
        position={[5, 2, 4]}
        intensity={0.18}
        distance={7}
        color="#8ee7ff"
      />
      <directionalLight
        position={[-4.5, -2.4, -3.6]}
        intensity={0.055}
        color="#6da0ff"
      />
    </>
  );
}

export default TrackingLight;
