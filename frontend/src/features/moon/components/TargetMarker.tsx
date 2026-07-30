import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { RetrievalResult } from "../../retrieval/api";
import { latLngToSpherical } from "../lib/sphericalCoordinates";

interface TargetMarkerProps {
  result: RetrievalResult;
  selected: boolean;
}

function TargetMarker({ result, selected }: TargetMarkerProps) {
  const markerPosition = useMemo(
    () => latLngToSpherical(result.lat, result.lng, 2.045),
    [result.lat, result.lng],
  );
  const markerRef = useRef<THREE.Group>(null);

  useFrame(({ clock, camera }) => {
    if (!markerRef.current) return;
    markerRef.current.lookAt(camera.position);
    const pulse =
      1 +
      Math.sin(clock.elapsedTime * (selected ? 5 : 3.2)) *
        (selected ? 0.08 : 0.035);
    markerRef.current.scale.setScalar(pulse);
  });

  return (
    <group ref={markerRef} position={markerPosition}>
      <mesh>
        <ringGeometry args={[0.045, 0.066, 48]} />
        <meshBasicMaterial
          color={selected ? "#7dd3fc" : "#f4d06f"}
          transparent
          opacity={selected ? 0.95 : 0.72}
          depthTest={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.014, 18, 18]} />
        <meshBasicMaterial
          color={selected ? "#f8fbff" : "#f4d06f"}
          depthTest={false}
        />
      </mesh>
    </group>
  );
}

export default TargetMarker;
