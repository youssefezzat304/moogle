import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { RetrievalResult } from "../../retrieval/api";
import { latLngToSpherical } from "../lib/sphericalCoordinates";

interface TargetMarkerProps {
  activeResult: RetrievalResult;
}

function TargetMarker({ activeResult }: TargetMarkerProps) {
  const markerPosition = useMemo(
    () => latLngToSpherical(activeResult.lat, activeResult.lng, 2.045),
    [activeResult.lat, activeResult.lng],
  );
  const markerRef = useRef<THREE.Group>(null);

  useFrame(({ clock, camera }) => {
    if (!markerRef.current) return;
    markerRef.current.lookAt(camera.position);
    const pulse = 1 + Math.sin(clock.elapsedTime * 5) * 0.08;
    markerRef.current.scale.setScalar(pulse);
  });

  return (
    <group ref={markerRef} position={markerPosition}>
      <mesh>
        <ringGeometry args={[0.045, 0.066, 48]} />
        <meshBasicMaterial
          color="#7dd3fc"
          transparent
          opacity={0.95}
          depthTest={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.014, 18, 18]} />
        <meshBasicMaterial color="#f8fbff" depthTest={false} />
      </mesh>
    </group>
  );
}

export default TargetMarker;
