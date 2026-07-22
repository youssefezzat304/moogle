import { useFrame, useThree } from "@react-three/fiber";
import { useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import type { RetrievalResult } from "../../retrieval/api";
import { latLngToSpherical } from "../lib/sphericalCoordinates";

interface CameraControllerProps {
  activeResult: RetrievalResult;
  cameraDistanceRef: MutableRefObject<number>;
  recenterNonce: number;
  userInteracting: boolean;
}

function CameraController({
  activeResult,
  cameraDistanceRef,
  recenterNonce,
  userInteracting,
}: CameraControllerProps) {
  const { camera } = useThree();
  const currentPosition = useRef(new THREE.Vector3(0, 0, 6.4));
  const isRecentering = useRef(true);
  const lastResultId = useRef(activeResult.id);
  const lastRecenterNonce = useRef(recenterNonce);

  useFrame(() => {
    if (
      activeResult.id !== lastResultId.current ||
      recenterNonce !== lastRecenterNonce.current
    ) {
      lastResultId.current = activeResult.id;
      lastRecenterNonce.current = recenterNonce;
      isRecentering.current = true;
      currentPosition.current.copy(camera.position);
      cameraDistanceRef.current = camera.position.length();
    }

    if (userInteracting) {
      isRecentering.current = false;
      cameraDistanceRef.current = camera.position.length();
      currentPosition.current.copy(camera.position);
      return;
    }

    if (!isRecentering.current) return;

    const cameraDistance = THREE.MathUtils.clamp(
      cameraDistanceRef.current,
      2.7,
      15,
    );
    const targetPosition = latLngToSpherical(
      activeResult.lat,
      activeResult.lng,
      cameraDistance,
    );

    currentPosition.current.lerp(targetPosition, 0.035);
    camera.position.copy(currentPosition.current);
    camera.lookAt(0, 0, 0);

    if (currentPosition.current.distanceTo(targetPosition) < 0.015) {
      isRecentering.current = false;
    }
  });

  return null;
}

export default CameraController;
