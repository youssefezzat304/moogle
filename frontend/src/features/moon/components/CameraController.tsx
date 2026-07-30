import { useFrame, useThree } from "@react-three/fiber";
import { useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import type { RetrievalResult } from "../../retrieval/api";
import { latLngToSpherical } from "../lib/sphericalCoordinates";

interface CameraControllerProps {
  activeResult: RetrievalResult;
  cameraDistanceRef: MutableRefObject<number>;
  interactionNonceRef: MutableRefObject<number>;
  recenterNonce: number;
}

function CameraController({
  activeResult,
  cameraDistanceRef,
  interactionNonceRef,
  recenterNonce,
}: CameraControllerProps) {
  const { camera } = useThree();
  const isRecentering = useRef(false);
  const lastRecenterNonce = useRef(recenterNonce);
  const lastInteractionNonce = useRef(0);
  const progress = useRef(0);
  const startDirection = useRef(new THREE.Vector3(0, 0, 1));
  const startUp = useRef(new THREE.Vector3(0, 1, 0));
  const targetDirection = useRef(new THREE.Vector3());
  const arcRotation = useRef(new THREE.Quaternion());
  const frameRotation = useRef(new THREE.Quaternion());
  const frameDirection = useRef(new THREE.Vector3());
  const frameUp = useRef(new THREE.Vector3());

  useFrame((_, delta) => {
    if (recenterNonce !== lastRecenterNonce.current) {
      lastRecenterNonce.current = recenterNonce;
      isRecentering.current = true;
      progress.current = 0;
      cameraDistanceRef.current = camera.position.length();
      startDirection.current.copy(camera.position).normalize();
      startUp.current
        .copy(camera.up)
        .addScaledVector(
          startDirection.current,
          -camera.up.dot(startDirection.current),
        )
        .normalize();
      targetDirection.current
        .copy(latLngToSpherical(activeResult.lat, activeResult.lng, 1))
        .normalize();
      arcRotation.current.setFromUnitVectors(
        startDirection.current,
        targetDirection.current,
      );
    }

    if (interactionNonceRef.current !== lastInteractionNonce.current) {
      lastInteractionNonce.current = interactionNonceRef.current;
      isRecentering.current = false;
      cameraDistanceRef.current = camera.position.length();
      return;
    }

    if (!isRecentering.current) {
      cameraDistanceRef.current = camera.position.length();
      return;
    }

    const cameraDistance = THREE.MathUtils.clamp(
      cameraDistanceRef.current,
      2.7,
      15,
    );
    progress.current = 1 - (1 - progress.current) * Math.exp(-2.15 * delta);
    frameRotation.current
      .identity()
      .slerp(arcRotation.current, progress.current);
    frameDirection.current
      .copy(startDirection.current)
      .applyQuaternion(frameRotation.current);
    frameUp.current
      .copy(startUp.current)
      .applyQuaternion(frameRotation.current)
      .normalize();

    camera.position.copy(frameDirection.current).multiplyScalar(cameraDistance);
    camera.up.copy(frameUp.current);
    camera.lookAt(0, 0, 0);

    if (progress.current > 0.9975) {
      camera.position
        .copy(targetDirection.current)
        .multiplyScalar(cameraDistance);
      camera.up
        .copy(startUp.current)
        .applyQuaternion(arcRotation.current)
        .normalize();
      camera.lookAt(0, 0, 0);
      isRecentering.current = false;
    }
  });

  return null;
}

export default CameraController;
