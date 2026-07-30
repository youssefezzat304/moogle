import { useThree } from "@react-three/fiber";
import { useEffect, useRef, type MutableRefObject } from "react";
import * as THREE from "three";

interface MoonControlsProps {
  cameraDistanceRef: MutableRefObject<number>;
  minDistance: number;
  maxDistance: number;
  onInteractionStart: () => void;
}

const ORIGIN = new THREE.Vector3();
const ROTATE_SPEED = 0.9;
const WHEEL_SPEED = 0.0012;

function MoonControls({
  cameraDistanceRef,
  minDistance,
  maxDistance,
  onInteractionStart,
}: MoonControlsProps) {
  const { camera, gl } = useThree();
  const activePointerId = useRef<number | null>(null);
  const previousPoint = useRef(new THREE.Vector3());
  const currentPoint = useRef(new THREE.Vector3());
  const localRotation = useRef(new THREE.Quaternion());
  const worldRotation = useRef(new THREE.Quaternion());
  const inverseCameraRotation = useRef(new THREE.Quaternion());
  const identityRotation = useRef(new THREE.Quaternion());

  useEffect(() => {
    const element = gl.domElement;

    const projectPointer = (event: PointerEvent, target: THREE.Vector3) => {
      const bounds = element.getBoundingClientRect();
      const radius = Math.min(bounds.width, bounds.height) * 0.5;
      const x = (event.clientX - bounds.left - bounds.width * 0.5) / radius;
      const y = (bounds.top + bounds.height * 0.5 - event.clientY) / radius;
      const distanceSquared = x * x + y * y;

      if (distanceSquared <= 1) {
        target.set(x, y, Math.sqrt(1 - distanceSquared));
      } else {
        const scale = 1 / Math.sqrt(distanceSquared);
        target.set(x * scale, y * scale, 0);
      }
    };

    const handlePointerDown = (event: PointerEvent) => {
      if (
        activePointerId.current !== null ||
        !event.isPrimary ||
        (event.pointerType === "mouse" && event.button !== 0)
      ) {
        return;
      }

      event.preventDefault();
      activePointerId.current = event.pointerId;
      projectPointer(event, previousPoint.current);
      element.setPointerCapture(event.pointerId);
      onInteractionStart();
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (event.pointerId !== activePointerId.current) return;

      event.preventDefault();
      projectPointer(event, currentPoint.current);
      localRotation.current.setFromUnitVectors(
        currentPoint.current,
        previousPoint.current,
      );
      localRotation.current.slerp(identityRotation.current, 1 - ROTATE_SPEED);

      inverseCameraRotation.current.copy(camera.quaternion).invert();
      worldRotation.current
        .copy(camera.quaternion)
        .multiply(localRotation.current)
        .multiply(inverseCameraRotation.current);

      camera.position.applyQuaternion(worldRotation.current);
      camera.up.applyQuaternion(worldRotation.current).normalize();
      camera.lookAt(ORIGIN);
      cameraDistanceRef.current = camera.position.length();
      previousPoint.current.copy(currentPoint.current);
    };

    const finishPointerInteraction = (event: PointerEvent) => {
      if (event.pointerId !== activePointerId.current) return;

      if (element.hasPointerCapture(event.pointerId)) {
        element.releasePointerCapture(event.pointerId);
      }
      activePointerId.current = null;
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      onInteractionStart();

      const currentDistance = camera.position.length();
      const wheelDelta = THREE.MathUtils.clamp(event.deltaY, -250, 250);
      const nextDistance = THREE.MathUtils.clamp(
        currentDistance * Math.exp(wheelDelta * WHEEL_SPEED),
        minDistance,
        maxDistance,
      );
      camera.position.setLength(nextDistance);
      cameraDistanceRef.current = nextDistance;
    };

    element.addEventListener("pointerdown", handlePointerDown);
    element.addEventListener("pointermove", handlePointerMove);
    element.addEventListener("pointerup", finishPointerInteraction);
    element.addEventListener("pointercancel", finishPointerInteraction);
    element.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      element.removeEventListener("pointerdown", handlePointerDown);
      element.removeEventListener("pointermove", handlePointerMove);
      element.removeEventListener("pointerup", finishPointerInteraction);
      element.removeEventListener("pointercancel", finishPointerInteraction);
      element.removeEventListener("wheel", handleWheel);
    };
  }, [
    camera,
    cameraDistanceRef,
    gl.domElement,
    maxDistance,
    minDistance,
    onInteractionStart,
  ]);

  return null;
}

export default MoonControls;
