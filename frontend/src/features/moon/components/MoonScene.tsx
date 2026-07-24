import { OrbitControls, Stars } from "@react-three/drei";
import { Suspense, type MutableRefObject } from "react";
import type { RetrievalResult } from "../../retrieval/api";
import type { MoonTarget } from "../types";
import CameraController from "./CameraController";
import EvidenceCallout from "./EvidenceCallout";
import Moon from "./Moon";
import MoonFallback from "./MoonFallback";
import TargetMarker from "./TargetMarker";
import TrackingLight from "./TrackingLight";

interface MoonSceneProps {
  activeResult: MoonTarget;
  cameraDistanceRef: MutableRefObject<number>;
  recenterNonce: number;
  userInteracting: boolean;
  onInteractionStart: () => void;
  onInteractionEnd: () => void;
  onPreviewResult: (result: RetrievalResult) => void;
}

function MoonScene({
  activeResult,
  cameraDistanceRef,
  recenterNonce,
  userInteracting,
  onInteractionStart,
  onInteractionEnd,
  onPreviewResult,
}: MoonSceneProps) {
  return (
    <>
      <TrackingLight activeResult={activeResult} />

      {activeResult && (
        <CameraController
          activeResult={activeResult}
          cameraDistanceRef={cameraDistanceRef}
          recenterNonce={recenterNonce}
          userInteracting={userInteracting}
        />
      )}

      <OrbitControls
        enableDamping
        dampingFactor={0.065}
        rotateSpeed={0.42}
        minDistance={2.7}
        maxDistance={15}
        enablePan={false}
        onStart={onInteractionStart}
        onEnd={onInteractionEnd}
        makeDefault
      />

      <Stars
        radius={130}
        depth={70}
        count={6500}
        factor={3.2}
        saturation={0.04}
        fade
        speed={0.006}
      />

      <Suspense fallback={<MoonFallback />}>
        <Moon
          targetCoords={
            activeResult
              ? { lat: activeResult.lat, lng: activeResult.lng }
              : null
          }
        />
        {activeResult && (
          <>
            <TargetMarker activeResult={activeResult} />
            <EvidenceCallout
              activeResult={activeResult}
              onPreviewResult={onPreviewResult}
            />
          </>
        )}
      </Suspense>
    </>
  );
}

export default MoonScene;
