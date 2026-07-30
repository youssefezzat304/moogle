import { Stars } from "@react-three/drei";
import { Suspense, type MutableRefObject } from "react";
import type { RetrievalResult } from "../../retrieval/api";
import type { MoonTarget } from "../types";
import CameraController from "./CameraController";
import EvidenceCallout from "./EvidenceCallout";
import Moon from "./Moon";
import MoonControls from "./MoonControls";
import MoonFallback from "./MoonFallback";
import RetrievalMeteorSearch from "./RetrievalMeteorSearch";
import TargetMarker from "./TargetMarker";
import TrackingLight from "./TrackingLight";

interface MoonSceneProps {
  results: RetrievalResult[];
  activeResult: MoonTarget;
  isLoading: boolean;
  cameraDistanceRef: MutableRefObject<number>;
  interactionNonceRef: MutableRefObject<number>;
  recenterNonce: number;
  onInteractionStart: () => void;
  onSelectResult: (result: RetrievalResult) => void;
  onPreviewResult: (result: RetrievalResult) => void;
}

function MoonScene({
  results,
  activeResult,
  isLoading,
  cameraDistanceRef,
  interactionNonceRef,
  recenterNonce,
  onInteractionStart,
  onSelectResult,
  onPreviewResult,
}: MoonSceneProps) {
  return (
    <>
      <TrackingLight activeResult={activeResult} />

      {activeResult && (
        <CameraController
          activeResult={activeResult}
          cameraDistanceRef={cameraDistanceRef}
          interactionNonceRef={interactionNonceRef}
          recenterNonce={recenterNonce}
        />
      )}

      <MoonControls
        cameraDistanceRef={cameraDistanceRef}
        minDistance={2.7}
        maxDistance={15}
        onInteractionStart={onInteractionStart}
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

      {isLoading && <RetrievalMeteorSearch />}

      <Suspense fallback={<MoonFallback />}>
        <Moon
          targetCoords={
            activeResult
              ? { lat: activeResult.lat, lng: activeResult.lng }
              : null
          }
        />
        {results.map((result) => {
          const selected = result.id === activeResult?.id;

          return (
            <group key={result.id}>
              <TargetMarker result={result} selected={selected} />
              <EvidenceCallout
                result={result}
                selected={selected}
                onSelectResult={onSelectResult}
                onPreviewResult={onPreviewResult}
              />
            </group>
          );
        })}
      </Suspense>
    </>
  );
}

export default MoonScene;
