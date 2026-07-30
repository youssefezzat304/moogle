import { Canvas } from "@react-three/fiber";
import { useCallback, useRef } from "react";
import * as THREE from "three";
import type { RetrievalResult } from "../../retrieval/api";
import { useMoonInteraction } from "../hooks/useMoonInteraction";
import type { DemoQueryCatalog } from "../../../shared/utils/demoQueries";
import type { MoonTarget } from "../types";
import MoonHud from "./MoonHud";
import MoonScene from "./MoonScene";

interface MoonCanvasProps {
  results: RetrievalResult[];
  activeResult: MoonTarget;
  onSelectResult: (result: RetrievalResult) => void;
  onPreviewResult: (result: RetrievalResult) => void;
  demoQueryCatalog: DemoQueryCatalog | null;
  demoQueryError: string | null;
  canRunDemoQuery: boolean;
  onRunDemoQuery: (query: string) => void;
}

function MoonCanvas({
  results,
  activeResult,
  onSelectResult,
  onPreviewResult,
  demoQueryCatalog,
  demoQueryError,
  canRunDemoQuery,
  onRunDemoQuery,
}: MoonCanvasProps) {
  const cameraDistanceRef = useRef(6.4);
  const {
    hasWandered,
    recenterNonce,
    interactionNonceRef,
    startInteraction,
    recenterTarget,
  } = useMoonInteraction();
  const centerResult = useCallback(
    (result: RetrievalResult) => {
      onSelectResult(result);
      recenterTarget();
    },
    [onSelectResult, recenterTarget],
  );

  return (
    <div className="moon-stage">
      <MoonHud
        activeResult={activeResult}
        hasWandered={hasWandered}
        onRecenter={recenterTarget}
        demoQueryCatalog={demoQueryCatalog}
        demoQueryError={demoQueryError}
        canRunDemoQuery={canRunDemoQuery}
        onRunDemoQuery={onRunDemoQuery}
      />

      <Canvas
        camera={{ position: [0, 0, 6.4], fov: 48 }}
        className="moon-canvas"
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 0.94,
          outputColorSpace: THREE.SRGBColorSpace,
        }}
      >
        <MoonScene
          results={results}
          activeResult={activeResult}
          cameraDistanceRef={cameraDistanceRef}
          recenterNonce={recenterNonce}
          interactionNonceRef={interactionNonceRef}
          onInteractionStart={startInteraction}
          onSelectResult={centerResult}
          onPreviewResult={onPreviewResult}
        />
      </Canvas>
    </div>
  );
}

export default MoonCanvas;
