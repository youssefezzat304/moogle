import { Canvas } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";
import type { RetrievalResult } from "../../retrieval/api";
import { useMoonInteraction } from "../hooks/useMoonInteraction";
import type { DemoQueryCatalog } from "../../../shared/utils/demoQueries";
import type { MoonTarget } from "../types";
import MoonHud from "./MoonHud";
import MoonScene from "./MoonScene";

interface MoonCanvasProps {
  activeResult: MoonTarget;
  onPreviewResult: (result: RetrievalResult) => void;
  demoQueryCatalog: DemoQueryCatalog | null;
  demoQueryError: string | null;
  canRunDemoQuery: boolean;
  onRunDemoQuery: (query: string) => void;
}

function MoonCanvas({
  activeResult,
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
    userInteracting,
    startInteraction,
    settleInteraction,
    recenterTarget,
  } = useMoonInteraction();

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
        onPointerDown={startInteraction}
        onPointerUp={settleInteraction}
        onPointerLeave={settleInteraction}
        onWheel={() => {
          startInteraction();
          settleInteraction();
        }}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 0.94,
          outputColorSpace: THREE.SRGBColorSpace,
        }}
      >
        <MoonScene
          activeResult={activeResult}
          cameraDistanceRef={cameraDistanceRef}
          recenterNonce={recenterNonce}
          userInteracting={userInteracting}
          onInteractionStart={startInteraction}
          onInteractionEnd={settleInteraction}
          onPreviewResult={onPreviewResult}
        />
      </Canvas>
    </div>
  );
}

export default MoonCanvas;
