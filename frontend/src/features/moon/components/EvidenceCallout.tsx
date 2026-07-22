import { Html, Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { resultLabel, type RetrievalResult } from "../../retrieval/api";
import { latLngToSpherical } from "../lib/sphericalCoordinates";

interface EvidenceCalloutProps {
  activeResult: RetrievalResult;
}

function EvidenceCallout({ activeResult }: EvidenceCalloutProps) {
  const markerPosition = useMemo(
    () => latLngToSpherical(activeResult.lat, activeResult.lng, 2.12),
    [activeResult.lat, activeResult.lng],
  );
  const calloutRef = useRef<THREE.Group>(null);
  const offset = useMemo<[number, number, number]>(() => [0.52, 0.32, 0], []);

  useFrame(({ camera }) => {
    calloutRef.current?.lookAt(camera.position);
  });

  return (
    <group ref={calloutRef} position={markerPosition}>
      <Line
        points={[
          [0, 0, 0],
          [offset[0] * 0.72, offset[1] * 0.72, 0],
        ]}
        color="#7dd3fc"
        transparent
        opacity={0.46}
        lineWidth={1}
        depthTest={false}
      />
      <Html
        center
        position={offset}
        zIndexRange={[30, 0]}
        className="target-callout-html"
      >
        <article className="target-evidence-card">
          <a
            href={activeResult.imageUrl}
            target="_blank"
            rel="noreferrer"
            className="target-evidence-thumb"
            aria-label={`Open ${resultLabel(activeResult)} image`}
            title={`Open ${resultLabel(activeResult)} image`}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={(event) => event.stopPropagation()}
            style={{
              backgroundImage: `linear-gradient(145deg, rgba(3, 8, 14, 0.16), rgba(125, 211, 252, 0.12)), url('${activeResult.imageUrl}')`,
            }}
          />
          <section>
            <strong>{resultLabel(activeResult)}</strong>
            <span>{activeResult.similarity.toFixed(3)} similarity</span>
          </section>
        </article>
      </Html>
    </group>
  );
}

export default EvidenceCallout;
