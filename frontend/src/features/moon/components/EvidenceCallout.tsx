import { Html, Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { resultLabel, type RetrievalResult } from "../../retrieval/api";
import { latLngToSpherical } from "../lib/sphericalCoordinates";

interface EvidenceCalloutProps {
  result: RetrievalResult;
  selected: boolean;
  onSelectResult: (result: RetrievalResult) => void;
  onPreviewResult: (result: RetrievalResult) => void;
}

function EvidenceCallout({
  result,
  selected,
  onSelectResult,
  onPreviewResult,
}: EvidenceCalloutProps) {
  const markerPosition = useMemo(
    () => latLngToSpherical(result.lat, result.lng, 2.12),
    [result.lat, result.lng],
  );
  const calloutRef = useRef<THREE.Group>(null);
  const offset = useMemo(() => calloutOffset(result.rank), [result.rank]);

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
        color={selected ? "#7dd3fc" : "#f4d06f"}
        transparent
        opacity={selected ? 0.46 : 0.28}
        lineWidth={1}
        depthTest={false}
      />
      <Html
        center
        position={offset}
        zIndexRange={[30, 0]}
        className="target-callout-html"
      >
        <article
          className={`target-evidence-card ${selected ? "selected" : ""}`}
        >
          <button
            type="button"
            className="target-evidence-center"
            aria-label={`Center ${resultLabel(result)} on the moon`}
            aria-pressed={selected}
            title={`Center ${resultLabel(result)} on the moon`}
            onPointerDown={(event) => event.stopPropagation()}
            onPointerUp={(event) => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation();
              onSelectResult(result);
            }}
          />
          <button
            type="button"
            className="target-evidence-thumb"
            aria-label={`Preview ${resultLabel(result)} image and description`}
            title={`Preview ${resultLabel(result)} image and description`}
            onPointerDown={(event) => event.stopPropagation()}
            onPointerUp={(event) => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation();
              onPreviewResult(result);
            }}
            style={{
              backgroundImage: `linear-gradient(145deg, rgba(3, 8, 14, 0.16), rgba(125, 211, 252, 0.12)), url('${result.wacImageUrl}')`,
            }}
          />
          <section>
            <strong>
              #{result.rank.toString().padStart(2, "0")} {resultLabel(result)}
            </strong>
            <span>{result.similarity.toFixed(3)} similarity</span>
          </section>
        </article>
      </Html>
    </group>
  );
}

const CALLOUT_OFFSETS: [number, number, number][] = [
  [0.52, 0.32, 0],
  [-0.52, 0.32, 0],
  [0.52, -0.32, 0],
  [-0.52, -0.32, 0],
  [0.68, 0, 0],
  [-0.68, 0, 0],
  [0, 0.5, 0],
  [0, -0.5, 0],
];

function calloutOffset(rank: number): [number, number, number] {
  return CALLOUT_OFFSETS[(rank - 1) % CALLOUT_OFFSETS.length];
}

export default EvidenceCallout;
