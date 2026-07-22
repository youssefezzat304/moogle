import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Line, OrbitControls, Stars } from "@react-three/drei";
import {
  Suspense,
  useCallback,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
} from "react";
import { LocateFixed } from "lucide-react";
import * as THREE from "three";
import Moon from "./Moon";
import {
  EVIDENCE_IMAGE_URL,
  formatCoords,
  type RetrievalResult,
} from "../../retrieval/mockData";

interface MoonCanvasProps {
  activeResult: RetrievalResult;
  hasRetrieved: boolean;
}

function latLngToSpherical(lat: number, lng: number, radius: number) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 90) * (Math.PI / 180);
  return new THREE.Vector3().setFromSphericalCoords(radius, phi, theta);
}

function CameraController({
  activeResult,
  cameraDistanceRef,
  recenterNonce,
  userInteracting,
}: {
  activeResult: RetrievalResult;
  cameraDistanceRef: MutableRefObject<number>;
  recenterNonce: number;
  userInteracting: boolean;
}) {
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

function TrackingLight({ activeResult }: { activeResult: RetrievalResult }) {
  const keyLight = useRef<THREE.DirectionalLight>(null);
  const glowLight = useRef<THREE.PointLight>(null);
  const lightPosition = useRef(new THREE.Vector3(6, 3, 4));

  useFrame(() => {
    const surfaceDirection = latLngToSpherical(
      activeResult.lat,
      activeResult.lng,
      1,
    ).normalize();
    const targetLightPosition = surfaceDirection
      .multiplyScalar(6.7)
      .add(new THREE.Vector3(0.75, 1.15, 0.55));

    lightPosition.current.lerp(targetLightPosition, 0.045);

    if (keyLight.current) {
      keyLight.current.position.copy(lightPosition.current);
      keyLight.current.target.position.set(0, 0, 0);
      keyLight.current.target.updateMatrixWorld();
    }

    if (glowLight.current) {
      glowLight.current.position.copy(lightPosition.current.clone().multiplyScalar(0.72));
    }
  });

  return (
    <>
      <ambientLight intensity={0.025} />
      <directionalLight
        ref={keyLight}
        position={[6, 3, 4]}
        intensity={1.55}
        color="#f8fbff"
      />
      <pointLight
        ref={glowLight}
        position={[5, 2, 4]}
        intensity={0.18}
        distance={7}
        color="#8ee7ff"
      />
      <directionalLight
        position={[-4.5, -2.4, -3.6]}
        intensity={0.055}
        color="#6da0ff"
      />
    </>
  );
}

function MoonSphere() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = clock.getElapsedTime() * 0.004;
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0, 0]}>
      <sphereGeometry args={[2, 64, 64]} />
      <meshStandardMaterial color="#4f5966" roughness={1} metalness={0} />
    </mesh>
  );
}

function TargetMarker({ activeResult }: { activeResult: RetrievalResult }) {
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

function TargetEvidenceCallouts({
  activeResult,
}: {
  activeResult: RetrievalResult;
}) {
  const markerPosition = useMemo(
    () => latLngToSpherical(activeResult.lat, activeResult.lng, 2.12),
    [activeResult.lat, activeResult.lng],
  );
  const calloutRef = useRef<THREE.Group>(null);
  const offsets = useMemo<[number, number, number][]>(
    () => [
      [0.52, 0.32, 0],
      [0.58, -0.3, 0],
      [-0.48, 0.02, 0],
    ],
    [],
  );

  useFrame(({ camera }) => {
    calloutRef.current?.lookAt(camera.position);
  });

  return (
    <group ref={calloutRef} position={markerPosition}>
      {activeResult.images.slice(0, 3).map((image, index) => {
        const offset =
          offsets[index] ?? ([0.72, 0.44, 0] as [number, number, number]);

        return (
          <group key={image.id}>
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
                  href={EVIDENCE_IMAGE_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="target-evidence-thumb"
                  aria-label={`Open ${image.title} image`}
                  title={`Open ${image.title} image`}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={(event) => event.stopPropagation()}
                  style={{
                    backgroundImage: `linear-gradient(145deg, rgba(3, 8, 14, 0.16), rgba(125, 211, 252, 0.12)), url('${EVIDENCE_IMAGE_URL}')`,
                    backgroundPosition: image.crop,
                  }}
                />
                <section>
                  <strong>{image.title}</strong>
                  <span>{Math.round(image.score * 100)} match</span>
                </section>
              </article>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

function ViewportHUD({
  activeResult,
  hasWandered,
  onRecenter,
}: {
  activeResult: RetrievalResult;
  hasWandered: boolean;
  onRecenter: () => void;
}) {
  return (
    <div className="viewport-hud">
      <div className="hud-target">
        <span className="eyebrow">target lock</span>
        <strong>{activeResult.title}</strong>
        <small>{formatCoords(activeResult.lat, activeResult.lng)}</small>
      </div>

      <div className="reticle" aria-hidden="true">
        <span />
      </div>

      {hasWandered && (
        <button
          type="button"
          className="recenter-button"
          onClick={onRecenter}
          aria-label="Return to active retrieval target"
          title="Return to active retrieval target"
        >
          <LocateFixed size={16} />
        </button>
      )}

      <div className="viewport-footer">
        <span>LROC texture stream · LDEM relief</span>
        <strong>dynamic terminator</strong>
      </div>
    </div>
  );
}

function MoonCanvas({ activeResult, hasRetrieved }: MoonCanvasProps) {
  const [hasWandered, setHasWandered] = useState(false);
  const [recenterNonce, setRecenterNonce] = useState(0);
  const [userInteracting, setUserInteracting] = useState(false);
  const cameraDistanceRef = useRef(6.4);
  const interactionTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const startInteraction = useCallback(() => {
    setHasWandered(true);
    setUserInteracting(true);
    if (interactionTimeout.current) clearTimeout(interactionTimeout.current);
  }, []);

  const settleInteraction = useCallback(() => {
    if (interactionTimeout.current) clearTimeout(interactionTimeout.current);
    interactionTimeout.current = setTimeout(
      () => setUserInteracting(false),
      2400,
    );
  }, []);

  const recenterTarget = useCallback(() => {
    setHasWandered(false);
    setUserInteracting(false);
    if (interactionTimeout.current) clearTimeout(interactionTimeout.current);
    setRecenterNonce((nonce) => nonce + 1);
  }, []);

  return (
    <div className="moon-stage">
      <ViewportHUD
        activeResult={activeResult}
        hasWandered={hasWandered}
        onRecenter={recenterTarget}
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
        <TrackingLight activeResult={activeResult} />

        <CameraController
          activeResult={activeResult}
          cameraDistanceRef={cameraDistanceRef}
          recenterNonce={recenterNonce}
          userInteracting={userInteracting}
        />

        <OrbitControls
          enableDamping
          dampingFactor={0.065}
          rotateSpeed={0.42}
          minDistance={2.7}
          maxDistance={15}
          enablePan={false}
          onStart={startInteraction}
          onEnd={settleInteraction}
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

        <Suspense fallback={<MoonSphere />}>
          <Moon targetCoords={{ lat: activeResult.lat, lng: activeResult.lng }} />
          <TargetMarker activeResult={activeResult} />
          {hasRetrieved && <TargetEvidenceCallouts activeResult={activeResult} />}
        </Suspense>
      </Canvas>
    </div>
  );
}

export default MoonCanvas;
