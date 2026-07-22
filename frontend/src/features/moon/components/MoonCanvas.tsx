import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Stars, OrbitControls } from "@react-three/drei";
import { Suspense, useState, useEffect, useRef, useCallback } from "react";
import Moon from "./Moon";
import * as THREE from "three";

// ─────────────────────────────────────────────
//  Location data
// ─────────────────────────────────────────────

interface LunarLocation {
  lat: number;
  lng: number;
  label: string;
  sublabel: string;
}

const LOCATIONS: LunarLocation[] = [
  {
    lat: 0.674,
    lng: 23.473,
    label: "Apollo 11",
    sublabel: "Mare Tranquillitatis",
  },
  {
    lat: -3.013,
    lng: -23.422,
    label: "Apollo 12",
    sublabel: "Oceanus Procellarum",
  },
  {
    lat: 26.132,
    lng: 3.634,
    label: "Apollo 14",
    sublabel: "Fra Mauro Highlands",
  },
  {
    lat: -43.192,
    lng: 339.602,
    label: "Tycho Crater",
    sublabel: "Southern Highlands",
  },
  { lat: 19.794, lng: -4.532, label: "Mare Imbrium", sublabel: "Lunar Mare" },
  {
    lat: 45.0,
    lng: 120.0,
    label: "Mare Frigoris",
    sublabel: "Northern Polar Region",
  },
  {
    lat: -20.157,
    lng: 30.772,
    label: "Apollo 16",
    sublabel: "Descartes Highlands",
  },
  {
    lat: 20.178,
    lng: 30.772,
    label: "Plato Crater",
    sublabel: "Mare Imbrium Border",
  },
];

// ─────────────────────────────────────────────
//  Utility
// ─────────────────────────────────────────────

function latLngToSpherical(
  lat: number,
  lng: number,
  radius: number,
): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 90) * (Math.PI / 180);
  return new THREE.Vector3().setFromSphericalCoords(radius, phi, theta);
}

// ─────────────────────────────────────────────
//  Camera controller (inside Canvas)
// ─────────────────────────────────────────────

interface CameraControllerProps {
  targetIndex: number;
  userDragging: boolean;
}

function CameraController({
  targetIndex,
  userDragging,
}: CameraControllerProps) {
  const { camera } = useThree();
  const currentTarget = useRef(new THREE.Vector3(0, 0, 5));

  useFrame(() => {
    if (userDragging) return;

    const loc = LOCATIONS[targetIndex];
    const dest = latLngToSpherical(loc.lat, loc.lng, 5);

    // Smooth lerp — 0.022 is slow enough to feel deliberate, fast enough to feel responsive
    currentTarget.current.lerp(dest, 0.022);
    camera.position.copy(currentTarget.current);
    camera.lookAt(0, 0, 0);
  });

  return null;
}

// ─────────────────────────────────────────────
//  Fallback sphere (shown while textures load)
// ─────────────────────────────────────────────

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
      <meshStandardMaterial color="#4a5060" roughness={1} metalness={0} />
    </mesh>
  );
}

// ─────────────────────────────────────────────
//  HUD — rendered in DOM over the canvas
// ─────────────────────────────────────────────

interface HUDProps {
  location: LunarLocation;
  locationIndex: number;
  total: number;
  loaded: boolean;
}

function HUD({ location, locationIndex, total, loaded }: HUDProps) {
  const lat = location.lat.toFixed(3);
  const lng = location.lng.toFixed(3);
  const latDir = location.lat >= 0 ? "N" : "S";
  const lngDir = location.lng >= 0 ? "E" : "W";

  return (
    <>
      {/* ── Top-left: coordinates ── */}
      <div
        className="absolute top-4 left-4 flex flex-col gap-1 font-mono pointer-events-none"
        style={{ zIndex: 10 }}
      >
        <div
          className="text-[9px] tracking-[0.2em] uppercase"
          style={{ color: "var(--color-amber)" }}
        >
          {location.label}
        </div>
        <div
          className="text-[9px] tracking-[0.12em]"
          style={{ color: "var(--color-muted)" }}
        >
          {location.sublabel}
        </div>
        <div
          className="text-[9px] tracking-[0.12em] tabular-nums mt-1"
          style={{ color: "var(--color-fg-dim)" }}
        >
          {Math.abs(parseFloat(lat))}° {latDir} &nbsp;·&nbsp;{" "}
          {Math.abs(parseFloat(lng))}° {lngDir}
        </div>
      </div>

      {/* ── Top-right: frame counter / index ── */}
      <div
        className="absolute top-4 right-4 flex flex-col items-end gap-1 font-mono pointer-events-none"
        style={{ zIndex: 10 }}
      >
        <div
          className="text-[9px] tracking-[0.2em] tabular-nums"
          style={{ color: "var(--color-muted)" }}
        >
          {String(locationIndex + 1).padStart(2, "0")} /{" "}
          {String(total).padStart(2, "0")}
        </div>
        <div
          className="flex items-center gap-1.5 mt-1"
          style={{ color: loaded ? "var(--color-green)" : "#6b5530" }}
        >
          <span
            className="w-1 h-1 rounded-full"
            style={{
              background: loaded
                ? "var(--color-green)"
                : "var(--color-amber-dim)",
              boxShadow: loaded ? "0 0 4px var(--color-green)" : "none",
            }}
          />
          <span className="text-[9px] tracking-[0.15em] uppercase">
            {loaded ? "TEXTURED" : "LOADING"}
          </span>
        </div>
      </div>

      {/* ── Center crosshair ── */}
      <div
        className="absolute inset-0 flex items-center justify-center pointer-events-none"
        style={{ zIndex: 10 }}
      >
        <svg
          width="32"
          height="32"
          viewBox="0 0 32 32"
          fill="none"
          opacity={0.35}
        >
          {/* Four bracket arms */}
          <line
            x1="16"
            y1="0"
            x2="16"
            y2="6"
            stroke="#c8a96e"
            strokeWidth="0.75"
          />
          <line
            x1="16"
            y1="26"
            x2="16"
            y2="32"
            stroke="#c8a96e"
            strokeWidth="0.75"
          />
          <line
            x1="0"
            y1="16"
            x2="6"
            y2="16"
            stroke="#c8a96e"
            strokeWidth="0.75"
          />
          <line
            x1="26"
            y1="16"
            x2="32"
            y2="16"
            stroke="#c8a96e"
            strokeWidth="0.75"
          />
          {/* Center dot */}
          <circle cx="16" cy="16" r="1" fill="#c8a96e" />
        </svg>
      </div>

      {/* ── Bottom-right: scale bar ── */}
      <div
        className="absolute right-4 flex flex-col items-end gap-1 font-mono pointer-events-none"
        style={{ zIndex: 10, bottom: "2.5rem" }}
      >
        <div className="flex items-center gap-0">
          <div
            style={{ width: 40, height: 1, background: "var(--color-muted)" }}
          />
          <div
            style={{ width: 1, height: 4, background: "var(--color-muted)" }}
          />
        </div>
        <span
          className="text-[8px] tracking-[0.12em]"
          style={{ color: "var(--color-muted)" }}
        >
          ~500 KM
        </span>
      </div>
    </>
  );
}

// ─────────────────────────────────────────────
//  Main component
// ─────────────────────────────────────────────

function MoonCanvas() {
  const [locationIndex, setLocationIndex] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [userDragging, setUserDragging] = useState(false);
  const dragTimeout = useRef<ReturnType<typeof setTimeout>>();

  // Listen for messages from ChatInterface
  useEffect(() => {
    const handleMessageSent = () => {
      setLocationIndex((prev) => (prev + 1) % LOCATIONS.length);
    };
    window.addEventListener("messageSent", handleMessageSent);
    return () => window.removeEventListener("messageSent", handleMessageSent);
  }, []);

  // Detect user orbit-drag so we can pause the auto-camera
  const onPointerDown = useCallback(() => {
    setUserDragging(true);
    clearTimeout(dragTimeout.current);
  }, []);

  const onPointerUp = useCallback(() => {
    // Resume auto-camera 2.5 s after the user lets go
    dragTimeout.current = setTimeout(() => setUserDragging(false), 2500);
  }, []);

  const location = LOCATIONS[locationIndex];

  return (
    <div className="relative h-full w-full bg-black">
      {/* ─ HUD overlay (DOM) ─ */}
      <HUD
        location={location}
        locationIndex={locationIndex}
        total={LOCATIONS.length}
        loaded={loaded}
      />

      {/* ─ Three.js canvas ─ */}
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        className="h-full w-full"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.1,
          outputColorSpace: THREE.SRGBColorSpace,
        }}
      >
        {/*
          Two-light rig:
          • Key light: hard sun from upper-right → strong terminator shadow
          • Fill light: very dim cool blue from the opposite side → slight earthshine fill
        */}
        <ambientLight intensity={0.04} />
        <directionalLight
          position={[6, 3, 4]}
          intensity={1.8}
          color="#fff8f0"
        />
        <directionalLight
          position={[-5, -2, -3]}
          intensity={0.06}
          color="#7799cc"
        />

        {/* Auto-camera (pauses while user drags) */}
        <CameraController
          targetIndex={locationIndex}
          userDragging={userDragging}
        />

        {/*
          OrbitControls enabled — lets the user manually inspect the surface.
          enableDamping makes the drag feel weighty and satisfying.
          Min/max distance keeps the moon from being clipped.
        */}
        <OrbitControls
          enableDamping
          dampingFactor={0.06}
          rotateSpeed={0.45}
          minDistance={2.6}
          maxDistance={18}
          enablePan={false}
          makeDefault
        />

        {/* Star field: dense, small, slow drift */}
        <Stars
          radius={120}
          depth={60}
          count={7000}
          factor={3.5}
          saturation={0.1}
          fade
          speed={0.008}
        />

        {/* Moon — Suspense falls back to a plain grey sphere while textures stream in */}
        <Suspense fallback={<MoonSphere />}>
          <Moon />
          {/* Side-effect: mark loaded once Three's Suspense resolves */}
          <LoadedSignal onLoaded={() => setLoaded(true)} />
        </Suspense>
      </Canvas>
    </div>
  );
}

// Tiny helper that fires a callback once it mounts (i.e. textures resolved)
function LoadedSignal({ onLoaded }: { onLoaded: () => void }) {
  const fired = useRef(false);
  useFrame(() => {
    if (!fired.current) {
      fired.current = true;
      onLoaded();
    }
  });
  return null;
}

export default MoonCanvas;
