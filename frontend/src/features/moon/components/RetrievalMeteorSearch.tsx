import { Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

interface MeteorConfig {
  radius: number;
  speed: number;
  phase: number;
  tilt: [number, number, number];
  color: string;
  size: number;
  tailLength: number;
}

const TAIL_LENGTH_MULTIPLIER = 10;
const ORBIT_SPEED_MULTIPLIER = 1.25;
const TRAIL_SEGMENTS = 56;

const METEORS: MeteorConfig[] = [
  {
    radius: 2.16,
    speed: 2.75,
    phase: 0.2,
    tilt: [0.3, 0.1, 0.18],
    color: "#38bdf8",
    size: 0.014,
    tailLength: 0.2,
  },
  {
    radius: 2.21,
    speed: -2.35,
    phase: 1.4,
    tilt: [1.02, -0.35, 0.25],
    color: "#60a5fa",
    size: 0.016,
    tailLength: 0.24,
  },
  {
    radius: 2.26,
    speed: 2.1,
    phase: 2.65,
    tilt: [-0.72, 0.58, -0.12],
    color: "#22d3ee",
    size: 0.012,
    tailLength: 0.18,
  },
  {
    radius: 2.31,
    speed: -2.8,
    phase: 3.7,
    tilt: [0.48, 1.12, 0.52],
    color: "#7dd3fc",
    size: 0.015,
    tailLength: 0.22,
  },
  {
    radius: 2.36,
    speed: 2.45,
    phase: 4.75,
    tilt: [-1.08, -0.42, 0.36],
    color: "#3b82f6",
    size: 0.017,
    tailLength: 0.26,
  },
  {
    radius: 2.41,
    speed: -1.95,
    phase: 5.6,
    tilt: [0.74, -1.06, -0.4],
    color: "#93c5fd",
    size: 0.013,
    tailLength: 0.19,
  },
  {
    radius: 2.19,
    speed: 2.2,
    phase: 0.85,
    tilt: [-0.22, 0.68, 0.48],
    color: "#0ea5e9",
    size: 0.012,
    tailLength: 0.17,
  },
  {
    radius: 2.24,
    speed: -2.65,
    phase: 1.95,
    tilt: [0.88, 0.32, -0.64],
    color: "#67e8f9",
    size: 0.014,
    tailLength: 0.21,
  },
  {
    radius: 2.29,
    speed: 2.95,
    phase: 3.15,
    tilt: [-0.56, -0.92, 0.2],
    color: "#2563eb",
    size: 0.013,
    tailLength: 0.2,
  },
  {
    radius: 2.34,
    speed: -2.15,
    phase: 4.25,
    tilt: [1.24, 0.72, 0.08],
    color: "#38bdf8",
    size: 0.016,
    tailLength: 0.23,
  },
  {
    radius: 2.39,
    speed: 2.55,
    phase: 5.25,
    tilt: [-1.3, 0.18, -0.34],
    color: "#818cf8",
    size: 0.012,
    tailLength: 0.18,
  },
  {
    radius: 2.17,
    speed: -3.05,
    phase: 0.45,
    tilt: [0.42, -0.68, 0.84],
    color: "#7dd3fc",
    size: 0.013,
    tailLength: 0.19,
  },
  {
    radius: 2.23,
    speed: 1.9,
    phase: 1.65,
    tilt: [-0.94, 1.18, 0.46],
    color: "#06b6d4",
    size: 0.015,
    tailLength: 0.22,
  },
  {
    radius: 2.3,
    speed: -2.45,
    phase: 2.9,
    tilt: [0.16, -1.28, -0.58],
    color: "#60a5fa",
    size: 0.012,
    tailLength: 0.18,
  },
  {
    radius: 2.37,
    speed: 2.7,
    phase: 4.05,
    tilt: [1.38, -0.24, 0.72],
    color: "#22d3ee",
    size: 0.014,
    tailLength: 0.21,
  },
  {
    radius: 2.43,
    speed: -2,
    phase: 5.85,
    tilt: [-0.38, 1.42, -0.76],
    color: "#93c5fd",
    size: 0.016,
    tailLength: 0.24,
  },
];

function RetrievalMeteorSearch() {
  const lightRef = useRef<THREE.PointLight>(null);

  useFrame(({ clock }) => {
    if (!lightRef.current) return;
    lightRef.current.intensity =
      0.55 + Math.sin(clock.elapsedTime * 5.5) * 0.16;
  });

  return (
    <group>
      <pointLight
        ref={lightRef}
        position={[2.8, 2.4, 3.8]}
        intensity={0.55}
        distance={8}
        color="#38bdf8"
      />
      {METEORS.map((meteor, index) => (
        <Meteor key={index} config={meteor} index={index} />
      ))}
    </group>
  );
}

interface MeteorProps {
  config: MeteorConfig;
  index: number;
}

function Meteor({ config, index }: MeteorProps) {
  const orbitRef = useRef<THREE.Group>(null);
  const headRef = useRef<THREE.Group>(null);
  const trail = useMemo(() => createCurvedTrail(config), [config]);

  useFrame(({ clock }) => {
    const elapsed = clock.elapsedTime;

    if (orbitRef.current) {
      orbitRef.current.rotation.z =
        config.phase + elapsed * config.speed * ORBIT_SPEED_MULTIPLIER;
    }

    if (headRef.current) {
      const pulse = 1 + Math.sin(elapsed * 8 + index * 1.7) * 0.14;
      headRef.current.scale.setScalar(config.size * pulse);
    }
  });

  return (
    <group rotation={config.tilt}>
      <mesh>
        <torusGeometry args={[config.radius, 0.002, 4, 128]} />
        <meshBasicMaterial
          color={config.color}
          transparent
          opacity={0.045}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </mesh>

      <group ref={orbitRef}>
        <group
          ref={headRef}
          position={[config.radius, 0, 0]}
          scale={config.size}
        >
          <mesh>
            <sphereGeometry args={[1, 16, 16]} />
            <meshBasicMaterial color="#effcff" toneMapped={false} />
          </mesh>
          <mesh scale={2.2}>
            <sphereGeometry args={[1, 14, 14]} />
            <meshBasicMaterial
              color={config.color}
              transparent
              opacity={0.2}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
              toneMapped={false}
            />
          </mesh>
        </group>

        <Line
          points={trail.points}
          vertexColors={trail.glowColors}
          lineWidth={2.2}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
        <Line
          points={trail.points}
          vertexColors={trail.coreColors}
          lineWidth={0.75}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </group>
    </group>
  );
}

type TrailColor = [number, number, number, number];

function createCurvedTrail(config: MeteorConfig) {
  const direction = Math.sign(config.speed);
  const tailLength = config.tailLength * TAIL_LENGTH_MULTIPLIER;
  const arcAngle = tailLength / config.radius;
  const color = new THREE.Color(config.color);
  const coreColor = new THREE.Color("#e8faff");
  const points: [number, number, number][] = [];
  const glowColors: TrailColor[] = [];
  const coreColors: TrailColor[] = [];

  for (let index = 0; index <= TRAIL_SEGMENTS; index += 1) {
    const progress = index / TRAIL_SEGMENTS;
    const angle = -direction * arcAngle * (1 - progress);
    const glowAlpha = Math.pow(progress, 1.45) * 0.62;
    const coreAlpha = Math.pow(progress, 2.1) * 0.9;
    const brightenedColor = color
      .clone()
      .lerp(coreColor, Math.pow(progress, 3) * 0.82);

    points.push([
      Math.cos(angle) * config.radius,
      Math.sin(angle) * config.radius,
      0,
    ]);
    glowColors.push([color.r, color.g, color.b, glowAlpha]);
    coreColors.push([
      brightenedColor.r,
      brightenedColor.g,
      brightenedColor.b,
      coreAlpha,
    ]);
  }

  return { points, glowColors, coreColors };
}

export default RetrievalMeteorSearch;
