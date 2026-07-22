import { useRef, useMemo } from "react";
import * as THREE from "three";
import { useTexture } from "@react-three/drei";
import { useThree, useFrame } from "@react-three/fiber";

interface MoonProps {
  targetCoords?: { lat: number; lng: number } | null;
}

function Moon({ targetCoords }: MoonProps) {
  const moonRef = useRef<THREE.Mesh>(null);
  const { gl } = useThree();

  const [colorMap, normalMap, displacementMap] = useTexture([
    "/lroc_color_16bit_srgb_8k.webp",
    "/ldem_16_uint_normal.png",
    "/ldem_16_uint.webp",
  ]);

  const maxAnisotropy = gl.capabilities.getMaxAnisotropy();

  useMemo(() => {
    for (const tex of [colorMap, normalMap, displacementMap]) {
      tex.anisotropy = maxAnisotropy;
      // Clamp wrapping avoids seam artefacts at the antimeridian
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.needsUpdate = true;
    }
  }, [colorMap, normalMap, displacementMap, maxAnisotropy]);

  // Slow idle axial drift when no target is being tracked
  useFrame(({ clock }) => {
    if (!moonRef.current) return;
    // Extremely subtle wobble — feels alive without being distracting
    moonRef.current.rotation.y = clock.getElapsedTime() * 0.004;
  });

  return (
    <group>
      {/* ── Main lunar sphere ── */}
      <mesh ref={moonRef} position={[0, 0, 0]}>
        {/*
          96 segments give a smoother silhouette vs the default 64,
          especially visible against the star field at the limb.
        */}
        <sphereGeometry args={[2, 96, 96]} />
        <meshStandardMaterial
          map={colorMap}
          normalMap={normalMap}
          normalScale={new THREE.Vector2(1.2, 1.2)}
          displacementMap={displacementMap}
          displacementScale={0.015}
          roughness={0.98}
          metalness={0.0}
          // Slight color tint — pulls the textures toward the amber palette
          color={new THREE.Color(0xddd5be)}
        />
      </mesh>

      {/* ── Terminator atmosphere rim ── */}
      {/*
        A slightly larger, inverted, fully-transparent sphere with additive
        blending creates a thin scattering halo around the moon limb
        without any actual atmosphere — pure optical trick.
      */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[2.018, 64, 64]} />
        <meshStandardMaterial
          color={new THREE.Color(0x8899bb)}
          transparent
          opacity={0.045}
          side={THREE.BackSide}
          roughness={1}
          metalness={0}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

export default Moon;
