import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useTexture } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";

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

  const [configuredColorMap, configuredNormalMap, configuredDisplacementMap] =
    useMemo(() => {
      const configureTexture = (
        source: THREE.Texture,
        colorSpace: THREE.ColorSpace,
      ) => {
        const tex = source.clone();
        tex.colorSpace = colorSpace;
        tex.anisotropy = maxAnisotropy;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.needsUpdate = true;
        return tex;
      };

      return [
        configureTexture(colorMap, THREE.SRGBColorSpace),
        configureTexture(normalMap, THREE.NoColorSpace),
        configureTexture(displacementMap, THREE.NoColorSpace),
      ];
    }, [colorMap, normalMap, displacementMap, maxAnisotropy]);

  useFrame(({ clock }) => {
    if (!moonRef.current) return;
    if (!targetCoords) {
      moonRef.current.rotation.y = clock.getElapsedTime() * 0.004;
    }
  });

  return (
    <group>
      <mesh ref={moonRef} position={[0, 0, 0]}>
        <sphereGeometry args={[2, 128, 128]} />
        <meshStandardMaterial
          map={configuredColorMap}
          normalMap={configuredNormalMap}
          normalScale={new THREE.Vector2(1.35, 1.35)}
          displacementMap={configuredDisplacementMap}
          displacementScale={0.018}
          roughness={0.92}
          metalness={0}
          color="#ffffff"
        />
      </mesh>

      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[2.018, 64, 64]} />
        <meshStandardMaterial
          color="#8fb9d8"
          transparent
          opacity={0.04}
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
