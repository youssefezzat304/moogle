import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";

function MoonFallback() {
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

export default MoonFallback;
