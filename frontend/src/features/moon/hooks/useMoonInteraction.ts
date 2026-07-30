import { useCallback, useRef, useState } from "react";

export function useMoonInteraction() {
  const [hasWandered, setHasWandered] = useState(false);
  const [recenterNonce, setRecenterNonce] = useState(0);
  const interactionNonceRef = useRef(0);

  const startInteraction = useCallback(() => {
    interactionNonceRef.current += 1;
    setHasWandered(true);
  }, []);

  const recenterTarget = useCallback(() => {
    setHasWandered(false);
    setRecenterNonce((nonce) => nonce + 1);
  }, []);

  return {
    hasWandered,
    recenterNonce,
    interactionNonceRef,
    startInteraction,
    recenterTarget,
  };
}
