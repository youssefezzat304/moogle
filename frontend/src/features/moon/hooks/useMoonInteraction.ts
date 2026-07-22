import { useCallback, useEffect, useRef, useState } from "react";

const INTERACTION_SETTLE_DELAY_MS = 2400;

export function useMoonInteraction() {
  const [hasWandered, setHasWandered] = useState(false);
  const [recenterNonce, setRecenterNonce] = useState(0);
  const [userInteracting, setUserInteracting] = useState(false);
  const interactionTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearInteractionTimeout = useCallback(() => {
    if (interactionTimeout.current) {
      clearTimeout(interactionTimeout.current);
      interactionTimeout.current = null;
    }
  }, []);

  useEffect(() => clearInteractionTimeout, [clearInteractionTimeout]);

  const startInteraction = useCallback(() => {
    setHasWandered(true);
    setUserInteracting(true);
    clearInteractionTimeout();
  }, [clearInteractionTimeout]);

  const settleInteraction = useCallback(() => {
    clearInteractionTimeout();
    interactionTimeout.current = setTimeout(
      () => setUserInteracting(false),
      INTERACTION_SETTLE_DELAY_MS,
    );
  }, [clearInteractionTimeout]);

  const recenterTarget = useCallback(() => {
    setHasWandered(false);
    setUserInteracting(false);
    clearInteractionTimeout();
    setRecenterNonce((nonce) => nonce + 1);
  }, [clearInteractionTimeout]);

  return {
    hasWandered,
    recenterNonce,
    userInteracting,
    startInteraction,
    settleInteraction,
    recenterTarget,
  };
}
