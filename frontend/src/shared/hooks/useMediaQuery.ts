"use client";

import { useState, useEffect } from "react";

const useMediaQuery = (query: string) => {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mediaQueryList = window.matchMedia(query)

    const mediaQueryHandler = () => {
      setMatches(mediaQueryList.matches)
    }

    mediaQueryList.addEventListener('change', mediaQueryHandler)

    return () => {
      mediaQueryList.removeEventListener('change', mediaQueryHandler)
    }
  }, [query]);

  return matches;
};

export default useMediaQuery;
