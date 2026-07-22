import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  retrieveLunarPatches,
  RetrievalApiError,
  type RetrievalResult,
} from "../../retrieval/api";
import type {
  SearchController,
  SearchMetadata,
  SearchPhase,
  SearchRequest,
} from "../types";

interface SearchState {
  phase: SearchPhase;
  query: string | null;
  results: RetrievalResult[];
  selectedResultId: string | null;
  error: string | null;
  metadata: SearchMetadata;
}

const INITIAL_STATE: SearchState = {
  phase: "idle",
  query: null,
  results: [],
  selectedResultId: null,
  error: null,
  metadata: {},
};

const requestRetrieval: SearchRequest = (query, signal) =>
  retrieveLunarPatches(query, { signal });

export function useSearch(
  searchRequest: SearchRequest = requestRetrieval,
): SearchController {
  const [state, setState] = useState<SearchState>(INITIAL_STATE);
  const [queryCount, setQueryCount] = useState(0);
  const requestRef = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      requestRef.current?.abort();
    },
    [],
  );

  const runSearch = useCallback(
    async (rawQuery: string) => {
      const query = rawQuery.trim();
      if (!query) return false;

      requestRef.current?.abort();
      const request = new AbortController();
      requestRef.current = request;
      setState({
        phase: "loading",
        query,
        results: [],
        selectedResultId: null,
        error: null,
        metadata: {},
      });

      try {
        const response = await searchRequest(query, request.signal);
        if (requestRef.current !== request) return false;

        requestRef.current = null;
        setState({
          phase: response.results.length > 0 ? "success" : "empty",
          query: response.query,
          results: response.results,
          selectedResultId: response.results[0]?.id ?? null,
          error: null,
          metadata: {
            modelId: response.modelId,
            indexSize: response.indexSize,
          },
        });
        setQueryCount((count) => count + 1);
        return true;
      } catch (requestError) {
        if (
          requestError instanceof DOMException &&
          requestError.name === "AbortError"
        ) {
          return false;
        }
        if (requestRef.current !== request) return false;

        requestRef.current = null;
        setState({
          phase: "error",
          query,
          results: [],
          selectedResultId: null,
          error: errorMessage(requestError),
          metadata: {},
        });
        return false;
      }
    },
    [searchRequest],
  );

  const selectResult = useCallback((result: RetrievalResult) => {
    setState((current) =>
      current.results.some((candidate) => candidate.id === result.id)
        ? { ...current, selectedResultId: result.id }
        : current,
    );
  }, []);

  const activeResult = useMemo(
    () =>
      state.results.find((result) => result.id === state.selectedResultId) ??
      null,
    [state.results, state.selectedResultId],
  );

  return {
    phase: state.phase,
    query: state.query,
    results: state.results,
    activeResult,
    error: state.error,
    metadata: state.metadata,
    queryCount,
    runSearch,
    selectResult,
  };
}

function errorMessage(error: unknown) {
  if (error instanceof RetrievalApiError) return error.message;
  if (error instanceof TypeError) {
    return "The configured API could not be reached.";
  }
  if (error instanceof Error) return error.message;
  return "The retrieval request failed.";
}
