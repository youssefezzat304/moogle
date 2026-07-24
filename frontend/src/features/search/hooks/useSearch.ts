import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  retrieveLunarPatches,
  RetrievalApiError,
  type RetrievalResult,
} from "../../retrieval/api";
import type {
  SearchController,
  SearchMessage,
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
  messages: SearchMessage[];
}

const INITIAL_STATE: SearchState = {
  phase: "idle",
  query: null,
  results: [],
  selectedResultId: null,
  error: null,
  metadata: {},
  messages: [],
};

const requestRetrieval: SearchRequest = (query, signal) =>
  retrieveLunarPatches(query, { signal });

export function useSearch(
  searchRequest: SearchRequest = requestRetrieval,
): SearchController {
  const [state, setState] = useState<SearchState>(INITIAL_STATE);
  const [queryCount, setQueryCount] = useState(0);
  const requestRef = useRef<AbortController | null>(null);
  const messageIdRef = useRef(0);

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
      const userMessage = createMessage(messageIdRef, "user", query);
      setState((current) => ({
        ...current,
        phase: "loading",
        query,
        results: [],
        selectedResultId: null,
        error: null,
        metadata: {},
        messages: [...current.messages, userMessage],
      }));

      try {
        const response = await searchRequest(query, request.signal);
        if (requestRef.current !== request) return false;

        requestRef.current = null;
        const phase = response.results.length > 0 ? "success" : "empty";
        const assistantMessage = createMessage(
          messageIdRef,
          "assistant",
          response.results.length > 0
            ? `Retrieved ${response.results.length} ranked lunar ${
                response.results.length === 1 ? "patch" : "patches"
              } for “${response.query}”.`
            : `No lunar patches were returned for “${response.query}”.`,
        );
        setState((current) => ({
          ...current,
          phase,
          query: response.query,
          results: response.results,
          selectedResultId: response.results[0]?.id ?? null,
          error: null,
          metadata: {
            modelId: response.modelId,
            indexSize: response.indexSize,
            elapsedMs: response.elapsedMs,
          },
          messages: [...current.messages, assistantMessage],
        }));
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
        const message = errorMessage(requestError);
        setState((current) => ({
          ...current,
          phase: "error",
          query,
          results: [],
          selectedResultId: null,
          error: message,
          metadata: {},
          messages: [
            ...current.messages,
            createMessage(
              messageIdRef,
              "assistant",
              `Retrieval failed: ${message}`,
              "error",
            ),
          ],
        }));
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
    messages: state.messages,
    queryCount,
    runSearch,
    selectResult,
  };
}

function createMessage(
  idRef: { current: number },
  role: SearchMessage["role"],
  content: string,
  tone: SearchMessage["tone"] = "default",
): SearchMessage {
  idRef.current += 1;
  return {
    id: `message-${idRef.current}`,
    role,
    content,
    tone,
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
