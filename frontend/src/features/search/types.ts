import type { RetrievalResponse, RetrievalResult } from "../retrieval/api";

export type SearchPhase = "idle" | "loading" | "success" | "empty" | "error";

export interface SearchMetadata {
  modelId?: string;
  indexSize?: number;
}

export type SearchRequest = (
  query: string,
  signal: AbortSignal,
) => Promise<RetrievalResponse>;

export interface SearchController {
  phase: SearchPhase;
  query: string | null;
  results: RetrievalResult[];
  activeResult: RetrievalResult | null;
  error: string | null;
  metadata: SearchMetadata;
  queryCount: number;
  runSearch: (query: string) => Promise<boolean>;
  selectResult: (result: RetrievalResult) => void;
}
