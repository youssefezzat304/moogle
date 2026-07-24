import type { RetrievalResponse, RetrievalResult } from "../retrieval/api";

export type SearchPhase = "idle" | "loading" | "success" | "empty" | "error";

export interface SearchMetadata {
  modelId?: string;
  indexSize?: number;
  elapsedMs?: number;
}

export interface SearchMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  tone?: "default" | "error";
}

export type SearchRequest = (
  query: string,
  signal: AbortSignal,
  topK: number,
) => Promise<RetrievalResponse>;

export interface SearchController {
  phase: SearchPhase;
  query: string | null;
  results: RetrievalResult[];
  activeResult: RetrievalResult | null;
  error: string | null;
  metadata: SearchMetadata;
  messages: SearchMessage[];
  runSearch: (query: string, topK: number) => Promise<boolean>;
  selectResult: (result: RetrievalResult) => void;
}
