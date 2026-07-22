const DEFAULT_API_BASE_URL = "/api";
const DEFAULT_RESULT_LIMIT = 6;

interface RetrievalApiResult {
  patch_id: number | string;
  image_url: string;
  latitude: number;
  longitude: number;
  similarity: number;
  description?: string;
  source_version?: string;
  prompt_style?: string;
}

interface RetrievalApiResponse {
  query?: string;
  model_id?: string;
  index_size?: number;
  results: RetrievalApiResult[];
}

export interface RetrievalResult {
  id: string;
  patchId: number | string;
  imageUrl: string;
  lat: number;
  lng: number;
  similarity: number;
  description?: string;
  sourceVersion?: string;
  promptStyle?: string;
}

export interface RetrievalResponse {
  query: string;
  modelId?: string;
  indexSize?: number;
  results: RetrievalResult[];
}

export class RetrievalApiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "RetrievalApiError";
    this.status = status;
  }
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL)
  .trim()
  .replace(/\/$/, "");

export async function retrieveLunarPatches(
  query: string,
  options: { signal?: AbortSignal; limit?: number } = {},
): Promise<RetrievalResponse> {
  const normalizedQuery = query.trim();
  if (!normalizedQuery) {
    throw new RetrievalApiError("Enter a retrieval query.");
  }

  const response = await fetch(`${apiBaseUrl}/retrieval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: normalizedQuery,
      top_k: options.limit ?? DEFAULT_RESULT_LIMIT,
    }),
    signal: options.signal,
  });

  if (!response.ok) {
    throw new RetrievalApiError(
      await responseErrorMessage(response),
      response.status,
    );
  }

  const payload: unknown = await response.json();
  if (!isRetrievalApiResponse(payload)) {
    throw new RetrievalApiError(
      "The retrieval service returned an invalid response.",
    );
  }

  return {
    query: payload.query?.trim() || normalizedQuery,
    modelId: payload.model_id,
    indexSize: payload.index_size,
    results: payload.results.map(toRetrievalResult),
  };
}

export function formatCoords(lat: number, lng: number) {
  const latDir = lat >= 0 ? "N" : "S";
  const lngDir = lng >= 0 ? "E" : "W";

  return `${Math.abs(lat).toFixed(2)} deg ${latDir} / ${Math.abs(lng).toFixed(
    2,
  )} deg ${lngDir}`;
}

export function resultLabel(result: RetrievalResult) {
  return `Patch ${result.patchId}`;
}

function toRetrievalResult(result: RetrievalApiResult): RetrievalResult {
  return {
    id: String(result.patch_id),
    patchId: result.patch_id,
    imageUrl: resolveImageUrl(result.image_url),
    lat: result.latitude,
    lng: result.longitude,
    similarity: result.similarity,
    description: result.description,
    sourceVersion: result.source_version,
    promptStyle: result.prompt_style,
  };
}

function resolveImageUrl(imageUrl: string) {
  if (/^(?:https?:|data:|blob:)/.test(imageUrl)) return imageUrl;

  const base = new URL(`${apiBaseUrl}/`, window.location.origin);
  return new URL(imageUrl, base).toString();
}

function isRetrievalApiResponse(value: unknown): value is RetrievalApiResponse {
  if (!isRecord(value) || !Array.isArray(value.results)) return false;
  if (value.query !== undefined && typeof value.query !== "string")
    return false;
  if (value.model_id !== undefined && typeof value.model_id !== "string") {
    return false;
  }
  if (value.index_size !== undefined && !isFiniteNumber(value.index_size)) {
    return false;
  }
  return value.results.every(isRetrievalApiResult);
}

function isRetrievalApiResult(value: unknown): value is RetrievalApiResult {
  return (
    isRecord(value) &&
    (typeof value.patch_id === "string" || isFiniteNumber(value.patch_id)) &&
    typeof value.image_url === "string" &&
    value.image_url.length > 0 &&
    isFiniteNumber(value.latitude) &&
    value.latitude >= -90 &&
    value.latitude <= 90 &&
    isFiniteNumber(value.longitude) &&
    value.longitude >= -180 &&
    value.longitude <= 180 &&
    isFiniteNumber(value.similarity) &&
    optionalString(value.description) &&
    optionalString(value.source_version) &&
    optionalString(value.prompt_style)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function optionalString(value: unknown) {
  return value === undefined || typeof value === "string";
}

async function responseErrorMessage(response: Response) {
  try {
    const payload: unknown = await response.json();
    if (isRecord(payload)) {
      const message = payload.message ?? payload.detail;
      if (typeof message === "string" && message.trim()) return message;
    }
  } catch {
    // The status text below is the best available error when the body is not JSON.
  }

  return (
    response.statusText || `Retrieval request failed (${response.status}).`
  );
}
