const DEFAULT_API_BASE_URL = "/api";
const DEFAULT_RESULT_LIMIT = 5;
const MAX_QUERY_LENGTH = 500;
const MAX_RESULT_LIMIT = 10;

interface RetrievalApiResult {
  rank: number;
  patch_id: number;
  similarity: number;
  description: string;
  source_version: string;
  prompt_style: string;
  image_url: string;
  latitude: number;
  longitude: number;
}

interface RetrievalApiResponse {
  schema_version: 1;
  query: string;
  model_id: string;
  index_size: number;
  elapsed_ms: number;
  results: RetrievalApiResult[];
}

interface RetrievalApiErrorResponse {
  error: {
    code: string;
    message: string;
    request_id: string;
  };
}

export interface RetrievalResult {
  id: string;
  rank: number;
  patchId: number;
  imageUrl: string;
  lat: number;
  lng: number;
  similarity: number;
  description: string;
  sourceVersion: string;
  promptStyle: string;
}

export interface RetrievalResponse {
  schemaVersion: 1;
  query: string;
  modelId: string;
  indexSize: number;
  elapsedMs: number;
  results: RetrievalResult[];
}

export class RetrievalApiError extends Error {
  readonly status?: number;
  readonly code?: string;
  readonly requestId?: string;

  constructor(
    message: string,
    options: { status?: number; code?: string; requestId?: string } = {},
  ) {
    super(message);
    this.name = "RetrievalApiError";
    this.status = options.status;
    this.code = options.code;
    this.requestId = options.requestId;
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
  if (normalizedQuery.length > MAX_QUERY_LENGTH) {
    throw new RetrievalApiError(
      `Retrieval queries cannot exceed ${MAX_QUERY_LENGTH} characters.`,
    );
  }

  const limit = options.limit ?? DEFAULT_RESULT_LIMIT;
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_RESULT_LIMIT) {
    throw new RetrievalApiError(
      `The result limit must be an integer from 1 to ${MAX_RESULT_LIMIT}.`,
    );
  }

  const response = await fetch(`${apiBaseUrl}/retrieval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: normalizedQuery,
      top_k: limit,
    }),
    signal: options.signal,
  });

  if (!response.ok) {
    const error = await responseError(response);
    throw new RetrievalApiError(error.message, {
      status: response.status,
      code: error.code,
      requestId: error.requestId,
    });
  }

  const parsed = parseRetrievalResponse(await response.json());
  return {
    ...parsed,
    results: parsed.results.map((result) => ({
      ...result,
      imageUrl: resolveImageUrl(result.imageUrl),
    })),
  };
}

export function parseRetrievalResponse(value: unknown): RetrievalResponse {
  if (!isRetrievalApiResponse(value)) {
    throw new RetrievalApiError(
      "The retrieval service returned an invalid response.",
    );
  }

  return {
    schemaVersion: value.schema_version,
    query: value.query,
    modelId: value.model_id,
    indexSize: value.index_size,
    elapsedMs: value.elapsed_ms,
    results: value.results.map(toRetrievalResult),
  };
}

export function formatCoords(lat: number, lng: number) {
  const latDir = lat >= 0 ? "N" : "S";
  const lngDir = lng >= 0 ? "E" : "W";

  return `${Math.abs(lat).toFixed(2)} deg ${latDir} / ${Math.abs(lng).toFixed(
    2,
  )} deg ${lngDir}`;
}

export function formatSimilarity(similarity: number) {
  return similarity.toFixed(3);
}

export function resultLabel(result: RetrievalResult) {
  return `Patch ${result.patchId}`;
}

function toRetrievalResult(result: RetrievalApiResult): RetrievalResult {
  return {
    id: String(result.patch_id),
    rank: result.rank,
    patchId: result.patch_id,
    imageUrl: result.image_url,
    lat: result.latitude,
    lng: result.longitude,
    similarity: result.similarity,
    description: result.description,
    sourceVersion: result.source_version,
    promptStyle: result.prompt_style,
  };
}

function resolveImageUrl(imageUrl: string) {
  if (/^https?:/.test(imageUrl)) return imageUrl;

  const base = new URL(`${apiBaseUrl}/`, window.location.origin);
  return new URL(imageUrl, base).toString();
}

function isRetrievalApiResponse(value: unknown): value is RetrievalApiResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schema_version",
      "query",
      "model_id",
      "index_size",
      "elapsed_ms",
      "results",
    ]) ||
    value.schema_version !== 1 ||
    !isTrimmedString(value.query, MAX_QUERY_LENGTH) ||
    !isTrimmedString(value.model_id) ||
    !isNonNegativeInteger(value.index_size) ||
    !isNonNegativeInteger(value.elapsed_ms) ||
    !Array.isArray(value.results) ||
    value.results.length > MAX_RESULT_LIMIT ||
    !value.results.every(isRetrievalApiResult)
  ) {
    return false;
  }

  const patchIds = new Set<number>();
  return value.results.every((result, index, results) => {
    if (result.rank !== index + 1 || patchIds.has(result.patch_id))
      return false;
    patchIds.add(result.patch_id);
    return index === 0 || results[index - 1].similarity >= result.similarity;
  });
}

function isRetrievalApiResult(value: unknown): value is RetrievalApiResult {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "rank",
      "patch_id",
      "similarity",
      "description",
      "source_version",
      "prompt_style",
      "image_url",
      "latitude",
      "longitude",
    ]) &&
    isInteger(value.rank) &&
    value.rank >= 1 &&
    isInteger(value.patch_id) &&
    isFiniteNumber(value.similarity) &&
    isTrimmedString(value.description) &&
    isTrimmedString(value.source_version) &&
    isTrimmedString(value.prompt_style) &&
    typeof value.image_url === "string" &&
    /^(?:\/|https?:\/\/)/.test(value.image_url) &&
    isFiniteNumber(value.latitude) &&
    value.latitude >= -90 &&
    value.latitude <= 90 &&
    isFiniteNumber(value.longitude) &&
    value.longitude >= -180 &&
    value.longitude < 180
  );
}

function isRetrievalApiErrorResponse(
  value: unknown,
): value is RetrievalApiErrorResponse {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["error"]) &&
    isRecord(value.error) &&
    hasExactKeys(value.error, ["code", "message", "request_id"]) &&
    typeof value.error.code === "string" &&
    /^[A-Z][A-Z0-9_]*$/.test(value.error.code) &&
    isTrimmedString(value.error.message) &&
    isTrimmedString(value.error.request_id)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return isInteger(value) && value >= 0;
}

function isTrimmedString(value: unknown, maxLength?: number): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value === value.trim() &&
    (maxLength === undefined || value.length <= maxLength)
  );
}

function hasExactKeys(
  value: Record<string, unknown>,
  expectedKeys: readonly string[],
) {
  const keys = Object.keys(value);
  return (
    keys.length === expectedKeys.length &&
    keys.every((key) => expectedKeys.includes(key))
  );
}

async function responseError(response: Response) {
  try {
    const payload: unknown = await response.json();
    if (isRetrievalApiErrorResponse(payload)) {
      return {
        message: payload.error.message,
        code: payload.error.code,
        requestId: payload.error.request_id,
      };
    }
  } catch {
    // The status text below is the best available error for a non-JSON body.
  }

  return {
    message:
      response.statusText ||
      `Retrieval request failed with status ${response.status}.`,
  };
}
