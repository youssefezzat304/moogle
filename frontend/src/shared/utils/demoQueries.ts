const MAX_QUERY_LENGTH = 500;

export interface DemoQuery {
  patchId: number;
  query: string;
}

export interface DemoQueryCatalog {
  sourceVersion: string;
  promptStyle: string;
  features: Record<string, DemoQuery[]>;
}

export function parseDemoQueryCatalog(value: unknown): DemoQueryCatalog {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.source_version) ||
    !isNonEmptyString(value.prompt_style) ||
    !isRecord(value.features)
  ) {
    throw new Error("Demo query catalog is invalid.");
  }

  const features = Object.fromEntries(
    Object.entries(value.features).map(([code, queries]) => {
      if (
        !isNonEmptyString(code) ||
        !Array.isArray(queries) ||
        !queries.every(isDemoQuery)
      ) {
        throw new Error(`Demo queries for feature ${code} are invalid.`);
      }

      return [
        code,
        queries.map((query) => ({
          patchId: query.patch_id,
          query: query.query,
        })),
      ];
    }),
  );

  return {
    sourceVersion: value.source_version,
    promptStyle: value.prompt_style,
    features,
  };
}

export function pickFeatureDemoQuery(
  catalog: DemoQueryCatalog | null,
  featureCode: string,
  random: () => number = Math.random,
): string | null {
  const queries = catalog?.features[featureCode] ?? [];
  if (queries.length === 0) return null;

  return queries[randomIndex(queries.length, random)]?.query ?? null;
}

export function pickRandomDemoSuggestions(
  catalog: DemoQueryCatalog | null,
  count: number,
  random: () => number = Math.random,
): string[] {
  if (!catalog || count <= 0) return [];

  const featureQueries = Object.values(catalog.features).filter(
    (queries) => queries.length > 0,
  );

  for (let index = featureQueries.length - 1; index > 0; index -= 1) {
    const swapIndex = randomIndex(index + 1, random);
    [featureQueries[index], featureQueries[swapIndex]] = [
      featureQueries[swapIndex],
      featureQueries[index],
    ];
  }

  return featureQueries.slice(0, count).flatMap((queries) => {
    const query = queries[randomIndex(queries.length, random)];
    return query ? [query.query] : [];
  });
}

function isDemoQuery(value: unknown): value is {
  patch_id: number;
  query: string;
} {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["patch_id", "query"]) &&
    Number.isInteger(value.patch_id) &&
    Number(value.patch_id) >= 0 &&
    isNonEmptyString(value.query) &&
    value.query.length <= MAX_QUERY_LENGTH &&
    value.query === value.query.trim()
  );
}

function randomIndex(length: number, random: () => number) {
  const value = random();
  const normalized = Number.isFinite(value)
    ? Math.min(Math.max(value, 0), 1 - Number.EPSILON)
    : 0;
  return Math.floor(normalized * length);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function hasExactKeys(value: Record<string, unknown>, keys: string[]) {
  const actualKeys = Object.keys(value).sort();
  const expectedKeys = [...keys].sort();
  return (
    actualKeys.length === expectedKeys.length &&
    actualKeys.every((key, index) => key === expectedKeys[index])
  );
}
