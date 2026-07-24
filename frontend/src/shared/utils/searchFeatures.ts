export interface SearchFeature {
  code: string;
  color: string;
  description: string;
  longDescription: string;
}

export function parseSearchFeatures(value: unknown): SearchFeature[] {
  if (!isRecord(value)) {
    throw new Error("Legend must be an object.");
  }

  return Object.entries(value).map(([code, entry]) => {
    if (
      !isRecord(entry) ||
      !isColor(entry.color) ||
      !isNonEmptyString(entry.description) ||
      !isNonEmptyString(entry.long_description)
    ) {
      throw new Error(`Legend entry ${code} is invalid.`);
    }

    return {
      code,
      color: entry.color,
      description: entry.description,
      longDescription: entry.long_description,
    };
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isColor(value: unknown): value is string {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value.trim());
}
