import { ListTree, X } from "lucide-react";
import { useEffect, useState } from "react";
import {
  parseSearchFeatures,
  type SearchFeature,
} from "../utils/searchFeatures";
import {
  pickFeatureDemoQuery,
  type DemoQueryCatalog,
} from "../utils/demoQueries";

const LEGEND_URL = `${import.meta.env.BASE_URL}legend.json`;

interface SearchFeatureLegendProps {
  demoQueryCatalog: DemoQueryCatalog | null;
  demoQueryError: string | null;
  canRunDemoQuery: boolean;
  onRunDemoQuery: (query: string) => void;
}

function SearchFeatureLegend({
  demoQueryCatalog,
  demoQueryError,
  canRunDemoQuery,
  onRunDemoQuery,
}: SearchFeatureLegendProps) {
  const [features, setFeatures] = useState<SearchFeature[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadFeatures() {
      try {
        const response = await fetch(LEGEND_URL, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Legend request failed with ${response.status}.`);
        }
        setFeatures(parseSearchFeatures(await response.json()));
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(
            cause instanceof Error
              ? cause.message
              : "The feature list could not be loaded.",
          );
        }
      }
    }

    void loadFeatures();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }

    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [isOpen]);

  return (
    <div className="search-feature-legend">
      {isOpen && (
        <section
          id="search-feature-list"
          className="search-feature-popover"
          role="region"
          aria-label="Searchable lunar features"
        >
          <header>
            <div>
              <span className="eyebrow">Geologic legend</span>
              <strong>Mapped lunar features</strong>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              aria-label="Close searchable features"
            >
              <X size={15} />
            </button>
          </header>

          <p>
            Select a mapped unit to run one of its real v2.0 example
            descriptions.
          </p>

          {error ? (
            <div className="search-feature-message error">{error}</div>
          ) : features.length === 0 ? (
            <div className="search-feature-message">Loading feature list…</div>
          ) : (
            <ul className="search-feature-list">
              {features.map((feature) => {
                const hasDemoQuery =
                  (demoQueryCatalog?.features[feature.code]?.length ?? 0) > 0;
                const isDisabled = !canRunDemoQuery || !hasDemoQuery;

                return (
                  <li key={feature.code}>
                    <button
                      type="button"
                      disabled={isDisabled}
                      title={
                        hasDemoQuery
                          ? `Run a v2.0 example for ${feature.longDescription}`
                          : `No v2.0 example is available for ${feature.longDescription}`
                      }
                      onClick={() => {
                        const query = pickFeatureDemoQuery(
                          demoQueryCatalog,
                          feature.code,
                        );
                        if (!query) return;
                        setIsOpen(false);
                        onRunDemoQuery(query);
                      }}
                    >
                      <span
                        className="search-feature-swatch"
                        style={{ backgroundColor: feature.color }}
                        aria-hidden="true"
                      />
                      <span>
                        <strong>{feature.longDescription}</strong>
                        <small>
                          {feature.code} · {feature.description}
                          {!hasDemoQuery && demoQueryCatalog && (
                            <> · No v2.0 example</>
                          )}
                        </small>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          {demoQueryError && (
            <div className="search-feature-message error">
              Demo descriptions could not be loaded.
            </div>
          )}
        </section>
      )}

      <button
        type="button"
        className="search-feature-trigger"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
        aria-controls="search-feature-list"
      >
        <ListTree size={15} />
        <span>Search vocabulary</span>
      </button>
    </div>
  );
}

export default SearchFeatureLegend;
