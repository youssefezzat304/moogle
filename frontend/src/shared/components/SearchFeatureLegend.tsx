import { ListTree, X } from "lucide-react";
import { useEffect, useState } from "react";
import {
  parseSearchFeatures,
  type SearchFeature,
} from "../utils/searchFeatures";

const LEGEND_URL = `${import.meta.env.BASE_URL}legend.json`;

function SearchFeatureLegend() {
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

          <p>Use these mapped unit names as terms in your terrain query.</p>

          {error ? (
            <div className="search-feature-message error">{error}</div>
          ) : features.length === 0 ? (
            <div className="search-feature-message">Loading feature list…</div>
          ) : (
            <ul className="search-feature-list">
              {features.map((feature) => (
                <li key={feature.code}>
                  <span
                    className="search-feature-swatch"
                    style={{ backgroundColor: feature.color }}
                    aria-hidden="true"
                  />
                  <span>
                    <strong>{feature.longDescription}</strong>
                    <small>
                      {feature.code} · {feature.description}
                    </small>
                  </span>
                </li>
              ))}
            </ul>
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
        {features.length > 0 && <strong>{features.length}</strong>}
      </button>
    </div>
  );
}

export default SearchFeatureLegend;
