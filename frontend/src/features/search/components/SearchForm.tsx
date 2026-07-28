import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  ChevronDown,
  Loader2,
  Search,
  Send,
  Settings2,
  Sparkles,
} from "lucide-react";
import type { DemoQueryRequest } from "../types";

interface SearchFormProps {
  isSubmitting: boolean;
  topK: number;
  onTopKChange: (topK: number) => void;
  onSubmit: (query: string, topK: number) => Promise<boolean>;
  suggestions?: string[];
  suggestionError?: string | null;
  demoQueryRequest?: DemoQueryRequest | null;
  onConsumeDemoQuery?: (requestId: number) => void;
  placement?: "initial" | "conversation";
}

function SearchForm({
  isSubmitting,
  topK,
  onTopKChange,
  onSubmit,
  suggestions = [],
  suggestionError = null,
  demoQueryRequest = null,
  onConsumeDemoQuery,
  placement = "initial",
}: SearchFormProps) {
  const [input, setInput] = useState(demoQueryRequest?.query ?? "");
  const settingsRef = useRef<HTMLDetailsElement>(null);
  const submittedDemoQueryIdRef = useRef<number | null>(null);

  const submitQuery = useCallback(
    async (rawQuery: string) => {
      const query = rawQuery.trim();
      if (!query || isSubmitting || topK === 0) return;

      settingsRef.current?.removeAttribute("open");
      if (await onSubmit(query, topK)) {
        setInput("");
      }
    },
    [isSubmitting, onSubmit, topK],
  );

  const submit = () => submitQuery(input);

  useEffect(() => {
    if (
      !demoQueryRequest ||
      !onConsumeDemoQuery ||
      submittedDemoQueryIdRef.current === demoQueryRequest.id
    ) {
      return;
    }

    submittedDemoQueryIdRef.current = demoQueryRequest.id;
    window.requestAnimationFrame(() => {
      void submitQuery(demoQueryRequest.query);
      onConsumeDemoQuery(demoQueryRequest.id);
    });
  }, [demoQueryRequest, onConsumeDemoQuery, submitQuery]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submit();
  };

  return (
    <section className={`search-form-section ${placement}`}>
      <div className="panel-section-title">
        <span>
          <Search size={13} />
          {placement === "initial" ? "Start a search" : "Next query"}
        </span>
        <strong>Semantic retrieval</strong>
      </div>

      <form className="composer-box" onSubmit={handleSubmit}>
        <textarea
          rows={placement === "initial" ? 4 : 2}
          value={input}
          maxLength={500}
          placeholder="Describe the lunar terrain to retrieve..."
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submit();
            }
          }}
          disabled={isSubmitting}
        />
        <div className="composer-actions">
          <details ref={settingsRef} className="retrieval-settings">
            <summary
              aria-label="Open retrieval settings"
              aria-disabled={isSubmitting}
              title={`Retrieval settings · ${topK} images · BPE-GEO`}
              onClick={(event) => {
                if (isSubmitting) event.preventDefault();
              }}
            >
              <Settings2 size={17} />
            </summary>

            <div className="retrieval-settings-popover">
              <header>
                <span>Retrieval settings</span>
                <strong>{topK} images</strong>
              </header>

              <label className="top-k-control">
                <span>
                  Top-k images
                  <strong>{topK}</strong>
                </span>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="1"
                  value={topK}
                  onChange={(event) => onTopKChange(Number(event.target.value))}
                  disabled={isSubmitting}
                  aria-valuetext={`${topK} retrieval images`}
                />
                <small>
                  <span>0</span>
                  <span>10</span>
                </small>
              </label>

              <label className="model-control">
                <span>Model</span>
                <span className="model-select">
                  <select
                    defaultValue="bpe_geo"
                    disabled={isSubmitting}
                    aria-label="Retrieval model"
                  >
                    <option value="bpe_geo">BPE-GEO</option>
                  </select>
                  <ChevronDown size={14} aria-hidden="true" />
                </span>
              </label>

              {topK === 0 && (
                <small className="retrieval-control-hint">
                  Select at least one image to run retrieval.
                </small>
              )}
            </div>
          </details>

          <button
            type="submit"
            className="send-button"
            disabled={!input.trim() || isSubmitting || topK === 0}
            aria-label="Run retrieval query"
            title="Run retrieval query"
          >
            {isSubmitting ? (
              <Loader2 size={16} className="spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>
      </form>

      {placement === "initial" && suggestions.length > 0 && (
        <div className="search-suggestions">
          <span>
            <Sparkles size={12} />
            Try an example
          </span>
          <div className="search-suggestion-list">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                title={suggestion}
                disabled={isSubmitting || topK === 0}
                onClick={() => {
                  setInput(suggestion);
                  window.requestAnimationFrame(() => {
                    void submitQuery(suggestion);
                  });
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {placement === "initial" && suggestionError && (
        <small className="search-suggestion-error">
          Example queries are temporarily unavailable.
        </small>
      )}
    </section>
  );
}

export default SearchForm;
