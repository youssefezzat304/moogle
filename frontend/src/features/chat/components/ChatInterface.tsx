import { useEffect, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Image as ImageIcon, Loader2, Search, Send } from "lucide-react";
import {
  formatCoords,
  resultLabel,
  RetrievalApiError,
  type RetrievalResponse,
  type RetrievalResult,
} from "../../retrieval/api";

interface ChatInterfaceProps {
  activeResult: RetrievalResult | null;
  results: RetrievalResult[];
  onQuery: (query: string, signal: AbortSignal) => Promise<RetrievalResponse>;
  onSelectResult: (result: RetrievalResult) => void;
}

function ChatInterface({
  activeResult,
  results,
  onQuery,
  onSelectResult,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);
  const [isRetrieving, setIsRetrieving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(
    () => () => {
      requestRef.current?.abort();
    },
    [],
  );

  const submitQuery = async () => {
    const query = input.trim();
    if (!query || isRetrieving) return;

    requestRef.current?.abort();
    const request = new AbortController();
    requestRef.current = request;
    setSubmittedQuery(query);
    setError(null);
    setIsRetrieving(true);

    try {
      await onQuery(query, request.signal);
      setInput("");
    } catch (requestError) {
      if (
        requestError instanceof DOMException &&
        requestError.name === "AbortError"
      ) {
        return;
      }
      setError(errorMessage(requestError));
    } finally {
      if (requestRef.current === request) {
        requestRef.current = null;
        setIsRetrieving(false);
      }
    }
  };

  return (
    <div className="chat-shell">
      <section className="query-section">
        <div className="panel-section-title">
          <span>
            <Search size={13} />
            Retrieval query
          </span>
          <strong>API required</strong>
        </div>

        <div className="composer-box">
          <textarea
            ref={textareaRef}
            rows={3}
            value={input}
            placeholder="Describe the lunar terrain to retrieve..."
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submitQuery();
              }
            }}
          />
          <button
            type="button"
            className="send-button"
            onClick={() => void submitQuery()}
            disabled={!input.trim() || isRetrieving}
            aria-label="Run retrieval query"
            title="Run retrieval query"
          >
            {isRetrieving ? (
              <Loader2 size={16} className="spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>
      </section>

      <AnimatePresence mode="wait">
        {isRetrieving && (
          <RetrievalState key="loading" tone="loading">
            <Loader2 size={14} className="spin" />
            Waiting for retrieval results for “{submittedQuery}”.
          </RetrievalState>
        )}

        {!isRetrieving && error && (
          <RetrievalState key="error" tone="error">
            <strong>Retrieval unavailable.</strong> {error}
          </RetrievalState>
        )}

        {!isRetrieving && !error && submittedQuery && results.length === 0 && (
          <RetrievalState key="empty" tone="empty">
            The retrieval service returned no patches for “{submittedQuery}”.
          </RetrievalState>
        )}

        {!isRetrieving && !error && !submittedQuery && (
          <RetrievalState key="idle" tone="empty">
            No target is selected. Results will appear only after the retrieval
            API returns matching patches.
          </RetrievalState>
        )}
      </AnimatePresence>

      {results.length > 0 && (
        <motion.section
          className="evidence-section"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.22 }}
        >
          <div className="panel-section-title">
            <span>
              <ImageIcon size={13} />
              Retrieved patches
            </span>
            <strong>{results.length} returned</strong>
          </div>

          <div className="evidence-grid">
            {results.map((result) => (
              <button
                key={result.id}
                type="button"
                className={`evidence-card ${
                  result.id === activeResult?.id ? "selected" : ""
                }`}
                onClick={() => onSelectResult(result)}
              >
                <img
                  className="evidence-thumb"
                  src={result.imageUrl}
                  alt={`Retrieved lunar ${resultLabel(result)}`}
                  loading="lazy"
                />
                <div className="evidence-copy">
                  <div className="evidence-title-row">
                    <strong>{resultLabel(result)}</strong>
                    <span>{result.similarity.toFixed(3)}</span>
                  </div>
                  {result.description && <p>{result.description}</p>}
                  <small>{formatCoords(result.lat, result.lng)}</small>
                </div>
              </button>
            ))}
          </div>
        </motion.section>
      )}
    </div>
  );
}

function RetrievalState({
  children,
  tone,
}: {
  children: ReactNode;
  tone: "loading" | "error" | "empty";
}) {
  return (
    <motion.div
      className={`retrieval-state ${tone}`}
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -5 }}
      transition={{ duration: 0.16 }}
      role={tone === "error" ? "alert" : "status"}
    >
      {children}
    </motion.div>
  );
}

function errorMessage(error: unknown) {
  if (error instanceof RetrievalApiError) return error.message;
  if (error instanceof TypeError) {
    return "The configured API could not be reached.";
  }
  if (error instanceof Error) return error.message;
  return "The retrieval request failed.";
}

export default ChatInterface;
