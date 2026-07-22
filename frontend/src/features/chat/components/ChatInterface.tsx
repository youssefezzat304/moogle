import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowUpRight,
  Bot,
  Image as ImageIcon,
  Loader2,
  Search,
  Send,
  Sparkles,
  User,
} from "lucide-react";
import {
  EVIDENCE_IMAGE_URL,
  formatCoords,
  type RetrievalResult,
} from "../../retrieval/mockData";

interface Message {
  id: number;
  role: "user" | "system";
  text: string;
  resultId?: string;
  ts: string;
}

interface ChatInterfaceProps {
  activeResult: RetrievalResult;
  hasRetrieved: boolean;
  results: RetrievalResult[];
  onQuery: (query: string) => RetrievalResult;
  onSelectResult: (result: RetrievalResult) => void;
}

let messageId = 0;

const exampleQueries = [
  "Show me Tycho central peak imagery",
  "Find polar shadow evidence near Shackleton",
  "Where are lunar swirls like Reiner Gamma?",
];

function timestamp() {
  const now = new Date();
  return now.toUTCString().split(" ")[4] + " UTC";
}

function ChatInterface({
  activeResult,
  hasRetrieved,
  results,
  onQuery,
  onSelectResult,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: ++messageId,
      role: "system",
      text: "Mock vector index is ready. Ask for a crater, mare, landing site, polar shadow, or albedo pattern.",
      resultId: activeResult.id,
      ts: timestamp(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking, activeResult.id]);

  const submitQuery = (query = input) => {
    const trimmed = query.trim();
    if (!trimmed || isThinking) return;

    const matchedResult = onQuery(trimmed);
    setInput("");
    setIsThinking(true);

    setMessages((current) => [
      ...current,
      {
        id: ++messageId,
        role: "user",
        text: trimmed,
        ts: timestamp(),
      },
    ]);

    window.setTimeout(() => {
      setMessages((current) => [
        ...current,
        {
          id: ++messageId,
          role: "system",
          resultId: matchedResult.id,
          text: `${matchedResult.summary} ${matchedResult.images.length} evidence frames surfaced at ${Math.round(
            matchedResult.confidence * 100,
          )}% confidence.`,
          ts: timestamp(),
        },
      ]);
      setIsThinking(false);
    }, 760);
  };

  const selectResult = (result: RetrievalResult) => {
    onSelectResult(result);
    setMessages((current) => [
      ...current,
      {
        id: ++messageId,
        role: "system",
        resultId: result.id,
        text: `Target changed to ${result.title}. Camera and scene light are slewing to ${formatCoords(
          result.lat,
          result.lng,
        )}.`,
        ts: timestamp(),
      },
    ]);
  };

  return (
    <div className="chat-shell">
      <div className="query-composer"></div>

      <AnimatePresence initial={false}>
        {hasRetrieved && (
          <motion.div
            className="evidence-section"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.22 }}
          >
            <div className="panel-section-title">
              <span>
                <ImageIcon size={13} />
                Vector evidence
              </span>
              <strong>{activeResult.images.length} frames</strong>
            </div>

            <div className="evidence-grid">
              <AnimatePresence mode="popLayout">
                {activeResult.images.map((image) => (
                  <motion.article
                    key={image.id}
                    className="evidence-card"
                    initial={{ opacity: 0, y: 10, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -8, scale: 0.98 }}
                    transition={{ duration: 0.22 }}
                  >
                    <a
                      href={EVIDENCE_IMAGE_URL}
                      target="_blank"
                      rel="noreferrer"
                      className="evidence-thumb-link"
                      aria-label={`Open ${image.title} image`}
                      title={`Open ${image.title} image`}
                    >
                      <span
                        className="evidence-thumb"
                        style={{
                          backgroundImage: `linear-gradient(145deg, rgba(4, 10, 18, 0.08), rgba(125, 211, 252, 0.12)), url('${EVIDENCE_IMAGE_URL}')`,
                          backgroundPosition: image.crop,
                        }}
                      />
                    </a>
                    <div className="evidence-copy">
                      <div className="evidence-title-row">
                        <h3>{image.title}</h3>
                        <span>{Math.round(image.score * 100)}</span>
                      </div>
                      <p>{image.caption}</p>
                      <small>{image.meta}</small>
                    </div>
                  </motion.article>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="message-log">
        <div className="panel-section-title">
          <span>
            <Bot size={13} />
            Session trace
          </span>
          <strong>{messages.length} events</strong>
        </div>

        <div className="messages">
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                className={`message ${message.role}`}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
              >
                <div className="message-icon">
                  {message.role === "user" ? (
                    <User size={13} />
                  ) : (
                    <Bot size={13} />
                  )}
                </div>
                <div>
                  <div className="message-meta">
                    <span>
                      {message.role === "user" ? "operator" : "engine"}
                    </span>
                    <small>{message.ts}</small>
                  </div>
                  <p>{message.text}</p>
                </div>
              </motion.div>
            ))}

            {isThinking && (
              <motion.div
                className="message system"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <div className="message-icon">
                  <Loader2 size={13} className="spin" />
                </div>
                <div>
                  <div className="message-meta">
                    <span>engine</span>
                    <small>retrieving</small>
                  </div>
                  <p>
                    Embedding query, selecting coordinate target, and loading
                    evidence tiles.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>
      </div>
      <div className="panel-section-title">
        <span>
          <Search size={13} />
          Retrieval query
        </span>
        <strong>mock backend</strong>
      </div>

      <div className="composer-box">
        <textarea
          ref={textareaRef}
          rows={3}
          value={input}
          placeholder="Ask about a lunar surface feature..."
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submitQuery();
            }
          }}
        />
        <button
          type="button"
          className="send-button"
          onClick={() => submitQuery()}
          disabled={!input.trim() || isThinking}
          aria-label="Send query"
          title="Send query"
        >
          {isThinking ? (
            <Loader2 size={16} className="spin" />
          ) : (
            <Send size={16} />
          )}
        </button>
      </div>

      <div className="recommendation-row">
        {exampleQueries.map((query) => (
          <button
            key={query}
            type="button"
            onClick={() => submitQuery(query)}
            disabled={isThinking}
          >
            <Sparkles size={12} />
            {query}
          </button>
        ))}

        {results.map((result) => (
          <button
            key={result.id}
            type="button"
            className={result.id === activeResult.id ? "selected" : ""}
            onClick={() => selectResult(result)}
          >
            <span>{result.title}</span>
            <ArrowUpRight size={12} />
          </button>
        ))}
      </div>
    </div>
  );
}

export default ChatInterface;
