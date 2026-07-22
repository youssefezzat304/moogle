import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  id: number;
  role: "user" | "system";
  text: string;
  ts: string;
}

let _id = 0;
const uid = () => ++_id;

const timestamp = () => {
  const now = new Date();
  return now.toUTCString().split(" ")[4] + " UTC";
};

// Mock system responses — swap out for real API responses later
const MOCK_RESPONSES = [
  "Scanning regolith database… 3,847 results matched. Refining by crater morphology.",
  "Query indexed. Nearest Apollo landing site: Apollo 11 — Mare Tranquillitatis, 0.6741°N 23.4731°E.",
  "Retrieval complete. LROC NAC imagery available at 0.5 m/px for selected coordinates.",
  "Cross-referencing LDEM elevation data… Peak elevation delta: +4,230 m over 180 km baseline.",
  "No surface feature matches found in current FOV. Expanding search radius to 50 km.",
];
let _mockIdx = 0;

function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isInitial = messages.length === 0;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isThinking) return;

    const userMsg: Message = {
      id: uid(),
      role: "user",
      text: trimmed,
      ts: timestamp(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsThinking(true);

    window.dispatchEvent(new CustomEvent("messageSent"));

    // Simulate async system response
    setTimeout(
      () => {
        const sysMsg: Message = {
          id: uid(),
          role: "system",
          text: MOCK_RESPONSES[_mockIdx % MOCK_RESPONSES.length],
          ts: timestamp(),
        };
        _mockIdx++;
        setMessages((prev) => [...prev, sysMsg]);
        setIsThinking(false);
      },
      1200 + Math.random() * 800,
    );
  };

  return (
    <div
      className="h-full flex flex-col font-mono"
      style={{ background: "var(--color-surface)" }}
    >
      {/* ── Panel header ── */}
      <div
        className="shrink-0 flex items-center justify-between px-4 py-2.5 border-b"
        style={{ borderColor: "var(--color-border)" }}
      >
        <div className="flex flex-col gap-0.5">
          <span
            className="text-[10px] tracking-[0.25em] uppercase"
            style={{ color: "var(--color-amber)" }}
          >
            Query Interface
          </span>
          <span
            className="text-[9px] tracking-[0.12em]"
            style={{ color: "var(--color-muted)" }}
          >
            {messages.length === 0
              ? "AWAITING INPUT"
              : `${messages.filter((m) => m.role === "user").length} QUERIES · ${messages.filter((m) => m.role === "system").length} RESULTS`}
          </span>
        </div>

        {/* Session indicator */}
        <div
          className="text-[9px] tracking-[0.12em] px-2 py-1 border"
          style={{
            color: "var(--color-muted)",
            borderColor: "var(--color-border)",
          }}
        >
          SES-{String(Math.floor(Math.random() * 9000) + 1000)}
        </div>
      </div>

      {/* ── Message list ── */}
      <div className="flex-1 overflow-y-auto min-h-0 flex flex-col">
        {isInitial ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.3 }}
            className="flex-1 flex flex-col items-center justify-center gap-6 px-6"
          >
            {/* Reticle art */}
            <div className="relative flex items-center justify-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
                style={{
                  width: 72,
                  height: 72,
                  borderRadius: "50%",
                  border: "1px solid var(--color-amber-dim)",
                  borderTopColor: "var(--color-amber)",
                }}
              />
              <div
                className="absolute text-2xl"
                style={{ color: "var(--color-amber-dim)" }}
              >
                ◎
              </div>
            </div>

            <div className="flex flex-col items-center gap-2 text-center">
              <p
                className="text-[11px] tracking-[0.2em] uppercase"
                style={{ color: "var(--color-amber)" }}
              >
                System Ready
              </p>
              <p
                className="text-[10px] leading-relaxed max-w-[240px]"
                style={{ color: "var(--color-muted)" }}
              >
                Enter a query to begin lunar surface retrieval. Coordinates,
                feature names, or natural language accepted.
              </p>
            </div>

            {/* Example queries */}
            <div className="flex flex-col gap-1.5 w-full max-w-[280px]">
              {[
                "Show craters near Apollo 11 site",
                "Elevation profile: Tycho crater",
                "Mare Tranquillitatis imagery",
              ].map((ex) => (
                <button
                  key={ex}
                  onClick={() => setInput(ex)}
                  className="text-left text-[9px] tracking-[0.1em] px-3 py-2 border transition-colors duration-150"
                  style={{
                    color: "var(--color-muted)",
                    borderColor: "var(--color-border)",
                    background: "transparent",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor =
                      "var(--color-amber-dim)";
                    (e.currentTarget as HTMLButtonElement).style.color =
                      "var(--color-amber)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor =
                      "var(--color-border)";
                    (e.currentTarget as HTMLButtonElement).style.color =
                      "var(--color-muted)";
                  }}
                >
                  ↗ {ex}
                </button>
              ))}
            </div>
          </motion.div>
        ) : (
          <div className="flex flex-col px-4 py-4 gap-5">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  className={`flex flex-col gap-1 ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  {/* Role label + timestamp */}
                  <div className="flex items-center gap-2">
                    <span
                      className="text-[9px] tracking-[0.15em] uppercase"
                      style={{
                        color:
                          msg.role === "user"
                            ? "var(--color-amber-dim)"
                            : "var(--color-muted)",
                      }}
                    >
                      {msg.role === "user" ? "OPERATOR" : "MOOGLE"}
                    </span>
                    <span
                      className="text-[8px] tabular-nums"
                      style={{ color: "var(--color-muted)" }}
                    >
                      {msg.ts}
                    </span>
                  </div>

                  {/* Bubble */}
                  <div
                    className="max-w-[88%] px-3 py-2.5 text-[11px] leading-relaxed"
                    style={
                      msg.role === "user"
                        ? {
                            borderRight: "2px solid var(--color-amber)",
                            color: "var(--color-fg)",
                            background: "rgba(200,169,110,0.05)",
                          }
                        : {
                            borderLeft: "2px solid var(--color-border)",
                            color: "var(--color-fg-dim)",
                            background: "rgba(255,255,255,0.02)",
                          }
                    }
                  >
                    {msg.text}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Thinking indicator */}
            <AnimatePresence>
              {isThinking && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col gap-1 items-start"
                >
                  <span
                    className="text-[9px] tracking-[0.15em] uppercase"
                    style={{ color: "var(--color-muted)" }}
                  >
                    MOOGLE
                  </span>
                  <div
                    className="px-3 py-2.5 flex items-center gap-1.5"
                    style={{
                      borderLeft: "2px solid var(--color-border)",
                      background: "rgba(255,255,255,0.02)",
                    }}
                  >
                    {[0, 1, 2].map((i) => (
                      <motion.span
                        key={i}
                        animate={{ opacity: [0.2, 1, 0.2] }}
                        transition={{
                          duration: 1.2,
                          repeat: Infinity,
                          delay: i * 0.2,
                        }}
                        className="w-1 h-1 rounded-full"
                        style={{ background: "var(--color-muted)" }}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* ── Input area ── */}
      <div
        className="shrink-0 border-t p-4"
        style={{ borderColor: "var(--color-border)" }}
      >
        {/* Corner-bracket input wrapper */}
        <div className="relative">
          {/* TL bracket */}
          <svg
            className="absolute top-0 left-0 pointer-events-none"
            width="10"
            height="10"
            viewBox="0 0 10 10"
          >
            <path
              d="M0 10 L0 0 L10 0"
              stroke="var(--color-amber-dim)"
              strokeWidth="1"
              fill="none"
            />
          </svg>
          {/* BR bracket */}
          <svg
            className="absolute bottom-0 right-0 pointer-events-none"
            width="10"
            height="10"
            viewBox="0 0 10 10"
            style={{ transform: "rotate(180deg)" }}
          >
            <path
              d="M0 10 L0 0 L10 0"
              stroke="var(--color-amber-dim)"
              strokeWidth="1"
              fill="none"
            />
          </svg>

          <textarea
            ref={textareaRef}
            rows={2}
            className="w-full resize-none text-[11px] leading-relaxed px-3 py-2.5 outline-none font-mono placeholder:opacity-100 transition-colors duration-150"
            placeholder="Enter query…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            style={{
              background: "var(--color-input-bg)",
              color: "var(--color-fg)",
              border: "1px solid var(--color-border)",
              caretColor: "var(--color-amber)",
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "var(--color-amber-dim)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "var(--color-border)";
            }}
          />
        </div>

        {/* Footer row */}
        <div className="flex items-center justify-between mt-2">
          <span
            className="text-[9px] tracking-[0.1em]"
            style={{ color: "var(--color-muted)" }}
          >
            ↵ SEND · SHIFT+↵ NEWLINE
          </span>

          <button
            onClick={handleSend}
            disabled={!input.trim() || isThinking}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[9px] tracking-[0.15em] uppercase border transition-all duration-150 disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              color: "var(--color-amber)",
              borderColor: "var(--color-amber-dim)",
              background: "transparent",
            }}
            onMouseEnter={(e) => {
              if (!isThinking && input.trim()) {
                (e.currentTarget as HTMLButtonElement).style.background =
                  "rgba(200,169,110,0.08)";
              }
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background =
                "transparent";
            }}
          >
            {isThinking ? "RETRIEVING…" : "TRANSMIT ↗"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
