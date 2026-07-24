import { Bot, MessageSquareText, UserRound } from "lucide-react";
import { useEffect, useRef } from "react";
import type { SearchMessage, SearchPhase } from "../types";

interface SearchConversationProps {
  messages: SearchMessage[];
  phase: SearchPhase;
}

function SearchConversation({ messages, phase }: SearchConversationProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [messages, phase]);

  return (
    <section className="conversation-window" aria-label="Search conversation">
      <div className="panel-section-title">
        <span>
          <MessageSquareText size={13} />
          Recent messages
        </span>
        <strong>{messages.length}</strong>
      </div>

      <div className="conversation-messages" aria-live="polite">
        {messages.map((message) => (
          <article
            key={message.id}
            className={`conversation-message ${message.role} ${
              message.tone === "error" ? "error" : ""
            }`}
          >
            <span className="conversation-avatar" aria-hidden="true">
              {message.role === "user" ? (
                <UserRound size={14} />
              ) : (
                <Bot size={14} />
              )}
            </span>
            <div>
              <strong>{message.role === "user" ? "You" : "Moogle"}</strong>
              <p>{message.content}</p>
            </div>
          </article>
        ))}

        {phase === "loading" && (
          <article className="conversation-message assistant pending">
            <span className="conversation-avatar" aria-hidden="true">
              <Bot size={14} />
            </span>
            <div>
              <strong>Moogle</strong>
              <p>Searching the lunar terrain index…</p>
            </div>
          </article>
        )}
        <div ref={endRef} />
      </div>
    </section>
  );
}

export default SearchConversation;
