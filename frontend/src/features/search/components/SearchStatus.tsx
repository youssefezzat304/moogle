import { AnimatePresence, motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import type { SearchMetadata, SearchPhase } from "../types";

interface SearchStatusProps {
  phase: SearchPhase;
  query: string | null;
  error: string | null;
  metadata: SearchMetadata;
}

function SearchStatus({ phase, query, error, metadata }: SearchStatusProps) {
  return (
    <AnimatePresence mode="wait">
      {phase === "loading" && (
        <StatusFrame key="loading" tone="loading">
          <Loader2 size={14} className="spin" />
          Waiting for retrieval results for “{query}”.
        </StatusFrame>
      )}

      {phase === "error" && (
        <StatusFrame key="error" tone="error">
          <strong>Retrieval unavailable.</strong> {error}
        </StatusFrame>
      )}

      {phase === "empty" && (
        <StatusFrame key="empty" tone="empty">
          The retrieval service returned no patches for “{query}”.
        </StatusFrame>
      )}

      {phase === "idle" && (
        <StatusFrame key="idle" tone="empty">
          No target is selected. Ranked patches will appear only after the
          retrieval API returns results.
        </StatusFrame>
      )}

      {phase === "success" && (metadata.modelId || metadata.indexSize) && (
        <span className="search-trace" key="trace">
          {metadata.modelId && <strong>{metadata.modelId}</strong>}
          {metadata.indexSize !== undefined && (
            <span>{metadata.indexSize.toLocaleString()} indexed patches</span>
          )}
          {metadata.elapsedMs !== undefined && (
            <span>{metadata.elapsedMs.toLocaleString()} ms</span>
          )}
        </span>
      )}
    </AnimatePresence>
  );
}

function StatusFrame({
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

export default SearchStatus;
