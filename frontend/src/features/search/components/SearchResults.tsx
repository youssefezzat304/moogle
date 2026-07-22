import { motion } from "framer-motion";
import { Image as ImageIcon } from "lucide-react";
import type { RetrievalResult } from "../../retrieval/api";
import SearchResultCard from "./SearchResultCard";

interface SearchResultsProps {
  results: RetrievalResult[];
  activeResult: RetrievalResult | null;
  onSelectResult: (result: RetrievalResult) => void;
}

function SearchResults({
  results,
  activeResult,
  onSelectResult,
}: SearchResultsProps) {
  if (results.length === 0) return null;

  return (
    <motion.section
      className="evidence-section"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      aria-label="Ranked retrieval results"
    >
      <div className="panel-section-title">
        <span>
          <ImageIcon size={13} />
          Ranked patches
        </span>
        <strong>{results.length} returned</strong>
      </div>

      <div className="evidence-grid">
        {results.map((result, index) => (
          <SearchResultCard
            key={result.id}
            result={result}
            rank={index + 1}
            selected={result.id === activeResult?.id}
            onSelect={onSelectResult}
          />
        ))}
      </div>
    </motion.section>
  );
}

export default SearchResults;
