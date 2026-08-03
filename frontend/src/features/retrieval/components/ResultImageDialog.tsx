import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useCallback, useEffect } from "react";
import {
  formatCoords,
  formatSimilarity,
  resultLabel,
  type RetrievalResult,
} from "../api";

interface ResultImageDialogProps {
  results: RetrievalResult[];
  activeResultId: string;
  onSelectResult: (result: RetrievalResult) => void;
  onClose: () => void;
}

function ResultImageDialog({
  results,
  activeResultId,
  onSelectResult,
  onClose,
}: ResultImageDialogProps) {
  const activeIndex = results.findIndex(
    (result) => result.id === activeResultId,
  );
  const result = results[activeIndex];

  const showResult = useCallback(
    (offset: number) => {
      if (results.length < 2 || activeIndex < 0) return;
      const nextIndex =
        (activeIndex + offset + results.length) % results.length;
      onSelectResult(results[nextIndex]);
    },
    [activeIndex, onSelectResult, results],
  );

  useEffect(() => {
    function handleKeyboardNavigation(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft") showResult(-1);
      if (event.key === "ArrowRight") showResult(1);
    }

    document.addEventListener("keydown", handleKeyboardNavigation);
    return () =>
      document.removeEventListener("keydown", handleKeyboardNavigation);
  }, [onClose, showResult]);

  if (!result) return null;

  return (
    <div
      className="result-image-backdrop"
      role="presentation"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <button
        type="button"
        className="result-image-navigation previous"
        onClick={() => showResult(-1)}
        disabled={results.length < 2}
        aria-label="Show previous retrieved image"
      >
        <ChevronLeft size={24} />
      </button>

      <section
        className="result-image-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="result-image-dialog-title"
      >
        <header>
          <div>
            <span className="eyebrow">Retrieved WAC patch</span>
            <h2 id="result-image-dialog-title">{resultLabel(result)}</h2>
            <span className="result-image-position">
              Image {activeIndex + 1} of {results.length}
            </span>
          </div>
          <button
            type="button"
            className="result-image-close"
            onClick={onClose}
            aria-label="Close image preview"
          >
            <X size={18} />
          </button>
        </header>

        <div className="result-image-stage">
          <img
            src={result.wacImageUrl}
            alt={`Retrieved lunar ${resultLabel(result)}`}
          />
        </div>

        <div className="result-image-details">
          <div>
            <span>Rank</span>
            <strong>#{result.rank.toString().padStart(2, "0")}</strong>
          </div>
          <div>
            <span>Similarity</span>
            <strong>{formatSimilarity(result.similarity)}</strong>
          </div>
          <div>
            <span>Coordinates</span>
            <strong>{formatCoords(result.lat, result.lng)}</strong>
          </div>
        </div>

        <p>{result.description}</p>
      </section>

      <button
        type="button"
        className="result-image-navigation next"
        onClick={() => showResult(1)}
        disabled={results.length < 2}
        aria-label="Show next retrieved image"
      >
        <ChevronRight size={24} />
      </button>
    </div>
  );
}

export default ResultImageDialog;
