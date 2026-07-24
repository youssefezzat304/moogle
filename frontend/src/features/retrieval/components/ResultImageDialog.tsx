import { X } from "lucide-react";
import { useEffect } from "react";
import {
  formatCoords,
  formatSimilarity,
  resultLabel,
  type RetrievalResult,
} from "../api";

interface ResultImageDialogProps {
  result: RetrievalResult;
  onClose: () => void;
}

function ResultImageDialog({ result, onClose }: ResultImageDialogProps) {
  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }

    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div
      className="result-image-backdrop"
      role="presentation"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
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

        <img
          src={result.wacImageUrl}
          alt={`Retrieved lunar ${resultLabel(result)}`}
        />

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
        <small>
          {result.sourceVersion} · {result.promptStyle}
        </small>
      </section>
    </div>
  );
}

export default ResultImageDialog;
