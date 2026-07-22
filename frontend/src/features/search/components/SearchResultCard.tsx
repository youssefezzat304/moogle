import {
  formatCoords,
  resultLabel,
  type RetrievalResult,
} from "../../retrieval/api";

interface SearchResultCardProps {
  result: RetrievalResult;
  rank: number;
  selected: boolean;
  onSelect: (result: RetrievalResult) => void;
}

function SearchResultCard({
  result,
  rank,
  selected,
  onSelect,
}: SearchResultCardProps) {
  return (
    <button
      type="button"
      className={`evidence-card ${selected ? "selected" : ""}`}
      onClick={() => onSelect(result)}
      aria-pressed={selected}
    >
      <img
        className="evidence-thumb"
        src={result.imageUrl}
        alt={`Retrieved lunar ${resultLabel(result)}`}
        loading="lazy"
      />
      <div className="evidence-copy">
        <div className="evidence-title-row">
          <strong>
            <b className="evidence-rank">#{rank.toString().padStart(2, "0")}</b>
            {resultLabel(result)}
          </strong>
          <span>{result.similarity.toFixed(3)}</span>
        </div>
        {result.description && <p>{result.description}</p>}
        <small>{formatCoords(result.lat, result.lng)}</small>
      </div>
    </button>
  );
}

export default SearchResultCard;
