import {
  formatCoords,
  formatSimilarity,
  resultLabel,
  type RetrievalResult,
} from "../../retrieval/api";

interface SearchResultCardProps {
  result: RetrievalResult;
  selected: boolean;
  onSelect: (result: RetrievalResult) => void;
  onPreview: (result: RetrievalResult) => void;
}

function SearchResultCard({
  result,
  selected,
  onSelect,
  onPreview,
}: SearchResultCardProps) {
  return (
    <article
      className={`evidence-card ${selected ? "selected" : ""}`}
    >
      <button
        type="button"
        className="evidence-card-select"
        onClick={() => onSelect(result)}
        aria-label={`Select ${resultLabel(result)}`}
        aria-pressed={selected}
      />
      <button
        type="button"
        className="evidence-thumb-button"
        onClick={() => onPreview(result)}
        aria-label={`Preview ${resultLabel(result)} image and description`}
        title={`Preview ${resultLabel(result)} image and description`}
      >
        <img
          className="evidence-thumb"
          src={result.wacImageUrl}
          alt=""
          loading="lazy"
        />
      </button>
      <div className="evidence-copy">
        <div className="evidence-title-row">
          <strong>
            <b className="evidence-rank">
              #{result.rank.toString().padStart(2, "0")}
            </b>
            {resultLabel(result)}
          </strong>
          <span>{formatSimilarity(result.similarity)}</span>
        </div>
        {result.description && <p>{result.description}</p>}
        <small>{formatCoords(result.lat, result.lng)}</small>
      </div>
    </article>
  );
}

export default SearchResultCard;
