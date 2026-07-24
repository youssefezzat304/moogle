import type { SearchController } from "../types";
import SearchConversation from "./SearchConversation";
import SearchForm from "./SearchForm";
import SearchResults from "./SearchResults";
import SearchStatus from "./SearchStatus";

interface SearchPanelProps {
  search: SearchController;
}

function SearchPanel({ search }: SearchPanelProps) {
  if (search.messages.length === 0) {
    return (
      <div className="search-panel initial">
        <div className="initial-search">
          <span className="eyebrow">Semantic lunar search</span>
          <h2>What terrain are you looking for?</h2>
          <p>
            Describe a landform, surface texture, or mapped geologic feature.
          </p>
          <SearchForm
            isSubmitting={search.phase === "loading"}
            onSubmit={search.runSearch}
            placement="initial"
          />
        </div>
      </div>
    );
  }

  return (
    <div className="search-panel conversation">
      {search.results.length > 0 && (
        <div className="search-results-window">
          <SearchResults
            results={search.results}
            activeResult={search.activeResult}
            onSelectResult={search.selectResult}
          />
        </div>
      )}
      <SearchStatus
        phase={search.phase}
        query={search.query}
        error={search.error}
        metadata={search.metadata}
      />
      <SearchConversation messages={search.messages} phase={search.phase} />
      <SearchForm
        isSubmitting={search.phase === "loading"}
        onSubmit={search.runSearch}
        placement="conversation"
      />
    </div>
  );
}

export default SearchPanel;
