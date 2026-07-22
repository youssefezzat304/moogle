import type { SearchController } from "../types";
import SearchForm from "./SearchForm";
import SearchResults from "./SearchResults";
import SearchStatus from "./SearchStatus";

interface SearchPanelProps {
  search: SearchController;
}

function SearchPanel({ search }: SearchPanelProps) {
  return (
    <div className="search-panel">
      <SearchForm
        isSubmitting={search.phase === "loading"}
        onSubmit={search.runSearch}
      />
      <SearchStatus
        phase={search.phase}
        query={search.query}
        error={search.error}
        metadata={search.metadata}
      />
      <SearchResults
        results={search.results}
        activeResult={search.activeResult}
        onSelectResult={search.selectResult}
      />
    </div>
  );
}

export default SearchPanel;
