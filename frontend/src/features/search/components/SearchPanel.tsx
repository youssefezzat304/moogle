import { useMemo } from "react";
import type { DemoQueryCatalog } from "../../../shared/utils/demoQueries";
import { pickRandomDemoSuggestions } from "../../../shared/utils/demoQueries";
import type { DemoQueryRequest, SearchController } from "../types";
import SearchConversation from "./SearchConversation";
import SearchForm from "./SearchForm";
import SearchResults from "./SearchResults";
import SearchStatus from "./SearchStatus";

interface SearchPanelProps {
  search: SearchController;
  topK: number;
  onTopKChange: (topK: number) => void;
  demoQueryCatalog: DemoQueryCatalog | null;
  demoQueryError: string | null;
  demoQueryRequest: DemoQueryRequest | null;
  onConsumeDemoQuery: (requestId: number) => void;
}

function SearchPanel({
  search,
  topK,
  onTopKChange,
  demoQueryCatalog,
  demoQueryError,
  demoQueryRequest,
  onConsumeDemoQuery,
}: SearchPanelProps) {
  const suggestions = useMemo(
    () => pickRandomDemoSuggestions(demoQueryCatalog, 3),
    [demoQueryCatalog],
  );

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
            key={`initial-${demoQueryRequest?.id ?? "manual"}`}
            isSubmitting={search.phase === "loading"}
            topK={topK}
            onTopKChange={onTopKChange}
            onSubmit={search.runSearch}
            suggestions={suggestions}
            suggestionError={demoQueryError}
            demoQueryRequest={demoQueryRequest}
            onConsumeDemoQuery={onConsumeDemoQuery}
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
        key={`conversation-${demoQueryRequest?.id ?? "manual"}`}
        isSubmitting={search.phase === "loading"}
        topK={topK}
        onTopKChange={onTopKChange}
        onSubmit={search.runSearch}
        demoQueryRequest={demoQueryRequest}
        onConsumeDemoQuery={onConsumeDemoQuery}
        placement="conversation"
      />
    </div>
  );
}

export default SearchPanel;
