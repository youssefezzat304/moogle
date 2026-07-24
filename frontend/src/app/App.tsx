import { useCallback, useState } from "react";
import MoonCanvas from "../features/moon/components/MoonCanvas";
import type { RetrievalResult } from "../features/retrieval/api";
import ResultImageDialog from "../features/retrieval/components/ResultImageDialog";
import SearchPanel from "../features/search/components/SearchPanel";
import { useSearch } from "../features/search/hooks/useSearch";
import SplitLayout from "../shared/components/layout/SplitLayout";
import "../styles/App.css";

function App() {
  const search = useSearch();
  const { selectResult } = search;
  const [previewResultId, setPreviewResultId] = useState<string | null>(null);
  const previewResult =
    search.results.find((result) => result.id === previewResultId) ?? null;
  const closePreview = useCallback(() => setPreviewResultId(null), []);
  const previewImage = useCallback(
    (result: RetrievalResult) => {
      selectResult(result);
      setPreviewResultId(result.id);
    },
    [selectResult],
  );

  return (
    <>
      <SplitLayout
        activeResult={search.activeResult}
        stats={{
          totalResults: search.results.length,
          queryCount: search.queryCount,
        }}
        leftPanel={
          <MoonCanvas
            activeResult={search.activeResult}
            onPreviewResult={previewImage}
          />
        }
        rightPanel={<SearchPanel search={search} />}
      />
      {previewResult && (
        <ResultImageDialog
          results={search.results}
          activeResultId={previewResult.id}
          onSelectResult={previewImage}
          onClose={closePreview}
        />
      )}
    </>
  );
}

export default App;
