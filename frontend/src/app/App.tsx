import { useCallback, useRef, useState } from "react";
import MoonCanvas from "../features/moon/components/MoonCanvas";
import type { RetrievalResult } from "../features/retrieval/api";
import ResultImageDialog from "../features/retrieval/components/ResultImageDialog";
import SearchPanel from "../features/search/components/SearchPanel";
import { useSearch } from "../features/search/hooks/useSearch";
import type { DemoQueryRequest } from "../features/search/types";
import SplitLayout from "../shared/components/layout/SplitLayout";
import { useDemoQueries } from "../shared/hooks/useDemoQueries";
import "../styles/App.css";

function App() {
  const search = useSearch();
  const demoQueries = useDemoQueries();
  const { selectResult } = search;
  const [topK, setTopK] = useState(5);
  const [demoQueryRequest, setDemoQueryRequest] =
    useState<DemoQueryRequest | null>(null);
  const demoQueryIdRef = useRef(0);
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
  const requestDemoQuery = useCallback(
    (query: string) => {
      if (search.phase === "loading" || topK === 0) return;
      demoQueryIdRef.current += 1;
      setDemoQueryRequest({
        id: demoQueryIdRef.current,
        query,
      });
    },
    [search.phase, topK],
  );
  const consumeDemoQuery = useCallback((requestId: number) => {
    setDemoQueryRequest((current) =>
      current?.id === requestId ? null : current,
    );
  }, []);

  return (
    <>
      <SplitLayout
        leftPanel={
          <MoonCanvas
            activeResult={search.activeResult}
            onPreviewResult={previewImage}
            demoQueryCatalog={demoQueries.catalog}
            demoQueryError={demoQueries.error}
            canRunDemoQuery={search.phase !== "loading" && topK > 0}
            onRunDemoQuery={requestDemoQuery}
          />
        }
        rightPanel={
          <SearchPanel
            search={search}
            topK={topK}
            onTopKChange={setTopK}
            demoQueryCatalog={demoQueries.catalog}
            demoQueryError={demoQueries.error}
            demoQueryRequest={demoQueryRequest}
            onConsumeDemoQuery={consumeDemoQuery}
          />
        }
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
