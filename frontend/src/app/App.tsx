import { useMemo, useState } from "react";
import ChatInterface from "../features/chat/components/ChatInterface";
import MoonCanvas from "../features/moon/components/MoonCanvas";
import {
  RETRIEVAL_RESULTS,
  resolveMockRetrieval,
  type RetrievalResult,
} from "../features/retrieval/mockData";
import SplitLayout from "../shared/components/layout/SplitLayout";
import "../styles/App.css";

function App() {
  const [activeResult, setActiveResult] = useState<RetrievalResult>(
    RETRIEVAL_RESULTS[0],
  );
  const [queryCount, setQueryCount] = useState(0);

  const runQuery = (query: string) => {
    const result = resolveMockRetrieval(query, queryCount);
    setActiveResult(result);
    setQueryCount((count) => count + 1);
    return result;
  };

  const retrievalStats = useMemo(
    () => ({
      totalResults: RETRIEVAL_RESULTS.length,
      queryCount,
    }),
    [queryCount],
  );

  return (
    <SplitLayout
      activeResult={activeResult}
      stats={retrievalStats}
      leftPanel={
        <MoonCanvas activeResult={activeResult} hasRetrieved={queryCount > 0} />
      }
      rightPanel={
        <ChatInterface
          activeResult={activeResult}
          hasRetrieved={queryCount > 0}
          results={RETRIEVAL_RESULTS}
          onQuery={runQuery}
          onSelectResult={setActiveResult}
        />
      }
    />
  );
}

export default App;
