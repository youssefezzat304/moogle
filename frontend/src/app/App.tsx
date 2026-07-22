import { useState } from "react";
import ChatInterface from "../features/chat/components/ChatInterface";
import MoonCanvas from "../features/moon/components/MoonCanvas";
import {
  retrieveLunarPatches,
  type RetrievalResponse,
  type RetrievalResult,
} from "../features/retrieval/api";
import SplitLayout from "../shared/components/layout/SplitLayout";
import "../styles/App.css";

function App() {
  const [activeResult, setActiveResult] = useState<RetrievalResult | null>(
    null,
  );
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [queryCount, setQueryCount] = useState(0);

  const runQuery = async (
    query: string,
    signal: AbortSignal,
  ): Promise<RetrievalResponse> => {
    setResults([]);
    setActiveResult(null);
    const response = await retrieveLunarPatches(query, { signal });
    setResults(response.results);
    setActiveResult(response.results[0] ?? null);
    setQueryCount((count) => count + 1);
    return response;
  };

  return (
    <SplitLayout
      activeResult={activeResult}
      stats={{ totalResults: results.length, queryCount }}
      leftPanel={<MoonCanvas activeResult={activeResult} />}
      rightPanel={
        <ChatInterface
          activeResult={activeResult}
          results={results}
          onQuery={runQuery}
          onSelectResult={setActiveResult}
        />
      }
    />
  );
}

export default App;
