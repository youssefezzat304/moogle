import MoonCanvas from "../features/moon/components/MoonCanvas";
import SearchPanel from "../features/search/components/SearchPanel";
import { useSearch } from "../features/search/hooks/useSearch";
import SplitLayout from "../shared/components/layout/SplitLayout";
import "../styles/App.css";

function App() {
  const search = useSearch();

  return (
    <SplitLayout
      activeResult={search.activeResult}
      stats={{
        totalResults: search.results.length,
        queryCount: search.queryCount,
      }}
      leftPanel={<MoonCanvas activeResult={search.activeResult} />}
      rightPanel={<SearchPanel search={search} />}
    />
  );
}

export default App;
