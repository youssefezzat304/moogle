import { useEffect, useState } from "react";
import {
  parseDemoQueryCatalog,
  type DemoQueryCatalog,
} from "../utils/demoQueries";

const DEMO_QUERIES_URL = `${import.meta.env.BASE_URL}demo-queries.json`;

interface DemoQueryState {
  catalog: DemoQueryCatalog | null;
  error: string | null;
}

const INITIAL_STATE: DemoQueryState = {
  catalog: null,
  error: null,
};

export function useDemoQueries(): DemoQueryState {
  const [state, setState] = useState<DemoQueryState>(INITIAL_STATE);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDemoQueries() {
      try {
        const response = await fetch(DEMO_QUERIES_URL, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Demo query request failed with ${response.status}.`);
        }
        setState({
          catalog: parseDemoQueryCatalog(await response.json()),
          error: null,
        });
      } catch (cause) {
        if (!controller.signal.aborted) {
          setState({
            catalog: null,
            error:
              cause instanceof Error
                ? cause.message
                : "Demo queries could not be loaded.",
          });
        }
      }
    }

    void loadDemoQueries();
    return () => controller.abort();
  }, []);

  return state;
}
