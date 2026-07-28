import { LocateFixed } from "lucide-react";
import SearchFeatureLegend from "../../../shared/components/SearchFeatureLegend";
import type { DemoQueryCatalog } from "../../../shared/utils/demoQueries";
import { formatCoords, resultLabel } from "../../retrieval/api";
import type { MoonTarget } from "../types";

interface MoonHudProps {
  activeResult: MoonTarget;
  hasWandered: boolean;
  onRecenter: () => void;
  demoQueryCatalog: DemoQueryCatalog | null;
  demoQueryError: string | null;
  canRunDemoQuery: boolean;
  onRunDemoQuery: (query: string) => void;
}

function MoonHud({
  activeResult,
  hasWandered,
  onRecenter,
  demoQueryCatalog,
  demoQueryError,
  canRunDemoQuery,
  onRunDemoQuery,
}: MoonHudProps) {
  return (
    <div className="viewport-hud">
      <div className="hud-target">
        <span className="eyebrow">
          {activeResult ? "retrieval target" : "no target"}
        </span>
        <strong>
          {activeResult ? resultLabel(activeResult) : "Awaiting retrieval"}
        </strong>
        <small>
          {activeResult
            ? formatCoords(activeResult.lat, activeResult.lng)
            : "Submit a query to the retrieval API"}
        </small>
      </div>

      <div className="reticle" aria-hidden="true">
        <span />
      </div>

      {activeResult && hasWandered && (
        <button
          type="button"
          className="recenter-button"
          onClick={onRecenter}
          aria-label="Return to active retrieval target"
          title="Return to active retrieval target"
        >
          <LocateFixed size={16} />
        </button>
      )}

      <SearchFeatureLegend
        demoQueryCatalog={demoQueryCatalog}
        demoQueryError={demoQueryError}
        canRunDemoQuery={canRunDemoQuery}
        onRunDemoQuery={onRunDemoQuery}
      />
    </div>
  );
}

export default MoonHud;
