import { type ReactNode, useState } from "react";
import { Orbit, PanelRightClose, PanelRightOpen } from "lucide-react";
import {
  formatCoords,
  resultLabel,
  type RetrievalResult,
} from "../../../features/retrieval/api";
import useMediaQuery from "../../hooks/useMediaQuery";

interface SplitLayoutProps {
  leftPanel: ReactNode;
  rightPanel: ReactNode;
  activeResult: RetrievalResult | null;
}

function SplitLayout({
  leftPanel,
  rightPanel,
  activeResult,
}: SplitLayoutProps) {
  const isMobile = useMediaQuery("(max-width: 845px)");
  const [isAsideCollapsed, setIsAsideCollapsed] = useState(false);

  if (isMobile) {
    return (
      <main className="app-shell mobile-shell">
        <section className="mobile-canvas">{leftPanel}</section>
        <section className="mobile-panel">{rightPanel}</section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">
            <Orbit size={18} strokeWidth={1.7} />
          </div>
          <div>
            <span className="eyebrow">MOOGLE</span>
            <h1>Lunar Retrieval Engine</h1>
          </div>
        </div>

        <div className="topbar-target">
          <span>
            {activeResult ? resultLabel(activeResult) : "No target selected"}
          </span>
          <strong>
            {activeResult
              ? formatCoords(activeResult.lat, activeResult.lng)
              : "Waiting for retrieval results"}
          </strong>
        </div>
      </header>

      <div
        className={`workspace-grid ${
          isAsideCollapsed ? "aside-collapsed" : ""
        }`}
      >
        <section className="viewport-panel">{leftPanel}</section>
        <button
          type="button"
          className="aside-edge-toggle"
          onClick={() => setIsAsideCollapsed((collapsed) => !collapsed)}
          aria-label={
            isAsideCollapsed ? "Show retrieval panel" : "Hide retrieval panel"
          }
          title={
            isAsideCollapsed ? "Show retrieval panel" : "Hide retrieval panel"
          }
        >
          {isAsideCollapsed ? (
            <PanelRightOpen size={16} />
          ) : (
            <PanelRightClose size={16} />
          )}
        </button>
        {!isAsideCollapsed && (
          <aside className="retrieval-panel">{rightPanel}</aside>
        )}
      </div>
    </main>
  );
}

export default SplitLayout;
