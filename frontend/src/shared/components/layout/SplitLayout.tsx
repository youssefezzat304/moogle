import { type ReactNode, useState } from "react";
import { Orbit, PanelRightClose, PanelRightOpen } from "lucide-react";
import useMediaQuery from "../../hooks/useMediaQuery";

interface SplitLayoutProps {
  leftPanel: ReactNode;
  rightPanel: ReactNode;
}

function SplitLayout({ leftPanel, rightPanel }: SplitLayoutProps) {
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

        <div className="topbar-logos">
          <a
            className="topbar-logo-link"
            href="https://www.tu-dortmund.de"
            aria-label="Visit the TU Dortmund website"
          >
            <img
              className="topbar-tudo-logo"
              src="/tudo-logo.svg"
              alt="Technische Universität Dortmund"
            />
          </a>
          <a
            className="topbar-logo-link"
            href="https://bv.etit.tu-dortmund.de"
            aria-label="Visit the Bildsignalverarbeitung website"
          >
            <img
              className="topbar-institution-logo"
              src="/bv-logo.svg"
              alt="Arbeitsgebiet Bildsignalverarbeitung"
            />
          </a>
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
