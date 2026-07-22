import { type ReactNode, useEffect, useState } from "react";
import {
  Activity,
  Database,
  Orbit,
  PanelRightClose,
  PanelRightOpen,
  Radio,
} from "lucide-react";
import {
  formatCoords,
  type RetrievalResult,
} from "../../../features/retrieval/mockData";
import useMediaQuery from "../../hooks/useMediaQuery";

interface SplitLayoutProps {
  leftPanel: ReactNode;
  rightPanel: ReactNode;
  activeResult: RetrievalResult;
  stats: {
    totalResults: number;
    queryCount: number;
  };
}

function getUtcClock() {
  const now = new Date();
  return `UTC ${now.toUTCString().split(" ")[4]}`;
}

function SplitLayout({
  leftPanel,
  rightPanel,
  activeResult,
  stats,
}: SplitLayoutProps) {
  const isMobile = useMediaQuery("(max-width: 845px)");
  const [isAsideCollapsed, setIsAsideCollapsed] = useState(false);
  const [clock, setClock] = useState(getUtcClock);

  useEffect(() => {
    const id = setInterval(() => setClock(getUtcClock()), 1000);
    return () => clearInterval(id);
  }, []);

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
          <span>{activeResult.title}</span>
          <strong>{formatCoords(activeResult.lat, activeResult.lng)}</strong>
        </div>

        <div className="status-cluster">
          <StatusIndicator
            icon={<Radio size={13} />}
            label="Telemetry"
            active
          />
          <StatusIndicator
            icon={<Database size={13} />}
            label={`${stats.totalResults} mock vectors`}
            active
          />
          <StatusIndicator
            icon={<Activity size={13} />}
            label={stats.queryCount ? `${stats.queryCount} queries` : "standby"}
            pulse
          />
          <div className="clock">{clock}</div>
        </div>
      </header>

      <div
        className={`workspace-grid ${
          isAsideCollapsed ? "aside-collapsed" : ""
        }`}
      >
        <section className="viewport-panel">
          {leftPanel}
        </section>
        <button
          type="button"
          className="aside-edge-toggle"
          onClick={() => setIsAsideCollapsed((collapsed) => !collapsed)}
          aria-label={
            isAsideCollapsed
              ? "Show retrieval panel"
              : "Hide retrieval panel"
          }
          title={
            isAsideCollapsed
              ? "Show retrieval panel"
              : "Hide retrieval panel"
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

function StatusIndicator({
  icon,
  label,
  active,
  pulse,
}: {
  icon: ReactNode;
  label: string;
  active?: boolean;
  pulse?: boolean;
}) {
  return (
    <div
      className={`status-pill ${active ? "is-active" : ""} ${
        pulse ? "is-pulse" : ""
      }`}
    >
      {icon}
      <span>{label}</span>
    </div>
  );
}

export default SplitLayout;
