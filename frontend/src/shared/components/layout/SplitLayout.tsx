import { type ReactNode, useEffect, useState } from "react";
import useMediaQuery from "../../hooks/useMediaQuery";

interface SplitLayoutProps {
  leftPanel: ReactNode;
  rightPanel: ReactNode;
}

function SplitLayout({ leftPanel, rightPanel }: SplitLayoutProps) {
  const isMobile = useMediaQuery("(max-width: 845px)");
  const [clock, setClock] = useState("");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const tick = () => {
      const now = new Date();
      const utc = now.toUTCString().split(" ")[4];
      setClock(`UTC ${utc}`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  if (isMobile) {
    return (
      <main
        className="h-full w-full flex flex-col"
        style={{ background: "var(--color-bg)" }}
      >
        {rightPanel}
      </main>
    );
  }

  return (
    <main
      className="h-full w-full flex flex-col"
      style={{ background: "var(--color-bg)" }}
    >
      {/* ── Global top bar ── */}
      <header
        className="shrink-0 flex items-center justify-between px-5 h-9 border-b"
        style={{
          borderColor: "var(--color-border)",
          background: "var(--color-surface)",
        }}
      >
        {/* Left: system identifier */}
        <div className="flex items-center gap-3">
          <span
            className="text-[10px] tracking-[0.25em] uppercase font-mono"
            style={{ color: "var(--color-amber)" }}
          >
            MOOGLE
          </span>
          <span
            className="text-[10px] tracking-[0.15em] uppercase font-mono"
            style={{ color: "var(--color-muted)" }}
          >
            // Lunar Retrieval System
          </span>
        </div>

        {/* Center: status dots */}
        <div className="flex items-center gap-4">
          <StatusIndicator label="TELEMETRY" active />
          <StatusIndicator label="RENDER" active />
          <StatusIndicator label="INDEX" pulse />
        </div>

        {/* Right: clock */}
        <div
          className="text-[10px] tracking-[0.15em] font-mono tabular-nums"
          style={{ color: "var(--color-muted)" }}
        >
          {mounted ? clock : ""}
        </div>
      </header>

      {/* ── Main grid ── */}
      <div className="flex flex-1 min-h-0">
        {/* Left: Moon canvas */}
        <section className="flex-1 relative" style={{ background: "#000" }}>
          {leftPanel}

          {/* Corner brackets — top-left */}
          <div className="absolute top-3 left-3 pointer-events-none">
            <Corner />
          </div>
          {/* Corner brackets — bottom-right */}
          <div
            className="absolute bottom-3 right-3 pointer-events-none"
            style={{ transform: "rotate(180deg)" }}
          >
            <Corner />
          </div>

          {/* Bottom label strip */}
          <div
            className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-1.5 pointer-events-none"
            style={{
              borderTop: "1px solid var(--color-border)",
              background: "rgba(2,4,10,0.72)",
              backdropFilter: "blur(6px)",
            }}
          >
            <span
              className="text-[9px] tracking-[0.2em] uppercase font-mono"
              style={{ color: "var(--color-muted)" }}
            >
              LROC · NAC · 0.5 m/px
            </span>
            <span
              className="text-[9px] tracking-[0.2em] uppercase font-mono"
              style={{ color: "var(--color-amber-dim)" }}
            >
              ◎ LIVE VIEW
            </span>
          </div>
        </section>

        {/* Divider */}
        <div
          className="w-px shrink-0"
          style={{
            background:
              "linear-gradient(to bottom, transparent, var(--color-amber-dim) 20%, var(--color-border) 80%, transparent)",
          }}
        />

        {/* Right: Chat panel */}
        <aside
          className="flex flex-col shrink-0"
          style={{
            width: "420px",
            background: "var(--color-surface)",
          }}
        >
          {rightPanel}
        </aside>
      </div>
    </main>
  );
}

function StatusIndicator({
  label,
  active,
  pulse,
}: {
  label: string;
  active?: boolean;
  pulse?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className={`w-1.5 h-1.5 rounded-full ${pulse ? "animate-pulse" : ""}`}
        style={{
          background:
            active || pulse ? "var(--color-green)" : "var(--color-muted)",
          boxShadow: active || pulse ? "0 0 4px var(--color-green)" : "none",
        }}
      />
      <span
        className="text-[9px] tracking-[0.15em] uppercase font-mono"
        style={{ color: "var(--color-muted)" }}
      >
        {label}
      </span>
    </div>
  );
}

function Corner() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path
        d="M0 16 L0 0 L16 0"
        stroke="var(--color-amber-dim)"
        strokeWidth="1"
        fill="none"
      />
    </svg>
  );
}

export default SplitLayout;
