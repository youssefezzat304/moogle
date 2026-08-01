import {
  Boxes,
  Clock,
  Cpu,
  Database,
  Gauge,
  Grid3X3,
  Image,
  Layers,
  List,
  Search,
  SlidersHorizontal,
  Target,
  Type,
  User,
  X,
} from "lucide-react";
import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";

type ArchitectureTab = "architecture" | "transformer" | "training";

interface ModelArchitectureDialogProps {
  modelId: string;
  onClose: () => void;
}

interface DiagramNodeProps {
  icon: ReactNode;
  eyebrow: string;
  title: string;
  detail: string;
  accent?: "cyan" | "gold" | "rose";
  step?: number;
  className?: string;
}

const MODEL_LABELS: Record<string, string> = {
  bpe_geo: "BPE-GEO",
};

function DiagramNode({
  icon,
  eyebrow,
  title,
  detail,
  accent = "cyan",
  step,
  className = "",
}: DiagramNodeProps) {
  return (
    <article className={`architecture-node ${accent} ${className}`}>
      {step !== undefined && (
        <span className="architecture-step" aria-hidden="true">
          {step}
        </span>
      )}
      <span className="architecture-node-icon" aria-hidden="true">
        {icon}
      </span>
      <span>{eyebrow}</span>
      <strong>{title}</strong>
      <small>{detail}</small>
    </article>
  );
}

function FlowConnector({ label }: { label?: string }) {
  return (
    <div className="architecture-connector" aria-hidden="true">
      {label && <span>{label}</span>}
      <i />
    </div>
  );
}

function ArchitectureOverview() {
  const overviewRef = useRef<HTMLDivElement>(null);
  const connectorMarkerId = useId().replace(/:/g, "");
  const [connector, setConnector] = useState<{
    width: number;
    height: number;
    path: string;
  } | null>(null);

  useEffect(() => {
    const overview = overviewRef.current;
    if (!overview) return;

    const similarityNode = overview.querySelector<HTMLElement>(
      ".similarity-search-node",
    );
    const vectorIndexNode =
      overview.querySelector<HTMLElement>(".vector-index-node");
    if (!similarityNode || !vectorIndexNode) return;
    const measuredOverview: HTMLDivElement = overview;
    const measuredSimilarityNode: HTMLElement = similarityNode;
    const measuredVectorIndexNode: HTMLElement = vectorIndexNode;

    function updateConnector() {
      const overviewBounds = measuredOverview.getBoundingClientRect();
      const similarityBounds = measuredSimilarityNode.getBoundingClientRect();
      const vectorBounds = measuredVectorIndexNode.getBoundingClientRect();
      const startX =
        vectorBounds.left + vectorBounds.width / 2 - overviewBounds.left;
      const startY = vectorBounds.top - overviewBounds.top;
      const endX =
        similarityBounds.left +
        similarityBounds.width / 2 -
        overviewBounds.left;
      const endY = similarityBounds.bottom - overviewBounds.top;
      const midpointY = endY + (startY - endY) / 2;

      setConnector({
        width: overviewBounds.width,
        height: overviewBounds.height,
        path: `M ${startX} ${startY} C ${startX} ${midpointY}, ${endX} ${midpointY}, ${endX} ${endY}`,
      });
    }

    const resizeObserver = new ResizeObserver(updateConnector);
    resizeObserver.observe(overview);
    resizeObserver.observe(similarityNode);
    resizeObserver.observe(vectorIndexNode);
    const scrollContainers = overview.querySelectorAll(
      ".architecture-flow-scroll",
    );
    scrollContainers.forEach((container) =>
      container.addEventListener("scroll", updateConnector, { passive: true }),
    );
    window.addEventListener("resize", updateConnector);
    updateConnector();

    return () => {
      resizeObserver.disconnect();
      scrollContainers.forEach((container) =>
        container.removeEventListener("scroll", updateConnector),
      );
      window.removeEventListener("resize", updateConnector);
    };
  }, []);

  return (
    <div ref={overviewRef} className="architecture-overview">
      {connector && (
        <svg
          className="architecture-index-connector"
          viewBox={`0 0 ${connector.width} ${connector.height}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <marker
              id={connectorMarkerId}
              markerWidth="8"
              markerHeight="8"
              refX="6"
              refY="4"
              orient="auto"
            >
              <path d="M 0 0 L 8 4 L 0 8 Z" />
            </marker>
          </defs>
          <path d={connector.path} markerEnd={`url(#${connectorMarkerId})`} />
        </svg>
      )}
      <div className="architecture-section-heading">
        <div>
          <span className="eyebrow">Online retrieval path</span>
          <h3>From natural-language query to ranked lunar patches</h3>
        </div>
        <span className="architecture-badge">live</span>
      </div>

      <div className="architecture-flow-scroll">
        <div className="architecture-flow query-flow">
          <DiagramNode
            icon={<User size={19} />}
            eyebrow="Input"
            title="User query"
            detail="A natural-language terrain description"
            step={1}
          />
          <FlowConnector label="text" />
          <DiagramNode
            icon={<Type size={19} />}
            eyebrow="Tokenize"
            title="BPE tokenizer"
            detail="Retrieval token + up to 384 tokens"
            step={2}
          />
          <FlowConnector />
          <DiagramNode
            icon={<Cpu size={19} />}
            eyebrow="Encode"
            title="Text transformer"
            detail="8 layers · 8 heads · 512 hidden"
            step={3}
          />
          <FlowConnector />
          <DiagramNode
            icon={<Layers size={19} />}
            eyebrow="Align"
            title="Text projection"
            detail="Linear map + L2 normalization"
            step={4}
          />
          <FlowConnector label="512-D" />
          <DiagramNode
            icon={<Search size={19} />}
            eyebrow="Compare"
            title="Similarity search"
            detail="Query vector against the image index"
            accent="gold"
            step={5}
            className="similarity-search-node"
          />
          <FlowConnector label="top-k" />
          <DiagramNode
            icon={<List size={19} />}
            eyebrow="Output"
            title="Ranked results"
            detail="Most similar patches, scores, and locations"
            accent="rose"
            step={6}
          />
        </div>
      </div>

      <div className="architecture-index-bridge" aria-hidden="true">
        <span>indexed image embeddings</span>
      </div>

      <div className="architecture-section-heading index-heading">
        <div>
          <span className="eyebrow">Offline indexing path</span>
          <h3>How searchable image embeddings are prepared</h3>
        </div>
        <span className="architecture-badge muted">precomputed</span>
      </div>

      <div className="architecture-flow-scroll">
        <div className="architecture-flow index-flow">
          <DiagramNode
            icon={<Image size={19} />}
            eyebrow="Source"
            title="Lunar geomaps"
            detail="512 × 512 RGB map patches"
            accent="rose"
          />
          <FlowConnector label="RGB" />
          <DiagramNode
            icon={<Grid3X3 size={19} />}
            eyebrow="Patchify"
            title="16 × 16 patches"
            detail="32 × 32 grid · 1,024 visual tokens"
          />
          <FlowConnector />
          <DiagramNode
            icon={<Cpu size={19} />}
            eyebrow="Encode"
            title="GEO transformer"
            detail="6 layers · 8 heads · 512 hidden"
          />
          <FlowConnector />
          <DiagramNode
            icon={<Layers size={19} />}
            eyebrow="Align"
            title="Image projection"
            detail="Linear map + L2 normalization"
          />
          <FlowConnector label="512-D" />
          <DiagramNode
            icon={<Database size={19} />}
            eyebrow="Store"
            title="Vector index"
            detail="Embedding paired with patch metadata"
            accent="gold"
            className="vector-index-node"
          />
        </div>
      </div>
    </div>
  );
}

interface EncoderStageProps {
  title: string;
  detail: string;
  emphasis?: boolean;
}

function EncoderStage({ title, detail, emphasis = false }: EncoderStageProps) {
  return (
    <div className={`encoder-stage ${emphasis ? "emphasis" : ""}`}>
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function StageArrow() {
  return <div className="encoder-stage-arrow" aria-hidden="true" />;
}

function TransformerUnit({ ffnDim }: { ffnDim: number }) {
  return (
    <div className="transformer-unit">
      <div className="transformer-unit-label">
        <span>Repeated transformer unit</span>
        <small>pretrained encoder block</small>
      </div>
      <div className="transformer-unit-stack">
        <span>Multi-head self-attention · 8 heads</span>
        <i>Residual connection + layer normalization</i>
        <span>Feed-forward network · 512 → {ffnDim} → 512</span>
        <i>GELU · dropout · residual + layer normalization</i>
      </div>
    </div>
  );
}

function TransformerArchitecture() {
  return (
    <div className="encoder-comparison">
      <article className="encoder-card text-encoder-card">
        <header>
          <span className="encoder-card-icon" aria-hidden="true">
            <Type size={20} />
          </span>
          <div>
            <span className="eyebrow">Query tower</span>
            <h3>BPE text encoder</h3>
          </div>
          <span className="encoder-depth">8×</span>
        </header>

        <div
          className="token-preview text-token-preview"
          aria-label="Text token sequence"
        >
          <span>[RETRIEVAL]</span>
          <span>[SOS]</span>
          <span>terrain</span>
          <span>tokens…</span>
          <span>[EOS]</span>
        </div>

        <div className="encoder-pipeline">
          <EncoderStage
            title="Token + position embeddings"
            detail="8,596-token vocabulary · maximum sequence length 384"
          />
          <StageArrow />
          <TransformerUnit ffnDim={1024} />
          <StageArrow />
          <EncoderStage
            title="Retrieval-token pooling"
            detail="Take hidden state at sequence position 0"
            emphasis
          />
          <StageArrow />
          <EncoderStage
            title="Text projection + normalization"
            detail="512 hidden → 512-D unit-length embedding"
          />
        </div>
      </article>

      <article className="encoder-card vision-encoder-card">
        <header>
          <span className="encoder-card-icon" aria-hidden="true">
            <Image size={20} />
          </span>
          <div>
            <span className="eyebrow">Image tower</span>
            <h3>GEO vision encoder</h3>
          </div>
          <span className="encoder-depth">6×</span>
        </header>

        <div
          className="token-preview vision-token-preview"
          aria-label="Image patch token grid"
        >
          {Array.from({ length: 20 }, (_, index) => (
            <span key={index} />
          ))}
          <strong>512 × 512 geomap</strong>
        </div>

        <div className="encoder-pipeline">
          <EncoderStage
            title="Convolutional patch embedding"
            detail="16 × 16 non-overlapping patches → 32 × 32 token grid"
          />
          <StageArrow />
          <EncoderStage
            title="Position + retrieval tokens"
            detail="Resize learned positions to 32 × 32; prepend [RETRIEVAL]"
          />
          <StageArrow />
          <TransformerUnit ffnDim={2048} />
          <StageArrow />
          <EncoderStage
            title="Retrieval-token pooling"
            detail="Take hidden state at sequence position 0"
            emphasis
          />
          <StageArrow />
          <EncoderStage
            title="Image projection + normalization"
            detail="512 hidden → 512-D unit-length embedding"
          />
        </div>
      </article>
    </div>
  );
}

interface TrainingStatProps {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  accent?: "cyan" | "gold" | "rose";
}

function TrainingStat({
  icon,
  label,
  value,
  detail,
  accent = "cyan",
}: TrainingStatProps) {
  return (
    <article className={`training-stat ${accent}`}>
      <span className="training-stat-icon" aria-hidden="true">
        {icon}
      </span>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </article>
  );
}

function SpecRow({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="training-spec-row">
      <span>{label}</span>
      <div>
        <strong>{value}</strong>
        {detail && <small>{detail}</small>}
      </div>
    </div>
  );
}

interface RecallRowProps {
  label: string;
  recallAt1: string;
  recallAt5: string;
  recallAt10: string;
  tone?: "cyan" | "rose";
}

function RecallRow({
  label,
  recallAt1,
  recallAt5,
  recallAt10,
  tone = "cyan",
}: RecallRowProps) {
  return (
    <div className={`recall-row ${tone}`}>
      <strong>{label}</strong>
      <span>{recallAt1}</span>
      <span>{recallAt5}</span>
      <span>{recallAt10}</span>
    </div>
  );
}

function TrainingCard({
  icon,
  eyebrow,
  title,
  className = "",
  children,
}: {
  icon: ReactNode;
  eyebrow: string;
  title: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <article className={`training-card ${className}`}>
      <header>
        <span aria-hidden="true">{icon}</span>
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h3>{title}</h3>
        </div>
      </header>
      <div className="training-card-content">{children}</div>
    </article>
  );
}

function TrainingOverview() {
  return (
    <div className="training-overview">
      <div className="training-stat-grid">
        <TrainingStat
          icon={<Boxes size={20} />}
          label="Model size"
          value="41.66M"
          detail="total parameters"
        />
        <TrainingStat
          icon={<Database size={20} />}
          label="Lunar dataset"
          value="22,578"
          detail="unique image patches"
          accent="rose"
        />
        <TrainingStat
          icon={<Gauge size={20} />}
          label="Optimization"
          value="20 epochs"
          detail="16,940 training steps"
          accent="gold"
        />
        <TrainingStat
          icon={<Target size={20} />}
          label="Full mean recall"
          value="73.37%"
          detail="selected checkpoint score"
        />
      </div>

      <div className="training-detail-grid">
        <TrainingCard
          icon={<Cpu size={18} />}
          eyebrow="Capacity"
          title="Model specifications"
          className="model-spec-card"
        >
          <SpecRow label="Text tower" value="21.69M" detail="BPE · 8 layers" />
          <SpecRow
            label="Vision tower"
            value="19.44M"
            detail="GEO · 6 layers"
          />
          <SpecRow
            label="Projection heads"
            value="524K"
            detail="two 512 × 512 maps"
          />
          <SpecRow
            label="Embedding space"
            value="512-D"
            detail="L2 normalized"
          />
          <SpecRow label="Input patch" value="512 × 512" detail="RGB geomap" />
        </TrainingCard>

        <TrainingCard
          icon={<Clock size={18} />}
          eyebrow="Corpus & runtime"
          title="Dataset and timing"
          className="dataset-card"
        >
          <SpecRow
            label="Training split"
            value="20,320"
            detail="90% of image patches"
          />
          <SpecRow
            label="Evaluation split"
            value="2,258"
            detail="full-index retrieval"
          />
          <SpecRow
            label="Caption policy"
            value="2 per image"
            detail="v1.0 + v2.0 descriptions"
          />
          <SpecRow
            label="Training time"
            value="Not recorded"
            detail="absent from checkpoint"
          />
          <SpecRow
            label="Full evaluation"
            value="7m 42s"
            detail="measured runtime"
          />
        </TrainingCard>

        <TrainingCard
          icon={<SlidersHorizontal size={18} />}
          eyebrow="Training recipe"
          title="Hyperparameters"
          className="hyperparameter-card"
        >
          <div className="hyperparameter-grid">
            <SpecRow label="Optimizer" value="AdamW" />
            <SpecRow label="Batch size" value="24" />
            <SpecRow label="CLIP learning rate" value="1.5 × 10⁻⁴" />
            <SpecRow label="Text learning rate" value="1.5 × 10⁻⁵" />
            <SpecRow label="Weight decay" value="0.01" />
            <SpecRow label="Temperature" value="0.07" />
            <SpecRow label="Gradient clipping" value="1.0" />
            <SpecRow label="Precision" value="32-bit" />
            <SpecRow label="Accelerator" value="1 GPU" />
            <SpecRow label="Random seed" value="42" />
          </div>
        </TrainingCard>

        <TrainingCard
          icon={<Target size={18} />}
          eyebrow="Full-index evaluation"
          title="Retrieval performance"
          className="performance-card"
        >
          <p className="recall-explanation">
            Recall@k measures how often a correct match appears among the first
            k results across the complete evaluation index.
          </p>
          <div className="recall-table">
            <div className="recall-table-header">
              <span>Direction</span>
              <span>R@1</span>
              <span>R@5</span>
              <span>R@10</span>
            </div>
            <RecallRow
              label="Text → image"
              recallAt1="49.60%"
              recallAt5="79.63%"
              recallAt10="88.80%"
            />
            <RecallRow
              label="Image → text"
              recallAt1="49.51%"
              recallAt5="81.58%"
              recallAt10="91.10%"
              tone="rose"
            />
          </div>

          <div className="rank-summary">
            <div>
              <span>Median rank</span>
              <strong>2</strong>
              <small>both directions</small>
            </div>
            <div>
              <span>Mean text → image rank</span>
              <strong>4.95</strong>
              <small>lower is better</small>
            </div>
            <div>
              <span>Mean image → text rank</span>
              <strong>4.13</strong>
              <small>lower is better</small>
            </div>
          </div>

          <div className="caption-performance">
            <div className="caption-performance-header">
              <span>Text source breakdown</span>
              <span>R@1</span>
              <span>R@5</span>
              <span>R@10</span>
            </div>
            <RecallRow
              label="Caption v1.0"
              recallAt1="39.24%"
              recallAt5="70.68%"
              recallAt10="83.35%"
            />
            <RecallRow
              label="Caption v2.0"
              recallAt1="60.10%"
              recallAt5="88.66%"
              recallAt10="94.29%"
              tone="rose"
            />
          </div>
        </TrainingCard>
      </div>
    </div>
  );
}

function ModelArchitectureDialog({
  modelId,
  onClose,
}: ModelArchitectureDialogProps) {
  const [activeTab, setActiveTab] = useState<ArchitectureTab>("architecture");
  const titleId = useId();
  const descriptionId = useId();
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const modelLabel = MODEL_LABELS[modelId] ?? modelId;

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    function handleEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }

    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus();
    };
  }, [onClose]);

  const tabs: Array<{ id: ArchitectureTab; label: string }> = [
    { id: "architecture", label: "Architecture" },
    { id: "transformer", label: "Transformer unit" },
    { id: "training", label: "Training" },
  ];

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    const activeIndex = tabs.findIndex((tab) => tab.id === activeTab);
    let nextIndex = activeIndex;
    if (event.key === "ArrowRight") nextIndex = (activeIndex + 1) % tabs.length;
    if (event.key === "ArrowLeft") {
      nextIndex = (activeIndex - 1 + tabs.length) % tabs.length;
    }
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabs.length - 1;
    if (nextIndex === activeIndex) return;

    event.preventDefault();
    setActiveTab(tabs[nextIndex].id);
    tabRefs.current[nextIndex]?.focus();
  }

  return (
    <div
      className="model-architecture-backdrop"
      role="presentation"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="model-architecture-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <header className="model-architecture-header">
          <div>
            <span className="eyebrow">Selected model · {modelLabel}</span>
            <h2 id={titleId}>How lunar retrieval works</h2>
            <p id={descriptionId}>
              Follow the model from query encoding to the final ranked output.
            </p>
          </div>
          <div
            className="model-architecture-summary"
            aria-label="Model summary"
          >
            <span>
              Text <strong>BPE</strong>
            </span>
            <span>
              Vision <strong>GEO</strong>
            </span>
            <span>
              Space <strong>512-D</strong>
            </span>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="model-architecture-close"
            onClick={onClose}
            aria-label="Close model architecture"
          >
            <X size={18} />
          </button>
        </header>

        <div
          className="model-architecture-tabs"
          role="tablist"
          aria-label="Model architecture views"
        >
          {tabs.map((tab, index) => (
            <button
              key={tab.id}
              ref={(node) => {
                tabRefs.current[index] = node;
              }}
              type="button"
              role="tab"
              id={`${titleId}-${tab.id}-tab`}
              aria-selected={activeTab === tab.id}
              aria-controls={`${titleId}-${tab.id}-panel`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={handleTabKeyDown}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="model-architecture-content">
          <div
            role="tabpanel"
            id={`${titleId}-${activeTab}-panel`}
            aria-labelledby={`${titleId}-${activeTab}-tab`}
          >
            {activeTab === "architecture" && <ArchitectureOverview />}
            {activeTab === "transformer" && <TransformerArchitecture />}
            {activeTab === "training" && <TrainingOverview />}
          </div>
        </div>
      </section>
    </div>
  );
}

export default ModelArchitectureDialog;
