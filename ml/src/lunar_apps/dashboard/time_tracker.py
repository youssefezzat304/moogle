import time
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


class TimeTracker:
    """
    Context-manager based timer for profiling pipeline stages.

    Usage
    -----
    tracker = TimeTracker()

    with tracker.track("data_loading"):
        dataset = load_dataset(...)

    for batch in dataloader:
        with tracker.track("batch", group="batches"):
            process(batch)

    tracker.report()                      # prints to stdout
    tracker.save_report("results/")       # writes txt report to disk
    """

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.start_wall: float = time.time()
        self.start_perf: float = time.perf_counter()

        # {label: [elapsed, ...]}  — one list entry per call
        self._records: dict[str, list[float]] = defaultdict(list)
        # group → set of labels that belong to it
        self._groups: dict[str, list[str]] = defaultdict(list)
        # labels recorded in insertion order (de-duped)
        self._label_order: list[str] = []

    # ------------------------------------------------------------------ #
    #  Context manager                                                   #
    # ------------------------------------------------------------------ #

    def track(self, label: str, group: Optional[str] = None):
        """Return a context manager that times the enclosed block."""
        return _TimedBlock(self, label, group)

    # ------------------------------------------------------------------ #
    #  Internal recording                                                #
    # ------------------------------------------------------------------ #

    def _record(self, label: str, elapsed: float, group: Optional[str]):
        if label not in self._records:
            self._label_order.append(label)
        self._records[label].append(elapsed)
        if group:
            if label not in self._groups[group]:
                self._groups[group].append(label)

    # ------------------------------------------------------------------ #
    #  Stats helpers                                                     #
    # ------------------------------------------------------------------ #

    def _stats(self, times: list[float]) -> dict:
        n = len(times)
        return {
            "count":   n,
            "total_s": round(sum(times), 4),
            "mean_s":  round(statistics.mean(times), 4),
            "min_s":   round(min(times), 4),
            "max_s":   round(max(times), 4),
            "std_s":   round(statistics.stdev(times), 4) if n > 1 else 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Report builders                                                   #
    # ------------------------------------------------------------------ #

    def _build_report_dict(self) -> dict:
        total_wall = time.perf_counter() - self.start_perf

        per_label: dict[str, dict] = {}
        for label in self._label_order:
            per_label[label] = self._stats(self._records[label])

        per_group: dict[str, dict] = {}
        for group, labels in self._groups.items():
            all_times = [t for lbl in labels for t in self._records[lbl]]
            per_group[group] = self._stats(all_times)
            per_group[group]["labels"] = labels

        return {
            "pipeline_name":       self.name,
            "started_at":          datetime.fromtimestamp(self.start_wall).isoformat(),
            "total_wall_time_s":   round(total_wall, 4),
            "stages":              per_label,
            "groups":              per_group,
        }

    def _format_report(self, data: dict) -> str:
        SEP  = "─" * 56
        SEP2 = "═" * 56
        lines = [
            SEP2,
            f"  TIMING REPORT — {data['pipeline_name']}",
            f"  Started : {data['started_at']}",
            f"  Total   : {data['total_wall_time_s']:.2f}s",
            SEP2,
        ]

        # ── Per-stage breakdown ──────────────────────────────────────── #
        lines.append("  STAGES")
        lines.append(SEP)
        header = f"  {'Label':<24} {'N':>4}  {'Total':>8}  {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}"
        lines.append(header)
        lines.append(SEP)

        for label, s in data["stages"].items():
            lines.append(
                f"  {label:<24} {s['count']:>4}  "
                f"{s['total_s']:>7.2f}s  {s['mean_s']:>7.2f}s  "
                f"{s['min_s']:>7.2f}s  {s['max_s']:>7.2f}s  "
                f"{s['std_s']:>7.2f}s"
            )

        # ── Per-group summary ────────────────────────────────────────── #
        if data["groups"]:
            lines.append("")
            lines.append("  GROUP SUMMARIES")
            lines.append(SEP)
            for group, s in data["groups"].items():
                pct = (s["total_s"] / data["total_wall_time_s"] * 100) if data["total_wall_time_s"] else 0
                lines.append(f"  [{group}]")
                lines.append(f"    Iterations : {s['count']}")
                lines.append(f"    Total time : {s['total_s']:.2f}s  ({pct:.1f}% of wall time)")
                lines.append(f"    Mean / iter: {s['mean_s']:.2f}s")
                lines.append(f"    Fastest    : {s['min_s']:.2f}s")
                lines.append(f"    Slowest    : {s['max_s']:.2f}s")
                lines.append(f"    Std dev    : {s['std_s']:.2f}s")
                lines.append(f"    Tracked via: {', '.join(s['labels'])}")

        lines.append(SEP2)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def report(self) -> None:
        """Print the timing report to stdout."""
        data = self._build_report_dict()
        print(self._format_report(data))

    def save_report(self, output_dir: str = ".", stem: str = "timing_report") -> Path:
        """
        Write the report to *output_dir* as plain text.

        Returns txt_path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        txt_path = out / f"{stem}.txt"
        txt_path.write_text(self._format_report(self._build_report_dict()))

        print(f"Timing report saved → {txt_path}")
        return txt_path


# ────────────────────────────────────────────────────────────────────────── #
#  Internal helper                                                           #
# ────────────────────────────────────────────────────────────────────────── #

class _TimedBlock:
    __slots__ = ("_tracker", "_label", "_group", "_t0")

    def __init__(self, tracker: TimeTracker, label: str, group: Optional[str]):
        self._tracker = tracker
        self._label   = label
        self._group   = group

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        self._tracker._record(self._label, elapsed, self._group)