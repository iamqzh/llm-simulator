"""
Swarm TUI Dashboard — Multi-worker monitoring with per-DP detail.

Uses multiprocessing.Manager to share state between workers and dashboard.

Layout:
┌─ Header ──────────────────────────────────────────────────────────┐
│ SWARM DASHBOARD — N Workers × M DP each                           │
├─ Left ────────────────────────┬─ Right ───────────────────────────┤
│ All DP Status (All Workers)   │ Application Logs                   │
│ W0: DP0 DP1 DP2 DP3           │ ┌─ Recent Logs ──┐                │
│ W1: DP0 DP1 DP2 DP3           │ │ ...             │                │
│ W2: DP0 DP1 DP2 DP3           │ └─────────────────┘                │
└───────────────────────────────┴───────────────────────────────────┘
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class DPStatus:
    """Status for a single DP worker."""

    worker_id: int
    dp_id: int
    port: int
    queue_depth: int = 0
    queue_tokens: int = 0
    batch_size: int = 0
    batch_tokens: int = 0
    compute_time: float = 0.0
    busy: bool = False
    iteration: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DPStatus":
        return cls(**d)


@dataclass
class WorkerStatus:
    """Status for a single worker."""

    worker_id: int
    base_port: int
    iteration: int = 0
    dp_statuses: List[DPStatus] = field(default_factory=list)
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "base_port": self.base_port,
            "iteration": self.iteration,
            "dp_statuses": [dp.to_dict() for dp in self.dp_statuses],
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerStatus":
        dp_statuses = [DPStatus.from_dict(dp) for dp in d.get("dp_statuses", [])]
        return cls(
            worker_id=d["worker_id"],
            base_port=d["base_port"],
            iteration=d.get("iteration", 0),
            dp_statuses=dp_statuses,
            last_update=d.get("last_update", 0.0),
        )


class SwarmLogHandler(logging.Handler):
    """
    Captures logs from the local (manager) process for TUI display.
    Cross-process worker logs arrive via log_queue and are drained
    by a dedicated background thread, never inside the render path.
    """

    def __init__(self, max_logs: int = 200) -> None:
        super().__init__()
        # Use deque with maxlen for O(1) append and automatic size limit
        # Root cause: list.pop(0) is O(n) operation; deque.popleft() is O(1)
        self._logs: deque = deque(maxlen=max_logs)
        self._lock = threading.Lock()

    # ── called from any thread that holds a Python logging.Logger ────────────

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = self.format(record)
            self._append(log_entry)
        except Exception:
            self.handleError(record)

    def append_raw(self, text: str) -> None:
        """Append a pre-formatted string (used by the queue-drain thread)."""
        self._append(text)

    def _append(self, text: str) -> None:
        with self._lock:
            self._logs.append(text)  # deque maxlen handles overflow automatically

    # ── called from the render thread (read-only) ─────────────────────────────

    def snapshot(self, count: int = 50) -> List[str]:
        """Return a thread-safe snapshot of the last `count` log lines."""
        with self._lock:
            tail = list(self._logs)[-count:] if len(self._logs) > count else list(self._logs)
        return tail


class SwarmDashboard:
    """Multi-worker TUI dashboard using shared state."""

    def __init__(
        self,
        n_workers: int,
        dp_per_worker: int,
        base_port: int,
        port_step: int,
        refresh_rate: float = 0.5,
        shared_state=None,          # Manager.dict proxy — kept as-is, never converted
        log_queue=None,             # Manager.Queue proxy for cross-process logs
    ) -> None:
        self.n_workers = n_workers
        self.dp_per_worker = dp_per_worker
        self.base_port = base_port
        self.port_step = port_step
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Keep the Manager proxy directly — never snapshot it here.
        # The render loop reads it on every frame, so it always sees live data.
        self._shared_state = shared_state

        # ── log handler (local process) ───────────────────────────────────────
        self._log_handler = SwarmLogHandler(max_logs=200)
        self._log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        for logger_name in ("prefill_swarm", "uvicorn", "uvicorn.error"):
            logger = logging.getLogger(logger_name)
            logger.handlers = []
            logger.addHandler(self._log_handler)
            logger.propagate = False

        # ── queue-drain thread (cross-process logs) ───────────────────────────
        # Draining happens in its own thread so the render path stays read-only.
        self._log_queue = log_queue
        self._drain_thread: Optional[threading.Thread] = None

    # ── shared-state helpers ──────────────────────────────────────────────────

    def _get_worker_status(self, worker_id: int) -> WorkerStatus:
        """Read live worker status from the Manager proxy.

        Root cause: Manager.dict() single-key read is atomic via IPC proxy,
        but the returned dict may be from a stale update. The version field
        helps detect inconsistencies but is informational only for now.
        """
        key = f"worker_{worker_id}"
        try:
            if self._shared_state is not None and key in self._shared_state:
                # Single atomic read from Manager.dict proxy
                data = self._shared_state[key]
                # Validate the payload has expected fields
                if "worker_id" in data and "dp_statuses" in data:
                    return WorkerStatus.from_dict(data)
                else:
                    logging.getLogger("prefill_swarm").debug("Incomplete status payload for worker %d", worker_id)
        except KeyError:
            # Worker key not yet created - normal during startup
            pass
        except (ConnectionError, EOFError):
            # Manager connection issue - return empty status
            logging.getLogger("prefill_swarm").debug("Manager connection error for worker %d", worker_id)
        except Exception as e:
            logging.getLogger("prefill_swarm").debug("Unexpected error reading worker %d status: %s", worker_id, e)

        # Fallback: empty status while worker is still starting up
        dp_statuses = [
            DPStatus(
                worker_id=worker_id,
                dp_id=d,
                port=self.base_port + worker_id * self.port_step + d,
            )
            for d in range(self.dp_per_worker)
        ]
        return WorkerStatus(
            worker_id=worker_id,
            base_port=self.base_port + worker_id * self.port_step,
            dp_statuses=dp_statuses,
        )

    # ── rendering ─────────────────────────────────────────────────────────────

    def _make_header(self) -> Panel:
        header_text = Text()
        header_text.append("PREFILL WORKER SWARM", style="bold cyan")
        header_text.append(
            f" — {self.n_workers} Workers × {self.dp_per_worker} DP each",
            style="white",
        )
        header_text.append(
            f" — Total Ports: {self.n_workers * self.dp_per_worker}",
            style="dim",
        )
        return Panel(Align.center(header_text), style="cyan on black", padding=(0, 1))

    def _make_worker_dp_table(self, worker_id: int) -> Table:
        worker = self._get_worker_status(worker_id)

        table = Table(
            title=(
                f"Worker {worker_id} "
                f"(Ports {worker.base_port}-{worker.base_port + self.dp_per_worker - 1}) "
                f"Iter={worker.iteration}"
            ),
            title_style="bold magenta",
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
            show_header=True,
            show_lines=False,
        )

        table.add_column("DP", style="bold", width=6)
        table.add_column("Queue", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=7)
        table.add_column("Batch", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=7)
        table.add_column("Compute", justify="right", width=9)
        table.add_column("Load", width=10)

        for dp in worker.dp_statuses:
            if dp.queue_depth == 0 and dp.batch_size == 0:
                status_style, emoji = "dim", "💤"
            elif dp.queue_depth > 5:
                status_style, emoji = "red", "🔥"
            elif dp.queue_depth > 0:
                status_style, emoji = "yellow", "⚡"
            else:
                status_style, emoji = "green", "✅"

            if dp.batch_size > 0:
                bar_length = min(dp.batch_size, 8)
                bar = "█" * bar_length + "░" * (8 - bar_length)
                load_text = f"[{status_style}]{bar}[/{status_style}]"
            else:
                load_text = "[dim]░░░░░░░░░[/dim]"

            table.add_row(
                f"[{status_style}]DP{dp.dp_id} {emoji}[/{status_style}]",
                f"[cyan]{dp.queue_depth}[/cyan]" if dp.queue_depth > 0 else "[dim]0[/dim]",
                f"[dim]{dp.queue_tokens}[/dim]" if dp.queue_tokens > 0 else "[dim]0[/dim]",
                f"[green]{dp.batch_size}[/green]" if dp.batch_size > 0 else "[dim]0[/dim]",
                f"[dim]{dp.batch_tokens}[/dim]" if dp.batch_tokens > 0 else "[dim]0[/dim]",
                f"[yellow]{dp.compute_time:.3f}s[/yellow]" if dp.compute_time > 0 else "[dim]0.000s[/dim]",
                load_text,
            )

        return table

    def _make_left_panel(self) -> Panel:
        tables = [self._make_worker_dp_table(w) for w in range(self.n_workers)]
        return Panel(
            Group(*tables),
            title="[bold cyan]All Workers Status[/bold cyan]",
            border_style="dim",
        )

    def _make_logs_panel(self) -> Panel:
        """Render-only: reads a snapshot, never modifies state."""
        logs = self._log_handler.snapshot(count=20)

        log_text = Text()
        for i, entry in enumerate(logs):
            truncated = entry if len(entry) <= 90 else entry[:87] + "..."
            log_text.append(Text.from_ansi(truncated))
            if i < len(logs) - 1:
                log_text.append("\n")

        return Panel(
            log_text,
            title="[bold magenta]Application Logs[/bold magenta]",
            border_style="dim",
            padding=(0, 1),
        )

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        return layout

    def _generate_layout(self) -> Layout:
        layout = self._make_layout()
        layout["header"].update(self._make_header())
        layout["left"].update(self._make_left_panel())
        layout["right"].update(self._make_logs_panel())
        return layout

    # ── background threads ────────────────────────────────────────────────────

    def _drain_log_queue(self) -> None:
        """
        Dedicated thread: drains the cross-process log_queue and feeds
        entries into SwarmLogHandler.  Never touches the render path.
        """
        while self._running:
            try:
                # Block with a short timeout so the thread exits promptly
                # when self._running is set to False.
                entry = self._log_queue.get(timeout=0.1)
                self._log_handler.append_raw(entry)
            except queue.Empty:
                # Normal case: no logs available, just continue looping
                pass
            except (EOFError, ConnectionError):
                # Manager shutdown or connection issue - exit gracefully
                break
            except Exception as e:
                # Log unexpected errors but don't crash the drain thread
                logging.getLogger("prefill_swarm").debug("Unexpected error in log drain: %s", e)

    def _run_loop(self) -> None:
        """Main TUI render loop."""
        self._running = True

        # Start the queue-drain thread now that _running is True
        if self._log_queue is not None:
            self._drain_thread = threading.Thread(
                target=self._drain_log_queue,
                daemon=True,
                name="swarm-log-drain",
            )
            self._drain_thread.start()

        with Live(
            self._generate_layout(),
            console=self.console,
            refresh_per_second=int(1 / self.refresh_rate) + 1,
            screen=True,
            transient=False,
        ) as live:
            while self._running:
                live.update(self._generate_layout())
                time.sleep(self.refresh_rate)

        # Drain thread will exit on its own timeout loop once _running=False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            logging.getLogger("prefill_swarm").warning("Dashboard already running")
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="swarm-tui-dashboard",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

        if self._drain_thread:
            self._drain_thread.join(timeout=1)
            self._drain_thread = None

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        for logger_name in ("prefill_swarm", "uvicorn", "uvicorn.error"):
            logger = logging.getLogger(logger_name)
            logger.removeHandler(self._log_handler)
            logger.propagate = True

        self.console.show_cursor()
