"""
Swarm TUI Dashboard — Multi-worker monitoring with per-DP detail.

Uses multiprocessing.Manager to share state between workers and dashboard.

Layout:
┌─ Header ──────────────────────────────────────────────────────────┐
│ SWARM DASHBOARD — N Workers × M DP each                           │
├─ Left ────────────────────────┬─ Right ───────────────────────────┤
│ All DP Status (All Workers)   │ Application Logs                   │
│ W0: DP0 DP1 DP2 DP3           │ ┌─ Recent Logs ──┐                │
│ W1: DP0 DP1 DP2 DP3           │ │ ...              │                │
│ W2: DP0 DP1 DP2 DP3           │ └─────────────────┘                │
└───────────────────────────────┴───────────────────────────────────┘
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from multiprocessing import Manager

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout


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
    """Captures logs for TUI display."""

    def __init__(self, max_logs: int = 200):
        super().__init__()
        self.logs: List[str] = []
        self.max_logs = max_logs
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = self.format(record)
            with self._lock:
                self.logs.append(log_entry)
                if len(self.logs) > self.max_logs:
                    self.logs.pop(0)
        except Exception:
            self.handleError(record)

    def get_logs(self, count: int = 50) -> List[str]:
        with self._lock:
            return self.logs[-count:] if count < len(self.logs) else self.logs.copy()


class SwarmDashboard:
    """Multi-worker TUI dashboard using shared state."""

    def __init__(
        self,
        n_workers: int,
        dp_per_worker: int,
        base_port: int,
        port_step: int,
        refresh_rate: float = 0.5,
        shared_state: Optional[Dict] = None,
    ) -> None:
        self.n_workers = n_workers
        self.dp_per_worker = dp_per_worker
        self.base_port = base_port
        self.port_step = port_step
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Shared state from Manager
        self.shared_state = shared_state

        # Log display config
        self._logs_per_worker = 6

        # Setup log handler for local logs (manager + worker startup)
        self.log_handler = SwarmLogHandler(max_logs=200)
        self.log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        # Intercept local loggers
        for logger_name in ("prefill_swarm", "uvicorn", "uvicorn.error"):
            logger = logging.getLogger(logger_name)
            logger.handlers = []
            logger.addHandler(self.log_handler)
            logger.propagate = False

    def _get_worker_status(self, worker_id: int) -> WorkerStatus:
        """Get status for a worker from shared state or create default."""
        if self.shared_state and f"worker_{worker_id}" in self.shared_state:
            try:
                data = self.shared_state[f"worker_{worker_id}"]
                return WorkerStatus.from_dict(data)
            except (KeyError, TypeError):
                pass
        
        # Return default status
        dp_statuses = []
        for d in range(self.dp_per_worker):
            dp_statuses.append(DPStatus(
                worker_id=worker_id,
                dp_id=d,
                port=self.base_port + worker_id * self.port_step + d,
            ))
        return WorkerStatus(
            worker_id=worker_id,
            base_port=self.base_port + worker_id * self.port_step,
            dp_statuses=dp_statuses,
        )

    def _make_header(self) -> Panel:
        """Create the swarm header panel."""
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

        return Panel(
            Align.center(header_text),
            style="cyan on black",
            padding=(0, 1),
        )

    def _make_worker_dp_table(self, worker_id: int) -> Table:
        """Create a DP status table for a single worker."""
        worker = self._get_worker_status(worker_id)
        
        table = Table(
            title=f"Worker {worker_id} (Ports {worker.base_port}-{worker.base_port + self.dp_per_worker - 1}) Iter={worker.iteration}",
            title_style="bold magenta",
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
            show_header=True,
            show_lines=False,
        )

        # Columns: DP | Queue | Tokens | Batch | Tokens | Compute | Load
        table.add_column("DP", style="bold", width=6)
        table.add_column("Queue", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=7)
        table.add_column("Batch", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=7)
        table.add_column("Compute", justify="right", width=9)
        table.add_column("Load", width=10)

        for dp in worker.dp_statuses:
            # Color code based on load
            if dp.queue_depth == 0 and dp.batch_size == 0:
                status_style = "dim"
                emoji = "💤"
            elif dp.queue_depth > 5:
                status_style = "red"
                emoji = "🔥"
            elif dp.queue_depth > 0:
                status_style = "yellow"
                emoji = "⚡"
            else:
                status_style = "green"
                emoji = "✅"

            # Load bar
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
        """Create the left panel with per-worker DP tables stacked vertically."""
        # Create a table for each worker and stack them
        tables = []
        for w in range(self.n_workers):
            table = self._make_worker_dp_table(w)
            tables.append(table)

        content = Group(*tables)

        return Panel(
            content,
            title="[bold cyan]All Workers Status[/bold cyan]",
            border_style="dim",
        )

    def _make_logs_panel(self) -> Panel:
        """Create the logs panel."""
        logs = self.log_handler.get_logs(20)

        log_text = Text()
        for i, log_entry in enumerate(logs):
            if len(log_entry) > 90:
                log_entry = log_entry[:87] + "..."
            log_text.append(Text.from_ansi(log_entry))
            if i < len(logs) - 1:
                log_text.append("\n")

        return Panel(
            log_text,
            title="[bold magenta]Application Logs[/bold magenta]",
            border_style="dim",
            padding=(0, 1),
        )

    def _make_layout(self) -> Layout:
        """Create the main layout."""
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
        """Generate the complete layout."""
        layout = self._make_layout()
        layout["header"].update(self._make_header())
        layout["left"].update(self._make_left_panel())
        layout["right"].update(self._make_logs_panel())
        return layout

    def _run_loop(self) -> None:
        """Main TUI loop."""
        self._running = True

        with Live(
            self._generate_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
            transient=False,
        ) as live:
            while self._running:
                live.update(self._generate_layout())
                time.sleep(self.refresh_rate)

    def start(self) -> None:
        """Start the TUI dashboard."""
        logger = logging.getLogger("prefill_swarm")

        if self._running:
            logger.warning("Dashboard already running")
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="swarm-tui-dashboard",
        )
        self._thread.start()
        logger.info("Swarm Dashboard started on %s", self.console)

    def stop(self) -> None:
        """Stop the TUI dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        # Restore logging
        for logger_name in ("prefill_swarm", "uvicorn", "uvicorn.error"):
            logger = logging.getLogger(logger_name)
            logger.removeHandler(self.log_handler)
            logger.propagate = True

        self.console.show_cursor()
        logger = logging.getLogger("prefill_swarm")
        logger.info("Swarm Dashboard stopped")
