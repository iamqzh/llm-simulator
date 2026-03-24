"""
Swarm TUI Dashboard — Multi-worker monitoring with per-DP detail.

Layout:
┌─ Header ──────────────────────────────────────────────────────────┐
│ SWARM DASHBOARD — N Workers × M DP each                           │
├─ Left ────────────────────────┬─ Right ───────────────────────────┤
│ All DP Status (All Workers)   │ Per-Worker Logs                   │
│ W0: DP0 DP1 DP2 DP3           │ ┌─ Worker 0 Logs ─┐               │
│ W1: DP0 DP1 DP2 DP3           │ │ ...               │               │
│ W2: DP0 DP1 DP2 DP3           │ ├─ Worker 1 Logs ─┤               │
│ ...                           │ │ ...               │               │
└───────────────────────────────┴───────────────────────────────────┘
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

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


@dataclass
class WorkerStatus:
    """Status for a single worker."""
    worker_id: int
    base_port: int
    iteration: int = 0
    dp_statuses: List[DPStatus] = field(default_factory=list)


class SwarmLogHandler(logging.Handler):
    """Captures logs per worker for TUI display."""

    def __init__(self, max_logs: int = 500):
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

    def get_logs(self, count: int = 100) -> List[str]:
        with self._lock:
            return self.logs[-count:] if count < len(self.logs) else self.logs.copy()

    def get_logs_for_worker(self, worker_id: int, count: int = 20) -> List[str]:
        """Get logs filtered by worker ID."""
        with self._lock:
            # Filter logs containing worker ID pattern like [prefill-worker-0]
            pattern = f"[prefill-worker-{worker_id}]"
            worker_logs = [
                log for log in self.logs
                if pattern in log
            ]
            return worker_logs[-count:] if count < len(worker_logs) else worker_logs


class SwarmDashboard:
    """Multi-worker TUI dashboard showing all DPs from all workers."""

    def __init__(
        self,
        n_workers: int,
        dp_per_worker: int,
        base_port: int,
        port_step: int,
        refresh_rate: float = 0.5,
    ) -> None:
        self.n_workers = n_workers
        self.dp_per_worker = dp_per_worker
        self.base_port = base_port
        self.port_step = port_step
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Log display config
        self._logs_per_worker = 6

        # Setup log handler
        self.log_handler = SwarmLogHandler(max_logs=1000)
        self.log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        # Intercept all loggers including per-worker loggers
        logger_names = [
            "prefill_swarm",
            "prefill_worker",
            "uvicorn",
            "uvicorn.error",
        ] + [f"prefill-worker-{w}" for w in range(n_workers)]
        
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            logger.handlers = []
            logger.addHandler(self.log_handler)
            logger.propagate = False

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

    def _parse_status_from_logs(self) -> Dict[int, WorkerStatus]:
        """Parse worker and DP status from captured logs."""
        # Initialize with empty statuses
        workers = {}
        for w in range(self.n_workers):
            dp_statuses = []
            for d in range(self.dp_per_worker):
                dp_statuses.append(DPStatus(
                    worker_id=w,
                    dp_id=d,
                    port=self.base_port + w * self.port_step + d,
                ))
            workers[w] = WorkerStatus(
                worker_id=w,
                base_port=self.base_port + w * self.port_step,
                dp_statuses=dp_statuses,
            )

        # Parse logs to extract status info
        all_logs = self.log_handler.get_logs(200)
        for log in all_logs:
            # Determine which worker this log belongs to
            worker_id = None
            for w in range(self.n_workers):
                if f"[prefill-worker-{w}]" in log:
                    worker_id = w
                    break
            
            if worker_id is None:
                continue

            # Parse iteration info
            if "iteration=" in log.lower():
                try:
                    import re
                    match = re.search(r'iteration=(\d+)', log, re.IGNORECASE)
                    if match:
                        workers[worker_id].iteration = int(match.group(1))
                except (ValueError, AttributeError):
                    pass

            # Parse DP status from ENQUEUE logs
            if "ENQUEUE" in log and "[DP" in log:
                try:
                    import re
                    dp_match = re.search(r'\[DP\s*(\d+)\]', log)
                    if dp_match:
                        dp_id = int(dp_match.group(1))
                        if 0 <= dp_id < self.dp_per_worker:
                            workers[worker_id].dp_statuses[dp_id].queue_depth += 1
                except (ValueError, AttributeError):
                    pass

        return workers

    def _make_dp_table(self) -> Panel:
        """Create the all-DP status table (left panel)."""
        table = Table(
            title="All DP Workers",
            title_style="bold magenta",
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
            show_header=True,
            show_lines=False,
        )

        # Columns: Worker | DP | Port | Queue | Batch | Compute | Status
        table.add_column("Worker", style="bold", width=7)
        table.add_column("DP", style="bold", width=4)
        table.add_column("Port", justify="right", width=6)
        table.add_column("Queue", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=7)
        table.add_column("Batch", justify="right", width=6)
        table.add_column("Compute", justify="right", width=9)
        table.add_column("Status", width=8)

        # Parse current status from logs
        workers = self._parse_status_from_logs()

        for w in range(self.n_workers):
            worker = workers.get(w, WorkerStatus(w, self.base_port + w * self.port_step))
            
            for dp in worker.dp_statuses:
                # Status indicator
                if dp.queue_depth > 0:
                    status = "[yellow]BUSY[/yellow]"
                else:
                    status = "[dim]IDLE[/dim]"

                table.add_row(
                    f"W{w}",
                    f"D{dp.dp_id}",
                    str(dp.port),
                    str(dp.queue_depth),
                    str(dp.queue_tokens) if dp.queue_tokens > 0 else "-",
                    str(dp.batch_size) if dp.batch_size > 0 else "-",
                    f"{dp.compute_time:.3f}s" if dp.compute_time > 0 else "-",
                    status,
                )

        return Panel(table, border_style="dim")

    def _make_worker_log_panel(self, worker_id: int) -> Panel:
        """Create a single worker's log panel."""
        log_text = Text()

        # Get logs for this worker
        worker_logs = self.log_handler.get_logs_for_worker(worker_id, self._logs_per_worker)

        for i, log_entry in enumerate(worker_logs):
            # Truncate long lines
            if len(log_entry) > 90:
                log_entry = log_entry[:87] + "..."
            # Remove the logger name prefix for cleaner display
            log_entry = self._clean_log_line(log_entry)
            log_text.append(Text.from_ansi(log_entry))
            if i < len(worker_logs) - 1:
                log_text.append("\n")

        # Fill empty lines to maintain height
        for _ in range(self._logs_per_worker - len(worker_logs)):
            log_text.append(" ")
            log_text.append("\n")

        return Panel(
            log_text,
            title=f"[bold cyan]Worker {worker_id}[/bold cyan]",
            border_style="dim",
            padding=(0, 1),
            height=self._logs_per_worker + 2,
        )

    def _clean_log_line(self, log_entry: str) -> str:
        """Clean up log line for display."""
        # Remove common prefixes to save space
        prefixes = [
            "prefill_worker",
            "prefill_swarm",
            "uvicorn",
        ]
        for prefix in prefixes:
            if prefix in log_entry.lower():
                # Keep only timestamp + level + message
                parts = log_entry.split(None, 2)  # Split into 3 parts
                if len(parts) >= 3:
                    return f"{parts[0]} {parts[1]:8} {parts[2]}"
        return log_entry

    def _make_logs_column(self) -> Panel:
        """Create the right column with all worker log panels."""
        # Stack worker log panels vertically
        panels = []
        for w in range(self.n_workers):
            panel = self._make_worker_log_panel(w)
            panels.append(panel)

        content = Group(*panels)

        return Panel(
            content,
            title="[bold magenta]Worker Logs[/bold magenta]",
            border_style="dim",
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
        layout["left"].update(self._make_dp_table())
        layout["right"].update(self._make_logs_column())
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
        logger_names = [
            "prefill_swarm",
            "prefill_worker",
            "uvicorn",
            "uvicorn.error",
        ] + [f"prefill-worker-{w}" for w in range(self.n_workers)]
        
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            logger.removeHandler(self.log_handler)
            logger.propagate = True

        self.console.show_cursor()
        logger = logging.getLogger("prefill_swarm")
        logger.info("Swarm Dashboard stopped")
