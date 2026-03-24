"""
Swarm TUI Dashboard — Multi-worker monitoring with scrollable logs.

Features:
- Left panel: Overview of all workers (summary table)
- Right panel: Per-worker log panels with scrollable view
- Keyboard navigation: up/down to scroll through workers' logs
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Dict, Any

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn


class SwarmLogHandler(logging.Handler):
    """Captures logs per worker for TUI display."""

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
    """
    Multi-worker TUI dashboard with scrollable per-worker logs.
    
    Layout:
    ┌─ Header ──────────────────────────────────────────────┐
    │ SWARM DASHBOARD — N Workers × M DP each               │
    ├─ Left ───────────────┬─ Right ───────────────────────┤
    │ Worker Summary Table │ Per-Worker Logs (scrollable)  │
    │ - Worker ID          │ ┌─ Worker 0 Logs ─┐          │
    │ - Status (IDLE/BUSY) │ │ ...               │          │
    │ - Total Queued       │ ├─ Worker 1 Logs ─┤          │
    │ - Total Active       │ │ ...               │          │
    │ - Iteration          │ └─ ... ───────────┘          │
    └──────────────────────┴───────────────────────────────┘
    """

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

        # Per-worker state tracking
        self._worker_iterations: Dict[int, int] = {i: 0 for i in range(n_workers)}
        self._total_iterations = 0
        
        # Log scroll offset (which logs to show for each worker)
        self._log_offset = 0
        self._logs_per_worker = 8  # Number of log lines per worker panel

        # Setup log handler
        self.log_handler = SwarmLogHandler(max_logs=500)
        self.log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        
        # Intercept all relevant loggers
        for logger_name in ("prefill_swarm", "prefill_worker", "uvicorn", "uvicorn.error"):
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
            f" — Ports {self.base_port}-{self.base_port + self.n_workers * self.port_step - 1}",
            style="dim",
        )

        return Panel(
            Align.center(header_text),
            style="cyan on black",
            padding=(0, 1),
        )

    def _make_worker_summary(self) -> Panel:
        """Create the worker summary table (left panel)."""
        table = Table(
            title="Worker Overview",
            title_style="bold magenta",
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
        )

        table.add_column("Worker", style="bold", width=8)
        table.add_column("Status", width=10)
        table.add_column("Ports", width=20)
        table.add_column("Iteration", justify="right", width=10)
        table.add_column("Load", width=12)

        for w in range(self.n_workers):
            worker_port_start = self.base_port + w * self.port_step
            worker_port_end = worker_port_start + self.dp_per_worker - 1
            ports = f"{worker_port_start}-{worker_port_end}"
            iteration = self._worker_iterations.get(w, 0)
            
            # Placeholder status - in real implementation would query workers
            status_text = "IDLE 💤"
            status_style = "green"
            
            # Simple load bar
            load_bar = "░" * 10

            table.add_row(
                f"W{w}",
                f"[{status_style}]{status_text}[/{status_style}]",
                ports,
                str(iteration),
                f"[dim]{load_bar}[/dim]",
            )

        return Panel(table, border_style="dim")

    def _make_worker_log_panel(self, worker_id: int, logs: List[str]) -> Panel:
        """Create a single worker's log panel."""
        log_text = Text()
        
        # Get logs for this worker (filter by worker ID in log message)
        worker_logs = [
            log for log in logs 
            if f"Worker {worker_id}" in log or f"worker {worker_id}" in log
        ]
        
        # Show last N logs with scrolling support
        display_logs = worker_logs[-self._logs_per_worker:]
        
        for i, log_entry in enumerate(display_logs):
            # Truncate long lines
            if len(log_entry) > 80:
                log_entry = log_entry[:77] + "..."
            log_text.append(Text.from_ansi(log_entry))
            if i < len(display_logs) - 1:
                log_text.append("\n")
        
        # Fill empty lines to maintain height
        for _ in range(self._logs_per_worker - len(display_logs)):
            log_text.append("\n")

        return Panel(
            log_text,
            title=f"[bold cyan]Worker {worker_id}[/bold cyan]",
            border_style="dim",
            padding=(0, 1),
            height=self._logs_per_worker + 2,  # Fixed height
        )

    def _make_logs_column(self) -> Panel:
        """Create the right column with all worker log panels."""
        all_logs = self.log_handler.get_logs(100)
        
        # Stack worker log panels vertically
        panels = []
        for w in range(self.n_workers):
            panel = self._make_worker_log_panel(w, all_logs)
            panels.append(panel)
        
        # Use Group to stack panels
        from rich.console import Group
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
            Layout(name="right", ratio=2),  # Right side gets more space for logs
        )
        
        return layout

    def _generate_layout(self) -> Layout:
        """Generate the complete layout."""
        layout = self._make_layout()
        layout["header"].update(self._make_header())
        layout["left"].update(self._make_worker_summary())
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
        for logger_name in ("prefill_swarm", "prefill_worker", "uvicorn", "uvicorn.error"):
            logger = logging.getLogger(logger_name)
            logger.removeHandler(self.log_handler)
            logger.propagate = True

        self.console.show_cursor()
        logger = logging.getLogger("prefill_swarm")
        logger.info("Swarm Dashboard stopped")
