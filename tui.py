"""
TUI Dashboard for Prefill Worker.

Real-time terminal dashboard for monitoring EP-group scheduler status.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from rich.align import Align

# Rich imports
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from prefill_worker import EPGroupScheduler


class TUILogHandler(logging.Handler):
    """Custom log handler that captures logs for TUI display."""

    def __init__(self, max_logs: int = 100):
        super().__init__()
        self.logs: list[str] = []
        self.max_logs = max_logs
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Capture log record."""
        try:
            log_entry = self.format(record)
            with self._lock:
                self.logs.append(log_entry)
                if len(self.logs) > self.max_logs:
                    self.logs.pop(0)
        except Exception:
            self.handleError(record)

    def get_logs(self, count: int = 50) -> list[str]:
        """Get the last N log entries."""
        with self._lock:
            return self.logs[-count:] if count < len(self.logs) else self.logs.copy()


class PrefillDashboard:
    """
    Real-time TUI dashboard for monitoring EP-group scheduler status.
    Left panel: DP load status, Right panel: Application logs.
    """

    def __init__(
        self,
        scheduler: EPGroupScheduler,
        base_port: int,
        refresh_rate: float = 0.5,
    ) -> None:
        self.scheduler = scheduler
        self.base_port = base_port
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False
        self._thread: threading.Thread | None = None

        # Track statistics
        self._total_requests = 0
        self._total_iterations = 0
        self._last_iteration = 0

        # Setup log handler - replace console output with TUI capture
        self.log_handler = TUILogHandler(max_logs=200)
        self.log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        # Intercept prefill_server and uvicorn loggers to prevent TUI flickering
        for logger_name in ("prefill_server", "uvicorn", "uvicorn.error", "uvicorn.access"):
            logger = logging.getLogger(logger_name)
            # Remove existing handlers to prevent console output
            logger.handlers = []
            logger.addHandler(self.log_handler)
            # Prevent propagation to root logger (which may have console handler)
            logger.propagate = False

    def _make_header(self) -> Panel:
        """Create the header panel."""
        header_text = Text()
        header_text.append("PREFILL WORKER DASHBOARD", style="bold cyan")
        header_text.append(
            f" — DP {self.scheduler.n_dp} Workers",
            style="white",
        )
        header_text.append(
            f" — Ports {self.base_port}-{self.base_port + self.scheduler.n_dp - 1}",
            style="dim",
        )

        return Panel(
            Align.center(header_text),
            style="cyan on black",
            padding=(0, 1),
        )

    def _make_group_status(self, status: dict) -> Panel:
        """Create the EP-group status panel."""
        busy = status["busy"]
        iteration = status["iteration"]

        # Update iteration counter
        if iteration > self._last_iteration:
            self._total_iterations += iteration - self._last_iteration
            self._last_iteration = iteration

        status_text = Text()
        status_text.append("Status: ", style="dim")
        if busy:
            status_text.append("BUSY 🔥", style="bold red")
        else:
            status_text.append("IDLE 💤", style="bold green")

        status_text.append("  |  ", style="dim")
        status_text.append(f"Iteration: {iteration}", style="yellow")
        status_text.append("  |  ", style="dim")
        status_text.append(f"Total Iters: {self._total_iterations}", style="cyan")

        return Panel(
            Align.center(status_text),
            style="white on black",
            padding=(0, 2),
        )

    def _make_dp_table(self, status: dict) -> Table:
        """Create the DP status table."""
        table = Table(
            title="Data Parallel Workers",
            title_style="bold magenta",
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
            show_header=True,
            show_lines=False,
        )

        table.add_column("DP", style="bold", width=6)
        table.add_column("Queue", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Batch", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Compute", justify="right", width=10)
        table.add_column("Load", width=16)

        for dp in status["dp_details"]:
            dp_id = dp["dp_id"]
            queue_depth = dp["queue_depth"]
            queue_tokens = dp["queue_tokens"]
            batch_size = dp["batch_size"]
            batch_tokens = dp["batch_tokens"]
            compute_time = dp["compute_time"]

            # Color code based on load
            if queue_depth == 0 and batch_size == 0:
                status_style = "dim"
                queue_emoji = "💤"
            elif queue_depth > 10:
                status_style = "red"
                queue_emoji = "🔥"
            elif queue_depth > 5:
                status_style = "yellow"
                queue_emoji = "⚠️"
            else:
                status_style = "green"
                queue_emoji = "✅"

            # Create load bar
            max_batch = status["max_batch_size"]
            batch_ratio = min(batch_size / max_batch, 1.0) if max_batch > 0 else 0
            bar_length = int(batch_ratio * 10)
            if batch_size > 0:
                bar = "█" * bar_length + "░" * (10 - bar_length)
                load_text = f"[{status_style}]{bar}[/{status_style}] {batch_size}/{max_batch}"
            else:
                load_text = "[dim]░░░░░░░░░░░[/dim]"

            table.add_row(
                f"[{status_style}]DP{dp_id} {queue_emoji}[/{status_style}]",
                f"[cyan]{queue_depth}[/cyan]",
                f"[dim]{queue_tokens}[/dim]",
                f"[green]{batch_size}[/green]" if batch_size > 0 else "[dim]0[/dim]",
                f"[dim]{batch_tokens}[/dim]",
                f"[yellow]{compute_time:.3f}s[/yellow]" if compute_time > 0 else "[dim]0.000s[/dim]",
                load_text,
            )

        return table

    def _make_summary(self, status: dict) -> Panel:
        """Create the summary panel."""
        dp_details = status["dp_details"]

        total_queued = sum(dp["queue_depth"] for dp in dp_details)
        total_processing = sum(dp["batch_size"] for dp in dp_details)
        total_queue_tokens = sum(dp["queue_tokens"] for dp in dp_details)
        total_batch_tokens = sum(dp["batch_tokens"] for dp in dp_details)
        max_compute = max(dp["compute_time"] for dp in dp_details)

        summary_text = Text()
        summary_text.append("Queued: ", style="dim")
        summary_text.append(f"{total_queued} req", style="cyan")
        summary_text.append(f" ({total_queue_tokens} tok)", style="dim")

        summary_text.append("  |  ", style="dim")
        summary_text.append("Active: ", style="dim")
        summary_text.append(f"{total_processing} req", style="green")
        summary_text.append(f" ({total_batch_tokens} tok)", style="dim")

        summary_text.append("  |  ", style="dim")
        summary_text.append("Max: ", style="dim")
        summary_text.append(f"{max_compute:.3f}s", style="yellow")

        return Panel(
            Align.center(summary_text),
            style="white on black",
            padding=(0, 2),
        )

    def _make_log_panel(self) -> Panel:
        """Create the log viewer panel."""
        logs = self.log_handler.get_logs(50)

        if not logs:
            log_text = Text("No logs yet...", style="dim")
        else:
            log_text = Text()
            for log_entry in logs:
                # Use from_ansi to parse Rich's color codes properly
                log_text.append(Text.from_ansi(log_entry))
                log_text.append("\n")

        log_panel = Panel(
            log_text,
            title=Text("Application Logs", style="bold cyan"),
            border_style="dim",
            padding=(0, 1),
        )

        return log_panel

    def _make_left_panel(self, status: dict) -> Layout:
        """Create the left panel with DP status."""
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=3),
            Layout(name="table", ratio=1),
            Layout(name="summary", size=3),
        )

        layout["status"].update(self._make_group_status(status))
        layout["table"].update(self._make_dp_table(status))
        layout["summary"].update(self._make_summary(status))

        return layout

    def _make_layout(self) -> Layout:
        """Create the main split layout (left: DP status, right: logs)."""
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
        """Generate the complete layout for Live display."""
        status = self.scheduler.get_detailed_status()

        layout = self._make_layout()
        layout["header"].update(self._make_header())
        layout["left"].update(self._make_left_panel(status))
        layout["right"].update(self._make_log_panel())

        return layout

    def _run_loop(self) -> None:
        """Main TUI loop using Rich Live for smooth updates."""
        self._running = True

        # Use Rich Live for flicker-free updates with alternate screen buffer
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
        """Start the TUI dashboard in a background thread."""
        logger = logging.getLogger("prefill_server")

        if self._running:
            logger.warning("Dashboard already running")
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="tui-dashboard",
        )
        self._thread.start()
        logger.info("TUI Dashboard started on %s", self.console)

    def stop(self) -> None:
        """Stop the TUI dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        # Restore logging: remove TUI handler and re-enable propagation
        logger = logging.getLogger("prefill_server")
        logger.removeHandler(self.log_handler)
        logger.propagate = True

        # Show cursor and clear
        self.console.show_cursor()
        logger.info("TUI Dashboard stopped")
