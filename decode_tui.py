"""
decode_tui.py  —  TUI dashboard for the Decode EP-Group simulator

Layout
──────
┌─ Header ──────────────────────────────────────────────────────────────────┐
│  DECODE EP-GROUP  •  N DP Workers  •  Step=K  •  Batch cap=M             │
├─ DP Status (left, 60%) ──────────────────────┬─ Logs (right, 40%) ───────┤
│ ┌─ DP 0 ──────────────────────────────────┐  │  12:34:01 INFO ...        │
│ │ Active 3/8  Waiting 1  Step 0.0023s     │  │  12:34:01 INFO ...        │
│ │ req_id  cur_len  gen/max  progress       │  │  ...                      │
│ │ 1b60b0d6  45      12/100  ████░░  12%   │  │                           │
│ │ ...                                     │  │                           │
│ └─────────────────────────────────────────┘  │                           │
│ ┌─ DP 1 ──────────────────────────────────┐  │                           │
│ │ ...                                     │  │                           │
│ └─────────────────────────────────────────┘  │                           │
└──────────────────────────────────────────────┴───────────────────────────┘
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── log capture ───────────────────────────────────────────────────────────────


class _LogCapture(logging.Handler):
    """Captures log records into an in-memory ring buffer."""

    def __init__(self, max_lines: int = 300) -> None:
        super().__init__()
        self._lines: List[str] = []
        self._max = max_lines
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self._lock:
                self._lines.append(line)
                if len(self._lines) > self._max:
                    self._lines.pop(0)
        except Exception:
            self.handleError(record)

    def snapshot(self, n: int = 40) -> List[str]:
        with self._lock:
            return self._lines[-n:] if len(self._lines) > n else list(self._lines)


# ── dashboard ─────────────────────────────────────────────────────────────────


class DecodeDashboard:
    """
    TUI dashboard for the Decode EP-Group simulator.

    Reads scheduler state via get_detailed_status() on every refresh;
    no shared mutable state between the render thread and the scheduler thread.
    """

    def __init__(
        self,
        scheduler,  # DecodeScheduler instance
        base_port: int,
        refresh_rate: float = 0.3,
    ) -> None:
        self._scheduler = scheduler
        self._base_port = base_port
        self._refresh_rate = refresh_rate
        self._console = Console()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # ── log capture ───────────────────────────────────────────────────────
        self._log_capture = _LogCapture(max_lines=300)
        self._log_capture.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        for name in ("decode_server", "uvicorn", "uvicorn.error"):
            lg = logging.getLogger(name)
            lg.handlers = []
            lg.addHandler(self._log_capture)
            lg.propagate = False

    # ── rendering ─────────────────────────────────────────────────────────────

    def _make_header(self, status: dict) -> Panel:
        t = Text()
        t.append("DECODE EP-GROUP", style="bold green")
        t.append(f"  •  {status['n_dp']} DP Workers", style="white")
        t.append(f"  •  Step={status['step']}", style="cyan")
        t.append(f"  •  Batch cap={status['max_batch_size']}", style="dim")
        t.append(f"  •  Max seq={status['max_seq_len']}", style="dim")
        if status["busy"]:
            t.append("  [RUNNING]", style="bold yellow")
        else:
            t.append("  [IDLE]", style="bold dim")
        return Panel(Align.center(t), style="green on black", padding=(0, 1))

    def _make_dp_panel(self, dp: dict) -> Panel:
        dp_id = dp["dp_id"]
        active = dp["active_size"]
        waiting = dp["waiting_depth"]
        step_t = dp["step_time_s"]
        max_b = self._scheduler.cfg.max_batch_size
        port = self._base_port + dp_id

        # ── summary line ──────────────────────────────────────────────────────
        summary = Text()
        fill_pct = active / max_b if max_b > 0 else 0
        bar_len = 12
        filled = int(fill_pct * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        if fill_pct >= 0.9:
            bar_style = "bold red"
        elif fill_pct >= 0.5:
            bar_style = "yellow"
        else:
            bar_style = "green"

        summary.append(f"[{bar_style}]{bar}[/{bar_style}]")
        summary.append(f" {active}/{max_b} active")
        if waiting > 0:
            summary.append(f"  ⏳ {waiting} waiting", style="cyan")
        summary.append(f"  step={step_t * 1000:.2f}ms", style="dim")
        summary.append(f"  :{port}", style="dim")

        if not dp["requests"]:
            content = Group(summary, Text("\n  (empty batch)", style="dim italic"))
            title_style = "dim"
        else:
            # ── per-request table ─────────────────────────────────────────────
            tbl = Table(
                show_header=True,
                header_style="bold cyan",
                border_style="dim",
                padding=(0, 1),
                show_lines=False,
                expand=True,
            )
            tbl.add_column("req_id", style="dim", width=10)
            tbl.add_column("cur_len", justify="right", width=8)
            tbl.add_column("gen/max", justify="right", width=10)
            tbl.add_column("progress", width=22)
            tbl.add_column("%", justify="right", width=5)

            for r in dp["requests"]:
                pct = r["progress_pct"]
                bar_w = 16
                filled_b = int(pct / 100 * bar_w)
                prog_bar = "█" * filled_b + "░" * (bar_w - filled_b)

                if pct >= 80:
                    prog_style = "bold red"
                elif pct >= 50:
                    prog_style = "yellow"
                else:
                    prog_style = "green"

                tbl.add_row(
                    r["req_id"],
                    str(r["current_len"]),
                    f"{r['generated']}/{r['max_tokens']}",
                    f"[{prog_style}]{prog_bar}[/{prog_style}]",
                    f"{pct:.0f}%",
                )

            content = Group(summary, Text(""), tbl)
            title_style = "bold green"

        return Panel(
            content,
            title=f"[{title_style}]DP {dp_id}[/{title_style}]",
            border_style="dim",
            padding=(0, 1),
        )

    def _make_status_panel(self, status: dict) -> Panel:
        panels = [self._make_dp_panel(dp) for dp in status["dp_details"]]
        return Panel(
            Group(*panels),
            title="[bold green]DP Workers[/bold green]",
            border_style="green",
            padding=(0, 0),
        )

    def _make_logs_panel(self) -> Panel:
        lines = self._log_capture.snapshot(n=35)
        t = Text()
        for i, line in enumerate(lines):
            truncated = line if len(line) <= 85 else line[:82] + "…"
            t.append(Text.from_ansi(truncated))
            if i < len(lines) - 1:
                t.append("\n")
        return Panel(
            t,
            title="[bold magenta]Logs[/bold magenta]",
            border_style="dim",
            padding=(0, 1),
        )

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="status", ratio=3),
            Layout(name="logs", ratio=2),
        )
        return layout

    def _render(self) -> Layout:
        status = self._scheduler.get_detailed_status()
        layout = self._make_layout()
        layout["header"].update(self._make_header(status))
        layout["status"].update(self._make_status_panel(status))
        layout["logs"].update(self._make_logs_panel())
        return layout

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._running = True
        with Live(
            self._render(),
            console=self._console,
            refresh_per_second=max(1, int(1 / self._refresh_rate)),
            screen=True,
            transient=False,
        ) as live:
            while self._running:
                live.update(self._render())
                time.sleep(self._refresh_rate)

    def start(self) -> None:
        if self._running:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="decode-tui",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        for name in ("decode_server", "uvicorn", "uvicorn.error"):
            lg = logging.getLogger(name)
            lg.removeHandler(self._log_capture)
            lg.propagate = True
        self._console.show_cursor()
