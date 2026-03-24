"""
prefill_server.py  —  Prefill EP-Group simulator server

One process = one EP Group = N DP workers sharing a single EP-sync Scheduler.

Architecture (matches the diagram)
────────────────────────────────────────────────────────────────────
  ┌─ One Server (this process) ─────────────────────────────────┐
  │                                                             │
  │  API-Server-0  :base_port+0  ──►  Queue-0  ─┐               │
  │  API-Server-1  :base_port+1  ──►  Queue-1  ─┤  Scheduler    │──► Execute (EP-sync)
  │  ...                                        │  (EP-group)   │
  │  API-Server-N  :base_port+N  ──►  Queue-N  ─┘               │
  └─────────────────────────────────────────────────────────────┘

EP-sync rule
────────────
  • Scheduler fires when ANY DP queue is non-empty AND the group is idle.
  • It snapshots ALL queues at once and packs a sub-batch per DP.
  • finish_time = now + max(compute_time_dp_i)   ← All-to-All barrier
  • While the group is BUSY, newly arrived requests enter the queues but
    cannot be picked up until the current iteration completes.
  • A DP with an empty queue contributes 0 compute time but still
    participates in the barrier (it just doesn't block the group).

Prefill compute model
──────────────────────
  t_dp_i = alpha * Σ(s_j²) + beta * Σ(s_j)    for requests on DP i
  group_duration = max(t_dp_i  for i in 0..N-1)

API  (OpenAI-compatible, on each DP port)
──────────────────────────────────────────
  POST /v1/completions
    Body: { "prompt": "...", "max_tokens": 32 }
    Returns: OpenAI completions response with random token output

  POST /v1/chat/completions
    Body: { "messages": [...], "max_tokens": 32 }
    Returns: OpenAI chat completions response with random token output

  GET  /health
    Returns: { "dp_id": N, "queue_depth": K, "group_busy": bool,
               "group_iteration": N, "active_batch_size": K }

Install
───────
  pip install fastapi uvicorn

Usage
─────
  # 4 DP workers on ports 8100-8103
  python prefill_server.py --n-dp 4 --base-port 8100

  # custom compute coefficients
  python prefill_server.py --n-dp 2 --base-port 8100 --alpha 1e-7 --beta 1e-5

  # test
  curl -s -X POST http://localhost:8100/v1/completions \\
       -H 'Content-Type: application/json' \\
       -d '{"prompt": "hello world", "max_tokens": 16}' | python -m json.tool

  curl -s http://localhost:8101/health | python -m json.tool
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import string
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Rich imports for TUI
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.align import Align
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("prefill_server")


# ── compute model ─────────────────────────────────────────────────────────────

@dataclass
class PrefillComputeConfig:
    alpha: float = 10   # attention quadratic coeff  [s / token²]
    beta:  float = 20   # MoE linear coeff           [s / token]
    max_batch_size: int = 32
    max_seq_len:    int = 4096


def prefill_duration(seq_lens: List[int], cfg: PrefillComputeConfig) -> float:
    """t = alpha * Σ(s_i²) + beta * Σ(s_i)"""
    if not seq_lens:
        return 0.0
    return cfg.alpha * sum(s * s for s in seq_lens) + cfg.beta * sum(seq_lens)


# ── token helpers ─────────────────────────────────────────────────────────────

_VOCAB = string.ascii_letters + string.digits + "  "

def _random_tokens(n: int) -> str:
    """Generate a random string resembling tokenised output."""
    words = []
    rng = random.Random()
    while len(words) < n:
        wlen = rng.randint(2, 8)
        words.append("".join(rng.choices(_VOCAB, k=wlen)))
    return " ".join(words[:n])


def _count_prompt_tokens(prompt: str) -> int:
    """Rough whitespace tokenisation (good enough for simulation)."""
    return max(1, len(prompt.split()))


# ── pending request ───────────────────────────────────────────────────────────

@dataclass
class PendingRequest:
    req_id:        str
    dp_id:         int
    prompt_tokens: int
    max_tokens:    int
    arrived_at:    float = field(default_factory=time.monotonic)

    # filled when the iteration completes
    result:        Optional[dict] = field(default=None, repr=False)
    _event:        asyncio.Event  = field(default=None,  repr=False)

    def set_event(self, loop: asyncio.AbstractEventLoop) -> None:
        """Must be called from within the target event loop."""
        self._event = asyncio.Event()

    async def wait(self) -> dict:
        await self._event.wait()
        return self.result

    def complete(self, result: dict, loop: asyncio.AbstractEventLoop) -> None:
        self.result = result
        loop.call_soon_threadsafe(self._event.set)


# ── EP-group scheduler (runs in a background thread) ─────────────────────────

class EPGroupScheduler:
    """
    Manages N DP queues with EP-synchronised iteration execution.

    The scheduler loop runs in a dedicated daemon thread.
    Requests are submitted via enqueue(); completion is signalled
    back to the FastAPI event loop via asyncio.Event.
    """

    def __init__(self, n_dp: int, cfg: PrefillComputeConfig) -> None:
        self.n_dp = n_dp
        self.cfg  = cfg

        # per-DP waiting queues
        self._queues: List[Deque[PendingRequest]] = [deque() for _ in range(n_dp)]
        self._lock = __import__("threading").Lock()

        # group state
        self._busy            = False
        self._iteration_count = 0
        self._active_batches: List[List[PendingRequest]] = [[] for _ in range(n_dp)]

        # wakeup: scheduler thread sleeps until a request arrives
        self._has_work = __import__("threading").Event()

        # reference to the running asyncio event loop (set on first enqueue)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # start scheduler thread
        import threading
        t = threading.Thread(target=self._scheduler_loop, daemon=True, name="ep-scheduler")
        t.start()
        log.info("EPGroupScheduler started  n_dp=%d  max_batch=%d  max_seq=%d", n_dp, cfg.max_batch_size, cfg.max_seq_len)

    # ── public ────────────────────────────────────────────────────────────────

    def enqueue(
        self,
        dp_id: int,
        req: PendingRequest,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Thread-safe enqueue from the FastAPI request handler coroutine."""
        req.set_event(loop)
        if self._loop is None:
            self._loop = loop
        with self._lock:
            self._queues[dp_id].append(req)
            log.info("[DP %d] ENQUEUE  req=%s  prompt_tokens=%d  queue_depth=%d",
                     dp_id, req.req_id[:8], req.prompt_tokens,
                     len(self._queues[dp_id]))
        self._has_work.set()

    def group_status(self) -> dict:
        with self._lock:
            return {
                "busy":           self._busy,
                "iteration":      self._iteration_count,
                "queue_depths":   [len(q) for q in self._queues],
                "active_batches": [len(b) for b in self._active_batches],
            }

    def dp_status(self, dp_id: int) -> dict:
        with self._lock:
            return {
                "dp_id":             dp_id,
                "queue_depth":       len(self._queues[dp_id]),
                "group_busy":        self._busy,
                "group_iteration":   self._iteration_count,
                "active_batch_size": len(self._active_batches[dp_id]),
            }

    def get_detailed_status(self) -> dict:
        """Get detailed status for TUI display."""
        with self._lock:
            dp_details = []
            for dp_id in range(self.n_dp):
                queue = self._queues[dp_id]
                batch = self._active_batches[dp_id]

                # Calculate total tokens in queue and batch
                queue_tokens = sum(req.prompt_tokens for req in queue)
                batch_tokens = sum(req.prompt_tokens for req in batch)

                # Calculate compute time for current batch
                seq_lens = [r.prompt_tokens for r in batch]
                compute_time = prefill_duration(seq_lens, self.cfg)

                dp_details.append({
                    "dp_id": dp_id,
                    "queue_depth": len(queue),
                    "queue_tokens": queue_tokens,
                    "batch_size": len(batch),
                    "batch_tokens": batch_tokens,
                    "compute_time": compute_time,
                })

            return {
                "n_dp": self.n_dp,
                "busy": self._busy,
                "iteration": self._iteration_count,
                "dp_details": dp_details,
                "max_batch_size": self.cfg.max_batch_size,
                "max_seq_len": self.cfg.max_seq_len,
            }

    # ── scheduler thread ──────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        while True:
            # wait until at least one request arrives
            self._has_work.wait()
            self._has_work.clear()

            while True:
                batches = self._try_form_batches()
                if not batches:
                    break   # all queues empty → go back to sleep

                self._run_iteration(batches)

    def _try_form_batches(self) -> Optional[List[List[PendingRequest]]]:
        """
        Snapshot all queues and pack a sub-batch for each DP.
        Returns None if every queue is empty.
        """
        with self._lock:
            # check if any queue has work
            if not any(self._queues):
                return None

            batches: List[List[PendingRequest]] = []
            for dp_id, q in enumerate(self._queues):
                batch: List[PendingRequest] = []
                leftover: Deque[PendingRequest] = deque()
                batch_tokens = 0  # Track cumulative tokens in the batch
                for req in q:
                    # Check if single request exceeds max_seq_len
                    if req.prompt_tokens > self.cfg.max_seq_len:
                        log.warning("[DP %d] DROP req=%s prompt_tokens=%d > max_seq_len=%d",
                                    dp_id, req.req_id[:8],
                                    req.prompt_tokens, self.cfg.max_seq_len)
                        # complete with an error result immediately
                        self._complete_request(req, error="prompt exceeds max_seq_len")
                        continue
                    # Check if batch size limit reached
                    if len(batch) >= self.cfg.max_batch_size:
                        leftover.append(req)
                        continue
                    # Check if cumulative tokens exceed max_seq_len
                    if batch_tokens + req.prompt_tokens > self.cfg.max_seq_len:
                        leftover.append(req)
                        continue
                    # Add request to batch
                    batch.append(req)
                    batch_tokens += req.prompt_tokens
                self._queues[dp_id] = leftover
                batches.append(batch)
                self._active_batches[dp_id] = batch

            self._busy = True
            return batches

    def _run_iteration(self, batches: List[List[PendingRequest]]) -> None:
        """
        Execute one EP-synchronised iteration.

        Duration = max(compute_time per DP)  ← All-to-All barrier
        """
        # compute per-DP durations
        durations: List[float] = []
        for dp_id, batch in enumerate(batches):
            seq_lens = [r.prompt_tokens for r in batch]
            t = prefill_duration(seq_lens, self.cfg)
            durations.append(t)
            log.info(
                "[DP %d] sub-batch size=%d  seq_lens=%s  compute=%.4fs",
                dp_id, len(batch), seq_lens, t,
            )
        print(durations)

        group_duration = max(durations) if durations else 0.0
        self._iteration_count += 1
        log.info(
            "[EP-group] iteration=%d  per_dp_times=%s  group_duration=%.4fs  (barrier=%.4fs)",
            self._iteration_count,
            [f"{d:.4f}" for d in durations],
            group_duration,
            group_duration - min(durations) if durations else 0.0,
        )

        # simulate the computation
        time.sleep(group_duration)

        # complete all requests in all DPs simultaneously (barrier point)
        now = time.monotonic()
        with self._lock:
            for dp_id, batch in enumerate(batches):
                for req in batch:
                    result = self._build_result(req, now)
                    self._complete_request(req, result=result)
                self._active_batches[dp_id] = []
            self._busy = False

        log.info("[EP-group] iteration=%d  DONE", self._iteration_count)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_result(self, req: PendingRequest, finish_time: float) -> dict:
        elapsed   = finish_time - req.arrived_at
        out_text  = _random_tokens(req.max_tokens)
        return {
            "id":      f"cmpl-{req.req_id}",
            "object":  "text_completion",
            "created": int(time.time()),
            "model":   "prefill-simulator",
            "choices": [
                {
                    "text":          out_text,
                    "index":         0,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens":     req.prompt_tokens,
                "completion_tokens": req.max_tokens,
                "total_tokens":      req.prompt_tokens + req.max_tokens,
            },
            "_sim": {
                "dp_id":          req.dp_id,
                "prefill_time_s": round(elapsed, 6),
            },
        }

    def _complete_request(
        self,
        req: PendingRequest,
        result: Optional[dict] = None,
        error: Optional[str]   = None,
    ) -> None:
        """Called under self._lock. Signals the waiting coroutine."""
        if error:
            result = {"error": error, "id": req.req_id}
        if self._loop is not None:
            req.complete(result, self._loop)


# ── TUI Dashboard ────────────────────────────────────────────────────────────────

class TUILogHandler(logging.Handler):
    """Custom log handler that captures logs for TUI display."""

    def __init__(self, max_logs: int = 100):
        super().__init__()
        self.logs = []
        self.max_logs = max_logs
        self._lock = __import__("threading").Lock()

    def emit(self, record):
        """Capture log record."""
        try:
            log_entry = self.format(record)
            with self._lock:
                self.logs.append(log_entry)
                if len(self.logs) > self.max_logs:
                    self.logs.pop(0)
        except Exception:
            self.handleError(record)

    def get_logs(self, count: int = 50) -> List[str]:
        """Get the last N log entries."""
        with self._lock:
            return self.logs[-count:] if count < len(self.logs) else self.logs.copy()


class PrefillDashboard:
    """
    Real-time TUI dashboard for monitoring EP-group scheduler status.
    Left panel: DP load status, Right panel: Application logs.
    """

    def __init__(self, scheduler: EPGroupScheduler, base_port: int, refresh_rate: float = 0.5):
        if not RICH_AVAILABLE:
            raise RuntimeError("Rich library is required for TUI. Install with: pip install rich")

        self.scheduler = scheduler
        self.base_port = base_port
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False
        self._thread = None

        # Track statistics
        self._total_requests = 0
        self._total_iterations = 0
        self._last_iteration = 0

        # Setup log handler
        self.log_handler = TUILogHandler(max_logs=200)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger("prefill_server").addHandler(self.log_handler)

    def _make_header(self) -> Panel:
        """Create the header panel."""
        header_text = Text()
        header_text.append("PREFILL WORKER DASHBOARD", style="bold cyan")
        header_text.append(f" — DP {self.scheduler.n_dp} Workers", style="white")
        header_text.append(f" — Ports {self.base_port}-{self.base_port + self.scheduler.n_dp - 1}", style="dim")

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
            log_text = Text("[dim]No logs yet...[/dim]", justify="left")
        else:
            log_text = Text()
            for log_entry in logs:
                # Parse and colorize log entry
                if "INFO" in log_entry:
                    log_entry = log_entry.replace("INFO", "[green]INFO[/green]")
                elif "WARNING" in log_entry:
                    log_entry = log_entry.replace("WARNING", "[yellow]WARNING[/yellow]")
                elif "ERROR" in log_entry:
                    log_entry = log_entry.replace("ERROR", "[red]ERROR[/red]")
                elif "DEBUG" in log_entry:
                    log_entry = log_entry.replace("DEBUG", "[dim]DEBUG[/dim]")

                # Highlight DP info
                if "[DP " in log_entry:
                    import re
                    log_entry = re.sub(r'\[DP (\d+)\]', r'[cyan][DP \1][/cyan]', log_entry)

                # Highlight EP-group info
                if "[EP-group]" in log_entry:
                    log_entry = log_entry.replace("[EP-group]", "[bold magenta][EP-group][/bold magenta]")

                log_text.append(log_entry + "\n")

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

    def _update_display(self) -> None:
        """Update the display once."""
        status = self.scheduler.get_detailed_status()

        layout = self._make_layout()
        layout["header"].update(self._make_header())
        layout["left"].update(self._make_left_panel(status))
        layout["right"].update(self._make_log_panel())

        self.console.clear()
        self.console.print(layout)

    def _run_loop(self) -> None:
        """Main TUI loop (runs in separate thread)."""
        self._running = True

        # Clear screen and hide cursor
        self.console.clear()
        self._update_display()

        import time as _time
        while self._running:
            _time.sleep(self.refresh_rate)
            if self._running:
                self._update_display()

    def start(self) -> None:
        """Start the TUI dashboard in a background thread."""
        if not RICH_AVAILABLE:
            log.warning("Rich library not available, TUI disabled")
            return

        if self._running:
            log.warning("Dashboard already running")
            return

        import threading
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="tui-dashboard")
        self._thread.start()
        log.info("TUI Dashboard started on %s", self.console)

    def stop(self) -> None:
        """Stop the TUI dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        # Remove log handler
        logging.getLogger("prefill_server").removeHandler(self.log_handler)

        # Show cursor and clear
        self.console.show_cursor()
        log.info("TUI Dashboard stopped")


# ── FastAPI app factory ───────────────────────────────────────────────────────

def make_app(dp_id: int, scheduler: EPGroupScheduler) -> FastAPI:
    """
    Create a FastAPI app for a single DP port.
    All apps share the same EPGroupScheduler instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("DP %d  API server ready", dp_id)
        yield
        log.info("DP %d  API server shutting down", dp_id)

    app = FastAPI(
        title=f"Prefill DP {dp_id}",
        description="Prefill EP-Group simulator",
        lifespan=lifespan,
    )

    # ── endpoints ─────────────────────────────────────────────────────────────

    @app.post("/v1/completions")
    async def completions(request: Request):
        loop = asyncio.get_event_loop()
        body = await request.json()

        prompt = body.get("prompt", "hello")
        max_tokens = body.get("max_tokens", 16)
        kv_transfer_params = body.get("kv_transfer_params")

        prompt_toks = _count_prompt_tokens(prompt)
        req = PendingRequest(
            req_id=str(uuid.uuid4()),
            dp_id=dp_id,
            prompt_tokens=prompt_toks,
            max_tokens=max_tokens,
        )
        scheduler.enqueue(dp_id, req, loop)
        result = await req.wait()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Log aborted requests if present
        if kv_transfer_params and kv_transfer_params.get("aborted_request"):
            log.info(f"Received aborted requests: {kv_transfer_params['aborted_request']}")
        if kv_transfer_params and kv_transfer_params.get("absorted_request"):
            log.info(f"Received aborted requests (typo variant): {kv_transfer_params['absorted_request']}")

        # Generate mock KV transfer parameters for router compatibility
        num_blocks = max(1, (prompt_toks + max_tokens + 127) // 128)
        result["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": req.req_id,
            "remote_block_ids": list(range(num_blocks)),
            "remote_host": None,
            "remote_port": None,
        }
        return JSONResponse(content=result)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        loop = asyncio.get_event_loop()
        body = await request.json()

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 16)
        kv_transfer_params = body.get("kv_transfer_params")

        # join all message contents to estimate prompt length
        full_prompt = " ".join(m.get("content", "") for m in messages if isinstance(m, dict) and "content" in m)
        prompt_toks = _count_prompt_tokens(full_prompt)
        req = PendingRequest(
            req_id=str(uuid.uuid4()),
            dp_id=dp_id,
            prompt_tokens=prompt_toks,
            max_tokens=max_tokens,
        )
        scheduler.enqueue(dp_id, req, loop)
        result = await req.wait()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Log aborted requests if present
        if kv_transfer_params and kv_transfer_params.get("aborted_request"):
            log.info(f"Received aborted requests: {kv_transfer_params['aborted_request']}")
        if kv_transfer_params and kv_transfer_params.get("absorted_request"):
            log.info(f"Received aborted requests (typo variant): {kv_transfer_params['absorted_request']}")

        # Generate mock KV transfer parameters for router compatibility
        # The router expects these parameters to continue decoding on a separate decoder instance
        num_blocks = max(1, (prompt_toks + max_tokens + 127) // 128)  # Simulate block allocation
        kv_transfer_response = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": req.req_id,
            "remote_block_ids": list(range(num_blocks)),
            "remote_host": None,  # Router will fill this in
            "remote_port": None,  # Router will fill this in
        }

        # wrap into chat completions format
        chat_result = {
            "id":      result["id"].replace("cmpl-", "chatcmpl-"),
            "object":  "chat.completion",
            "created": result["created"],
            "model":   result["model"],
            "choices": [
                {
                    "index":         0,
                    "message":       {"role": "assistant", "content": result["choices"][0]["text"]},
                    "finish_reason": "length",
                }
            ],
            "usage": result["usage"],
            "kv_transfer_params": kv_transfer_response,
            "_sim":  result["_sim"],
        }
        return JSONResponse(content=chat_result)

    @app.get("/health")
    async def health():
        return JSONResponse(content=scheduler.dp_status(dp_id))

    @app.get("/group/status")
    async def group_status():
        """Full EP-group view (available on every DP port)."""
        return JSONResponse(content=scheduler.group_status())

    return app


# ── multi-server launcher ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefill EP-Group simulator server")
    p.add_argument("--n-dp",       type=int,   default=2,    help="Number of DP workers")
    p.add_argument("--base-port",  type=int,   default=8100, help="First DP port (DP-i uses base+i)")
    p.add_argument("--host",       type=str,   default="0.0.0.0")
    p.add_argument("--alpha",      type=float, default=30/100, help="Attention quadratic coeff [s/tok²]")
    p.add_argument("--beta",       type=float, default=5/100, help="MoE linear coeff [s/tok]")
    p.add_argument("--max-batch",  type=int,   default=32,   help="Max requests per DP per iteration")
    p.add_argument("--max-seq-len",type=int,   default=4096, help="Max prompt token length")
    p.add_argument("--log-level",  type=str,   default="info",
                   choices=["debug", "info", "warning", "error"])
    p.add_argument("--tui",        action="store_true", help="Enable TUI dashboard (requires rich)")
    p.add_argument("--no-tui",     action="store_true", help="Disable TUI dashboard")
    p.add_argument("--tui-refresh",type=float, default=0.5, help="TUI refresh rate in seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    cfg = PrefillComputeConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_batch_size=args.max_batch,
        max_seq_len=args.max_seq_len,
    )
    scheduler = EPGroupScheduler(n_dp=args.n_dp, cfg=cfg)

    # build one FastAPI app per DP
    apps = [make_app(dp_id=i, scheduler=scheduler) for i in range(args.n_dp)]

    log.info(
        "Starting %d DP servers on ports %d-%d",
        args.n_dp, args.base_port, args.base_port + args.n_dp - 1,
    )

    import threading

    servers = []
    threads = []
    dashboard = None

    for i, app in enumerate(apps):
        port   = args.base_port + i
        config = uvicorn.Config(
            app,
            host=args.host,
            port=port,
            log_level=args.log_level,
            # each server runs its own asyncio loop in its own thread
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        servers.append(server)

        t = threading.Thread(
            target=server.run,
            name=f"dp-{i}-uvicorn",
            daemon=True,
        )
        threads.append(t)
        t.start()
        log.info("  DP %d  → http://%s:%d", i, args.host, port)

    # Start TUI dashboard if enabled
    enable_tui = args.tui
    if not enable_tui and not args.no_tui and RICH_AVAILABLE:
        # Auto-enable TUI if rich is available and no explicit flag
        enable_tui = sys.stdout.isatty()  # Only enable if running in terminal

    if enable_tui:
        try:
            dashboard = PrefillDashboard(
                scheduler=scheduler,
                base_port=args.base_port,
                refresh_rate=args.tui_refresh,
            )
            dashboard.start()
        except Exception as e:
            log.warning("Failed to start TUI dashboard: %s", e)
            dashboard = None

    # keep the main thread alive; Ctrl-C shuts everything down
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Shutting down …")
        if dashboard:
            dashboard.stop()
        for s in servers:
            s.should_exit = True


if __name__ == "__main__":
    main()
