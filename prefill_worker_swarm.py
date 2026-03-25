"""
prefill_worker_swarm.py — Multi-worker swarm manager

Manages multiple prefill worker processes with a unified dashboard.
Each worker is an independent process with its own EP-group scheduler.
Uses multiprocessing.Manager to share state with the dashboard.

Architecture
────────────
  ┌─ Swarm Manager (this process) ──────────────────────────────┐
  │                                                             │
  │  ┌─ Worker 0 ──┐  ┌─ Worker 1 ──┐  ┌─ Worker N ──┐          │
  │  │ EP-Group 0  │  │ EP-Group 1  │  │ EP-Group N  │          │
  │  │ Ports 8100+ │  │ Ports 8200+ │  │ Ports 8300+ │          │
  │  └─────────────┘  └─────────────┘  └─────────────┘          │
  │         │                │                │                 │
  │         └────────────────┴────────────────┘                 │
  │                      │                                      │
  │              Shared State (Manager.dict)                    │
  │                      │                                      │
  │         Unified TUI Dashboard (reads state)                 │
  └─────────────────────────────────────────────────────────────┘

Usage
─────
  python prefill_worker_swarm.py --n-workers 3 --dp-per-worker 4 --base-port 8100
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import signal
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import Manager
from typing import List, Optional

# TUI Dashboard (optional)
try:
    from swarm_tui import SwarmDashboard

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    SwarmDashboard = None  # type: ignore

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("prefill_swarm")


@dataclass
class WorkerConfig:
    """Configuration for a single worker."""

    worker_id: int
    n_dp: int
    base_port: int
    host: str = "0.0.0.0"
    alpha: float = 0.3
    beta: float = 0.05
    max_batch: int = 32
    max_seq_len: int = 4096


class QueueLogHandler(logging.Handler):
    """Logging handler that sends pre-formatted log strings to a multiprocessing.Queue."""

    def __init__(self, q):
        super().__init__()
        self._queue = q

    def emit(self, record):
        try:
            self._queue.put_nowait(self.format(record))
        except Exception:
            self.handleError(record)


def run_worker(config: WorkerConfig, shared_state, log_queue) -> None:
    """Run a single worker process with state reporting."""
    import uvicorn

    from prefill_worker import EPGroupScheduler, PrefillComputeConfig, make_app

    # ── redirect ALL loggers in this process to the cross-process queue ───────
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  W{wid}  %(name)s  %(message)s".format(wid=config.worker_id),
        datefmt="%H:%M:%S",
    )
    queue_handler = QueueLogHandler(log_queue)
    queue_handler.setFormatter(fmt)

    # Attach to the root logger so every logger in this process is captured,
    # including prefill_server, uvicorn, asyncio, etc.
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(logging.INFO)

    # ── optional process title ────────────────────────────────────────────────
    try:
        import setproctitle
        setproctitle.setproctitle(f"prefill-worker-{config.worker_id}")
    except ImportError:
        pass

    cfg = PrefillComputeConfig(
        alpha=config.alpha,
        beta=config.beta,
        max_batch_size=config.max_batch,
        max_seq_len=config.max_seq_len,
    )
    scheduler = EPGroupScheduler(n_dp=config.n_dp, cfg=cfg)

    # Build one FastAPI app per DP
    apps = [make_app(dp_id=i, scheduler=scheduler) for i in range(config.n_dp)]

    log.info(
        "Worker %d starting %d DP servers on ports %d-%d",
        config.worker_id,
        config.n_dp,
        config.base_port,
        config.base_port + config.n_dp - 1,
    )

    servers = []
    threads = []

    for i, app in enumerate(apps):
        port = config.base_port + i
        uvicorn_config = uvicorn.Config(
            app,
            host=config.host,
            port=port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
        )
        server = uvicorn.Server(uvicorn_config)
        servers.append(server)

        t = threading.Thread(
            target=server.run,
            name=f"worker-{config.worker_id}-dp-{i}-uvicorn",
            daemon=True,
        )
        threads.append(t)
        t.start()

    # ── status reporting thread ───────────────────────────────────────────────
    def report_status():
        while True:
            try:
                status = scheduler.get_detailed_status()
                dp_statuses = [
                    {
                        "worker_id": config.worker_id,
                        "dp_id": dp["dp_id"],
                        "port": config.base_port + dp["dp_id"],
                        "queue_depth": dp["queue_depth"],
                        "queue_tokens": dp["queue_tokens"],
                        "batch_size": dp["batch_size"],
                        "batch_tokens": dp["batch_tokens"],
                        "compute_time": dp["compute_time"],
                        "busy": status["busy"],
                        "iteration": status["iteration"],
                    }
                    for dp in status["dp_details"]
                ]
                shared_state[f"worker_{config.worker_id}"] = {
                    "worker_id": config.worker_id,
                    "base_port": config.base_port,
                    "iteration": status["iteration"],
                    "dp_statuses": dp_statuses,
                    "last_update": time.time(),
                }
            except Exception as e:
                logging.getLogger(__name__).debug("Failed to report status: %s", e)
            time.sleep(0.2)

    threading.Thread(target=report_status, daemon=True, name="status-reporter").start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Worker %d shutting down...", config.worker_id)
        for s in servers:
            s.should_exit = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefill Worker Swarm Manager")
    p.add_argument("--n-workers", type=int, default=2, help="Number of worker processes")
    p.add_argument("--dp-per-worker", type=int, default=4, help="Number of DP workers per process")
    p.add_argument("--base-port", type=int, default=8100, help="Base port for first worker")
    p.add_argument("--port-step", type=int, default=100, help="Port increment between workers")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--alpha", type=float, default=0.3, help="Attention quadratic coeff [s/tok²]")
    p.add_argument("--beta", type=float, default=0.05, help="MoE linear coeff [s/tok]")
    p.add_argument("--max-batch", type=int, default=32, help="Max requests per DP per iteration")
    p.add_argument("--max-seq-len", type=int, default=4096, help="Max prompt token length")
    p.add_argument("--tui", action="store_true", help="Enable TUI dashboard (requires rich)")
    p.add_argument("--no-tui", action="store_true", help="Disable TUI dashboard")
    p.add_argument("--tui-refresh", type=float, default=0.5, help="TUI refresh rate in seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    manager = Manager()
    shared_state = manager.dict()
    log_queue = manager.Queue(maxsize=2000)

    worker_configs = [
        WorkerConfig(
            worker_id=w,
            n_dp=args.dp_per_worker,
            base_port=args.base_port + w * args.port_step,
            host=args.host,
            alpha=args.alpha,
            beta=args.beta,
            max_batch=args.max_batch,
            max_seq_len=args.max_seq_len,
        )
        for w in range(args.n_workers)
    ]

    log.info(
        "Starting swarm with %d workers, %d DP each, ports %d-%d",
        args.n_workers,
        args.dp_per_worker,
        args.base_port,
        args.base_port + args.n_workers * args.port_step - 1,
    )

    processes: List[multiprocessing.Process] = []
    for config in worker_configs:
        p = multiprocessing.Process(
            target=run_worker,
            # Pass the Manager proxy objects directly — they are picklable
            args=(config, shared_state, log_queue),
            name=f"prefill-worker-{config.worker_id}",
            daemon=True,
        )
        p.start()
        processes.append(p)
        log.info("Started worker %d (PID: %d)", config.worker_id, p.pid)

    # ── TUI dashboard ─────────────────────────────────────────────────────────
    dashboard: Optional[SwarmDashboard] = None
    enable_tui = args.tui
    if not enable_tui and not args.no_tui and RICH_AVAILABLE:
        enable_tui = sys.stdout.isatty()

    if enable_tui and RICH_AVAILABLE:
        try:
            dashboard = SwarmDashboard(
                n_workers=args.n_workers,
                dp_per_worker=args.dp_per_worker,
                base_port=args.base_port,
                port_step=args.port_step,
                refresh_rate=args.tui_refresh,
                shared_state=shared_state,   # ← pass the live Manager proxy, NOT dict(...)
                log_queue=log_queue,          # ← pass the live Manager.Queue proxy
            )
            dashboard.start()
        except Exception as e:
            log.warning("Failed to start TUI dashboard: %s", e)
            import traceback
            traceback.print_exc()
            dashboard = None

    # ── signal handling ───────────────────────────────────────────────────────
    def shutdown(sig=None, frame=None):
        log.info("Shutting down swarm...")
        if dashboard:
            dashboard.stop()
        for p in processes:
            p.terminate()
            p.join(timeout=5)
        manager.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── main loop: only monitors worker health ────────────────────────────────
    # No longer patches dashboard.shared_state — the proxy stays live on its own.
    try:
        while True:
            for i, p in enumerate(processes):
                if not p.is_alive():
                    log.error("Worker %d (PID: %d) died unexpectedly!", i, p.pid)
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    # Required for multiprocessing on macOS / some Linux configs
    multiprocessing.set_start_method("spawn", force=True)
    main()
