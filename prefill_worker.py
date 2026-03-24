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
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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

    # keep the main thread alive; Ctrl-C shuts everything down
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Shutting down …")
        for s in servers:
            s.should_exit = True


if __name__ == "__main__":
    main()
