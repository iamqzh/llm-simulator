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

# TUI Dashboard (optional)
try:
    from tui import PrefillDashboard

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    PrefillDashboard = None  # type: ignore

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("prefill_server")


# ── compute model ─────────────────────────────────────────────────────────────


@dataclass
class PrefillComputeConfig:
    alpha: float = 10  # attention quadratic coeff  [s / token²]
    beta: float = 20  # MoE linear coeff           [s / token]
    max_batch_size: int = 32
    max_seq_len: int = 4096


def prefill_duration(seq_lens: List[int], cfg: PrefillComputeConfig) -> float:
    """t = alpha * Σ(s_i²) + beta * Σ(s_i)"""
    if not seq_lens:
        return 0.0
    return cfg.alpha * sum(s * s for s in seq_lens) + cfg.beta * sum(seq_lens)


# ── token helpers ─────────────────────────────────────────────────────────────

_VOCAB = string.ascii_letters + string.digits + "  "

# Typical vocabulary size for large LLMs (DeepSeek V3 uses ~100K tokens)
_VOCAB_SIZE = 100000


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


# ── KV transfer helpers ───────────────────────────────────────────────────────

# KV cache block size in tokens (matches vLLM default)
_BLOCK_SIZE = 128


def _calc_num_blocks(prompt_tokens: int) -> int:
    """Number of KV cache blocks needed for prompt_tokens."""
    return max(1, (prompt_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE)


def _alloc_block_ids(num_blocks: int) -> List[int]:
    """
    Simulate KV cache block allocation.
    Returns a list of num_blocks distinct block IDs drawn from a plausible
    pool (1..4095), mirroring real vLLM block allocator behaviour where
    block 0 is reserved and IDs are not necessarily contiguous.
    """
    pool_size = 4096
    start = random.randint(1, pool_size - num_blocks)
    # Simulate slight fragmentation: mostly contiguous with occasional gaps
    ids = []
    cursor = start
    for _ in range(num_blocks):
        ids.append(cursor)
        cursor += random.choice([1, 1, 1, 2])  # 75% contiguous, 25% skip one
    return ids


def _make_engine_id(n_dp: int, dp_id: int) -> str:
    """
    Generate a stable engine ID that matches the real vLLM format:
        "{n_dp}-{hex32}_dp{dp_id}"
    e.g. "8-1f6fdee284a3426d84eb25b4ebdda6fe_dp0"
    The hex part is a fixed random value per process (stable across requests).
    """
    return _ENGINE_HEX_PART.format(n_dp=n_dp, dp_id=dp_id)


# Module-level stable hex part, generated once per process.
_ENGINE_HEX_RAW = uuid.uuid4().hex + uuid.uuid4().hex[:0]  # 32 hex chars
_ENGINE_HEX_PART = "{{n_dp}}-{hex}_dp{{dp_id}}".format(hex=_ENGINE_HEX_RAW)


def _build_kv_transfer_params(
    req_host: str,
    n_dp: int,
    dp_id: int,
    prompt_tokens: int,
    last_token_id: int,
) -> dict:
    """
    Build the kv_transfer_params block returned in the Prefill response.

    Direction flip vs. the request:
      request:  do_remote_decode=True,  do_remote_prefill=False
      response: do_remote_prefill=True, do_remote_decode=False

    Fields derived from the real capture:
      remote_block_ids              — allocated KV cache blocks
      remote_engine_id              — "{n_dp}-{hex32}_dp{dp_id}"
      remote_host                   — this server's host
      remote_port                   — ephemeral port (simulated)
      remote_pcp_size               — # prompt-cache pages  (= num_blocks)
      remote_dcp_size               — # decode-cache pages  (= num_blocks)
      last_token_id                 — vocab ID of the last prefilled token
      remote_multi_nodes_meta_mapping — one entry per block
      num_prompt_blocks             — total allocated blocks
    """
    num_blocks = _calc_num_blocks(prompt_tokens)
    block_ids = _alloc_block_ids(num_blocks)
    engine_id = _make_engine_id(n_dp, dp_id)

    # Ephemeral port: real vLLM uses a random high port for KV-transfer RDMA/TCP
    remote_port = random.randint(40000, 65535)

    # One meta entry per block – all on this node (single-node simulation)
    meta_mapping = {
        str(i): {"host": req_host, "engine_id": engine_id}
        for i in range(num_blocks)
    }

    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": block_ids,
        "remote_engine_id": engine_id,
        "remote_host": req_host,
        "remote_port": remote_port,
        "remote_pcp_size": num_blocks,
        "remote_dcp_size": num_blocks,
        "last_token_id": last_token_id,
        "remote_multi_nodes_meta_mapping": meta_mapping,
        "num_prompt_blocks": num_blocks,
    }


# ── pending request ───────────────────────────────────────────────────────────


@dataclass
class PendingRequest:
    req_id: str
    dp_id: int
    prompt_tokens: int
    max_tokens: int
    arrived_at: float = field(default_factory=time.monotonic)

    # filled when the iteration completes
    result: Optional[dict] = field(default=None, repr=False)
    _event: asyncio.Event = field(default=None, repr=False)

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
        self.cfg = cfg

        # per-DP waiting queues
        self._queues: List[Deque[PendingRequest]] = [deque() for _ in range(n_dp)]
        self._lock = __import__("threading").Lock()

        # group state
        self._busy = False
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
        log.info(
            "EPGroupScheduler started  n_dp=%d  max_batch=%d  max_seq=%d", n_dp, cfg.max_batch_size, cfg.max_seq_len
        )

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
            log.info(
                "[DP %d] ENQUEUE  req=%s  prompt_tokens=%d  queue_depth=%d",
                dp_id,
                req.req_id[:8],
                req.prompt_tokens,
                len(self._queues[dp_id]),
            )
        self._has_work.set()

    def group_status(self) -> dict:
        with self._lock:
            return {
                "busy": self._busy,
                "iteration": self._iteration_count,
                "queue_depths": [len(q) for q in self._queues],
                "active_batches": [len(b) for b in self._active_batches],
            }

    def dp_status(self, dp_id: int) -> dict:
        with self._lock:
            return {
                "dp_id": dp_id,
                "queue_depth": len(self._queues[dp_id]),
                "group_busy": self._busy,
                "group_iteration": self._iteration_count,
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

                dp_details.append(
                    {
                        "dp_id": dp_id,
                        "queue_depth": len(queue),
                        "queue_tokens": queue_tokens,
                        "batch_size": len(batch),
                        "batch_tokens": batch_tokens,
                        "compute_time": compute_time,
                    }
                )

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
                    break  # all queues empty → go back to sleep

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
                        log.warning(
                            "[DP %d] DROP req=%s prompt_tokens=%d > max_seq_len=%d",
                            dp_id,
                            req.req_id[:8],
                            req.prompt_tokens,
                            self.cfg.max_seq_len,
                        )
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
                dp_id,
                len(batch),
                seq_lens,
                t,
            )

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
        elapsed = finish_time - req.arrived_at
        # last_token_id: random vocab ID for the single prefill-output token
        last_token_id = random.randint(0, _VOCAB_SIZE - 1)
        return {
            "req_id": req.req_id,
            "prompt_tokens": req.prompt_tokens,
            "max_tokens": req.max_tokens,
            "dp_id": req.dp_id,
            "prefill_time_s": round(elapsed, 6),
            "last_token_id": last_token_id,
        }

    def _complete_request(
        self,
        req: PendingRequest,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Called under self._lock. Signals the waiting coroutine."""
        if error:
            result = {"error": error, "req_id": req.req_id}
        if self._loop is not None:
            req.complete(result, self._loop)


# ── response builders ─────────────────────────────────────────────────────────


def _build_chat_response(
    sim: dict,
    request_id: str,
    model: str,
    server_host: str,
    n_dp: int,
) -> dict:
    """
    Build a chat.completion response that exactly mirrors the real Prefill
    node output format, including all null fields vLLM emits.

    The scheduler returns only the minimal sim dict; everything else is
    assembled here so the scheduler stays model-agnostic.
    """
    req_id = sim["req_id"]
    prompt_tokens = sim["prompt_tokens"]
    max_tokens = sim["max_tokens"]
    dp_id = sim["dp_id"]
    last_token_id = sim["last_token_id"]

    # The single output token: one random ASCII character (mimics max_tokens=1)
    out_char = random.choice(string.ascii_letters)

    kv_params = _build_kv_transfer_params(
        req_host=server_host,
        n_dp=n_dp,
        dp_id=dp_id,
        prompt_tokens=prompt_tokens,
        last_token_id=last_token_id,
    )

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": out_char,
                    "refusal": None,
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                    "reasoning": None,
                    "reasoning_content": None,
                },
                "logprobs": None,
                "finish_reason": "length",
                "stop_reason": None,
                "token_ids": None,
            }
        ],
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
            "prompt_tokens_details": None,
        },
        "prompt_logprobs": None,
        "prompt_token_ids": None,
        "kv_transfer_params": kv_params,
    }


def _build_completions_response(
    sim: dict,
    request_id: str,
    model: str,
    server_host: str,
    n_dp: int,
) -> dict:
    """
    Build a text completion response with the same kv_transfer_params structure.
    """
    req_id = sim["req_id"]
    prompt_tokens = sim["prompt_tokens"]
    max_tokens = sim["max_tokens"]
    dp_id = sim["dp_id"]
    last_token_id = sim["last_token_id"]

    out_char = random.choice(string.ascii_letters)

    kv_params = _build_kv_transfer_params(
        req_host=server_host,
        n_dp=n_dp,
        dp_id=dp_id,
        prompt_tokens=prompt_tokens,
        last_token_id=last_token_id,
    )

    return {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": out_char,
                "logprobs": None,
                "finish_reason": "length",
                "stop_reason": None,
                "token_ids": None,
            }
        ],
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
            "prompt_tokens_details": None,
        },
        "prompt_logprobs": None,
        "prompt_token_ids": None,
        "kv_transfer_params": kv_params,
    }


# ── FastAPI app factory ───────────────────────────────────────────────────────


def make_app(dp_id: int, scheduler: EPGroupScheduler, server_host: str="0.0.0.0", model: str="deepseek_v3") -> FastAPI:
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
        max_tokens = int(body.get("max_tokens", 1))
        kv_transfer_params = body.get("kv_transfer_params")

        # Use X-Request-Id header if present, otherwise generate one
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        prompt_toks = _count_prompt_tokens(prompt)
        req = PendingRequest(
            req_id=request_id,
            dp_id=dp_id,
            prompt_tokens=prompt_toks,
            max_tokens=max_tokens,
        )
        scheduler.enqueue(dp_id, req, loop)
        sim = await req.wait()
        if "error" in sim:
            raise HTTPException(status_code=400, detail=sim["error"])

        # Log aborted requests if present
        if kv_transfer_params:
            for key in ("aborted_request", "absorted_request"):
                if kv_transfer_params.get(key):
                    log.info("Received aborted requests (%s): %s", key, kv_transfer_params[key])

        result = _build_completions_response(
            sim=sim,
            request_id=request_id,
            model=model,
            server_host=server_host,
            n_dp=scheduler.n_dp,
        )
        return JSONResponse(content=result)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        loop = asyncio.get_event_loop()
        body = await request.json()

        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 1))
        kv_transfer_params = body.get("kv_transfer_params")

        # Use X-Request-Id header if present, otherwise generate one
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # join all message contents to estimate prompt length
        full_prompt = " ".join(
            m.get("content", "") for m in messages if isinstance(m, dict) and "content" in m
        )
        prompt_toks = _count_prompt_tokens(full_prompt)

        req = PendingRequest(
            req_id=request_id,
            dp_id=dp_id,
            prompt_tokens=prompt_toks,
            max_tokens=max_tokens,
        )
        scheduler.enqueue(dp_id, req, loop)
        sim = await req.wait()
        if "error" in sim:
            raise HTTPException(status_code=400, detail=sim["error"])

        # Log aborted requests if present
        if kv_transfer_params:
            for key in ("aborted_request", "absorted_request"):
                if kv_transfer_params.get(key):
                    log.info("Received aborted requests (%s): %s", key, kv_transfer_params[key])

        result = _build_chat_response(
            sim=sim,
            request_id=request_id,
            model=model,
            server_host=server_host,
            n_dp=scheduler.n_dp,
        )
        return JSONResponse(content=result)

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
    p.add_argument("--n-dp", type=int, default=2, help="Number of DP workers")
    p.add_argument("--base-port", type=int, default=8100, help="First DP port (DP-i uses base+i)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--server-host", type=str, default=None, help="Host address reported in kv_transfer_params (defaults to --host or auto-detected)")
    p.add_argument("--model", type=str, default="deepseek_v3", help="Model name returned in responses")
    p.add_argument("--alpha", type=float, default=30 / 100, help="Attention quadratic coeff [s/tok²]")
    p.add_argument("--beta", type=float, default=5 / 100, help="MoE linear coeff [s/tok]")
    p.add_argument("--max-batch", type=int, default=32, help="Max requests per DP per iteration")
    p.add_argument("--max-seq-len", type=int, default=4096, help="Max prompt token length")
    p.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error"])
    p.add_argument("--tui", action="store_true", help="Enable TUI dashboard (requires rich)")
    p.add_argument("--no-tui", action="store_true", help="Disable TUI dashboard")
    p.add_argument("--tui-refresh", type=float, default=0.5, help="TUI refresh rate in seconds")
    return p.parse_args()


def _detect_host() -> str:
    """Best-effort local IP detection for use in kv_transfer_params."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    # Resolve the host address to embed in kv_transfer_params
    if args.server_host:
        server_host = args.server_host
    elif args.host not in ("0.0.0.0", ""):
        server_host = args.host
    else:
        server_host = _detect_host()

    log.info("kv_transfer_params.remote_host will be reported as: %s", server_host)

    cfg = PrefillComputeConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_batch_size=args.max_batch,
        max_seq_len=args.max_seq_len,
    )
    scheduler = EPGroupScheduler(n_dp=args.n_dp, cfg=cfg)

    # build one FastAPI app per DP
    apps = [
        make_app(dp_id=i, scheduler=scheduler, server_host=server_host, model=args.model)
        for i in range(args.n_dp)
    ]

    log.info(
        "Starting %d DP servers on ports %d-%d",
        args.n_dp,
        args.base_port,
        args.base_port + args.n_dp - 1,
    )

    import threading

    servers = []
    threads = []
    dashboard = None

    for i, app in enumerate(apps):
        port = args.base_port + i
        config = uvicorn.Config(
            app,
            host=args.host,
            port=port,
            log_level=args.log_level,
            access_log=False,  # Disable access log to prevent TUI flickering
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
