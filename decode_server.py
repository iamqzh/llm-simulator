"""
decode_server.py  —  Decode EP-Group simulator server

One process = one EP Group = N DP workers sharing a single EP-sync Scheduler.

Architecture
────────────────────────────────────────────────────────────────────
  ┌─ One Server (this process) ─────────────────────────────────┐
  │                                                             │
  │  API-Server-0  :base_port+0  ──►  ActiveBatch-0  ─┐         │
  │  API-Server-1  :base_port+1  ──►  ActiveBatch-1  ─┤  Sched  │──► Step (EP-sync)
  │  ...                                              │  (EP)   │
  │  API-Server-N  :base_port+N  ──►  ActiveBatch-N  ─┘         │
  └─────────────────────────────────────────────────────────────┘

Decode compute model (per step)
────────────────────────────────
  t_dp_i = alpha * Σ(s_i)  +  beta * batch_size_i
    where s_i is the CURRENT sequence length of each active request
  group_step_duration = max(t_dp_i  for i in 0..N-1)   ← All-to-All barrier

Continuous batching
────────────────────
  • Each DP has a fixed active-batch capacity: max_batch_size
  • After every step, finished requests are removed and waiting requests
    are admitted from the per-DP queue (up to capacity).
  • Newly arrived requests join the waiting queue and are admitted at
    the next step boundary.

Streaming (SSE) response format
─────────────────────────────────
  • First chunk:  delta = {"role":"assistant","content":"","reasoning_content":null}
                  + "prompt_token_ids": null  (top-level)
  • Body chunks:  delta = {"content": <token>, "reasoning_content": null}
                  + "token_ids": null  (in choice)
  • Final line:   data: [DONE]

API
────
  POST /v1/chat/completions   (stream: true required for real simulation;
                               stream: false also supported — waits for all tokens)
  GET  /health
  GET  /group/status

Usage
─────
  python decode_server.py --n-dp 4 --base-port 9100
  python decode_server.py --n-dp 2 --base-port 9100 --alpha 1e-4 --beta 2e-3 --max-batch 8
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import string
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Deque, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# TUI Dashboard (optional)
try:
    from decode_tui import DecodeDashboard
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    DecodeDashboard = None  # type: ignore

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("decode_server")

# ── vocabulary helpers ────────────────────────────────────────────────────────

_VOCAB_SIZE = 100000

# Poetic word pool for aesthetic decode output
_TOKEN_POOL = [
    "Serendipity ",   # 意外的好运
    "Ephemeral ",     # 短暂的
    "Luminous ",      # 发光的
    "Solitude ",      # 孤独
    "Ethereal ",      # 轻盈的，飘渺的
    "Sonorous ",      # 洪亮的声音
    "Labyrinth ",     # 迷宫
    "Velvet ",        # 天鹅绒，柔软的
    "Aurora ",        # 极光
    "Whimsy ",        # 异想天开
    "Mellifluous ",   # 悦耳的声音
    "Petrichor ",     # 雨后泥土的芬芳
    "Sunshine ",      # 阳光
    "Moonlight ",     # 月光
    "Starlight ",     # 星光
    "Whisper ",       # 低语
    "Breeze ",        # 微风
    "Cascade ",       # 瀑布
    "Tranquility ",   # 宁静
    "Harmony ",       # 和谐
]


def _next_token() -> str:
    """Return one plausible token string."""
    return random.choice(_TOKEN_POOL)


def _count_prompt_tokens(messages: list) -> int:
    """Rough token count from a messages list."""
    text = " ".join(
        m.get("content", "") for m in messages
        if isinstance(m, dict) and "content" in m
    )
    return max(1, len(text.split()))


# ── compute model ─────────────────────────────────────────────────────────────

@dataclass
class DecodeComputeConfig:
    alpha: float = 2e-4    # linear-attention coeff   [s / token]
    beta: float  = 1e-3    # batch-size coeff         [s / request]
    max_batch_size: int = 8
    max_seq_len: int = 2048  # hard cap; request aborted if exceeded


def decode_step_duration(seq_lens: List[int], cfg: DecodeComputeConfig) -> float:
    """
    t = alpha * Σ(s_i)  +  beta * batch_size
    seq_lens: current total sequence length of each active request
    """
    if not seq_lens:
        return 0.0
    return cfg.alpha * sum(seq_lens) + cfg.beta * len(seq_lens)


# ── active request ─────────────────────────────────────────────────────────────

@dataclass
class ActiveRequest:
    """
    One live decode request inside a DP's active batch.

    Lifecycle:
      WAITING  → admitted into active batch → RUNNING → DONE / ABORTED
    """
    req_id:        str
    dp_id:         int
    prompt_tokens: int          # tokens from the prefill phase
    max_tokens:    int          # max NEW tokens to generate
    arrived_at:    float = field(default_factory=time.monotonic)

    # mutable decode state
    generated:     int  = 0     # tokens produced so far
    current_len:   int  = 0     # prompt_tokens + generated  (updated each step)

    # asyncio queue: scheduler pushes token strings; handler reads them
    # sentinel None = stream finished
    _token_queue:  Optional[asyncio.Queue] = field(default=None, repr=False)
    _loop:         Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.current_len = self.prompt_tokens

    def init_async(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        # maxsize=0 → unbounded; scheduler never blocks
        self._token_queue = asyncio.Queue()

    def push_token(self, token: Optional[str]) -> None:
        """Called from the scheduler thread; thread-safe."""
        assert self._loop is not None and self._token_queue is not None
        self._loop.call_soon_threadsafe(self._token_queue.put_nowait, token)

    async def token_stream(self) -> AsyncIterator[Optional[str]]:
        """Async generator consumed by the HTTP handler."""
        while True:
            tok = await self._token_queue.get()
            yield tok
            if tok is None:
                break

    @property
    def is_done(self) -> bool:
        return self.generated >= self.max_tokens

    @property
    def is_overlong(self) -> bool:
        return self.current_len >= self.max_seq_len


# ── EP-group decode scheduler ─────────────────────────────────────────────────

class DecodeScheduler:
    """
    Manages N DP active batches with EP-synchronised step execution.

    Step loop (runs in a dedicated daemon thread):
      1. For each DP: fill empty slots from the waiting queue.
      2. Compute per-DP step duration.  group_duration = max(...).
      3. Sleep for group_duration (simulate computation).
      4. For each active request: generate one token, update seq len.
         If done/overlong: push sentinel None, remove from active batch.
      5. Goto 1.

    Requests are submitted via enqueue(); the scheduler never blocks on
    per-request I/O — it only pushes tokens into per-request asyncio queues.
    """

    def __init__(self, n_dp: int, cfg: DecodeComputeConfig) -> None:
        self.n_dp   = n_dp
        self.cfg    = cfg

        # per-DP: waiting queue (admitted before next step)
        self._waiting: List[Deque[ActiveRequest]] = [deque() for _ in range(n_dp)]
        # per-DP: active batch (currently decoding)
        self._active:  List[List[ActiveRequest]]  = [[]    for _ in range(n_dp)]

        self._lock = threading.Lock()

        # group state (informational, for TUI/health)
        self._step_count   = 0
        self._group_busy   = False

        # wakeup: step thread sleeps until there is work
        self._has_work = threading.Event()

        # loop reference — set on first enqueue
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        t = threading.Thread(
            target=self._step_loop, daemon=True, name="decode-ep-scheduler"
        )
        t.start()
        log.info(
            "DecodeScheduler started  n_dp=%d  max_batch=%d  max_seq=%d",
            n_dp, cfg.max_batch_size, cfg.max_seq_len,
        )

    # ── public ────────────────────────────────────────────────────────────────

    def enqueue(
        self,
        dp_id: int,
        req:   ActiveRequest,
        loop:  asyncio.AbstractEventLoop,
    ) -> None:
        req.init_async(loop)
        if self._loop is None:
            self._loop = loop
        with self._lock:
            self._waiting[dp_id].append(req)
            log.info(
                "[DP %d] ENQUEUE  req=%.8s  prompt_tokens=%d  max_tokens=%d  waiting=%d",
                dp_id, req.req_id, req.prompt_tokens, req.max_tokens,
                len(self._waiting[dp_id]),
            )
        self._has_work.set()

    def dp_status(self, dp_id: int) -> dict:
        with self._lock:
            return {
                "dp_id":           dp_id,
                "active_batch":    len(self._active[dp_id]),
                "waiting_queue":   len(self._waiting[dp_id]),
                "group_busy":      self._group_busy,
                "group_step":      self._step_count,
                "max_batch_size":  self.cfg.max_batch_size,
            }

    def group_status(self) -> dict:
        with self._lock:
            return {
                "busy":       self._group_busy,
                "step":       self._step_count,
                "active":     [len(b) for b in self._active],
                "waiting":    [len(q) for q in self._waiting],
            }

    def get_detailed_status(self) -> dict:
        """Rich status snapshot for the TUI dashboard."""
        with self._lock:
            dp_details = []
            for dp_id in range(self.n_dp):
                active  = self._active[dp_id]
                waiting = self._waiting[dp_id]

                seq_lens = [r.current_len for r in active]
                step_t   = decode_step_duration(seq_lens, self.cfg)

                # per-request progress info
                req_info = [
                    {
                        "req_id":       r.req_id[:8],
                        "current_len":  r.current_len,
                        "generated":    r.generated,
                        "max_tokens":   r.max_tokens,
                        "progress_pct": round(100 * r.generated / max(r.max_tokens, 1), 1),
                    }
                    for r in active
                ]

                dp_details.append({
                    "dp_id":          dp_id,
                    "active_size":    len(active),
                    "waiting_depth":  len(waiting),
                    "avg_seq_len":    round(sum(seq_lens) / len(seq_lens), 1) if seq_lens else 0,
                    "max_seq_len_in_batch": max(seq_lens) if seq_lens else 0,
                    "step_time_s":    round(step_t, 6),
                    "requests":       req_info,
                })

            return {
                "n_dp":          self.n_dp,
                "busy":          self._group_busy,
                "step":          self._step_count,
                "dp_details":    dp_details,
                "max_batch_size": self.cfg.max_batch_size,
                "max_seq_len":   self.cfg.max_seq_len,
            }

    # ── step loop (daemon thread) ─────────────────────────────────────────────

    def _step_loop(self) -> None:
        while True:
            # Sleep until at least one request exists anywhere
            self._has_work.wait()
            self._has_work.clear()

            while True:
                if not self._any_work():
                    break
                self._run_one_step()

    def _any_work(self) -> bool:
        with self._lock:
            return any(self._active) or any(self._waiting)

    def _admit_waiting(self) -> None:
        """Fill empty slots in each DP's active batch from its waiting queue. Must hold lock."""
        for dp_id in range(self.n_dp):
            slots = self.cfg.max_batch_size - len(self._active[dp_id])
            while slots > 0 and self._waiting[dp_id]:
                req = self._waiting[dp_id].popleft()
                self._active[dp_id].append(req)
                log.info(
                    "[DP %d] ADMIT  req=%.8s  active=%d/%d",
                    dp_id, req.req_id,
                    len(self._active[dp_id]), self.cfg.max_batch_size,
                )
                slots -= 1

    def _run_one_step(self) -> None:
        # ── 1. admit new requests into empty slots ────────────────────────────
        with self._lock:
            self._admit_waiting()
            self._group_busy = True

            # snapshot current active batches for this step
            batches: List[List[ActiveRequest]] = [list(b) for b in self._active]

        # ── 2. compute EP-sync step duration ─────────────────────────────────
        durations: List[float] = []
        for dp_id, batch in enumerate(batches):
            seq_lens = [r.current_len for r in batch]
            t = decode_step_duration(seq_lens, self.cfg)
            durations.append(t)

        group_duration = max(durations) if durations else 0.0

        with self._lock:
            self._step_count += 1
        step_num = self._step_count

        log.debug(
            "[EP-decode] step=%d  per_dp=%s  barrier=%.4fs",
            step_num,
            [f"{d:.4f}" for d in durations],
            group_duration,
        )

        # ── 3. simulate computation (sleep) ──────────────────────────────────
        time.sleep(group_duration)

        # ── 4. generate one token per active request; handle completions ──────
        with self._lock:
            for dp_id, batch in enumerate(batches):
                finished: List[ActiveRequest] = []
                for req in batch:
                    token = _next_token()
                    req.generated   += 1
                    req.current_len += 1

                    done = req.generated >= req.max_tokens
                    over = req.current_len >= self.cfg.max_seq_len

                    if done or over:
                        reason = "length" if done else "max_seq_len"
                        log.info(
                            "[DP %d] DONE  req=%.8s  generated=%d  reason=%s",
                            dp_id, req.req_id, req.generated, reason,
                        )
                        req.push_token(token)   # last real token
                        req.push_token(None)    # sentinel: stream finished
                        finished.append(req)
                    else:
                        req.push_token(token)

                # remove finished requests from active batch
                for req in finished:
                    self._active[dp_id].remove(req)

                # immediately admit next waiting requests for removed slots
                slots = self.cfg.max_batch_size - len(self._active[dp_id])
                while slots > 0 and self._waiting[dp_id]:
                    new_req = self._waiting[dp_id].popleft()
                    self._active[dp_id].append(new_req)
                    log.info(
                        "[DP %d] ADMIT(post-finish)  req=%.8s  active=%d/%d",
                        dp_id, new_req.req_id,
                        len(self._active[dp_id]), self.cfg.max_batch_size,
                    )
                    slots -= 1

            self._group_busy = False

        # Re-signal if there's still work
        if self._any_work():
            self._has_work.set()


# ── SSE chunk builders ────────────────────────────────────────────────────────

def _sse(data: str) -> str:
    return f"data: {data}\n\n"


def _first_chunk(request_id: str, model: str, created: int) -> str:
    """Role-announcement chunk — matches real decode output exactly."""
    import json
    chunk = {
        "id":      f"chatcmpl-{request_id}",
        "object":  "chat.completion.chunk",
        "created": created,
        "model":   model,
        "choices": [
            {
                "index":        0,
                "delta":        {"role": "assistant", "content": "", "reasoning_content": None},
                "logprobs":     None,
                "finish_reason": None,
            }
        ],
        "prompt_token_ids": None,
    }
    return _sse(json.dumps(chunk, separators=(",", ":")))


def _token_chunk(
    request_id: str,
    model:      str,
    created:    int,
    token:      str,
    finish_reason: Optional[str] = None,
) -> str:
    import json
    chunk = {
        "id":      f"chatcmpl-{request_id}",
        "object":  "chat.completion.chunk",
        "created": created,
        "model":   model,
        "choices": [
            {
                "index":        0,
                "delta":        {"content": token, "reasoning_content": None},
                "logprobs":     None,
                "finish_reason": finish_reason,
                "token_ids":    None,
            }
        ],
    }
    return _sse(json.dumps(chunk, separators=(",", ":")))


def _done_line() -> str:
    return "data: [DONE]\n\n"


# ── FastAPI app factory ───────────────────────────────────────────────────────

def make_app(
    dp_id:       int,
    scheduler:   DecodeScheduler,
    model:       str = "deepseek_v3",
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("Decode DP %d  API server ready", dp_id)
        yield
        log.info("Decode DP %d  API server shutting down", dp_id)

    app = FastAPI(
        title=f"Decode DP {dp_id}",
        description="Decode EP-Group simulator",
        lifespan=lifespan,
    )

    # ── /v1/chat/completions ──────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        loop = asyncio.get_event_loop()
        body = await request.json()

        messages   = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 64))
        stream     = bool(body.get("stream", True))
        kv_params  = body.get("kv_transfer_params", {})

        # Use X-Request-Id header if present, otherwise generate one
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Derive prompt length:
        # If the request carries kv_transfer_params from a real Prefill node,
        # prefer num_prompt_blocks * block_size as a lower-bound estimate;
        # otherwise fall back to whitespace tokenisation.
        if kv_params and kv_params.get("num_prompt_blocks"):
            # Each block = 16 tokens (vLLM default); use pcp_size if available
            pcp = kv_params.get("remote_pcp_size") or kv_params["num_prompt_blocks"]
            prompt_tokens = pcp * 16
        else:
            prompt_tokens = _count_prompt_tokens(messages)

        req = ActiveRequest(
            req_id        = request_id,
            dp_id         = dp_id,
            prompt_tokens = prompt_tokens,
            max_tokens    = max_tokens,
        )
        scheduler.enqueue(dp_id, req, loop)

        created = int(time.time())

        # ── streaming response ─────────────────────────────────────────────
        async def token_generator() -> AsyncIterator[str]:
            yield _first_chunk(request_id, model, created)

            tokens_sent = 0
            last_token: Optional[str] = None

            async for tok in req.token_stream():
                if tok is None:
                    # sentinel — stream is finished
                    # emit last buffered token with finish_reason
                    if last_token is not None:
                        yield _token_chunk(
                            request_id, model, created,
                            last_token, finish_reason="length",
                        )
                    break

                # buffer one token so we can attach finish_reason to the last one
                if last_token is not None:
                    yield _token_chunk(request_id, model, created, last_token)
                last_token = tok
                tokens_sent += 1

            yield _done_line()

        if stream:
            return StreamingResponse(
                token_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── non-streaming: collect all tokens then return ──────────────────
        all_tokens: List[str] = []
        async for tok in req.token_stream():
            if tok is None:
                break
            all_tokens.append(tok)

        content = "".join(all_tokens)
        result = {
            "id":      f"chatcmpl-{request_id}",
            "object":  "chat.completion",
            "created": created,
            "model":   model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role":              "assistant",
                        "content":           content,
                        "refusal":           None,
                        "annotations":       None,
                        "audio":             None,
                        "function_call":     None,
                        "tool_calls":        [],
                        "reasoning":         None,
                        "reasoning_content": None,
                    },
                    "logprobs":     None,
                    "finish_reason": "length",
                    "stop_reason":  None,
                    "token_ids":    None,
                }
            ],
            "service_tier":       None,
            "system_fingerprint": None,
            "usage": {
                "prompt_tokens":      req.prompt_tokens,
                "completion_tokens":  len(all_tokens),
                "total_tokens":       req.prompt_tokens + len(all_tokens),
                "prompt_tokens_details": None,
            },
            "prompt_logprobs":   None,
            "prompt_token_ids":  None,
        }
        return JSONResponse(content=result)

    # ── /health ───────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return JSONResponse(content=scheduler.dp_status(dp_id))

    # ── /group/status ─────────────────────────────────────────────────────────

    @app.get("/group/status")
    async def group_status():
        return JSONResponse(content=scheduler.group_status())

    return app


# ── multi-server launcher ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode EP-Group simulator server")
    p.add_argument("--n-dp",       type=int,   default=2,    help="Number of DP workers")
    p.add_argument("--base-port",  type=int,   default=9100, help="First DP port")
    p.add_argument("--host",       type=str,   default="0.0.0.0")
    p.add_argument("--model",      type=str,   default="deepseek_v3")
    p.add_argument("--alpha",      type=float, default=2e-4, help="Linear-attention coeff [s/tok]")
    p.add_argument("--beta",       type=float, default=1e-3, help="Batch-size coeff [s/req]")
    p.add_argument("--max-batch",  type=int,   default=8,    help="Max active requests per DP")
    p.add_argument("--max-seq-len",type=int,   default=2048, help="Hard sequence length cap")
    p.add_argument("--log-level",  type=str,   default="info",
                   choices=["debug", "info", "warning", "error"])
    p.add_argument("--tui",        action="store_true", help="Enable TUI dashboard")
    p.add_argument("--no-tui",     action="store_true", help="Disable TUI dashboard")
    p.add_argument("--tui-refresh",type=float, default=0.3,  help="TUI refresh interval (s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    cfg = DecodeComputeConfig(
        alpha          = args.alpha,
        beta           = args.beta,
        max_batch_size = args.max_batch,
        max_seq_len    = args.max_seq_len,
    )
    scheduler = DecodeScheduler(n_dp=args.n_dp, cfg=cfg)

    apps = [
        make_app(dp_id=i, scheduler=scheduler, model=args.model)
        for i in range(args.n_dp)
    ]

    log.info(
        "Starting %d Decode DP servers on ports %d-%d",
        args.n_dp, args.base_port, args.base_port + args.n_dp - 1,
    )

    servers = []
    threads = []
    dashboard = None

    for i, app in enumerate(apps):
        port = args.base_port + i
        config = uvicorn.Config(
            app,
            host      = args.host,
            port      = port,
            log_level = args.log_level,
            access_log= False,
            loop      = "asyncio",
        )
        server = uvicorn.Server(config)
        servers.append(server)

        t = threading.Thread(
            target=server.run,
            name=f"decode-dp-{i}-uvicorn",
            daemon=True,
        )
        threads.append(t)
        t.start()
        log.info("  Decode DP %d  → http://%s:%d", i, args.host, port)

    enable_tui = args.tui
    if not enable_tui and not args.no_tui and RICH_AVAILABLE:
        enable_tui = sys.stdout.isatty()

    if enable_tui:
        try:
            dashboard = DecodeDashboard(
                scheduler    = scheduler,
                base_port    = args.base_port,
                refresh_rate = args.tui_refresh,
            )
            dashboard.start()
        except Exception as e:
            log.warning("Failed to start TUI dashboard: %s", e)
            dashboard = None

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
