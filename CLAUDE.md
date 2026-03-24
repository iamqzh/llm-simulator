# LLM Simulator - Prefill Worker

This project is a simulator for LLM inference prefill workers with EP (Execution Parallelism) group synchronization.

## TUI Dashboard

The simulator includes a real-time Text User Interface (TUI) dashboard for monitoring EP-group status.

### Enabling TUI

```bash
# Enable TUI explicitly
python prefill_worker.py --n-dp 4 --base-port 8100 --tui

# Auto-enabled if rich is installed and running in a terminal
python prefill_worker.py --n-dp 4 --base-port 8100

# Disable TUI
python prefill_worker.py --n-dp 4 --base-port 8100 --no-tui

# Custom refresh rate (default 0.5s)
python prefill_worker.py --n-dp 4 --base-port 8100 --tui --tui-refresh 0.2
```

### TUI Display

The dashboard shows:

1. **Header**: Title and configuration (number of DPs, port range)
2. **EP-Group Status**: Current state (BUSY/IDLE), iteration count
3. **DP Table**: Per-worker information
   - Queue depth and tokens
   - Active batch size and tokens
   - Compute time
   - Visual load bar
4. **Summary**: Total queued/processing requests, max compute time

### Installing Rich

```bash
pip install rich
```

## Project Overview

This simulator demonstrates the EP-group synchronization behavior in distributed LLM inference, where multiple Data Parallel (DP) workers share a single EP-sync scheduler.

## Architecture

```
┌─ One Server (this process) ─────────────────────────────────┐
│                                                              │
│  API-Server-0  :base_port+0  ──►  Queue-0  ─┐               │
│  API-Server-1  :base_port+1  ──►  Queue-1  ─┤  Scheduler    │──► Execute (EP-sync)
│  ...                                         │  (EP-group)   │
│  API-Server-N  :base_port+N  ──►  Queue-N  ─┘               │
└─────────────────────────────────────────────────────────────┘
```

## EP-Sync Rule

- Scheduler fires when **ANY** DP queue is non-empty AND the group is idle
- It snapshots **ALL** queues at once and packs a sub-batch per DP
- `finish_time = now + max(compute_time_dp_i)` ← All-to-All barrier
- While the group is BUSY, newly arrived requests enter the queues but cannot be picked up until the current iteration completes
- A DP with an empty queue contributes 0 compute time but still participates in the barrier

## Prefill Compute Model

```
t_dp_i = alpha * Σ(s_j²) + beta * Σ(s_j)    for requests on DP i
group_duration = max(t_dp_i  for i in 0..N-1)
```

## Current Implementation

### Features

1. **EP-Group Scheduler** (`EPGroupScheduler` class)
   - Manages N DP queues with EP-synchronized iteration execution
   - Thread-safe enqueue from FastAPI request handlers
   - Background scheduler loop processes batches
   - Proper barrier synchronization (all DPs wait for slowest)

2. **Batch Formation Logic** (`_try_form_batches` method)
   - Checks if single request exceeds `max_seq_len`
   - Enforces `max_batch_size` limit per DP
   - **Cumulative token length check**: Ensures total tokens in batch ≤ `max_seq_len`
   - Properly handles leftover requests that don't fit in current batch

3. **OpenAI-Compatible API**
   - `POST /v1/completions` - Text completion endpoint
   - `POST /v1/chat/completions` - Chat completion endpoint
   - `GET /health` - Health check endpoint
   - `GET /group/status` - Full EP-group status endpoint

4. **Router Compatibility**
   - Supports `kv_transfer_params` for distributed inference
   - Returns KV transfer parameters in response
   - Handles aborted requests tracking
   - Compatible with load balance proxy server pattern

### Key Modifications

1. **Request Handling**
   - Removed Pydantic models (were causing 422 errors)
   - Uses direct JSON parsing via `await request.json()`
   - Supports both "aborted_request" and "absorted_request" spellings

2. **KV Transfer Parameters**
   - Accepts `kv_transfer_params` in requests
   - Returns mock `kv_transfer_params` in responses
   - Simulates KV cache block allocation (128 tokens per block)
   - Logs aborted requests for debugging

3. **Batch Validation**
   - Fixed batch formation to check cumulative token length
   - Properly validates batch size and sequence length constraints
   - Prevents individual requests from exceeding max_seq_len

## Usage

### Start Server

```bash
# 4 DP workers on ports 8100-8103
python prefill_worker.py --n-dp 4 --base-port 8100

# Custom compute coefficients
python prefill_worker.py --n-dp 2 --base-port 8100 --alpha 0.3 --beta 0.05
```

### Command Line Arguments

- `--n-dp`: Number of DP workers (default: 2)
- `--base-port`: First DP port (default: 8100)
- `--host`: Host address (default: 0.0.0.0)
- `--alpha`: Attention quadratic coefficient [s/tok²] (default: 0.3)
- `--beta`: MoE linear coefficient [s/tok] (default: 0.05)
- `--max-batch`: Max requests per DP per iteration (default: 32)
- `--max-seq-len`: Max prompt token length (default: 4096)
- `--log-level`: Logging level (default: info)

### API Examples

#### Completions

```bash
curl -X POST http://localhost:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Tell me a joke",
    "max_tokens": 16
  }'
```

#### Chat Completions

```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 16
  }'
```

#### With KV Transfer Params

```bash
curl -X POST http://localhost:8103/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Request-id: 98BA8BB5-1208-40C5-9CAF-BDB7AABF88FF' \
  -d '{
    "model": "demo",
    "prompt": "Tell me a joke",
    "min_tokens": 1,
    "max_tokens": 1,
    "stream": false,
    "kv_transfer_params": {
      "do_remote_decode": true,
      "do_remote_prefill": false,
      "remote_engine_ids": null,
      "remote_block_ids": null,
      "remote_host": null,
      "remote_port": null,
      "absorted_request": []
    }
  }'
```

## Expected Log Output

When sending a request to DP 3:

```
[DP 3] ENQUEUE  req=071cf6b0  prompt_tokens=4  queue_depth=1
[DP 0] sub-batch size=0  seq_lens=[]  compute=0.0000s
[DP 1] sub-batch size=0  seq_lens=[]  compute=0.0000s
[DP 2] sub-batch size=0  seq_lens=[]  compute=0.0000s
[DP 3] sub-batch size=1  seq_lens=[4]  compute=0.9200s
[0.0, 0.0, 0.0, 0.92]
[EP-group] iteration=1  per_dp_times=['0.0000', '0.0000', '0.0000', '0.9200']  group_duration=0.9200s  (barrier=0.9200s)
```

Key observations:
- Only DP 3 has work (size=1)
- Other DPs are idle (size=0, compute=0)
- `group_duration = max([0, 0, 0, 0.92]) = 0.92`
- `barrier = 0.92 - 0 = 0.92` (all DPs wait for slowest)

## Testing

Run the unit tests to verify EP-group synchronization:

```bash
python test_prefill_unit.py
```

Tests include:
1. **Single Request Test**: Verifies one DP processing while others idle
2. **Multiple DPs Test**: Verifies concurrent processing with barrier sync

## Design Decisions

### Direct JSON Parsing
Pydantic models were removed because they were causing 422 errors when defined inside the `make_app` function scope. Direct JSON parsing provides more flexibility and better error handling.

### Cumulative Token Validation
The batch formation logic was updated to check the **cumulative** token length of all requests in a batch, not just individual request lengths. This prevents the batch from exceeding `max_seq_len`.

### Thread-Based Scheduler
The EP-group scheduler runs in a dedicated daemon thread to handle synchronous blocking operations (simulating compute time) without blocking the FastAPI async event loop.

## Dependencies

```
fastapi
uvicorn
httpx
```

Install via:
```bash
pip install fastapi uvicorn httpx
```

## Future Enhancements

- [ ] Add decoder worker implementation
- [ ] Implement actual KV cache transfer between prefiller and decoder
- [ ] Add metrics/monitoring endpoints
- [ ] Support for streaming responses
- [ ] Add request cancellation handling
- [ ] Implement dynamic scaling (add/remove DPs)
