# LLM 推理模拟器运行机制文档

本文档详细说明 LLM Simulator 的内部运行机制，涵盖架构设计、调度算法、计算模型和关键技术实现。

---

## 一、整体架构

### 1.1 模拟器目标

模拟分布式 LLM 推理系统中的 **Prefill（预填充）** 和 **Decode（解码）** 两个阶段，特别关注 **EP-Group (Execution Parallelism Group)** 同步机制。

### 1.2 核心架构图

```
┌─ 单进程 = 一个 EP-Group ─────────────────────────────────────────┐
│                                                                   │
│  ┌─ API Layer (FastAPI + Uvicorn) ──────────────────────────────┐ │
│  │  API-Server-0 (:base_port+0)  ──►  Queue-0/ActiveBatch-0     │ │
│  │  API-Server-1 (:base_port+1)  ──►  Queue-1/ActiveBatch-1     │ │
│  │  ...                                                          │ │
│  │  API-Server-N (:base_port+N)  ──►  Queue-N/ActiveBatch-N     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─ Scheduler Layer ────────────────────────────────────────────┐ │
│  │  EP-Group Scheduler (单线程后台运行)                          │ │
│  │  - 管理 N 个 DP 队列                                          │ │
│  │  - 执行 EP-Sync 批处理                                        │ │
│  │  - All-to-All Barrier 同步                                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─ Compute Layer ──────────────────────────────────────────────┐ │
│  │  time.sleep(compute_duration)  // 模拟计算耗时               │ │
│  │  EP-sync barrier: 所有 DP 同时完成                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─ TUI Dashboard (可选) ───────────────────────────────────────┐ │
│  │  Rich-based 实时终端监控                                      │ │
│  └───────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### 1.3 进程模型

| 模块 | 进程 | 线程 | 说明 |
|------|------|------|------|
| Prefill Worker | 1 | N+2 | 1 主线程 + N Uvicorn 线程 + 1 Scheduler 线程 + 1 TUI 线程 |
| Decode Server | 1 | N+2 | 同上 |
| Swarm Manager | 1 | M+1 | 1 主线程 + M Worker 进程管理 |

---

## 二、EP-Group 同步机制

### 2.1 什么是 EP-Group？

EP-Group (Execution Parallelism Group) 是分布式推理中的并行执行单元：

- **一个 EP-Group = N 个 Data Parallel (DP) Workers**
- 所有 DP 共享同一个调度器
- 每次迭代必须 **同步执行** (All-to-All Barrier)

### 2.2 EP-Sync 规则

```
触发条件:
  • 任意 DP 队列非空 AND 组空闲

执行流程:
  1. 同时快照所有队列
  2. 为每个 DP 打包一个 sub-batch
  3. 计算每个 DP 的预估耗时
  4. group_duration = max(所有 DP 耗时)  ← Barrier
  5. sleep(group_duration)  ← 模拟计算
  6. 同时完成所有请求 (Barrier 点)

关键特性:
  • 组 BUSY 时，新请求入队但不调度
  • 空 DP 耗时为 0，但仍参与 Barrier
  • Barrier = group_duration - min(per_dp_time)
```

### 2.3 竞态条件修复

```python
# 错误模式 (会导致请求丢失):
def _scheduler_loop(self):
    while True:
        self._has_work.wait()
        self._has_work.clear()  # ← 立即清除，可能丢失新请求
        while True:
            batches = self._try_form_batches()
            if not batches:
                break

# 正确模式 (修复后):
def _scheduler_loop(self):
    while True:
        self._has_work.wait()
        while True:
            batches = self._try_form_batches()
            if not batches:
                self._has_work.clear()  # ← 确认空后才清除
                break
            self._run_iteration(batches)
```

---

## 三、Prefill Worker 机制

### 3.1 计算模型

Prefill 阶段处理 prompt，计算复杂度为 **二次方**（Attention 机制）：

```
t_dp_i = α × Σ(s_j²) + β × Σ(s_j)

其中:
  s_j  = 第 j 个请求的 prompt token 数
  α    = Attention 二次系数 (默认 1e-3 s/tok²)
  β    = MoE 线性系数 (默认 1e-4 s/tok)

group_duration = max(t_dp_i for i in 0..N-1)
```

**示例计算:**

| DP | Batch | Seq Lens | Compute Time |
|----|-------|----------|--------------|
| DP0 | 1 | [4] | α×16 + β×4 = 0.001s |
| DP1 | 0 | [] | 0s |
| DP2 | 0 | [] | 0s |
| DP3 | 2 | [8, 12] | α×208 + β×20 = 0.002s |

→ `group_duration = max(0.001, 0, 0, 0.002) = 0.002s`

### 3.2 批处理流程

```
┌─ Phase 1: 快速快照 (持锁) ────────────────────────────────────────┐
│  with self._lock:                                                │
│      queue_snapshots = [list(q) for q in self._queues]          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Phase 2: 批处理计算 (无锁) ───────────────────────────────────────┐
│  for dp_id, snapshot in enumerate(queue_snapshots):             │
│      batch = []                                                  │
│      leftover = deque()                                          │
│      batch_tokens = 0                                            │
│                                                                  │
│      for req in snapshot:                                        │
│          if req.prompt_tokens > max_seq_len:                     │
│              DROP request                                        │
│          elif len(batch) >= max_batch_size:                      │
│              leftover.append(req)                                │
│          elif batch_tokens + req.prompt_tokens > max_seq_len:    │
│              leftover.append(req)                                │
│          else:                                                    │
│              batch.append(req)                                   │
│              batch_tokens += req.prompt_tokens                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Phase 3: 更新共享状态 (持锁) ──────────────────────────────────────┐
│  with self._lock:                                                │
│      for dp_id in range(n_dp):                                  │
│          self._queues[dp_id].extendleft(leftovers[dp_id])       │
│          self._active_batches[dp_id] = batches[dp_id]           │
│      self._busy = True                                           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Phase 4: 完成错误请求 (无锁) ──────────────────────────────────────┐
│  for req, error in requests_to_complete:                        │
│      self._complete_request(req, error=error)                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 批处理约束

| 约束 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 单请求最大长度 | max_seq_len | 4096 | 超过则丢弃请求 |
| 每批最大请求数 | max_batch_size | 32 | 每个批次最多 32 个请求 |
| 批次累计 token | max_seq_len | 4096 | 批次内所有请求 prompt token 总和上限 |

### 3.4 KV Transfer 参数

Prefill 完成后返回 KV Cache 信息供 Decode 使用：

```json
{
  "kv_transfer_params": {
    "do_remote_prefill": true,
    "do_remote_decode": false,
    "remote_block_ids": [1551, 1552, 1553],
    "remote_engine_id": "2-1f6fdee284a3426d84eb25b4ebdda6fe_dp0",
    "remote_host": "172.28.239.227",
    "remote_port": 56193,
    "remote_pcp_size": 3,
    "remote_dcp_size": 3,
    "last_token_id": 95316,
    "num_prompt_blocks": 3
  }
}
```

---

## 四、Decode Server 机制

### 4.1 计算模型

Decode 阶段生成 token，计算复杂度为 **线性**（单次解码）：

```
t_dp_i = α × Σ(s_i) + β × batch_size_i

其中:
  s_i          = 第 i 个请求的当前序列长度
  batch_size_i = DP i 的活跃请求数
  α            = 线性系数 (默认 2e-4 s/tok)
  β            = 批次系数 (默认 1e-3 s/req)

group_duration = max(t_dp_i for i in 0..N-1)
```

### 4.2 Continuous Batching

Decode 使用 **Continuous Batching** 模式：

```
每个 DP 有固定容量 (max_batch_size = 8)

状态流转:
  WAITING  → 入队等待
  ACTIVE   → 被调度执行，每个 step 生成 1 token
  DONE     → 生成了 max_tokens 个 token
  ABORTED  → 序列长度超过 max_seq_len

每步流程:
  1. Admit: 从 waiting 队列填充 active 批次的空槽
  2. Compute: 计算各 DP 耗时
  3. Sleep: 模拟计算
  4. Generate: 每个请求生成 1 token
  5. Check: 完成/超长请求移除，立即补充新请求
```

### 4.3 流式响应 (SSE)

```
SSE 格式:

data: {"delta":{"role":"assistant","content":""}}    ← 首包
data: {"delta":{"content":"Starlight "}}             ← Token 1
data: {"delta":{"content":"Sunshine "}}              ← Token 2
data: {"delta":{"content":"Cascade ", "finish_reason":"length"}} ← 最后一个
data: [DONE]                                         ← 结束标记
```

### 4.4 Token Queue 机制

```
┌─ HTTP Handler (async) ──────────────────────────────────────────┐
│  req = ActiveRequest(...)                                       │
│  scheduler.enqueue(req)                                         │
│  async for tok in req.token_stream():                           │
│      if tok is None: break                                      │
│      yield tok  // SSE 输出                                      │
└──────────────────────────────────────────────────────────────────┘
                              │ asyncio.Queue (bounded)
                              ▼
┌─ Scheduler Thread ───────────────────────────────────────────────┐
│  for req in batch:                                              │
│      token = _next_token()                                      │
│      req.generated += 1                                         │
│      req.push_token(token)  // 通过 call_soon_threadsafe        │
│      if req.is_done:                                            │
│          req.push_token(None)  // Sentinel                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 五、线程模型与并发控制

### 5.1 线程架构

```
┌─ Main Thread ───────────────────────────────────────────────────┐
│  - 启动所有组件                                                  │
│  - 等待 Ctrl-C                                                  │
│  - 协调关闭                                                      │
└──────────────────────────────────────────────────────────────────┘

┌─ Uvicorn Threads (N个) ─────────────────────────────────────────┐
│  DP-0 Thread:                                                    │
│    - 运行 asyncio event loop                                    │
│    - 处理 HTTP 请求                                              │
│    - enqueue() 调用                                              │
│                                                                  │
│  DP-1 Thread:                                                    │
│    - 同上                                                        │
│  ...                                                             │
└──────────────────────────────────────────────────────────────────┘

┌─ Scheduler Thread ───────────────────────────────────────────────┐
│  - 后台 daemon 线程                                              │
│  - 运行调度循环                                                  │
│  - 执行 EP-Sync 批处理                                           │
│  - 通过 call_soon_threadsafe 回调 asyncio Event                 │
└──────────────────────────────────────────────────────────────────┘

┌─ TUI Thread (可选) ──────────────────────────────────────────────┐
│  - Rich Live 渲染                                                │
│  - 定时读取 scheduler 状态                                       │
│  - 渲染 Dashboard                                                │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 线程安全机制

```python
# 共享数据结构
self._queues: List[Deque]       # N 个等待队列
self._active: List[List]        # N 个活跃批次
self._lock: threading.Lock      # 保护共享状态

# 锁的最小化范围 (修复后):
with self._lock:
    snapshot = [list(q) for q in self._queues]  # 快速快照

# ... 无锁进行批处理计算 ...

with self._lock:
    self._queues[dp_id] = leftover               # 快速更新
    self._busy = True

# 跨线程通信
loop.call_soon_threadsafe(event.set)            # 通知 asyncio
loop.call_soon_threadsafe(queue.put_nowait)     # 推送 token
```

### 5.3 asyncio 与线程的桥接

```
┌─ FastAPI Coroutine ─────────────────────────────────────────────┐
│  req = PendingRequest(...)                                      │
│  scheduler.enqueue(req, loop)                                   │
│  await req.wait()  ← 阻塞等待                                   │
│  return req.result                                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ asyncio.Event
                              │
                              ▼
┌─ Scheduler Thread ───────────────────────────────────────────────┐
│  # 计算完成                                                      │
│  req.result = {...}                                             │
│  loop.call_soon_threadsafe(req._event.set)  ← 通知              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 六、API 接口

### 6.1 OpenAI 兼容 API

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/v1/completions` | POST | 文本补全 | 返回随机 token |
| `/v1/chat/completions` | POST | 聊天补全 | 支持 messages 格式 |
| `/health` | GET | 健康检查 | 返回 DP 状态 |
| `/group/status` | GET | 组状态 | 返回 EP-Group 状态 |

### 6.2 请求格式

```json
// POST /v1/completions
{
  "prompt": "Hello world",
  "max_tokens": 16,
  "kv_transfer_params": { ... }  // 可选，分布式推理参数
}

// POST /v1/chat/completions
{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 16,
  "stream": true  // Decode Server 支持
}
```

### 6.3 响应格式

```json
// Prefill 响应
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "choices": [{"text": "k", "finish_reason": "length"}],
  "usage": {"prompt_tokens": 4, "completion_tokens": 1},
  "kv_transfer_params": {...}
}

// Decode 响应 (非流式)
{
  "id": "chatcmpl-xxx",
  "choices": [{
    "message": {"content": "Starlight Sunshine Cascade ..."},
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 1, "completion_tokens": 16}
}
```

---

## 七、输入验证与安全

### 7.1 请求参数限制

| 参数 | 限制 | 错误消息 |
|------|------|----------|
| prompt 长度 | ≤ 100000 chars | `Prompt exceeds maximum length` |
| max_tokens | 1 ~ 8192 | `max_tokens must be between 1 and 8192` |
| messages 数量 | ≤ 100 | `Too many messages (max 100)` |
| message 内容 | ≤ 50000 chars | `Message content too long` |

### 7.2 CLI 参数验证

```python
if args.n_dp < 1:
    p.error("--n-dp must be at least 1")
if args.max_batch < 1:
    p.error("--max-batch must be at least 1")
if args.alpha < 0:
    p.error("--alpha must be non-negative")
```

### 7.3 JSON 解析异常处理

```python
try:
    body = await request.json()
except Exception as e:
    log.warning("Failed to parse JSON body: %s", e)
    raise HTTPException(status_code=400, detail="Invalid JSON body")
```

---

## 八、TUI Dashboard

### 8.1 功能概览

```
┌─ PREFILL WORKER DASHBOARD ──────────────────────────────────────┐
│  4 DP Workers — Ports 8100-8103                                  │
├─ EP-Group Status ───────────────────────────────────────────────┤
│  Status: IDLE 💤  |  Iteration: 42  |  Total Iters: 42          │
├─ Data Parallel Workers ──────────────────────────────────────────┤
│  DP0 💤   Queue: 0   Tokens: 0   Batch: 0   Compute: 0.000s      │
│  DP1 ✅   Queue: 3   Tokens: 12  Batch: 2   Compute: 0.002s      │
│  DP2 ⚠️   Queue: 8   Tokens: 45  Batch: 4   Compute: 0.004s      │
│  DP3 🔥   Queue: 15  Tokens: 80  Batch: 8   Compute: 0.008s      │
├─ Summary ────────────────────────────────────────────────────────┤
│  Queued: 26 req (137 tok)  |  Active: 14 req  |  Max: 0.008s     │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 刷新机制

- 使用 Rich Live 实现无闪烁更新
- 默认刷新率 0.5s (可配置)
- 日志使用 `deque(maxlen)` 实现 O(1) 旋转

---

## 九、Swarm Manager (多进程模式)

### 9.1 架构

```
┌─ Swarm Manager Process ──────────────────────────────────────────┐
│                                                                  │
│  ┌─ Worker 0 Process ───────────────────────────────────────────┐│
│  │  EP-Group 0: Ports 8100-8103                                 ││
│  │  独立 Scheduler                                               ││
│  └───────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─ Worker 1 Process ───────────────────────────────────────────┐│
│  │  EP-Group 1: Ports 8200-8203                                 ││
│  │  独立 Scheduler                                               ││
│  └───────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─ Worker 2 Process ───────────────────────────────────────────┐│
│  │  EP-Group 2: Ports 8300-8303                                 ││
│  └───────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─ Shared State (multiprocessing.Manager) ──────────────────────┐│
│  │  Manager.dict(): 跨进程状态共享                              ││
│  │  Manager.Queue(): 跨进程日志传输                             ││
│  └───────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─ Unified TUI Dashboard ──────────────────────────────────────┐│
│  │  监控所有 Worker 的状态                                       ││
│  └───────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 状态共享机制

```python
# Worker 进程内
shared_state[f"worker_{worker_id}"] = {
    "worker_id": worker_id,
    "iteration": iteration_count,
    "dp_statuses": [...]
}

# Dashboard 进程内
for worker_id in range(n_workers):
    status = shared_state[f"worker_{worker_id}"]
    # 渲染 Worker 状态
```

---

## 十、启动与测试

### 10.1 启动命令

```bash
# Prefill Worker (单进程)
uv run python prefill_worker.py --n-dp 4 --base-port 8100 --tui

# Decode Server (单进程)
uv run python decode_server.py --n-dp 4 --base-port 9100 --tui

# Swarm Manager (多进程)
uv run python prefill_worker_swarm.py --n-workers 3 --dp-per-worker 4 --tui
```

### 10.2 参数说明

| 参数 | Prefill 默认 | Decode 默认 | 说明 |
|------|-------------|-------------|------|
| --n-dp | 2 | 2 | DP Worker 数量 |
| --base-port | 8100 | 9100 | 起始端口 |
| --alpha | 1e-3 | 2e-4 | 计算系数 |
| --beta | 1e-4 | 1e-3 | 计算系数 |
| --max-batch | 32 | 8 | 批次大小 |
| --max-seq-len | 4096 | 2048 | 序列长度上限 |

### 10.3 测试请求

```bash
# Prefill 测试
curl -X POST http://localhost:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello", "max_tokens": 16}'

# Decode 测试 (流式)
curl -X POST http://localhost:9100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role":"user","content":"Hello"}], "max_tokens": 10, "stream": true}'

# 健康检查
curl http://localhost:8100/health
curl http://localhost:8100/group/status
```

---

## 十一、关键设计决策

### 11.1 为什么使用线程而非协程？

- Scheduler 需要执行阻塞操作 (`time.sleep`)
- 不能阻塞 asyncio event loop
- 独立线程可并行处理请求入队和批处理

### 11.2 为什么锁范围要最小化？

- 锁范围过大导致入队阻塞
- 高并发下请求堆积
- 修复后：快照-计算-更新三阶段分离

### 11.3 为什么 asyncio.Queue 要有界？

- 无界队列可被恶意请求耗尽内存
- `maxsize = max_tokens + 1` 限制内存使用
- Sentinel None 不计入 token 数

### 11.4 为什么用 deque 替代 list？

- `list.pop(0)` 是 O(n) 操作
- `deque(maxlen)` 是 O(1) 且自动溢出
- 日志高频写入需要高效操作

---

## 十二、总结

本模拟器实现了分布式 LLM 推理的核心机制：

1. **EP-Group 同步** - All-to-All Barrier 模拟
2. **Prefill 计算** - 二次方复杂度 Attention 模型
3. **Decode 计算** - 线性复杂度 Continuous Batching
4. **线程安全** - 最小化锁范围 + asyncio 桥接
5. **OpenAI 兼容** - 标准 API 格式 + KV Transfer
6. **实时监控** - Rich TUI Dashboard

该模拟器可用于：
- 分布式推理系统原型验证
- EP-Group 调度算法测试
- 负载均衡策略模拟
- 性能调优实验