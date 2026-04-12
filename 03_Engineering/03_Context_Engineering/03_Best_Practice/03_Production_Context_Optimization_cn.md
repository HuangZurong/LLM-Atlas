# 生产环境中的上下文优化

*前置要求：[02_Context_Quality_and_Evaluation_cn.md](02_Context_Quality_and_Evaluation_cn.md)。*

---

## 1. 延迟优化

上下文组装涉及多个 I/O 操作（vector DB lookup、memory retrieval、history fetch）。这些步骤必须并行化。

### 异步并行组装

```python
import asyncio

async def assemble_context_async(query: str, session_id: str) -> dict:
    # 并行执行所有检索
    memory_task = asyncio.create_task(retrieve_memory(session_id, query))
    rag_task = asyncio.create_task(retrieve_rag(query))
    history_task = asyncio.create_task(fetch_history(session_id))

    memory, rag_results, history = await asyncio.gather(
        memory_task, rag_task, history_task
    )
    return {"memory": memory, "rag": rag_results, "history": history}
```

**目标延迟**：上下文组装 < 100ms p95。若超出，应分析是哪一步检索成为瓶颈。

### 预热前缀缓存

对于高流量应用，可在启动时发送一个只包含静态前缀（system prompt + few-shots）的“预热”请求。这样正式流量到来前，KV cache 就已建立完成。

### 惰性加载

不要在每次请求时都加载全部上下文层：

| 条件 | 可跳过 |
| :--- | :--- |
| 第一轮，简单 query | Memory retrieval |
| 事实型 Q&A | Conversation history |
| 轻松闲聊 | RAG retrieval |
| 简短、明确的 query | Memory retrieval |

---

## 2. 成本优化

### 前缀缓存投入产出比（ROI）

```
每日节省 = (cached_tokens_per_request × requests_per_day × cache_discount × price_per_token)

示例（Anthropic claude-3-5-sonnet）：
  cached_tokens = 1000（system prompt + few-shots）
  requests/day  = 10,000
  cache_discount = 90%（Anthropic）
  price          = $0.003 / 1K input tokens

  Daily savings = 1000 × 10,000 × 0.90 × $0.000003 = $27/day = ~$810/month
```

### 分层模型策略

| 任务 | 模型 | 原因 |
| :--- | :--- | :--- |
| 记忆摘要 | claude-haiku / gpt-4o-mini | 便宜、快、效果足够 |
| 实体抽取 | claude-haiku / gpt-4o-mini | 结构化输出，复杂度低 |
| RAG 重排 | claude-haiku / gpt-4o-mini | 二分类相关性判断 |
| Query 分类 | claude-haiku / gpt-4o-mini | 简单分类 |
| 最终用户回复 | claude-sonnet / gpt-4o | 这里质量最关键 |

### 语义去重

在注入多个 RAG chunks 之前，先去掉近重复内容：

```python
def deduplicate_chunks(chunks: list[str], threshold: float = 0.92) -> list[str]:
    """移除与已选 chunk 的 cosine similarity 超过阈值的 chunk。"""
    embeddings = embed(chunks)
    selected = [0]
    for i in range(1, len(chunks)):
        similarities = cosine_similarity(embeddings[i], [embeddings[j] for j in selected])
        if max(similarities) < threshold:
            selected.append(i)
    return [chunks[i] for i in selected]
```

---

## 3. 可靠性模式

### 优雅降级

为每个上下文层定义 fallback 行为：

| 层 | Primary | Fallback |
| :--- | :--- | :--- |
| RAG context | Vector DB retrieval | 空（模型退回参数记忆） |
| Memory | Memory service | 空（无状态响应） |
| History | Session store | 从请求 payload 中只取最近 2 轮 |
| System prompt | Config service | 硬编码默认值 |

### 上下文校验

在发送给 LLM 前，校验组装后的上下文：

```python
def validate_context(context: str) -> list[str]:
    issues = []
    tokens = count_tokens(context)
    if tokens > MAX_INPUT_TOKENS * 0.95:
        issues.append(f"Context too large: {tokens} tokens")
    if detect_prompt_injection(context):
        issues.append("Potential prompt injection detected")
    if detect_pii(context):
        issues.append("PII detected in context")
    return issues
```

### 幂等组装

在相同输入下，上下文组装应是确定性的。应避免：
- 检索中的随机采样（使用确定性的 top-k）
- 静态层中的时间相关内容
- 非确定性压缩（摘要时使用 temperature=0）

---

## 4. 可观测性

### 关键指标

| 指标 | 告警阈值 | Dashboard |
| :--- | :--- | :--- |
| Avg input tokens / request | > 窗口的 80% | Yes |
| Output truncation rate | >5% | Yes |
| Context assembly latency p95 | >200ms | Yes |
| Cache hit rate | <30% | Yes |
| Context relevance score avg | <0.5 | Yes |
| Cost per 1K requests | >budget | Yes |
| Compression trigger rate | >20% of requests | Warning |

### 分布式追踪

将每一步组装过程都打成 span：

```
request_id: abc123
├── retrieve_memory        12ms
├── retrieve_rag           45ms  ← 瓶颈
├── fetch_history           8ms
├── assemble_context        3ms
│   ├── token_count        1ms
│   ├── priority_trim      1ms
│   └── sandwich_wrap      1ms
└── llm_call             820ms
    ├── ttft              340ms
    └── generation        480ms
```

### 上下文差异日志

记录轮次之间的上下文变化，用来调试异常行为：

```python
context_diff = {
    "turn": turn_number,
    "added_layers": [...],
    "removed_layers": [...],
    "token_delta": current_tokens - previous_tokens,
    "compression_applied": bool,
}
```

### 成本归因

按上下文层拆解成本，以识别优化空间：

```
Request cost breakdown:
  system_prompt:    $0.0015  (500T, cached → $0.00015 effective)
  rag_context:      $0.0120  (4000T, not cached)
  history:          $0.0090  (3000T, not cached)
  user_query:       $0.0006  (200T)
  output:           $0.0240  (800T at output rate)
  ─────────────────────────
  Total:            $0.0471
```

---

## 关键参考文献

1. **Anthropic. (2025). Prompt Caching.** Anthropic API Documentation.
2. **OpenAI. (2024). Latency Optimization Guide.** OpenAI Documentation.
3. **Arize AI. (2024). LLM Observability Best Practices.** Arize Documentation.
