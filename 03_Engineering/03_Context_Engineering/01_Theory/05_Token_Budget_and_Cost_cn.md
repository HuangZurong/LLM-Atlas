# 05 · Token 预算管理与成本控制

*前置要求：[04_Context_Composition_cn.md](04_Context_Composition_cn.md)。*
*在 CE 流水线中的位置：第 2 步（Budget & Sort）和第 3 步（Compress & Degrade）*

---

Token 预算管理（Token Budget Management）是在所有上下文层之间**分配、监控并优化**有限 token 空间的实践。它是上下文工程与成本工程的交汇点。

## 1. 预算模型

每次 LLM 调用都有一个固定的总预算：

```
总预算 = 模型的最大上下文窗口（例如 128K tokens）

分配示例（16K 预算）：
┌──────────────────────────────────────────────┐
│ System Prompt（固定）               ~500T    │
│ Background / Few-shots（固定）      ~1000T   │
│ Retrieved Memory（可变）            ~2000T   │
│ RAG Context（可变）                 ~4000T   │
│ Conversation History（可变）        ~3000T   │
│ Current User Query                  ~200T    │
│ Tool Results（可变）                ~1000T   │
│ ──────────────────────────────────────────── │
│ 预留给输出                           ~4300T   │
│ ──────────────────────────────────────────── │
│ 总计                                16000T   │
└──────────────────────────────────────────────┘
```

**关键规则（Critical Rule）**：始终至少为输出预留**20–30% 的预算**。如果你把 95% 的上下文都塞给输入，模型就没有空间去推理或生成完整响应。

## 2. 基于优先级的分配算法

```python
def allocate_budget(
    total_budget: int,
    system_tokens: int,
    query_tokens: int,
    output_reserve_ratio: float = 0.25,
) -> dict[str, int]:
    output_reserve = int(total_budget * output_reserve_ratio)
    available = total_budget - system_tokens - query_tokens - output_reserve

    return {
        "recent_history": int(available * 0.35),  # P2: 保持逐字内容
        "rag_context":    int(available * 0.35),  # P3: 减少 chunks
        "memory":         int(available * 0.20),  # P4: 压缩
        "old_history":    int(available * 0.10),  # P5: 摘要/丢弃
    }
```

当某一层超出它的分配额度时，应用 `04_Context_Composition_cn.md` 中优先级表定义的裁剪策略。

## 3. 压缩策略

当必须裁剪时，应为不同层选择正确的策略：

| 策略 | 机制 | 信息损失 | 成本 |
| :--- | :--- | :--- | :--- |
| **截断（Truncation）** | 丢掉最老的 tokens | 高（丢失开头） | 零 |
| **滑动窗口（Sliding Window）** | 保留最后 N 个 tokens | 中（丢失长程信息） | 零 |
| **抽取式摘要（Extractive Summary）** | 保留关键句子 | 低到中 | 低（正则/启发式） |
| **抽象式摘要（Abstractive Summary）** | 由 LLM 生成摘要 | 低 | 中（一次 LLM 调用） |
| **实体压缩（Entity Compression）** | 抽取结构化事实 | 很低 | 中 |
| **语义去重（Semantic Deduplication）** | 删除近重复 chunks | 低 | 中（embedding） |

**经验法则（Rule of thumb）**：对对话历史使用便宜策略（截断、滑动窗口）。对长期记忆和关键文档使用更高质量的策略（抽象式摘要、实体压缩）。

## 4. 成本优化策略

### 4.1 前缀缓存

缓存静态前缀（System Prompt + Few-shots），避免每次请求都重复计算 KV 状态。各供应商的具体设置见 `03_Context_Window_Mechanics_cn.md`。

**预期节省**：对于高流量应用，输入 token 成本可降低 30–60%。

### 4.2 分层模型策略

把更便宜的模型用于上下文预处理，把更贵的模型只用于最终响应：

```
用户查询
    │
    ▼
[便宜模型：gpt-4o-mini / claude-haiku]
    ├── 记忆摘要
    ├── 实体抽取
    ├── 查询分类
    └── RAG 重排
    │
    ▼
[昂贵模型：gpt-4o / claude-sonnet]
    └── 面向最终用户的回复
```

### 4.3 惰性加载上下文

不要在每次请求时都注入所有上下文层。按条件加载：

| 条件 | 加载 Memory？ | 加载 RAG？ | 加载完整历史？ |
| :--- | :--- | :--- | :--- |
| 新会话第一轮 | ✅（用户画像） | 视查询而定 | ❌ |
| 用户提到过去上下文 | ✅ | ❌ | ✅ |
| 事实型 / 知识型查询 | ❌ | ✅ | ❌ |
| 轻松闲聊 | ❌ | ❌ | ✅（仅最近） |

### 4.4 语义缓存

对语义相似的查询缓存完整 LLM 响应。如果一个新查询与缓存查询的 cosine distance < 0.05，就直接返回缓存响应。

**最适合**：FAQ 类应用、在同一数据集上重复出现的分析型查询。

## 5. 监控与告警

在生产环境中追踪这些指标：

| 指标 | 告警阈值 | 动作 |
| :--- | :--- | :--- |
| **平均每请求输入 Tokens** | > 上下文窗口的 80% | 审查 memory/RAG 注入逻辑 |
| **输出截断率** | >5% 的响应 | 提高输出预留比例 |
| **上下文组装延迟** | >200ms p95 | 优化检索或降低 top-k |
| **记忆检索延迟** | >500ms p95 | 优化向量索引 |
| **单次对话成本** | >$0.50 | 审核所注入的上下文是否全部必要 |
| **缓存命中率** | <30% | 审查前缀结构或语义缓存阈值 |

---

## 关键参考文献

1. **Anthropic. (2025). Prompt Caching.** Anthropic API Documentation.
2. **OpenAI. (2024). Prompt Caching.** OpenAI API Documentation.
3. **Zhu, Y., et al. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context.** *ICML 2023*.
