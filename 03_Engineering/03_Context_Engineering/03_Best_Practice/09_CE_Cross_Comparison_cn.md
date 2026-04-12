# 09 · 上下文工程横向比较

*对 Anthropic、OpenAI、Google/ADK、Cursor 以及主要代理框架的 CE 实践进行结构化比较。*

---

## 1. 理念对比

| 厂商 / 框架 | 核心隐喻 | CE 理念 |
|--------|--------------|---------------|
| **Anthropic** | 上下文是一份有限的注意力预算 | 最小高信号 token 集；Context Rot 是核心敌人 |
| **OpenAI** | LLM 是 CPU，上下文是 RAM，开发者是 OS | Write / Select / Compress / Isolate 四种操作 |
| **Google/ADK** | State is the Context Bus | 结构化状态传递 > 对话历史；三种作用域前缀 |
| **Cursor** | 分层检索流水线 | 选择质量胜过窗口大小；两阶段检索 |
| **LangGraph** | 带 reducers 的 Graph State | 通过按字段合并策略控制上下文流 |
| **CrewAI** | 四层记忆架构 | 短期 / 长期 / 实体 / 用户记忆分离 |
| **MemGPT/Letta** | 虚拟内存管理 | 由代理自己管理上下文的换入 / 换出 |
| **Devin** | 环境就是记忆 | 文件系统 / 终端 / 浏览器是外部存储；按需重读 |
| **DSPy** | 程序化优化 | 自动搜索最优的上下文配置 |

---

## 2. 上下文编排策略

### 2.1 什么进入上下文

| 组成部分 | Anthropic | OpenAI | Google/ADK | Cursor |
|-----------|-----------|--------|------------|--------|
| **System prompt** | “正确高度”—— 合同格式 | CTCO 模式（Context/Task/Constraints/Output） | 带 `{state}` 模板变量的 instruction | `.cursorrules` 项目级规则 |
| **Tools** | 最小工具集、优秀描述、响应 < 25K tokens | tool search 实现惰性加载；也可通过 fine-tune 降 token | 在 Agent 上定义；通过 `ContextCacheConfig` 缓存 | N/A（工具是 IDE 动作） |
| **Few-shot examples** | 多样、典型、非穷尽 | 纳入 system prompt | 通过 instruction text | N/A |
| **History** | Compaction + context editing | trimming 或 compression | `ContextFilterPlugin`、`include_contents="none"` | 按来源分配 token 预算 |
| **Retrieved data** | 通过 tools 做 Just-in-Time 加载 | 通过 tool outputs 做 RAG | Memory services（RAG、Memory Bank） | 两阶段检索（embedding + reranker） |
| **External memory** | `claude-progress.txt`、CLAUDE.md | YAML frontmatter + Markdown notes | `app:` / `user:` / `temp:` state scopes | `.cursorrules` |

### 2.2 动态与静态上下文

| 方式 | 使用者 | 做法 |
|----------|-------------|-----|
| **静态指令** | 所有人 | system prompt、rules files |
| **动态指令** | OpenAI、Google/ADK | 在运行时生成 instruction 的函数 |
| **模板注入** | Google/ADK | 在 instruction 中使用 `{variable}`，再由 state 解析 |
| **基于工具的加载** | Anthropic、OpenAI、Devin | 代理通过工具按需获取上下文 |
| **检索流水线** | Cursor、CrewAI | Embedding search → reranking → injection |

---

## 3. 上下文窗口管理

### 3.1 压缩与上下文紧缩（Compaction）

| 厂商 / 框架 | 机制 | 触发条件 | 保留什么 | 丢弃什么 |
|--------|-----------|---------|-----------------|-----------------|
| **Anthropic** | LLM 摘要 | 接近窗口上限 | 架构决策、未解决 bug、实现细节 | 冗余工具输出、旧消息 |
| **OpenAI** | 加密 compaction item | `compact_threshold`（可配置 token 数） | 用户消息原样保留；压缩项中保留模型潜在理解 | assistant 消息、tool calls、reasoning |
| **Google/ADK** | `ContextFilterPlugin` | `num_invocations_to_keep` | 最近 N 次 invocation | 更老 invocation |
| **LangGraph** | 摘要节点 | 自定义触发器 | 旧消息的摘要 | 原始旧消息 |
| **AutoGen** | Head-and-Tail | 缓冲区大小 | 前 K + 后 N 条消息 | 中间部分 |

### 3.2 降级链

**Anthropic**（最成熟）：
```
Level 1: 清除旧工具结果（替换为 placeholder）
Level 2: 清除旧 thinking blocks
Level 3: 对历史做 LLM 摘要
Level 4: 丢弃可选上下文源
Level 5: 借助 progress file 做完整 compaction
```

**OpenAI**：
```
Level 1: Context trimming（丢弃旧轮次）
Level 2: Context compression（摘要旧轮次）
Level 3: 服务端 compaction（加密 compaction item）
```

**Google/ADK**：
```
Level 1: ContextFilterPlugin（仅保留最近 N 次 invocation）
Level 2: include_contents="none"（完全剥离历史，只通过 state 注入）
Level 3: 基于 branch 的隔离（按代理分支过滤）
```

---

## 4. 多代理上下文隔离

| 厂商 / 框架 | 隔离机制 | 粒度 |
|--------|-------------------|-------------|
| **Anthropic** | 每个 subagent 独立上下文窗口；orchestrator-worker 模式 | 完全隔离（独立 sessions） |
| **OpenAI** | handoff 上的 `input_filter`；`remove_all_tools`；nested handoff history | 每次 handoff 可配置 |
| **Google/ADK** | `include_contents="none"` + state injection；按 branch 过滤 events | 按 agent 或 branch |
| **LangGraph** | 图状态作用域；每个 node 只看自己的输入 state | 按 node |
| **CrewAI** | 任务级上下文；每个 agent 获取任务描述 + 相关 memories | 按 task |
| **AutoGen** | `ChatCompletionContext` 协议；自定义 context classes | 按 agent |
| **MemGPT** | 每个 agent 独立管理自己的 memory | 按 agent（自管理） |

---

## 5. 记忆架构

### 5.1 记忆层级

| 层级 | Anthropic | OpenAI | Google/ADK | CrewAI | MemGPT |
|------|-----------|--------|------------|--------|--------|
| **Working**（上下文窗口内） | 当前上下文 | 当前消息 | 当前 state + history | 当前任务上下文 | Main context（core memory + recent messages） |
| **Session**（当前 session） | 工具结果、对话 | 通过 `save_note` 保存的 session notes | `temp:` 前缀 state | Short-term memory（vector） | Recall memory |
| **Persistent**（跨 session） | `claude-progress.txt`、CLAUDE.md、MEMORY.md | YAML frontmatter、整合后的全局记忆 | `app:` 和 `user:` 前缀 state；Memory Bank | Long-term memory（SQLite）、User memory | Archival memory（vector） |
| **Entity**（关系跟踪） | — | — | — | Entity memory（graph） | — |

### 5.2 记忆生命周期

**OpenAI 四阶段流水线**：
```
Inject → Distill → Consolidate → Reinject
```

**Anthropic**：
```
Load（CLAUDE.md + progress file）→ Work → Compact → Save（progress file + git commit）
```

**MemGPT**：
```
Compile context → Work → Self-manage（insert/search/replace/delete）→ Persist
```

**CrewAI**：
```
Load（四种 memory）→ 每步 RAG 检索 → 累积 → Persist
```

---

## 6. 缓存策略

| 厂商 / 框架 | 缓存机制 | 关键细节 |
|--------|------------------|-------------|
| **Anthropic** | 使用 `cache_control` 的 prompt caching | 最多 4 个断点；顺序为 tools → system → messages；最少 1,024 tokens；5 分钟 TTL（新模型可达 1 小时） |
| **OpenAI** | 自动缓存 | 50–90% 折扣；静态前置、动态后置；Responses API 可将利用率提高到 80% |
| **Google/ADK** | `ContextCacheConfig` + Gemini cached content API | 基于 fingerprint；第二次匹配请求时创建；30 分钟 TTL；后续请求会剥离已缓存内容 |
| **Cursor** | Speculative edits（预计算补全） | 预先计算下一步可能编辑并缓存，实现瞬时交付 |

---

## 7. 检索策略

| 厂商 / 框架 | 检索方式 | 关键创新 |
|--------|-------------------|----------------|
| **Anthropic** | 通过 tools 做 Just-in-Time 加载 | 轻量引用 → 按需加载；上下文可减少 95% |
| **OpenAI** | 基于工具的检索 + tool search 惰性加载 | 只有需要时才加载工具定义 |
| **Google/ADK** | Memory services（InMemory、Vertex RAG、Memory Bank） | Memory Bank 存“事实”而不是原始 events |
| **Cursor** | Embedding → cross-encoder reranking | 两阶段检索相较纯 embedding 显著提高 precision |
| **CrewAI** | 横跨四种 memory 的 RAG | 每种 memory 各自有检索机制 |
| **MemGPT** | 由代理主动 `memory_search` | 由代理决定何时检索、检索什么 |

---

## 8. 各家的独特创新

| 厂商 / 框架 | 独特创新 | 说明 |
|--------|------------------|-------------|
| **Anthropic** | Tool Result Clearing | 用 placeholder 替换旧工具结果；token 降低 84% |
| **Anthropic** | Thinking Block Clearing | 清除旧 reasoning blocks，仅保留最近部分 |
| **OpenAI** | 加密 compaction items | 不透明但语义丰富的压缩上下文 |
| **OpenAI** | handoff 上的 `input_filter` | 在代理交接时做外科手术式上下文过滤 |
| **Google/ADK** | `include_contents="none"` | 完全剥离历史，只用 state 注入 |
| **Google/ADK** | 三作用域 state 前缀 | 用 `app:` / `user:` / `temp:` 控制可见性 |
| **Cursor** | Cross-encoder reranking | 在 embedding recall 之后做第二阶段精筛 |
| **Cursor** | Speculative edits | 预计算上下文，实现即刻建议 |
| **LangGraph** | Reducer-based state merging | 按字段定义合并策略 |
| **CrewAI** | Entity Memory | 存储实体关系图，而不仅是文本 |
| **MemGPT** | 自管理记忆 | 代理用工具管理自己的上下文窗口 |
| **Devin** | Environment-as-memory | 把文件系统 / 终端作为外部存储 |
| **AutoGen** | Head-and-Tail context | 保留前 K + 后 N，丢掉中间 |
| **DSPy** | 程序化优化 | 自动搜索最优上下文配置 |

---

## 9. 性能数据比较

| 指标 | 来源 | 数据 |
|--------|--------|------|
| Context editing token reduction | Anthropic | **84%** |
| Memory tool + context editing improvement | Anthropic | **39%** over baseline |
| Multi-agent vs. single-agent | Anthropic | **90.2%** improvement on research tasks |
| Token usage → performance correlation | Anthropic | **80%** of performance variance explained |
| Multi-agent token overhead | Anthropic | **~15x** more tokens than single-agent |
| Just-in-Time context reduction | Anthropic | **95%** via tool lazy loading |
| Long context query placement improvement | Anthropic | Up to **30%** with queries at end |
| Prompt caching cost reduction | Anthropic | Up to **90%** |
| Prompt caching latency reduction | Anthropic | Up to **85%** |
| Chat → Responses API cache utilization | OpenAI | 40% → **80%** |
| Cached token cost reduction（o4-mini） | OpenAI | **75%** cheaper |
| Planning prompt improvement（SWE-bench） | OpenAI | **4%** pass rate increase |
| Reasoning performance degradation | OpenAI | Starts at ~**3,000 tokens** |
| Performance ceiling | Anthropic | ~**1M tokens** |

---

## 10. 决策矩阵：何时用什么

| 场景 | 推荐方式 | 灵感来源 |
|----------|---------------------|-------------|
| 简单单代理聊天 | 清晰 system prompt + few-shot examples | Anthropic Level 1-3 |
| 多轮且历史持续增长 | trimming + summarization | OpenAI Phase 4, LangGraph |
| 多代理流水线 | 基于 state 的上下文传递 + 隔离 | Google/ADK 的 `output_key` + `include_contents="none"` |
| 代码感知上下文 | AST chunking + embedding + reranking | Cursor 两阶段检索 |
| 长时间运行的自主任务 | progress files + git checkpoints + compaction | Anthropic harness pattern |
| 跨 session 个性化 | 四阶段记忆流水线 | OpenAI inject → distill → consolidate → reinject |
| 实体关系复杂的领域 | 带关系跟踪的 entity memory | CrewAI entity memory |
| 需要近乎无限上下文 | 自管理虚拟内存 | MemGPT/Letta |
| 成本敏感的生产环境 | prompt caching + model routing | Anthropic caching + model routing |
| 对延迟极度敏感的功能 | speculative pre-loading + custom small models | Cursor speculative edits |

---

## 11. 各家最值得拿走的一点

| 来源 | 最有价值的洞见 |
|--------|----------------------|
| **Anthropic** | **降级链**（clear → summarize → drop）而不是单一策略 |
| **OpenAI** | **四阶段记忆流水线**（inject → distill → consolidate → reinject） |
| **Google/ADK** | **`include_contents="none"` + state injection** — 最干净的隔离方式 |
| **Cursor** | **两阶段检索**（embedding recall → cross-encoder rerank） |
| **LangGraph** | **Reducer 模式** — 每个 state 字段都能有自己的合并策略 |
| **CrewAI** | **Entity memory** — 跟踪关系图，而不只是存文本 |
| **MemGPT** | **自管理记忆** — 由 LLM 决定记住什么、忘掉什么 |
| **Devin** | **Environment as memory** — 不要全记住，要知道去哪里找 |
| **AutoGen** | **Head-and-Tail** — 锚住开头、保留最近、丢弃中间 |
| **DSPy** | **程序化优化** — 自动搜索最优上下文配置 |
| **学术界** | **Lost in the Middle** — 重要信息应放开头或结尾，不要埋中间 |

---

## 参考资料

详见各单独实践文档：
- [04_Anthropic_CE_Practices](04_Anthropic_CE_Practices_cn.md)
- [05_OpenAI_CE_Practices](05_OpenAI_CE_Practices_cn.md)
- [06_Google_CE_Practices](06_Google_CE_Practices_cn.md)
- [07_Cursor_CE_Practices](07_Cursor_CE_Practices_cn.md)
- [08_Frameworks_CE_Practices](08_Frameworks_CE_Practices_cn.md)
