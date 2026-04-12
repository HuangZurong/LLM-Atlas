# 10 · 上下文工程框架设计文档

## 1. 执行摘要

**Context Engineering（CE）Framework** 是一个可配置、可插拔、且与厂商无关的 LLM 代理上下文生命周期管理框架。Prompt Engineering 关注的是如何与模型“说话”，而这个框架关注的是：哪些信息、按什么顺序、在什么约束下进入 LLM 那份有限的注意力预算。

该框架旨在通过系统化方式加载、选择、压缩并隔离上下文，从而缓解 **Context Rot** 与 **预算分配低效** 问题，并支持多代理协作与长时间运行的 session。

---

## 2. 设计原则

- **“Do the simplest thing that works”**：受 Anthropic 启发。框架采用 opt-in 设计；简单代理不应被额外开销拖累。
- **配置驱动的机械操作**：预算、缓存、截断规则都用 YAML 配置。
- **插件驱动的语义操作**：摘要、重排、蒸馏等能力作为可插拔策略实现。
- **厂商无关**：可接入任意 LLM provider（Gemini、Claude、GPT 等）和任意编排框架（ADK、LangGraph、OpenAI SDK）。
- **默认可观测**：每一步上下文组装都内建 token 统计与精度指标。
- **渐进式降级**：上下文不是非黑即白地截断，而是沿 “Clear -> Summarize -> Drop” 的降级链处理。

---

## 3. 架构概览

```text
┌─────────────────────────────────────────────────────────────┐
│                    Context Engineering Framework            │
│                                                             │
│  ┌───────────────────┐      ┌─────────────────────────┐     │
│  │  Context Registry │      │     Budget Manager      │     │
│  │ (Sources & Cache) │      │  (Counting & Triggers)  │     │
│  └─────────┬─────────┘      └────────────┬────────────┘     │
│            │                             │                  │
│  ┌─────────▼─────────┐      ┌────────────▼────────────┐     │
│  │  Retrieval Engine │      │   Compression Pipeline  │     │
│  │ (Recall & Rerank) │      │   (Degradation Chain)   │     │
│  └─────────┬─────────┘      └────────────┬────────────┘     │
│            │                             │                  │
│  ┌─────────▼─────────┐      ┌────────────▼────────────┐     │
│  │   State Router    │      │     Memory Manager      │     │
│  │ (Scopes & Merging)│      │  (Hierarchical Tiers)   │     │
│  └─────────┬─────────┘      └────────────┬────────────┘     │
│            │                             │                  │
│            └──────────────┬──────────────┘                  │
│                           │                                 │
│                ┌──────────▼──────────┐                      │
│                │ Isolation Controller│                      │
│                │  (Context Sandboxing)                      │
│                └──────────┬──────────┘                      │
└───────────────────────────┼─────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     Integration Layer     │
              │ (ADK / LangGraph / OpenAI)│
              └───────────────────────────┘
```

### 3.1 七个核心模块

1. **Context Registry**：管理来源注册（DB、文件、历史），具备分层优先级系统和 TTL 缓存。
2. **Budget Manager**：按每个 agent 和每个来源的限制跟踪 token 使用量；阈值越界时触发压缩。
3. **Retrieval Engine**：实现两阶段管线（Semantic Recall -> Cross-Encoder Rerank），筛出高信号 chunks。
4. **Compression Pipeline**：按渐进式策略序列，把信息压缩到预算内。
5. **Memory Manager**：管理三层信息：Working（窗口内）、Short-term（session）、Persistent（长期）。
6. **Isolation Controller**：控制代理间上下文可见性（None、Partial、Full），防止交叉污染。
7. **State Router**：通过带作用域的前缀（`app:`、`user:`、`temp:`）与自定义合并策略，编排结构化数据流。

---

## 4. 模块规范

### 4.1 上下文注册表

Registry 负责定义信息**来自何处**，以及它的**固有价值**。

- **分层优先级**：从 `CRITICAL`（永不丢弃）到 `LOW`（最先丢弃）
- **Just-in-Time Loading**：借鉴 Anthropic；只有被显式引用时才加载工具定义或文档
- **带 TTL 的缓存**：支持内存或 Redis 的重复来源加载缓存

### 4.2 预算管理器

Budget Manager 相当于**操作系统中的内存控制器**。

- **阈值触发器**：`Warning`（60%）、`Compress`（75%）、`Compaction`（90%）
- **可插拔分词器**：支持 Tiktoken、Google GenAI 和 HuggingFace tokenizers

### 4.3 检索引擎

其设计灵感来自 Cursor 的两阶段精确检索。

- **阶段 1（Recall）**：快速向量搜索取回候选 chunks
- **阶段 2（Rerank）**：用 Cross-Encoder 或快速 LLM 重新打分，筛出 Top-K 高信号 chunks
- **AST-Aware**：可选的结构化代码分块

### 4.4 压缩流水线

实现**渐进式降级链**。

```yaml
compression:
  chain:
    - strategy: clear_thinking      # remove old reasoning blocks
    - strategy: clear_tool_results  # keep only latest N results
      keep: 3
    - strategy: degrade_images      # High-res -> Low-res -> Text captions (for multimodal)
    - strategy: perplexity_pruning  # info-theoretic compression of low-signal tokens
    - strategy: summarize           # LLM-based abstractive summary of history
    - strategy: drop_optional       # discard Priority.LOW sources
```

### 4.5 记忆管理器

灵感来自 OpenAI 的四阶段生命周期：**Inject -> Distill -> Consolidate -> Reinject**。

- **Distillation**：定期从 session history 中提取“事实”
- **Consolidation**：把 session 级事实迁移到 user 级持久存储

### 4.6 隔离控制器

它负责控制代理之间那堵“墙”。

- **Level 1（None）**：共享完整对话历史
- **Level 2（Partial）**：过滤后的历史（例如去掉 tool call JSON）
- **Level 3（Full）**：不传历史，只注入 state（Google 模式）

### 4.7 状态路由器

它负责管理结构化的 context bus。

- **Scopes**：`app:`（全局）、`user:`（持久）、`session:`（本次运行）、`temp:`（子任务）
- **Merge Strategies**：`overwrite`、`append`（列表）、`reducer`（自定义逻辑）

---

## 5. 配置模式（示例）

```yaml
# context-engineering.yaml
global:
  default_budget: 8000
  compaction_threshold: 0.75

sources:
  user_profile:
    type: state
    keys: ["user:name", "user:history"]
    priority: critical

  code_snippets:
    type: retrieval
    engine: "two_stage"
    params: { top_k: 5 }
    priority: high

agents:
  coder:
    budget: 16000
    isolation: partial
    sources: [user_profile, code_snippets]
    overflow:
      - strategy: clear_thinking
      - strategy: clear_tool_results
      - strategy: perplexity_pruning
      - strategy: summarize
```

---

## 6. 集成模式

- **Google ADK**：通过 `before_agent_callback`
- **OpenAI Agents SDK**：通过 `RunContextWrapper`
- **LangGraph**：通过 state reducers 和 node wrappers

---

## 7. 实施路线图

1. **Phase 1（Core）**：Registry、Budgeting 与 Truncation
2. **Phase 2（Selection）**：Embedding search 与 Cross-Encoder reranking
3. **Phase 3（Lifecycle）**：四阶段 memory pipeline 与 State Router
4. **Phase 4（Advanced）**：实现学术范式（Perplexity pruning、结构化数据 verbalization、多模态预算降级）
5. **Phase 5（Autonomous）**：自动化上下文优化（DSPy 风格的自愈上下文）

---

## 8. 参考资料

- [04_Anthropic_CE_Practices](04_Anthropic_CE_Practices_cn.md)
- [05_OpenAI_CE_Practices](05_OpenAI_CE_Practices_cn.md)
- [06_Google_CE_Practices](06_Google_CE_Practices_cn.md)
- [07_Cursor_CE_Practices](07_Cursor_CE_Practices_cn.md)
- [08_Frameworks_CE_Practices](08_Frameworks_CE_Practices_cn.md)
- [09_CE_Cross_Comparison](09_CE_Cross_Comparison_cn.md)
