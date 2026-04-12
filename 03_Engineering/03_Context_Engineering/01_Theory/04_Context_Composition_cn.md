# 04 · 上下文编排

*前置要求：[03_Context_Window_Mechanics_cn.md](03_Context_Window_Mechanics_cn.md)。*
*在 CE 流水线中的位置：第 2 步（Budget & Sort）和第 4 步（Assemble）*

---

上下文编排（Context Composition）是一门决定**向上下文窗口中放入什么内容、按什么顺序放入、以及赋予什么优先级**的学问。它是上下文工程的核心技能。

## 1. 生产级上下文的解剖结构

一次生产环境中的 LLM 调用，会从多个来源组装上下文，而每个来源都扮演不同角色：

```
┌─────────────────────────────────────────────────────────────┐
│ 第 1 层：核心系统提示词          [固定、可缓存]      ~500T  │
│   角色定义、人格设定、核心约束                            │
├─────────────────────────────────────────────────────────────┤
│ 第 2 层：情境化指令              [半静态]            ~300T  │
│   风格指南、格式规则、当前任务范围                        │
├─────────────────────────────────────────────────────────────┤
│ 第 3 层：背景知识                [半静态]           ~1000T  │
│   领域规则、few-shot 示例、参考数据                       │
├─────────────────────────────────────────────────────────────┤
│ 第 4 层：检索到的记忆            [动态]             ~2000T  │
│   过去会话中的相关事实（vector DB 查找）                  │
├─────────────────────────────────────────────────────────────┤
│ 第 5 层：RAG 上下文              [动态]             ~4000T  │
│   与当前查询相关的文档 chunk                              │
├─────────────────────────────────────────────────────────────┤
│ 第 6 层：对话历史                [动态]             ~3000T  │
│   最近轮次（逐字保留）+ 更早轮次（摘要化）                │
├─────────────────────────────────────────────────────────────┤
│ 第 7 层：工具结果                [动态]             ~1000T  │
│   函数调用输出、API 响应、代码输出                        │
├─────────────────────────────────────────────────────────────┤
│ 第 8 层：当前用户与状态          [动态]              ~200T  │
│   当前日期/时间、用户 ID、即时请求                        │
└─────────────────────────────────────────────────────────────┘
                                          TOTAL:    ~12000T
                                          Output Reserve: ~4000T
```

## 2. 排序原则

各层的顺序很重要，因为注意力模式会影响模型对内容的读取（见 `03_Context_Window_Mechanics_cn.md`）。

### 三明治结构（Sandwich Pattern）

把最关键的指令放在上下文的**两端**：

```
[System Prompt + Core Instructions]   ← 首因偏差：模型会认真读取
[Background / Examples]
[Retrieved Context / Tool Results]
[Conversation History]
[Current Query]
[Reminder of Key Constraints]         ← 近因偏差：模型会据此行动
```

### 静态内容前置（Static Before Dynamic）

始终把稳定、可缓存的内容放在动态内容之前：

```
✅ [System][Few-shots][RAG][History][Query]
   ←── 可缓存前缀 ──→ ←── 动态内容 ──→

❌ [History][Query][System][Few-shots][RAG]
   ← 每轮都变化 → ← 缓存未命中 →
```

### 解耦系统提示词（前缀缓存安全）

很多开发者把 `System Prompt` 当成一个整体块，但为了最大化前缀缓存命中率，它应该在结构上被拆开。

```text
✅ 最优解耦方式
[Core Persona & Rules]      ← 静态。前缀缓存命中率：99%
[Formatting Guidelines]     ← 静态。前缀缓存命中率：99%
...
[Current Date: 2025-03-20]  ← 动态。移到第 8 层（User Query）
[User Preferences]          ← 动态。移到第 8 层（User Query）
```

不要把时间戳、UUID 或每次请求参数这类动态变量混入高层静态层中。

### 相关内容近端放置（Recency for Relevance）

最相关的检索 chunk 应该放在**离 query 最近的位置**，而不是埋在冗长 RAG 块的中间。

### 2.1 从 U 型注意力分布看这套结构

上面的 8 层结构首先是一份**分层解剖图**，它解决的是“上下文里有哪些组成部分、各自承担什么职责”，而不是一份已经对位置偏置做完优化的最终模板。

从 `Lost in the Middle` 的角度看，它**部分符合** U 型注意力规律，但并不算彻底：
- **符合的部分**：高优先级规则层（System、Instructions、Background）被放在前部，能够利用开头位置的强注意力。
- **也符合的部分**：当前用户请求被放在最后一层，能利用结尾位置的近因优势。
- **不足的部分**：`Retrieved Memory`、`RAG Context`、`Conversation History`、`Tool Results` 这些最容易膨胀的大层，仍主要堆在中部。如果层内不再排序，最关键的信息仍可能被埋进“中部弱注意区”。

因此，工程上应把这 8 层理解为：
- **第一层含义**：信息分类与职责划分
- **第二层含义**：预算与优先级管理
- **第三层含义**：还需要结合位置偏置做进一步落位优化

### 2.2 更符合业界实践的落位方式

结合 Anthropic、OpenAI、Google / Gemini 与 LangChain 等一手资料，更接近真实生产实践的做法通常是：

1. **顶部放稳定规则**
   - system prompt
   - 核心约束
   - 输出格式要求
   - 必要时放长文主体或背景主体

2. **中部放可压缩材料**
   - 次重要背景知识
   - 一般相关的 memory
   - 普通 RAG chunks
   - 历史摘要
   - 已压缩的工具结果

3. **靠近 query 处放最关键证据**
   - 最相关的 RAG chunk
   - 最近且最关键的工具结果
   - 最近 1–2 轮里决定性修正信息

4. **结尾放当前请求与约束重申**
   - 当前 query
   - 必要时重复关键限制或输出要求

这意味着生产系统里常见的不是“严格按 8 层顺序原样堆叠”，而是：
- 先按层分类
- 再按优先级筛选
- 最后按位置偏置重新摆放

### 2.3 一个更稳妥的工程判断

因此，对这套“上下文解剖结构”，更准确的理解应当是：

- 它是**教学友好的分层框架**
- 它对预算分配和职责划分很有帮助
- 但如果直接拿来作为生产 prompt 的最终摆放模板，通常还不够

真正上线时，往往还要额外做三件事：
- **层内重排**：把最关键 chunk 推到更靠后的位置
- **内容压缩**：避免中部被长块内容占满
- **约束重复**：把关键规则在结尾再强调一次

这也是为什么现代业界实践更强调：
- `write`
- `select`
- `compress`
- `isolate`

而不是只停留在“先列出 8 层上下文结构”。

## 3. 优先级层级

当组装后的总上下文超过预算时，应按以下顺序裁剪：

| 优先级 | 层 | 裁剪策略 |
| :--- | :--- | :--- |
| **P0 — 永不裁剪** | System Prompt | 如果放不下，就缩短提示词本身 |
| **P1 — 最后才裁剪** | 当前用户 Query | 仅在极端超长时截断（极少见） |
| **P2 — 保持逐字内容** | 最近历史（最近 2–4 轮） | 对话连贯性所必需 |
| **P3 — 减量** | RAG 上下文 | 减少 chunk 数（top-3 → top-1）或缩短摘录 |
| **P4 — 压缩** | 检索到的记忆 | 进行摘要或减少 top-k |
| **P5 — 最先裁剪** | 旧对话历史 | 做摘要或直接丢弃 |
| **P6 — 条件保留** | 工具结果 | 仅保留最近或最相关的结果 |

## 4. 代理系统中的上下文槽位

在代理式工作流中，上下文还必须容纳工具定义和工具结果。它们会与其他信息竞争同一份 token 预算：

```
┌──────────────────────────────────────────────────────────────┐
│ System Prompt + Agent Persona              ~500T             │
│ Tool Definitions (JSON schemas)            ~1000T  ← 固定    │
│ Scratchpad / Reasoning trace               ~2000T  ← 持续增长│
│ Tool Call Results (accumulated)            ~3000T  ← 持续增长│
│ Original Task + Constraints                ~500T             │
└──────────────────────────────────────────────────────────────┘
```

### 4.1 防止“工具结果老化”

代理开发中的一个常见错误，是把原始 JSON 数组或庞大的 API 响应直接注入上下文窗口。这不仅会快速吞噬 token，也会让模型淹没在无关噪声中，从而削弱推理能力。

**面向工具结果的工程规则：**
- **绝不要注入原始 JSON 表格**：应把结构化数据转换为 token 更高效的 Markdown 表格或自然语言摘要。
- **实现硬上限**：对数组返回结果设置严格的 Top-K 限制（例如 SQL 查询只返回前 5 条记录）。若还需要更多，再让代理翻页。
- **预先摘要大输出**：如果某个工具返回了一份 10 页文档，应先用一个便宜的小模型（例如 Llama-3-8B）提取相关答案，再把结果加入主代理的 scratchpad。

## 5. 多模态上下文编排

当上下文中包含图片、音频或结构化数据时，token 预算会发生显著变化：

| 内容类型 | 近似 Token 成本 |
| :--- | :--- |
| 文本（1 页） | ~500T |
| 图片（低细节） | ~85T |
| 图片（高细节，1024×1024） | ~765T |
| 图片（高细节，2048×2048） | ~2000T |
| 音频（1 分钟） | ~1500T（依模型而定） |

对于多模态代理来说，图片分辨率管理是一项一等公民级别的上下文工程问题。

---

## 6. 实现前缀缓存

前缀缓存要求把静态内容放在前面（见第 2 节），并在 API 层面启用缓存。

### Anthropic（`cache_control`）

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # 缓存这个块
        },
        {
            "type": "text",
            "text": FEW_SHOT_EXAMPLES,
            "cache_control": {"type": "ephemeral"},  # 这个块也缓存
        },
    ],
    messages=[
        {"role": "user", "content": user_query},  # 动态内容 — 不缓存
    ],
)
# 在响应中检查缓存使用情况：
# response.usage.cache_creation_input_tokens  （首次请求）
# response.usage.cache_read_input_tokens      （后续请求）
```

**规则**：至少 1024 tokens 才有资格进入缓存。缓存 TTL 为 5 分钟（ephemeral）。每个请求最多可设置 4 个 cache breakpoint。

### OpenAI（自动）

OpenAI 会自动缓存，无需修改 API。只要 prompt 的前 1024+ tokens 与先前请求完全一致，就会进入缓存。

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 自动缓存
        {"role": "user",   "content": FEW_SHOT_BLOCK},  # 自动缓存
        {"role": "user",   "content": user_query},      # 动态内容
    ],
)
# 检查缓存使用情况：
# response.usage.prompt_tokens_details.cached_tokens
```

**规则**：缓存前缀必须逐字节完全一致。任何变化（哪怕只是空白字符）都会导致缓存失效。

---

## 关键参考文献

1. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
2. **Liu, N. F., et al. (2024). Lost in the Middle.** *TACL, 12*, 157–173.
3. **OpenAI. (2024). Vision API — Token Costs for Images.** OpenAI Documentation.
