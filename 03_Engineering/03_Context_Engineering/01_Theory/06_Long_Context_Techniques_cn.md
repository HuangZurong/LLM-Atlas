# 06 · 长上下文处理技术

*前置要求：[04_Context_Composition_cn.md](04_Context_Composition_cn.md)。*

---

长上下文处理（Long Context Handling）要解决的问题是：当文档、代码库或对话历史接近甚至超过模型的有效上下文窗口时，应该如何处理。它与 RAG 不同（RAG 处理的是从外部存储中检索）— 这里关注的是上下文窗口**内部**发生的事。

## 1. 长上下文问题空间

```
文档大小 vs. 策略：

< 8K tokens    → 可完整放入上下文，无需特殊处理
8K – 32K       → 选择性纳入：优先保留相关片段
32K – 128K     → 分块 + 分层处理
> 128K         → RAG（见 [05_RAG module](../../05_RAG/01_Theory/01_Architecture.md)）或多轮处理
```

挑战不仅仅是“塞得下” token — 更在于如何在长跨度内容上保持**推理质量**。模型会以两种方式退化：
1. **检索退化（Retrieval degradation）**：找不到具体事实（Lost in the Middle）。
2. **推理退化（Reasoning degradation）**：当推理链跨越大量 token 时，连贯性变差。

## 2. Needle-in-a-Haystack（NIAH）测试

NIAH 是评估长上下文可靠性的标准基准。做法是在一份很长的“草堆”文档中插入一个“针”事实，然后让模型去找出来。

```
Haystack: [Filler text ... ] [NEEDLE: "The secret code is 7392"] [... Filler text]
Query: "What is the secret code?"
```

### NIAH 揭示了什么

- **位置敏感性**：大多数模型在 0% 和 100% 深度表现较好，在 40–60% 深度表现较差。
- **上下文长度断崖（Context length cliff）**：性能通常会在标称窗口的 60–70% 之后急剧下滑。
- **模型特有模式**：不同模型有不同的“盲区”。

### 部署前运行 NIAH

在上线任何长上下文功能之前，务必对你的“模型 + 上下文长度”组合跑 NIAH 测试。不要只依赖已发表的基准结果 — 它们使用的 haystack 内容与你的领域内容不同。

**进阶 NIAH：Multi-Needle Reasoning**  
标准 NIAH 只测试模型能否*找到*单个字符串。而在生产环境中，模型往往需要综合多个事实。更高级的评估（如 `08_Advanced_Context_Paradigms_cn.md` 所述）会把 3–5 个相互关联的 needle 分布插入 100K 窗口中，测试模型能否把它们逻辑连接起来，这对真正的长上下文能力要求严格得多。

## 3. 分块策略

当一个文档必须被切分时，分块策略决定了哪些信息被保留，哪些信息在边界处丢失。

### 定长分块（Fixed-Size Chunking）

```
[Token 0 – 512] [Token 513 – 1024] [Token 1025 – 1536] ...
```

- 简单、可预测。
- 会任意切断句子和段落。
- 只有在文档结构无关紧要时才使用。

### 按语义分块（Semantic Chunking）

按自然边界切分：段落、章节、句子。保留结构单元。

```python
# 按双换行切分（段落边界）
chunks = text.split("\n\n")
# 然后合并小块、再拆分大块，使之接近目标大小
```

### 分层式分块（Hierarchical Chunking）

维护一个两层表示：
- **摘要层（Summary level）**：每个章节一个摘要（每个约 ~100T）。
- **细节层（Detail level）**：每个章节完整文本（每个约 ~500T）。

先用摘要层做初步检索，再只为相关章节加载完整细节层。

### 带重叠的滑动窗口

```
Window 1: [Token 0 – 512]
Window 2: [Token 384 – 896]   ← 128-token overlap
Window 3: [Token 768 – 1280]  ← 128-token overlap
```

重叠能防止边界处的信息丢失。通常使用 10–25% 的 overlap。

## 4. 多轮处理

对于无法在单轮中完成处理的超长文档，应使用迭代式策略：

### 映射归约（Map-Reduce）

```
Document
    │
    ├── Chunk 1 → [LLM: extract key facts] → Summary 1
    ├── Chunk 2 → [LLM: extract key facts] → Summary 2
    ├── Chunk 3 → [LLM: extract key facts] → Summary 3
    │
    └── [LLM: synthesize summaries] → Final Answer
```

最适合：摘要、信息抽取、针对长文档的问答。

### 迭代细化（Refine）

```
Chunk 1 → [LLM] → Running Summary v1
Chunk 2 + Summary v1 → [LLM] → Running Summary v2
Chunk 3 + Summary v2 → [LLM] → Running Summary v3
...
Final Summary vN → Answer
```

最适合：需要保持叙事连贯的任务（例如顺序很重要的文档摘要）。

### 摘要树

```
Level 0 (raw):    [C1][C2][C3][C4][C5][C6][C7][C8]
Level 1 (pairs):  [S12]    [S34]    [S56]    [S78]
Level 2 (quads):  [S1234]           [S5678]
Level 3 (root):   [S12345678]
```

最适合：极长文档，单次 map-reduce 会损失过多细节。

## 5. 位置感知的上下文放置

考虑到注意力模式，应有策略地摆放内容：

| 内容类型 | 最优位置 | 原因 |
| :--- | :--- | :--- |
| 任务指令 | 开头 + 结尾 | 首因偏差 + 近因偏差 |
| 最相关 chunk | 结尾（紧挨 query 前） | 近因偏差 |
| 背景上下文 | 开头（system prompt 之后） | 首因偏差 |
| 相关性较低的 chunks | 中间 | 可接受的损失区 |
| 输出格式规范 | 结尾 | 模型会按最后的指令行动 |

## 6. 代理循环中的长上下文

*注：关于在自主代理中管理长上下文的权威模式，请参见 `07_Dynamic_Context_Management_cn.md` 和 `08_Advanced_Context_Paradigms_cn.md`。*

在多步代理执行中，上下文会随着每次工具调用而增长。必须主动管理：

```
Turn 1:  [System][Task][Tool Schema]                    → 2K tokens
Turn 5:  [System][Task][Tool Schema][4x Tool Results]   → 8K tokens
Turn 15: [System][Task][Tool Schema][14x Tool Results]  → 25K tokens
Turn 30: [System][Task][Tool Schema][29x Tool Results]  → 50K tokens  ← danger zone
```

**策略**：
- **结果修剪（Result pruning）**：每次工具调用后，丢弃不再需要的结果。
- **Scratchpad 摘要化**：每隔 N 轮，将推理轨迹摘要成紧凑状态。
- **上下文检查点（Context checkpointing）**：把当前状态保存到外部记忆中，重置上下文，再注入 checkpoint 摘要。

---

## 关键参考文献

1. **Liu, N. F., et al. (2024). Lost in the Middle.** *TACL, 12*, 157–173.
2. **Hsieh, C., et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models?** *arXiv:2404.06654*.
3. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
4. **LangChain. (2024). Document Transformers — Text Splitters.** LangChain Documentation.
