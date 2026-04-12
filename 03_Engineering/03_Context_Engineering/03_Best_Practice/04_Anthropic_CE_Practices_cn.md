# 04 · Anthropic 上下文工程实践

*整理自 Anthropic 官方工程博客、API 文档、Claude Code 最佳实践与公开研究。*

---

## 1. 核心理念

### 1.1 如何定义上下文工程

Anthropic 将 Context Engineering 视为 Prompt Engineering 的自然延伸：

> Building with LLMs is becoming "less about finding the right words and phrases for your prompts, and more about answering the broader question of: **what configuration of context is most likely to generate our model's desired behavior?**"

上下文包括模型在生成响应前能看到的一切：系统提示词、工具、few-shot 示例、消息历史、检索到的文档，以及其他任何数据。Prompt engineering 只是 context engineering 的一个子集 — 前者解决的是“把指令写好”，后者解决的是“把模型要看的全部内容组织好”。

“context engineering” 一词由 Shopify CEO Tobi Lutke 于 2025 年 6 月 19 日提出，随后被 Anthropic 接纳并正式化。

### 1.2 指导原则

> 鉴于 LLM 受到有限注意力预算的约束，好的上下文工程意味着**找到尽可能小的一组高信号 token，以最大化某个期望结果发生的概率。**

---

## 2. 上下文腐化（Context Rot）：核心问题

### 2.1 定义

**Context Rot**：随着上下文窗口中的 token 数量增加，模型从这些上下文中准确回忆信息的能力会下降。

这源于 Transformer 架构本身 — 每个 token 都会与其他 token 建立注意力关系，形成 n² 的成对关系。上下文越长，模型的“注意力预算”就越被摊薄。

### 2.2 关键数据点

| 发现 | 数据 |
|---------|------|
| 性能上限 | 约 ~100 万 tokens，超过这一点后性能会明显下降 |
| 普遍性 | Chroma 测试了 18 个前沿模型 — **所有模型** 都会随着输入长度增加而变差 |
| MRCR v2 基准 | Claude Opus 4.6 在 100 万 tokens 下取得 76% 准确率 |

### 2.3 启示

每一个不必要的词、每一段冗余的工具描述、每一条陈旧数据，都会真实地削弱代理性能。上下文并不是越多越好。

---

## 3. 系统提示词设计

### 3.1 “正确高度”（Right Altitude）

Anthropic 建议系统提示词应落在两个失败模式之间的**适中区间（Goldilocks zone）**：

| 失败模式 | 描述 |
|-------------|-------------|
| **过于规定性（too low）** | 用硬编码、复杂、脆弱的逻辑强迫模型产出精确行为 — 易碎且难维护 |
| **过于模糊（too high）** | 只给高层指导，缺少具体信号，或错误假设与模型共享上下文 |

### 3.2 推荐结构（合同格式）

```
- Role（1 行）
- Success Criteria（要点）
- Constraints（要点）
- Uncertainty Handling Rule
- Output Format Specification
```

### 3.3 关键心智模型

> 把 Claude 想成一位**聪明但刚入职的新员工**，他还不了解你的规范和工作流程。你必须清楚、明确。

### 3.4 长时程任务的提示词写法

对于使用 compaction 的代理，Anthropic 建议在 prompt 中加入：

```
Your context window will be automatically compacted as it approaches its limit,
allowing you to continue working indefinitely from where you left off.
Therefore, do not stop tasks early due to token budget concerns.
As you approach your token budget limit, save your current progress and state
to memory before the context window refreshes.
Always be as persistent and autonomous as possible and complete tasks fully.
```

### 3.5 自检区块（Self-Check Blocks）

对重要 prompt，在尾部追加一个小型自检块。Claude 在被明确要求检查自身输出时，常能主动发现错误。

---

## 4. 工具设计：为代理写，而不是为开发者写

### 4.1 范式转移

工具是一类新型软件，体现的是**确定性系统与非确定性代理之间的契约**。你不能再按“给开发者写 API”的思路来设计它，而要按“给代理使用”来设计。

### 4.2 具体最佳实践

| 实践 | 原因 |
|----------|-----------|
| **用第三人称写描述** | 描述会被注入 system prompt；视角不一致会影响工具发现 |
| **描述要具体并包含触发条件** | 不只说明工具做什么，还说明在什么上下文/触发条件下该用 |
| **给工具做命名空间与区分** | 当工具多达 100+ 时，清晰独特的名字能减少代理选择混乱 |
| **返回有意义的上下文** | 人类可读字段与简化输出优于原始技术 ID |
| **优化 token 效率** | 实现分页、截断、过滤；**保持响应低于 25,000 tokens** |
| **提供输入示例** | 尤其适用于嵌套对象、可选参数或格式敏感的复杂工具 |
| **维护最小可行工具集** | 如果人类工程师都说不清该用哪个工具，AI 代理更不可能选对 |

### 4.3 工具描述的影响

对工具描述做很小的调整，往往就能带来明显收益。Claude Sonnet 3.5 之所以能在 SWE-bench Verified 上取得 SOTA，其中一个重要原因，就是工具描述经过了精细打磨。

**真实案例**：Anthropic 在发布 Claude 的 web search tool 时发现，Claude 会无谓地在搜索 query 后追加 “2025”，导致结果带偏。后来通过改进工具描述解决了这个问题。

### 4.4 以评估驱动工具改进

> “Simply concatenate the transcripts from your evaluation agents and paste them into Claude Code. Claude is an expert at analyzing transcripts and refactoring lots of tools all at once.”

---

## 5. Few-shot 示例

Anthropic 强烈建议使用 few-shot prompting，但也特别提醒一个常见误区：为了覆盖所有规则，把一长串边缘情况全部塞进 prompt。

建议做法是：整理一组**多样且具有代表性的典型样例**，用来准确呈现期望行为。**多样性与代表性，比穷尽式覆盖更重要。**

---

## 6. 长上下文提示技巧

| 技巧 | 说明 |
|-----------|-------------|
| **把长篇数据放顶部** | 将 20K+ token 文档放在上方，位于 query / instructions / examples 之前 |
| **把 query 放末尾** | 将 query 放在末尾可让响应质量最高提升 **30%**，尤其在复杂多文档输入下 |
| **使用 XML tags 组织结构** | 给每个文档包上 `<document>`、`<document_content>`、`<source>` 标签 |
| **让回答先基于引文** | 对长文档任务，先要求 Claude 引用相关片段，再开展任务 |

---

## 7. 提示缓存（Prompt Caching）

### 7.1 效果

对长 prompt，可将成本最多降低 **90%**，延迟最多降低 **85%**。

### 7.2 两种方式

| 方式 | 说明 |
|----------|-------------|
| **Automatic（推荐）** | 在顶层添加 `cache_control={"type": "ephemeral"}`，系统自动处理其他部分 |
| **Explicit breakpoints** | 把 `cache_control` 放在具体内容块上，以进行更细粒度控制 |

### 7.3 关键规则

- 缓存前缀按顺序创建：**tools → system → messages**
- 最多可定义 **4 个 cache breakpoints**
- **缓存 key 是累积的**：每个块的 hash 依赖于它之前的全部内容
- **最小 token 要求**：每个缓存检查点至少 1,024 tokens
- **TTL**：默认 5 分钟，每次命中都会重置。Claude Opus 4.5 / Haiku 4.5 / Sonnet 4.5 还支持 1 小时 TTL
- **20-block lookback window**：连续 20 次检查仍未命中后，系统就停止继续回看

### 7.4 多轮最佳实践

始终在对话末尾显式设置一个 cache breakpoint，以最大化命中率。多轮缓存会逐步缓存整个对话历史。

---

## 8. 上下文压缩（Compaction）：长时程任务的第一杠杆

### 8.1 定义

Compaction 指的是：当一段对话接近上下文窗口上限时，先将其压缩成摘要，再以该摘要为起点开启新的上下文窗口。

### 8.2 Claude Code 如何实现 Compaction

- 将 message history 交给模型进行摘要与压缩
- **保留**：架构决策、未解决 bug、实现细节
- **丢弃**：冗余工具输出或消息
- 代理会用压缩后的上下文**加上最近访问过的 5 个文件**继续工作

### 8.3 调整压缩策略

- 关键在于取舍：哪些必须保留，哪些可以丢弃
- **先最大化 recall**（捕获所有相关信息），再**迭代提升 precision**（去掉多余内容）
- 清理工具结果，是一种安全且轻量的 compaction 形式

### 8.4 配套实践

| 实践 | 说明 |
|----------|-------------|
| **Git commits 作为检查点** | 用清晰 commit message 提交进度，让代理可以用 `git log` 和 `git diff` 重建状态 |
| **Progress files** | 把摘要写进 `claude-progress.txt`，使代理在 compaction 后仍能理解“已完成 / 进行中 / 阻塞中”的工作 |

### 8.5 服务端 Compaction API

该能力以 Beta 形式提供（`compact-2026-01-12`）。启用后，Claude 在接近设定的 token 阈值时，会自动总结对话、生成摘要、创建 compaction block，并丢弃此前所有消息块。

### 8.6 自定义压缩策略

在 `CLAUDE.md` 中加入诸如 “When compacting, always preserve the full list of modified files and any test commands” 之类的说明，以确保关键信息得以保留。

---

## 9. 上下文编辑（Beta）

### 9.1 Tool Result Clearing（`clear_tool_uses_20250919`）

- 当对话上下文增长到某一阈值后，清除工具结果
- 可配置：触发点、保留最近多少次 tool use、最少清理多少 tokens、哪些工具可豁免
- 被清掉的内容会替换成 placeholder 文本，让模型知道有内容被移除了

### 9.2 Thinking Block Clearing（`clear_thinking_20251015`）

- 删除更早轮次中的 thinking / reasoning blocks，仅保留最近的部分
- 可通过 `keep` 参数控制优先保缓存性能还是优先释放上下文窗口

### 9.3 性能影响

在一项 100 轮 web search 评估中：
- 开启 context editing 后，代理能完成原本会因上下文耗尽而失败的工作流
- **Token 消耗降低 84%**
- 将 memory tool 与 context editing 结合后，性能较基线提升 **39%**

---

## 10. 即时上下文加载（Just-in-Time）

### 10.1 策略

不是一开始就把所有内容都装进上下文，而是由代理维护轻量引用，在运行时按需加载数据。

### 10.2 实现模式

- 维护**轻量标识符**（file paths、stored queries、web links、API endpoints）
- 借助这些引用，通过工具在运行时**动态把数据加载进上下文**
- Claude Code 的工具惰性加载通过“不在需要前加载工具定义”，使上下文减少 **95%**

### 10.3 预加载与即时加载

| 方式 | 特征 |
|----------|----------------|
| **Pre-loading** | 预先处理全部相关数据；上下文会变得臃肿 |
| **Just-in-Time** | 仅在使用当下加载需要的数据；使上下文保持精简与聚焦 |

---

## 11. 多代理架构：上下文隔离

### 11.1 编排者—执行者模式（Orchestrator-Worker Pattern）

- **主代理**（orchestrator）负责维护高层策略和用户意图
- **子代理**（workers）在彼此隔离的上下文窗口中运行，接收具体指令、执行任务，并只回传摘要或结果
- 每个代理只能访问与其角色相关的工具和上下文

### 11.2 性能数据

- 多代理研究系统（Opus 4 负责统筹 + Sonnet 4 子代理）在研究任务上比单个 Opus 4 代理高出 **90.2%**
- **仅 token usage 就解释了复杂评估中 80% 的性能方差**

### 11.3 委派最佳实践

| 实践 | 说明 |
|----------|-------------|
| **明确目标** | 将 query 划分为带具体目标、输出格式、工具指导与边界的子任务 |
| **基于工件通信** | 子代理把输出保存到外部系统，并只把轻量引用返回，避免信息丢失并降低 token 开销 |
| **通过限制工具进行沙箱化** | 去掉 Edit 等工具，为代理构建“不能破坏构建”的沙箱 |

### 11.4 成本导向的模型路由

多代理系统的 token 消耗大约是单代理交互的 **15 倍**。因此要做模型路由：

| 模型 | 使用场景 |
|-------|----------|
| **Haiku** | Linting 与简单逻辑 |
| **Sonnet** | 大多数编码与调试 |
| **Opus** | 编排者或复杂架构推理 |

### 11.5 何时使用多代理

| Good Fit | Poor Fit |
|----------|----------|
| 需要重并行 | 需要所有代理共享完整上下文的领域 |
| 信息超出单个上下文窗口 | 代理之间依赖很多 |
| 需要对接多个复杂工具 | |

---

## 12. 长时运行代理支架（Agent Harness）

### 12.1 初始化代理 + 编码代理模式

| Agent | 职责 |
|-------|---------------|
| **Initializer** | 只在第一个上下文窗口运行；为后续 coding agents 建好全部必要上下文 |
| **Coding Agent** | 在后续每个上下文窗口运行；持续增量推进，并留下清晰工件 |

### 12.2 `claude-progress.txt` 模式

外部工件变成了代理的记忆：

- **进度文件**可跨 session 持久存在。每个新 session 开始时都会读取它们，以恢复完整状态
- **Git history** 提供已完成工作的日志以及可恢复的检查点
- **Feature checklists** 用于跟踪哪些完成了、哪些进行中、哪些被阻塞

### 12.3 会话结束协议

在一次 session 结束前，代理要更新 progress log，写明已完成与剩余事项，确保下一次 session 有准确起点。

### 12.4 压力测试：构建一个 C 编译器

Anthropic 曾用 16 个代理验证这一方法：从零开始编写一个基于 Rust 的 C 编译器，且能够编译 Linux 内核。经过近 **2,000 次 Claude Code sessions** 与 **$20,000 的 API 成本**，这些代理产出了一个 10 万行的编译器，能够在 x86、ARM 与 RISC-V 上构建 Linux 6.9。

---

## 13. CLAUDE.md 与自动记忆

### 13.1 CLAUDE.md 文件

这是一类持久化 Markdown 指令文件，会在每次 Claude Code session 开始时读取。它们采用分层组织，具有级联优先级（项目级会覆盖全局用户级）。

**最佳实践**：

| 实践 | 原因 |
|----------|-----------|
| 每个文件控制在 200 行以内 | 文件越长，消耗的上下文越多，遵循度越低 |
| 使用 markdown 标题与要点 | 便于分组相关指令 |
| 指令要具体且可验证 | 例如写 “Use 2-space indentation”，而不是 “Format code properly” |
| 用 `.claude/rules/` 做拆分 | 把大指令集拆成多个文件 |
| 用 `CLAUDE.local.md` 存放私有设置 | 会自动加入 `.gitignore` |

### 13.2 自动记忆（MEMORY.md）

Claude Code 会自动创建并维护 `MEMORY.md`，记录：

- 调试模式（反复出现的问题、根因、修复方式）
- 项目上下文（架构决策、命名规范、坑点）
- 偏好（代码结构、方案偏好、测试方法）

### 13.3 “记忆褪色”陷阱

当 CLAUDE.md 文件越来越大、越来越单体化时，模型定位其中最相关信息的能力会减弱 — 信号会淹没在噪声里。记忆文件必须简洁且结构良好。

---

## 14. 策略层级总结

从最简单到最复杂的上下文管理策略：

```
Level  1: 在“正确高度”写清晰的 system prompts
Level  2: 维护最小工具集，并写出优秀描述
Level  3: Few-shot examples —— 多样、典型，而非穷尽
Level  4: 长文档放顶部，query 放底部
Level  5: 启用 Prompt Caching
Level  6: 实现 Context Editing（清理工具结果、清理 thinking blocks）
Level  7: 结合 git checkpoints 与 progress files 实现 Compaction
Level  8: 用 Just-in-Time Context Loading 取代 upfront pre-loading
Level  9: 采用多代理架构实现上下文隔离
Level 10: 基于 Initializer / Coding Agent 模式构建 agent harness
```

总建议：**“Do the simplest thing that works.”** 只有在必要时才增加复杂度。先从简单方案开始，测量，再迭代。

---

## 参考资料

- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Anthropic Engineering Blog (Sep 2025)
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — Anthropic Engineering Blog (2026)
- [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — Anthropic Engineering Blog
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — Anthropic Research
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Anthropic Engineering Blog
- [Prompting best practices (Claude 4)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices) — Claude API Docs
- [Long context prompting tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) — Claude API Docs
- [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — Claude Platform Docs
- [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) — Claude Platform Docs
- [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) — Claude Platform Docs
- [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) — Claude Platform Docs
- [How Claude remembers your project](https://code.claude.com/docs/en/memory) — Claude Code Docs
- [Claude Code best practices](https://www.anthropic.com/engineering/claude-code-best-practices) — Anthropic Engineering Blog
