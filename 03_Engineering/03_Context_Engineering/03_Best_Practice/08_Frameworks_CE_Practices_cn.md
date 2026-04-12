# 08 · 代理框架的上下文工程实践

*整理自 LangGraph、CrewAI、AutoGen、MemGPT/Letta、Devin、Vercel AI SDK、DSPy、Semantic Kernel 与 Haystack 的官方文档、源码、博文与学术论文。*

---

## 1. LangChain / LangGraph

### 1.1 核心概念：图状态作为上下文载体

LangGraph 采用图状态机架构，把共享的 `State` 对象（TypedDict 或 Pydantic model）作为节点之间传递上下文的主要载体。

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # append merge
    plan: str                                 # overwrite merge
    facts: Annotated[list, operator.add]     # list append
```

### 1.2 Reducer：按字段定义的合并策略

每个节点都会读写共享 state。**Reducer functions** 决定字段更新如何合并：

| Reducer | 行为 | 使用场景 |
|---------|----------|----------|
| `add_messages` | 追加新消息，并处理去重 | Chat history |
| `operator.add` | 拼接列表 | 累积事实 / 发现 |
| Default（无 reducer） | 覆盖 | 最新 plan、当前状态 |
| Custom function | 任意逻辑 | 冲突解决、投票 |

这在代理框架中很有代表性 — **每个 state 字段都可以拥有自己的合并策略**，从而细粒度地控制上下文流动。

### 1.3 检查点与记忆

| 特性 | 说明 |
|---------|-------------|
| **Persistent checkpointing** | `MemorySaver`、`SqliteSaver`、`PostgresSaver` — 在每个 super-step 保存完整图状态 |
| **Time-travel debugging** | 可回放并检查任意历史状态 |
| **Human-in-the-loop** | 可暂停执行、修改 state，再继续 |
| **Thread-level memory** | 在单个会话线程内持久存在 |
| **Cross-thread memory（Store API）** | LangGraph Platform 的 `Store` 用于跨会话持久化用户偏好、事实、学到的行为 |

### 1.4 上下文窗口管理工具

| 工具 | 说明 |
|---------|-------------|
| `trim_messages()` | 只保留最后 N 条消息或最后 N tokens |
| **Summarization node** | 用节点摘要旧消息，并以摘要替换原始内容 |
| **Filtering** | 按消息类型选择性删除（例如移除陈旧 tool results） |

### 1.5 LangChain 的上下文工程分类法

LangChain 在 2025 年 6 月的博客中定义了五种上下文原语：

1. **Instruction context**：system prompts、rules、guidelines
2. **Knowledge context**：retrieved documents、tool outputs、search results
3. **Conversation context**：message history，经 trimming / summarization 处理
4. **Structured output context**：指导输出格式的 schemas 与 examples
5. **Tool context**：可用工具及其描述

---

## 2. CrewAI

### 2.1 四层记忆架构

CrewAI 拥有代理框架中最清晰、也最成体系的记忆系统之一：

| 记忆类型 | 存储 | 范围 | 说明 |
|-------------|---------|-------|-------------|
| **Short-Term Memory** | Vector store（RAG） | 当前执行 | 存储当前 crew run 的信息；由代理交互自动填充；通过 embeddings 搜索 |
| **Long-Term Memory** | SQLite | 跨执行 | 持久化任务结果与学习到的洞见；让代理随时间改进 |
| **Entity Memory** | Vector store + graph | 当前执行 | 跟踪实体（人物、组织、概念），构建关系知识图谱 |
| **User Memory** | Persistent store | 跨 session | 存储用户偏好和历史 |

### 2.2 每一步的上下文组装

在每个 agent 动作前，CrewAI 会组装：

```
1. 任务描述与期望输出
2. 相关 short-term memories（RAG 搜索）
3. 相关 long-term memories
4. 实体信息
5. 上游任务结果（task dependencies）
6. 先前步骤中的工具输出
```

### 2.3 知识来源

CrewAI 支持将外部知识（PDF、文本文件、JSON）切分并向量化，再通过 RAG 暴露给代理。知识既可以限定给某个 agent，也可以由整个 crew 共享。

### 2.4 关键洞见：实体记忆（Entity Memory）

Entity Memory 是 CrewAI 的一大特色 — 它不只是存储文本，而是**构建实体关系图**。这对需要追踪多个对象之间关系的场景尤其有用（例如“哪个客户从哪个供应商那里买了哪个产品”）。

---

## 3. Microsoft AutoGen

### 3.1 ChatCompletionContext 协议（AutoGen 0.4+）

AutoGen 0.4 为“什么该进入 LLM 上下文窗口”定义了正式抽象：

| 实现 | 策略 | 说明 |
|---------------|----------|-------------|
| `BufferedChatCompletionContext` | Sliding window | 仅保留最后 N 条消息 |
| `HeadAndTailChatCompletionContext` | Anchor + recency | 保留前 K 条消息（system context、初始指令）和最后 N 条消息（最近上下文），丢掉中间 |
| Custom implementation | 任意 | 自己实现协议以适配领域策略 |

### 3.2 首尾保留模式（Head-and-Tail）

这是一个在其他框架中较少见的模式：

```
[System message + first K messages]  ← 锚定保留
[... middle messages dropped ...]    ← 中间遗忘
[Last N messages]                    ← 最近保留
```

理由：对话开头通常包含关键初始化上下文（任务描述、约束、示例），不应被丢弃；而最近消息则包含当前工作状态。

### 3.3 多代理上下文共享

| 模式 | 说明 |
|---------|-------------|
| **Group Chat Manager** | 编排哪个代理接下来发言；维护共享 message history |
| **Selector Group Chat** | 用 LLM 根据当前上下文决定哪个代理应回应 |
| **Swarm pattern** | 代理之间做带上下文转移的 handoff |
| **Handoffs** | 代理把相关上下文转给下一个代理 |

---

## 4. MemGPT / Letta

### 4.1 核心概念：虚拟内存管理

MemGPT 把 LLM 上下文窗口看作操作系统管理的虚拟内存：

```
┌─────────────────────────────┐
│ Main Context（RAM）          │  ← 由代理主动管理
│  - Core Memory（可编辑）     │    memory_insert / memory_replace
│  - Recent Messages          │
└─────────────────────────────┘
         ↕ page in / page out
┌─────────────────────────────┐
│ Archival Memory（Disk）      │  ← Vector store，无限容量
│ Recall Memory（Disk）        │  ← 对话历史存储
└─────────────────────────────┘
```

### 4.2 自管理记忆

其关键创新是：**由代理自己决定哪些内容要换入 / 换出上下文窗口**，通过显式记忆工具实现：

| 工具 | 说明 |
|------|-------------|
| `memory_insert` | 向 archival memory 写入新信息 |
| `memory_search` | 在 archival memory 中搜索相关信息 |
| `memory_replace` | 更新 core memory 中的现有信息 |
| `memory_delete` | 从 memory 中移除信息 |

LLM 不再只是上下文的消费者，而是自己上下文窗口的**主动管理者**。

### 4.3 上下文编译

在每次 LLM 调用前，MemGPT 会从多个来源编译上下文窗口：

```
System prompt
+ Core memory blocks（可编辑 persona、user info 等）
+ Relevant archival memory results（来自 search）
+ Recent message buffer
+ Current user message
= Compiled context window
```

### 4.4 关键洞见

这是上下文工程中最激进的一条路线：不再由开发者预先规定什么该进、什么该出，而是**让 LLM 自己学会管理记忆**。代价是需要额外的 LLM 调用来完成这些记忆操作。

---

## 5. Devin（Cognition Labs）

### 5.1 核心概念：环境就是记忆

Devin 的关键洞见：

> **不要试图把所有内容都塞进上下文窗口。相反，让代理知道该去环境中的哪里找信息。**

文件系统、终端和浏览器共同构成了可按需查询的外部记忆。

### 5.2 架构

| 组件 | 说明 |
|-----------|-------------|
| **Planner model** | 维护高层计划，持有战略性上下文 |
| **Worker models** | 执行单步操作，获取战术性上下文 |
| **Scratchpad** | session 内持久笔记 — 发现、决策、中间结果 |
| **Event-based history** | 以事件流形式保存 session history（命令、编辑、浏览器动作） |

### 5.3 上下文组装

每次 LLM 调用时，上下文由以下部分组成：

```
Current plan state
+ Relevant recent events
+ Scratchpad contents
+ Current environment state（正在编辑的文件、终端输出）
= Context for this step
```

### 5.4 环境即记忆模式

| 与其…… | Devin 更倾向于…… |
|---------------|---------------|
| 把所有文件内容都放进上下文 | 需要时用 `cat` / `grep` 重新读取 |
| 把所有命令输出都保存在上下文 | 重跑命令，或查看终端历史 |
| 记住所有搜索结果 | 需要时重新搜索 |
| 保留完整对话历史 | 把关键发现写到 scratchpad |

### 5.5 关键洞见

环境本身就是最大的外部存储。与其把信息压缩到能塞进上下文窗口，不如让代理通过工具按需读取环境。对于会持续数小时的长任务，这尤其强大。

---

## 6. Vercel AI SDK

### 6.1 核心消息处理

AI SDK 使用 `messages` 数组作为主要上下文载体，兼容 OpenAI 风格的 `{ role, content }` 格式，并支持工具调用与多模态内容。

### 6.2 上下文管理工具

| 工具 | 说明 |
|---------|-------------|
| `trimMessages()` | 裁剪消息历史以适配 token 限制；支持保留 system message + 最近 N 条消息 |
| `convertToLanguageModelMessage()` | 统一不同 provider 间的消息格式 |
| Token counting utilities | 帮助在不同模型之间管理上下文预算 |

### 6.3 多步工具调用（Agentic Loops）

`maxSteps` 参数允许进入代理循环，模型可调用工具并继续。每一步的工具结果都会自动追加到下一步消息上下文中。

### 6.4 与提供方无关的上下文封装

AI SDK 屏蔽了各 provider（OpenAI、Anthropic、Google 等）在上下文格式上的差异。

---

## 7. DSPy（Stanford）

### 7.1 程序化上下文工程

DSPy 走的是截然不同的一条路 — **通过程序优化来决定该包含哪些上下文**：

| 概念 | 说明 |
|---------|-------------|
| **Signatures** | 定义输入输出字段，从而隐式规定需要哪些上下文 |
| **Modules** | 可组合单元，每个单元管理自己的上下文组装 |
| **Optimizers（Teleprompters）** | 自动搜索最优上下文配置（few-shot examples、instructions 等） |
| **Assertions** | 运行时检查，可触发上下文修改与重试 |

### 7.2 关键洞见

DSPy 不是手工工程化上下文，而是**自动搜索**最优配置（哪些样例、哪些指令、哪些检索参数）。这是最接近“机器学习式上下文工程”的路线。

---

## 8. Microsoft Semantic Kernel

### 8.1 上下文管理

| 特性 | 说明 |
|---------|-------------|
| **KernelArguments** | 在函数 / 插件之间传递上下文 |
| **Chat History management** | 内置 token-aware 截断 |
| **Memory connectors** | 与向量存储集成，实现 RAG 式上下文 |
| **Planner** | 生成同时包含每一步上下文需求的计划 |

---

## 9. Haystack（deepset）

### 9.1 基于流水线的上下文管理

| 特性 | 说明 |
|---------|-------------|
| **DocumentStores** | 与后端无关的上下文文档存储 |
| **Retrievers** | 多种检索策略：BM25、embedding、hybrid |
| **PromptBuilder** | 基于 Jinja2 的模板化上下文组装 |
| **Pipeline branching** | 按 query 类型走不同的上下文组装路径 |

---

## 10. 跨框架模式

### 10.1 正在收敛的模式

| 模式 | 说明 | 使用方 |
|---------|-------------|---------|
| **层级记忆** | 工作 / 短期 / 长期记忆分层 | CrewAI、MemGPT/Letta、LangGraph |
| **自管理记忆** | 代理决定存什么、取什么 | MemGPT/Letta、Devin |
| **上下文编译** | 从多个来源组装成一致的 prompt | 所有框架 |
| **Sliding Window + Anchors** | 保留开头和结尾，丢掉中间 | AutoGen、Vercel AI SDK |
| **RAG-Augmented Context** | 在决策时检索相关文档 / 代码 | LlamaIndex、Haystack、CrewAI |
| **Environment-as-Memory** | 通过工具重新访问信息，而不是一直保存在上下文中 | Devin、Copilot Agent、Windsurf |
| **Graph-Based Context Flow** | 用显式状态图控制上下文传播 | LangGraph、Semantic Kernel |
| **Programmatic Optimization** | 自动调优上下文编排以提升性能 | DSPy |
| **Reducer-Based Merging** | 对 state update 做按字段合并策略 | LangGraph |
| **Entity Tracking** | 为上下文中的实体建立关系图 | CrewAI |

### 10.2 共通主题

1. **上下文工程 > 提示词工程**：行业共识正从“写好 prompt”转向“动态组装正确上下文”
2. **层级记忆正在成为标准**：几乎所有成熟代理框架都实现了分层记忆（工作记忆、短期会话记忆、长期持久记忆）
3. **代理应管理自己的上下文**：MemGPT 的洞见 — 让 LLM 自己决定该换入 / 换出什么 — 正越来越多地被采纳
4. **代码场景的上下文工程是独立问题**：AST-aware 索引、依赖图遍历、符号解析，已超出通用文本 RAG
5. **环境就是外部记忆**：现代代理把文件系统、搜索、浏览器等工具视为可按需查询的外部记忆
6. **Token 预算管理已成为一等问题**：每个框架都提供裁剪、摘要或其他预算管理手段

---

## 11. 学术基础

### 11.1 关键论文

| 论文 | 关键发现 | CE 启示 |
|-------|-------------|----------------|
| **“Lost in the Middle”**（Liu et al., 2023） | 当相关信息位于长上下文中间时，LLM 表现更差；对开头和结尾注意更多 | 重要信息应放在 prompt 开头或结尾 |
| **“MemGPT”**（Packer et al., 2023） | 把上下文窗口视为分层存储的 OS 虚拟内存 | 自主记忆检索、上下文编译、记忆整合 |
| **“Reflexion”**（Shinn et al., 2023） | 能从失败中反思并把反思保存为未来上下文的代理更强 | 反思本身是一种高价值压缩上下文 |
| **“Voyager”**（Wang et al., 2023） | Minecraft 代理会构建技能库 — 学到的技能（代码）被作为上下文存储与检索 | 课程式上下文生成 |
| **RAG Survey**（Gao et al., 2024） | 将 RAG 分为 Naive、Advanced、Modular | Query rewriting、HyDE、迭代检索、自适应检索 |
| **Context Length Extension Survey**（2024） | RoPE scaling、ALiBi、landmark attention 等技术 | 更长窗口会改变工程权衡 |

### 11.2 学术分类法

学术界普遍认可 LLM 代理有三层记忆：

```
Working Memory    → 上下文窗口（有限、快）
Short-Term Memory → 当前 session store（中等、会话级）
Long-Term Memory  → 持久存储（无限、跨 session）
```

规划与反思模块会把经验压缩成可复用上下文，形成持续提升上下文质量的反馈回路。

---

## 参考资料

- [LangGraph Memory Concepts](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [LangChain Context Engineering Blog](https://blog.langchain.dev/context-engineering-for-agents/)
- [CrewAI Memory Documentation](https://docs.crewai.com/concepts/memory)
- [Microsoft AutoGen Documentation](https://microsoft.github.io/autogen/)
- [MemGPT / Letta Documentation](https://docs.letta.com/)
- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- Liu et al., "Lost in the Middle" (2023)
- Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023)
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
- Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (2023)
