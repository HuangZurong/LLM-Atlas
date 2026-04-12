# 07 · Cursor 上下文工程实践

*整理自 Cursor / Anysphere 博文、文档、创始人访谈（Lex Fridman、Latent Space podcast）以及对其可观察行为的分析。*

---

## 1. 核心理念

### 1.1 更好的选择胜过更大的窗口

Cursor 的中心论点：

> **Better context selection beats larger context windows.** 即使拥有 128K 或 200K token 的窗口，把所有内容都塞进去也会降低质量。真正难的问题是选择*什么*应该被放进去。

Cursor 团队认为，他们的核心差异化并不在于使用哪家 LLM，而在于如何为它组装上下文。

### 1.2 分层上下文架构

Cursor 把上下文组织成四个优先级递减的层级：

```
Tier 1: Immediate Context   （始终加载，优先级最高）
Tier 2: Local Context       （高优先级，近期活跃）
Tier 3: Global Context      （按需检索）
Tier 4: Static Rules        （项目级指令）
```

---

## 2. 代码库索引与向量嵌入

### 2.1 索引流水线

当一个工作区被打开时，Cursor 会索引整个代码库：

| 步骤 | 说明 |
|------|-------------|
| **Chunking** | 文件会按代码结构（函数、类、逻辑块）切分成语义单元，而不是固定大小窗口 |
| **Embedding** | 每个 chunk 都用为代码优化的自训练 embedding 模型向量化，而不是现成通用模型 |
| **Storage** | 向量存到用户机器上的本地 vector database |
| **Updates** | 索引在后台异步构建，并随文件变化做增量更新 |

### 2.2 过滤

匹配 `.gitignore` 的文件、二进制文件、`node_modules`、构建产物等，都会被完全排除出索引。

---

## 3. 面向对话的上下文选择

当用户在 Cursor chat 中提问或请求改代码时：

### 3.1 通过 `@` 显式附加上下文

用户可以手动通过引用附加上下文：

| 引用 | 说明 |
|-----------|-------------|
| `@file` | 将指定文件固定加入上下文 |
| `@folder` | 包含整个文件夹 |
| `@symbol` | 包含某个函数 / 类 / 类型 |
| `@web` | 搜索网络信息 |
| `@docs` | 搜索文档 |
| `@codebase` | 触发完整代码库检索 |
| `@git` | 纳入 git 历史 / diff 信息 |

### 3.2 自动代码库检索（`@codebase`）

当启用 `@codebase`（或自动 codebase context）时，Cursor 会执行多阶段检索流水线：

```
Step 1: 用同一 embedding 模型向量化用户 query
    ↓
Step 2: Vector similarity search → 取回 top-N 候选 chunks
    ↓
Step 3: 用 cross-encoder model 做 reranking → 重新打分
    ↓
Step 4: 在 token 预算内组装排名最高的 chunks 进 prompt
```

**reranking 是关键步骤** — 初始 embedding 检索 recall 尚可，但 precision 普通；cross-encoder reranker 会显著改善最终真正进入上下文的 chunks 质量。

### 3.3 优先级加权

- 最近编辑过的文件和打开标签页，会在检索中得到更高权重
- 当前文件与光标位置始终作为高优先级上下文纳入

---

## 4. Tab 补全的上下文

Tab completion 是 Cursor 对延迟最敏感的功能。它的上下文工程与 chat 模式根本不同。

### 4.1 自研模型

Cursor 训练并部署自己的小型快速模型来做 tab completion（不是 GPT-4 或 Claude）。速度是第一优先级 — 补全必须在约 200–300ms 内返回。

### 4.2 更小的上下文窗口

tab completion 模型的上下文窗口远小于 chat 模型。每个 token 都非常重要。

### 4.3 上下文组装

| 来源 | 说明 |
|--------|-------------|
| **光标附近的当前文件** | 光标上下文一定范围内的代码 |
| **最近编辑** | 当前文件中最近编辑过的行 / 区域 |
| **其他近期文件** | 最近访问 / 编辑文件中的片段 |
| **Imports and types** | 与当前作用域相关的 import 语句和类型定义 |
| **File metadata** | 语言标识和文件路径 |

### 4.4 Fill-in-the-Middle（FIM）

tab completion 使用 FIM 格式，让模型同时看到光标前后的代码，并预测中间缺失部分：

```
<prefix>code above cursor</prefix>
<suffix>code below cursor</suffix>
<middle>← model predicts this</middle>
```

### 4.5 推测式编辑（Speculative Edits）

Cursor 会根据最近编辑预计算可能的后续修改并进行缓存，因此当你按下 Tab 时，建议几乎可以立即出现。这是一种**预测式上下文预加载**。

---

## 5. 多文件编辑上下文（Composer / Agent Mode）

### 5.1 依赖图感知

Cursor 会分析 imports、references 和 call sites，理解哪些文件彼此相关。当你编辑某个文件时，它会把 import 当前文件或被当前文件 import 的相关文件一并拉入。

### 5.2 代理模式中的工具使用

在 agent mode 中，Cursor 会给 LLM 提供搜索代码库、读取文件、执行终端命令等工具。模型是**按需迭代地收集上下文**，而不是预先一次性喂入全部内容 — 这是一种 tool-use / agentic 模式，而不仅仅是纯 RAG。

### 5.3 基于差异的上下文

在多文件编辑场景下，Cursor 会把它已经在其他文件中做出的 diff / 变更展示给模型，让模型在跨文件修改时保持一致性。

### 5.4 应用模型（Apply Model）

Cursor 使用另一个独立、快速的 “apply” 模型，把 LLM 建议的改动合并到真实文件中。它和生成编辑建议的模型不是同一个模型 — 前者专门负责产出干净 diff 这一机械任务。

---

## 6. Token 预算分配

### 6.1 预算分配

Cursor 会给不同来源分配不同的“token 预算”：

| 来源 | 优先级 | 预算占比 |
|--------|----------|-------------|
| 当前文件 | 最高 | 最大份额 |
| 显式 `@` 引用 | 高 | 保证纳入 |
| 自动检索到的代码库 chunks | 中 | 填充剩余预算 |
| 对话历史 | 低 | 必要时裁剪 |

### 6.2 带优先级的截断

当上下文超出预算时，低优先级内容会先被截断：

```
1. 自动检索的 chunks（最低优先级，最先裁）
2. 更早的对话历史
3. 相关性更低的打开标签页
4. 当前文件（最高优先级，永不裁）
5. 显式 @ 引用（保证纳入，永不裁）
```

---

## 7. Cursor Rules（`.cursorrules`）

### 7.1 项目级规则

Cursor 支持项目级规则文件（`.cursorrules` 或 `.cursor/rules/` 下的文件），它们会把持久指令注入到每个 prompt 中：

- 项目特定规范
- 技术栈细节
- 编码模式与偏好
- 架构决策

### 7.2 实现方式

这些规则会被预置到 system prompt 中，或作为高优先级上下文注入。这与 Anthropic 的 `CLAUDE.md` 模式非常相似。

---

## 8. 架构总结

```
用户动作（tab / chat / composer）
    │
    ▼
上下文组装流水线
    ├── 当前文件 + 光标位置                （Tier 1: Immediate）
    ├── 最近编辑 / 打开的标签页            （Tier 2: Local）
    ├── 显式 @ 引用                        （Tier 2: Local）
    ├── 从代码库索引做向量检索             （Tier 3: Global）
    ├── Reranking（cross-encoder）         （Tier 3: Global）
    ├── Token budgeting & truncation
    └── .cursorrules 注入                  （Tier 4: Static）
    │
    ▼
Prompt Construction
    ├── System prompt（包含规则）
    ├── Retrieved context chunks
    ├── Conversation history（for chat）
    └── User query / 当前代码状态
    │
    ▼
Model Inference
    ├── Tab:   自研快速模型（FIM 格式）
    ├── Chat:  Claude / GPT-4 / etc.
    └── Apply: 自研快速 apply 模型
    │
    ▼
Response Processing
    ├── Tab:      内联建议
    ├── Chat:     流式响应
    └── Composer: 生成 diff + apply model
```

---

## 9. 对 CE 框架设计的关键启示

| 洞见 | 说明 |
|---------|-------------|
| **两阶段检索** | Embedding recall → cross-encoder rerank 远比仅用 embedding 有效 |
| **分层优先级** | 不是所有上下文都等价；显式引用 > 自动检索 > 历史 |
| **按功能定制上下文** | tab completion、chat、composer 三者的上下文需求本质不同 |
| **为速度使用自定义模型** | 对延迟敏感的上下文任务（tab completion、apply）应使用小而快的模型 |
| **推测式预加载** | 预计算下一步最可能需要的上下文，以实现瞬时响应 |
| **AST-aware chunking** | 在函数、类等逻辑边界切分代码，优于固定大小窗口 |
| **环境也是上下文源** | 终端输出、linter errors、IDE signals 都是隐式上下文 |
| **客户端侧组装** | 大量上下文工程工作是在客户端完成后，才把最终 prompt 发给模型 |

---

## 参考资料

- [Cursor Blog — Codebase Indexing](https://cursor.com/blog)
- [Cursor Blog — Speculative Edits](https://cursor.com/blog)
- [Cursor Blog — Custom Models](https://cursor.com/blog)
- [Cursor Documentation](https://docs.cursor.com)
- Lex Fridman Podcast — Interview with Anysphere founders
- Latent Space Podcast — Cursor team interviews
- Various technical analyses and reverse-engineering blog posts
