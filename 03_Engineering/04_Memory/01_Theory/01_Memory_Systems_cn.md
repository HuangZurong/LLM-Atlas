# LLM 记忆系统

*前置要求：[../../03_Context_Engineering/01_Theory/01_Context_Window_Mechanics.md](../../03_Context_Engineering/01_Theory/01_Context_Window_Mechanics.md)。*
*另见：[../../05_RAG/01_Theory/01_Architecture.md](../../05_RAG/01_Theory/01_Architecture.md)（将 RAG 视为外部记忆），[../../../02_Scientist/01_Architecture/12_Long_Context.md](../../../02_Scientist/01_Architecture/12_Long_Context.md)（长上下文理论）。*

LLM 默认是无状态的，每一次 API 调用彼此独立，不会记住先前的交互。记忆系统正是用来弥合这一缺口，使其能够在多轮对话与跨会话场景中保持连续性。

---

## 1. 为什么需要记忆？

没有记忆时，每一段对话都要从零开始。记忆要解决三个问题：

| 问题 | 没有记忆时 | 有记忆时 |
|---|---|---|
| **连续性** | “我刚刚问了什么？”模型并不知道 | 模型能够回忆前面的轮次 |
| **个性化** | 对每位用户都给出同样的通用回答 | 能根据用户偏好和历史进行适配 |
| **上下文累积** | 每次都必须重新提供全部上下文 | 系统会自动检索相关上下文 |

根本约束在于：**上下文窗口是有限的**。在多轮对话加工具输出的场景中，128K 的窗口很快就会被填满。记忆，就是应对这一约束的工程解法。

## 2. 记忆的分类体系

### 2.1 短期记忆（上下文内）

指单个会话中的对话历史，通常以发送给模型的消息数组形式保存。

```
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},       # turn 1
    {"role": "assistant", "content": "..."},   # turn 1
    {"role": "user", "content": "..."},       # turn 2
    {"role": "assistant", "content": "..."},   # turn 2
    ...  # grows with every turn
]
```

**当上下文装满时的管理策略**：

| 策略 | 做法 | 取舍 |
|---|---|---|
| **截断** | 丢弃最早的轮次 | 简单，但会失去早期上下文 |
| **滑动窗口** | 只保留最近 N 轮 | 成本可预测，但会丢失长程上下文 |
| **摘要** | 由 LLM 将旧轮次压缩为一段摘要 | 保留关键信息，但有损 |
| **混合** | 旧轮次做摘要 + 最近 N 轮逐字保留 | 在成本与质量之间取得最佳平衡 |

### 2.2 长期记忆（持久化）

指跨会话持久存在的记忆，存放在外部系统中（数据库、向量库）。

**使用场景**：
- 用户偏好（“我更喜欢 Python 而不是 JavaScript”）
- 在先前会话中学到的事实（“用户在 X 公司工作”）
- 项目上下文（“我们正在为法律文档构建一个 RAG 系统”）

**存储方式**：

| 后端 | 最适合 | 查询方式 |
|---|---|---|
| **向量数据库**（Pinecone、Qdrant、Chroma） | 对过往交互做语义检索 | 嵌入相似度搜索 |
| **键值存储**（Redis） | 快速查找结构化事实 | 精确键匹配 |
| **关系型数据库**（PostgreSQL） | 针对结构化记忆做复杂查询 | SQL |
| **图数据库**（Neo4j） | 实体关系 | 图遍历 |

### 2.3 工作记忆（草稿板）

指单个复杂任务执行过程中的临时状态，不会跨会话持久化。

- 智能体草稿板（中间推理步骤）
- 等待综合的工具调用结果
- 多步流水线中的局部输出

## 3. 记忆架构

### 3.1 缓冲记忆

最简单的方法：保存完整对话，超过 token 上限时再截断。

```
if token_count(messages) > MAX_TOKENS:
    messages = messages[-N:]  # keep last N messages
```

**优点**：在窗口内没有信息损失。
**缺点**：一旦发生截断，上下文会突然丢失。对长对话而言成本很高。

### 3.2 摘要记忆

定期将对话压缩为滚动摘要：

```
[Turn 1-10 Summary]: "User asked about RAG architecture. We discussed
 chunking strategies and decided on 512-token chunks with 50-token overlap."
[Turn 11]: (verbatim)
[Turn 12]: (verbatim)
```

**优点**：无论对话多长，记忆占用都能保持恒定。
**缺点**：摘要是有损的，具体细节（数字、代码片段）可能会丢失。

### 3.3 实体记忆

从对话中抽取并追踪命名实体：

```json
{
  "user": {"name": "Alice", "role": "ML Engineer", "preference": "Python"},
  "project": {"name": "LegalRAG", "stack": "LangChain + Qdrant", "status": "prototyping"},
  "decision": {"chunking": "512 tokens", "embedding": "BGE-M3"}
}
```

**优点**：结构化、可查询、无冗余。
**缺点**：抽取并不完美，还需要做实体消歧（“the project” 是否就是 “LegalRAG”？）。

### 3.4 检索增强记忆

将所有过往交互存入向量数据库。对于每个新查询，检索最相关的历史上下文：

```
User: "What chunking strategy did we decide on?"
    →
Embed query → Search vector DB → Retrieve relevant past turns
    →
Inject retrieved context into the prompt
```

**优点**：可扩展到近乎无限长的历史。只取回相关内容。
**缺点**：检索质量依赖于嵌入模型。也可能漏掉那些虽然相关但语义距离较远的上下文。

## 4. 实现模式

### 4.1 将记忆作为系统组件

```
User Query
    →
┌─────────────────────┐
│  Memory Manager     │
│  1. Read: retrieve  │→ query long-term memory
│  2. Inject: add to  │→ add to prompt context
│     prompt context  │
│  3. Write: store    │→ save new interactions
│     new interactions│
└─────────────────────┘
    →
LLM generates response
```

### 4.2 将记忆作为工具

在智能体系统中，记忆可以被暴露为工具，由智能体自主决定何时使用：

```
Tools available:
- memory_read(query): Search past conversations for relevant context
- memory_write(key, value): Store a fact for future reference
- memory_forget(key): Delete a stored fact
```

这让智能体能够显式控制：记住什么，何时回忆。

### 4.3 框架支持

| 框架 | 记忆类型 | 说明 |
|---|---|---|
| **LangChain** | Buffer、Summary、Entity、VectorStore | 最全面，模块化程度最高 |
| **LlamaIndex** | Chat memory + 基于索引的检索 | 与 RAG 紧密集成 |
| **Mem0** | 专门的记忆层 | 专为长期用户记忆设计 |
| **Custom** | 任意组合 | 控制最强，但工程投入更高 |

## 5. 挑战

### 5.1 上下文窗口与完整性

注入更多记忆上下文意味着连续性更好，但也意味着成本更高，并可能带来“中部遗失”问题。注入多少记忆才合适，取决于具体任务。

### 5.2 陈旧记忆

过时信息可能持续存在，并与当前现实相矛盾：
- “用户偏好 React”（但他们在 3 个月前已经转向 Vue）
- 解决办法：为记忆加时间戳，对旧记忆做衰减，并允许显式更新

### 5.3 隐私与被遗忘权

什么应当被记住，什么应当被遗忘？
- 用户可能希望删除特定记忆
- 监管要求（GDPR 的删除权）
- 解决办法：提供由用户控制的显式记忆管理 API

### 5.4 一致性

跨会话可能出现相互冲突的记忆：
- Session 1: “Budget is $10K”
- Session 5: “Budget is $50K”
- 解决办法：为记忆加版本与时间戳，并优先采用最新版本
