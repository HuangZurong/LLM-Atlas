# 05 · OpenAI 上下文工程实践

*整理自 OpenAI 官方 API 文档、Cookbook、Agents SDK 文档、GPT-4.1/5/5.2 prompting 指南与公开研究。*

---

## 1. 核心理念

### 1.1 操作系统隐喻

> The LLM is a CPU, the context window is RAM, and the developer's job is to be the operating system — loading working memory with exactly the right code and data for each task. — Andrej Karpathy（2025 年 6 月）

### 1.2 指导原则

在有限注意力预算下，找到**尽可能小的一组高信号 token**，以最大化期望结果出现的概率。研究显示，LLM 的推理表现会在大约 3,000 tokens 左右开始退化，远低于技术上的理论最大值。即便 GPT-5 拥有 272K 的输入窗口，也仍然可能被未经整理的历史、冗余工具结果或带噪检索淹没。

---

## 2. 四种上下文工程策略

| 策略 | 说明 | OpenAI 实现 |
|----------|-------------|----------------------|
| **Write** | 将上下文持久化到外部（scratchpads、notes、memory stores） | `RunContextWrapper` 状态对象、memory distillation tools |
| **Select** | 只检索相关内容（RAG、语义搜索） | Dynamic instructions、tool-based retrieval |
| **Compress** | 只保留必要 token（摘要、紧缩） | Compaction API、session compression、context summarization |
| **Isolate** | 将上下文拆分到不同代理的独立窗口 | Handoffs、agents-as-tools、handoff 上的 `input_filter` |

---

## 3. 系统消息 / 指令设计

### 3.1 通用原则

- 指令必须**清晰、具体、无歧义** — 大多数 prompt 失败并不是模型不够强，而是指令本身含糊
- 避免冲突指令（如既要求 “be brief” 又要求 “be comprehensive”，却不给优先级）
- 避免过长的 system message — 它们会占用上下文窗口，挤占用户内容空间
- 如果输出格式很重要，必须明确写出来
- 生产应用应固定到具体模型快照（如 `gpt-4.1-2025-04-14`）

### 3.2 GPT-4.1 Agentic System Prompt 结构

所有代理型 system prompt 的三个关键提醒：

1. **Persistence**："You are an agent — please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user."
2. **Tool-Calling**："If you're unsure, use your tools to read files, search, or verify before answering."
3. **Planning**："Think step-by-step before every tool call. Reflect after each one. Plan, act, and verify before responding."（这让 SWE-bench 通过率提升了 4%。）

### 3.3 GPT-5.2 的 CTCO 模式

GPT-5.2 推荐的 prompt 结构：

```
C — Context: Background information
T — Task: What to do
C — Constraints: Rules and boundaries
O — Output: Expected format
```

应删除诸如 “Take a deep breath” 或 “You are a world-class expert” 之类的“人格填充”，因为 GPT-5.2 会把它们当噪声。

### 3.4 长上下文中的指令摆放

对于长上下文 prompt，应把指令放在**上下文的开头与结尾**。如果只能放一次，放在上下文之前比放在之后更好。

### 3.5 动态指令模式（Agents SDK）

```python
def dynamic_instructions(context: RunContextWrapper, agent: Agent) -> str:
    return f"The user's name is {context.context.user_name}. Today is {date.today()}."

agent = Agent(
    name="my_agent",
    instructions=dynamic_instructions,  # 函数，而不是静态字符串
)
```

---

## 4. 对话状态管理

### 4.1 Responses API：两种方式

| 方式 | 说明 |
|----------|-------------|
| **Automatic（服务端）** | 通过 `previous_response_id` 串联响应；由 OpenAI 管理历史。所有历史输入 tokens 仍会计费 |
| **Manual（客户端）** | 将输出收集到一个列表中，再作为下次输入提交；你拥有完全控制权 |

### 4.2 面向推理模型的关键规则

当推理模型（o3、o4-mini）涉及工具调用时，你**必须**把 reasoning items 带回对话历史 — 可以通过 `previous_response_id`，也可以显式加入 input。省略它们会明显损害表现。

---

## 5. 上下文压缩（Compaction）

### 5.1 两种模式

**服务端（自动）**：
```python
response = client.responses.create(
    model="gpt-5",
    input=[...],
    context_management={
        "compact_threshold": 50000  # token 数跨过该阈值时触发
    }
)
```

**独立调用（显式）**：直接调用 `/responses/compact` 端点。发送完整窗口，返回压缩后的窗口。

### 5.2 工作机制

- 所有历史 **user messages** 都会原样保留
- 历史 assistant messages、tool calls、tool results 与加密 reasoning 会被替换为一个**单独的加密 compaction item**
- compaction item 会保留模型的潜在理解，同时保持不透明，并兼容 ZDR
- compaction 之后，你可以丢弃最近一个 compaction item 之前的所有条目

### 5.3 压缩后如何继续链式调用

两种模式：
1. **无状态 input-array chaining**：把输出（包括 compaction items）附加到下一次 input array
2. **`previous_response_id` chaining**：每轮只传新的用户消息，并持续携带 ID

---

## 6. Agents SDK 上下文管理

### 6.1 RunContextWrapper：依赖注入

```python
@dataclass
class MyContext:
    user_name: str
    user_id: str
    logger: Logger
    db: DatabaseConnection

context = MyContext(user_name="Alice", user_id="123", logger=logger, db=db)
result = await Runner.run(agent, input="Hello", context=context)
```

关键规则：
- 这个 context 对象**不会发送给 LLM** — 它只用于本地依赖注入
- 一次 run 中的每个 agent、tool、handoff 和生命周期钩子都必须使用**同一种 context type**
- 适用内容：用户元数据、logger、数据库连接、feature flags 等

### 6.2 基于会话的短期记忆

两种技术：

| 技术 | 优点 | 缺点 |
|-----------|------|------|
| **Context Trimming**（保留最后 N 轮） | 确定性、零额外延迟、最近工作逐字保留 | 会突然忘掉长程上下文；关键约束可能消失 |
| **Context Compression**（摘要旧轮次） | 保留长程上下文，还可能修正先前错误（“clean room” 效应） | 会增加延迟（额外一次模型调用），摘要器也可能有波动 |

### 6.3 如何让 LLM 获得这些数据

由于 `RunContextWrapper` 对 LLM 不可见，你必须借助以下之一：
1. **Agent instructions**（system prompt）— 静态字符串或动态函数
2. **Tool outputs** — 从工具返回数据
3. **Conversation history** — 注入到历史消息里

---

## 7. 记忆管理流水线

OpenAI 推荐的四阶段记忆流水线：

### 阶段 1：记忆注入（Memory Injection）

在 session 开始时，只注入与当前任务相关的状态：
- 用 **YAML frontmatter** 表达结构化、机器可读元数据
- 用 **Markdown notes** 表达灵活、可读性高的记忆
- state-based memory 优于 retrieval-based memory（结构化字段具有清晰优先级，而相似度搜索较脆弱）

### 阶段 2：记忆蒸馏（Memory Distillation）

在对话进行过程中，通过专门工具（如 `save_note`）捕捉动态洞见。代理可以在发现新事实时，持续写入 session notes。

### 阶段 3：记忆整合（Memory Consolidation）

在 session 结束后，把 session 级笔记整合成一组紧凑、无冲突的全局记忆：
- 修剪陈旧、被覆盖或低信号记忆
- 随时间进行激进去重
- “先做 note-taking，再做 consolidation”的两阶段流程，比一次性搭完整记忆系统更可靠

### 阶段 4：上下文窗口管理

只保留最后 N 个用户轮次。当发生 trimming 时，在下一轮把 session 级记忆重新注入 system prompt。

---

## 8. 多代理上下文隔离

### 8.1 两种模式

| 模式 | 说明 | 适用场景 |
|---------|-------------|----------|
| **Agents as Tools**（Manager Pattern） | 中央代理将专家代理作为工具调用，保留控制权与上下文 | 专家代理只需完成一个边界明确的子任务，而不应接管对话 |
| **Handoffs**（去中心化） | 一个代理直接把控制权转给另一个；目标代理接收对话历史 | 下一个代理应该主导后续互动 |

### 8.2 通过 `input_filter` 做上下文隔离

```python
from agents.extensions.handoff_filters import remove_all_tools

handoff = Handoff(
    agent=refund_agent,
    input_filter=remove_all_tools  # 从历史中剥离 tool calls
)
```

### 8.3 嵌套交接历史（Beta）

当通过 `RunConfig.nest_handoff_history` 启用后，runner 会把之前的转录折叠成一个 assistant 摘要，并包在 `<CONVERSATION HISTORY>` 块里。你也可以用 `RunConfig.handoff_history_mapper` 自定义映射函数。

---

## 9. 函数调用与工具结果管理

### 9.1 Token 使用效率

- 函数定义会被注入到 system message 中，并且**会计入上下文限制 / 输入 token 计费**
- 降低 token 使用的方法：减少初始加载的函数数、缩短描述、使用 **tool search** 实现延迟加载
- 对函数规范做微调也可减少 token 开销

### 9.2 工具结果过滤

对数据密集型应用，只返回工具调用响应中的关键字段。避免返回成千上万 token 的原始数据，而实际上只用到几个字段。

### 9.3 结构化输出（Structured Outputs）

用 function calling 连接工具；用 `response_format` 约束面向用户的结构化响应。底层会把 JSON Schema 定义转换为上下文无关文法。

---

## 10. 面向成本与延迟优化的缓存

为缓存而组织 prompt：
- **静态内容放前面**：system instructions、few-shot examples、tool definitions
- **可变内容放后面**：用户消息、query-specific data
- OpenAI 提供**自动缓存**，不同模型上可得到 50–90% 折扣
- 在测试中，从 Chat Completions 切换到 Responses API，让缓存利用率从 40% 提升到 80%
- 对 o4-mini 来说，缓存输入 token 比未缓存便宜 **75%**

---

## 11. 安全注意事项

- 在可用时，把不可信数据放进 `untrusted_text` 块
- 引文文本、多模态数据、文件附件和工具输出，默认都被视为无权威性的不可信输入
- 如果格式处理不当，不可信输入中可能包含 prompt injection，而模型很难把它与开发者指令区分开
- 如果你打算持久化或传输序列化状态，不要把 secrets 放在 `RunContextWrapper.context` 里

---

## 12. 以评估驱动上下文工程

OpenAI 一再强调：**“Evals is all you need for context engineering too.”**

- AI engineering 本质上是经验学科；LLM 天生具有非确定性
- 建立信息量高的 evals，并频繁迭代
- 固定到具体模型快照，构建能够衡量 prompt 表现的评估
- 从 `reasoning_effort="medium"` 开始，再根据 eval 结果调参

---

## 参考资料

- [OpenAI Prompt Engineering Guide](https://developers.openai.com/api/docs/guides/prompt-engineering/) — Official API Docs
- [GPT-4.1 Prompting Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide) — Agentic prompting best practices
- [GPT-5 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide) — GPT-5 specific guidance
- [GPT-5.2 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide) — CTCO pattern, enterprise workloads
- [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide) — Compaction for multi-hour reasoning
- [A Practical Guide to Building Agents (PDF)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) — OpenAI 的 32 页代理构建指南
- [Context Engineering for Personalization (Cookbook)](https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization/) — 长期记忆模式
- [Short-Term Memory Management with Sessions (Cookbook)](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory/) — trimming 与 compression
- [Agents SDK Context Management](https://openai.github.io/openai-agents-python/context/) — RunContextWrapper 文档
- [Agents SDK Multi-Agent Orchestration](https://openai.github.io/openai-agents-python/multi_agent/) — handoffs 与 agents-as-tools
- [Compaction Guide](https://developers.openai.com/api/docs/guides/compaction/) — 原生上下文 compaction API
- [Conversation State Guide](https://platform.openai.com/docs/guides/conversation-state) — 跨轮次管理状态
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — 工具定义与 token 管理
